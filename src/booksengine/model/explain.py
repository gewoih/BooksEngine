"""Объяснение выдачи смеси и слоёв «толпа + вкус»: балл книги — сумма вкладов книг, оценённых человеком.

Смесь линейна по входу, поэтому разбор точный, а не приближение:
- ALS: x = A⁻¹·Σ w_i·y_i (`ALS.fold_in_system`), балл книги j — Σ w_i·(A⁻¹y_i)·y_j (Hu, Koren, Volinsky 2008);
- EASE: балл j — Σ v_i·B[i, j], v_i — вес оценки (1★ −2 … 5★ +2);
- z-нормировка — сдвиг на среднее по 30 000 книг EASE и деление на разброс: вклад i тоже сдвигается
  на своё среднее по этим книгам. Сумма вкладов по i равна баллу `Mix.score` (проверяется тестом).
"""
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from booksengine.model.mix import Mix

# Подпись проверена удалением на 400 тестовых пользователях:
# без названных книг рекомендация выпадает из топ-20 в 86% случаев, без стольких же случайных — в 6%
MAX_BECAUSE = 3          # сколько книг называть в «потому что»
MIN_OF_LEADER = 0.25     # книга называется, если её вклад не меньше этой доли от самого большого
DESPITE_OF_LEADER = 0.5  # отрицательный вклад по модулю не меньше этой доли от самого большого — «несмотря на»
# вкус называется, если сдвигает балл книги хотя бы на столько (z-балл с весом вкуса; разброс вкуса по книгам — ~1.5)
TASTE_SHOWN = 0.5


def contributions(mix: Mix, x: sp.csr_matrix, cols: np.ndarray,
                  dnf: sp.csr_matrix | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Вклады книг входа одного человека (x — строка 1 × книги ядра, dnf — недочитанные, как в `Mix.score`)
    в балл смеси книг cols. Возвращает (столбцы входа, матрица вкладов вход × cols). cols — только книги EASE."""
    top = mix.ease.top_cols
    pos = np.searchsorted(top, cols)
    if not np.array_equal(top[np.minimum(pos, len(top) - 1)], cols):
        raise ValueError("объясняются только книги EASE: у остальных нет балла смеси")
    in_cols = x.indices
    s_als, s_ease = mix.components(x, dnf)
    w_als = mix.als_weight

    A, Yu, w = mix.als.fold_in_system(in_cols, x.data)
    G = np.linalg.solve(A, Yu.T).T * w[:, None]                 # вклад книги i в вектор человека
    Y = mix.als.item_factors.astype(np.float64)
    c_als = G @ Y[cols].T - (G @ Y[top].mean(axis=0))[:, None]

    v = mix.ease_inputs(x, dnf)                                   # 1 × 30 000, вес оценки у книг EASE
    rows = sp.csr_matrix((v.data, (np.searchsorted(in_cols, top[v.indices]), v.indices)),
                         shape=(len(in_cols), len(top)))       # вход × книги EASE
    c_ease_all = rows @ mix.ease._B
    c_ease_all = c_ease_all.toarray() if sp.issparse(c_ease_all) else np.asarray(c_ease_all)
    c_ease = c_ease_all[:, pos] - c_ease_all.mean(axis=1, keepdims=True)

    sd_als, sd_ease = (max(float(s.std()), 1e-9) for s in (s_als[0], s_ease[0]))
    return in_cols, w_als * c_als / sd_als + (1 - w_als) * c_ease / sd_ease


@dataclass
class Reason:
    because: list[int]      # номера книг входа, по убыванию вклада; пусто — ни одна не тянет вверх
    despite: int | None     # книга входа, которая сильнее всех тянет вниз, если тянет заметно


def reason(c: np.ndarray) -> Reason:
    """Подпись к одной книге по вкладам c (по книгам входа)."""
    order = np.argsort(-c, kind="stable")
    leader = c[order[0]] if len(c) else 0.0
    if leader <= 0:
        return Reason([], None)
    low = int(np.argmin(c))
    return Reason([int(i) for i in order[:MAX_BECAUSE] if c[i] >= MIN_OF_LEADER * leader],
                  low if -c[low] >= DESPITE_OF_LEADER * leader else None)


def taste_contributions(taste, x: sp.csr_matrix, cols: np.ndarray, top: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Вклады оценённых книг в z-балл модели вкуса книг cols (z — по книгам top) и постоянная часть книги.

    Прогноз p_j = μ + b_j + [1, y_j]·θ, θ = A⁻¹·Σ z_i·d_i (`Taste.fold_in_system`), d_i — «насколько выше или
    ниже, чем книгу обычно оценивают». После нормировки: вклад i = d_i·([1, y_j] − среднее)·A⁻¹z_i / разброс,
    постоянная часть — (b_j − среднее b) / разброс («как книгу обычно оценивают»); личная щедрость сокращается."""
    in_cols = x.indices
    A, Z, d = taste.fold_in_system(in_cols, x.data)
    F = taste._F().astype(np.float64)
    Ft = F[top]
    p_top = taste.mu + taste.item_bias[top] + Ft @ np.linalg.solve(A, Z.T @ d)
    sd = max(float(p_top.std()), 1e-9)
    G = np.linalg.solve(A, Z.T)                                  # (k+1) × входа
    c = ((F[cols] - Ft.mean(axis=0)) @ G).T * d[:, None] / sd     # вход × cols
    const = (taste.item_bias[cols] - taste.item_bias[top].mean()) / sd
    return c, const


def layers_parts(layers, x: sp.csr_matrix, cols: np.ndarray,
                 dnf: sp.csr_matrix | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Вклады книг входа в балл слоёв «толпа + вкус» (`Layers.score`) книг cols — отдельно толпа и вкус (с его весом),
    и постоянная часть вкуса («как книгу обычно оценивают»). Толпа + вкус + постоянная = балл, точно.
    Возвращает (вход, толпа, вкус, постоянная)."""
    v = layers.variant
    if v.crowd == "like":
        m, w = layers.like_mix, v.als_weight
    elif v.crowd == "mix":
        m, w = layers.mix, layers.mix.als_weight
    else:
        raise ValueError(f"объяснение для толпы «{v.crowd}» не сделано")
    w0 = m.als_weight
    try:
        m.configure(als_weight=w, ease_input=m.ease_input)
        in_cols, c = contributions(m, x, cols, dnf)
    finally:
        m.configure(als_weight=w0, ease_input=m.ease_input)
    t, const = np.zeros_like(c), np.zeros(len(cols))
    if v.taste_weight:
        from booksengine.model.layers import without_dnf
        xt = without_dnf(x, dnf)                  # недочитанные во вкус не подаются — их вклад во вкус 0
        ct, kt = taste_contributions(layers.taste, xt, cols, m.ease.top_cols)
        t[np.searchsorted(in_cols, xt.indices)] = v.taste_weight * ct
        const = v.taste_weight * kt
    return in_cols, c, t, const


def layers_contributions(layers, x: sp.csr_matrix, cols: np.ndarray,
                         dnf: sp.csr_matrix | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Вклады книг входа в балл слоёв (толпа и вкус вместе) и постоянная часть. Возвращает (вход, вклады, постоянная)."""
    in_cols, c, t, const = layers_parts(layers, x, cols, dnf)
    return in_cols, c + t, const


@dataclass
class TasteNote:
    sign: int               # +1 — вкус за книгу, −1 — против
    book: int | None        # номер книги входа, сильнее всех сдвигающей балл; None — «как книгу обычно оценивают»


def taste_note(t: np.ndarray, const: float) -> TasteNote | None:
    """Что вкус говорит о книге по вкладам t (по книгам входа, с весом вкуса) и постоянной части. Молчит, если вкус
    сдвигает балл меньше чем на TASTE_SHOWN."""
    total = float(t.sum() + const)
    if abs(total) < TASTE_SHOWN:
        return None
    s = 1 if total > 0 else -1
    i = int(np.argmax(s * t)) if len(t) else -1
    if i < 0 or s * const >= s * t[i]:
        return TasteNote(s, None)
    return TasteNote(s, i)
