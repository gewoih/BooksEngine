"""Объяснение выдачи смеси (TODO п. 11): балл книги — сумма вкладов книг, оценённых человеком.

Смесь линейна по входу, поэтому разбор точный, а не приближение:
- ALS: x = A⁻¹·Σ w_i·y_i (`ALS.fold_in_system`), балл книги j — Σ w_i·(A⁻¹y_i)·y_j (Hu, Koren, Volinsky 2008);
- EASE: балл j — Σ v_i·B[i, j], v_i — вес оценки (1★ −2 … 5★ +2);
- z-нормировка — сдвиг на среднее по 20 000 книг EASE и деление на разброс: вклад i тоже сдвигается
  на своё среднее по этим книгам. Сумма вкладов по i равна баллу `Mix.score` (проверяется тестом).
"""
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from booksengine.model.mix import Mix

# Подпись проверена удалением на 400 тестовых пользователях (docs/resheniya.md, «объяснение выдачи»):
# без названных книг рекомендация выпадает из топ-20 в 86% случаев, без стольких же случайных — в 6%
MAX_BECAUSE = 3          # сколько книг называть в «потому что»
MIN_OF_LEADER = 0.25     # книга называется, если её вклад не меньше этой доли от самого большого
DESPITE_OF_LEADER = 0.5  # отрицательный вклад по модулю не меньше этой доли от самого большого — «несмотря на»


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

    v = mix.ease_inputs(x, dnf)                                   # 1 × 20 000, вес оценки у книг EASE
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
