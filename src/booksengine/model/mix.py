"""Смесь ALS и EASE^R — основная модель (TODO п. 24, docs/resheniya.md, «смесь ALS и EASE»).

Балл книги = w·z(ALS) + (1 − w)·z(EASE), z — нормировка баллов человека по 20 000 книг EASE (среднее 0,
разброс 1); остальные книги ядра — −∞: EASE их не знает, а ALS почти не советует (0.2% рекомендаций).
EASE не видит оценок, поэтому вход ему взвешен по оценке: 1★ −2, 2★ −1, 3★ 0, 4★ 1, 5★ 2 (решение
пользователя) — без этого он советует книги, похожие на оценённые низко, и «Сумерки» за «Голодные игры».

Своего обучения нет: `fit` загружает готовые компоненты (models/als_neg, models/ease) и запоминает их
отпечатки; если компонент переобучен, `load` падает — иначе смесь молча стала бы другой моделью.
"""
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from booksengine.model import metrics
from booksengine.model.als import ALS
from booksengine.model.base import fingerprint, read_params, write_params
from booksengine.model.ease import EASE
from booksengine.model.matrix import RatingMatrix

EASE_INPUT = (-2.0, -1.0, 0.0, 1.0, 2.0)  # вес книги входа EASE по оценке 1..5


def _z(m: np.ndarray) -> np.ndarray:
    return (m - m.mean(axis=1, keepdims=True)) / np.maximum(m.std(axis=1, keepdims=True), 1e-9)


class Mix:
    name = "mix"

    def __init__(self, als_dir: str, ease_dir: str):
        self.als_dir, self.ease_dir = str(als_dir), str(ease_dir)
        self.als_weight, self.ease_input = 0.5, EASE_INPUT
        self.als: ALS | None = None
        self.ease: EASE | None = None
        self._fps: dict[str, str] = {}

    def _load_components(self) -> None:
        self.als, self.ease = ALS.load(Path(self.als_dir)), EASE.load(Path(self.ease_dir))
        if self.als.item_factors.shape[0] != self.ease.n_items:
            raise ValueError(f"ALS ({self.als.item_factors.shape[0]} книг) и EASE ({self.ease.n_items}) "
                             "обучены на разных каталогах")
        self._fps = {"als": fingerprint(Path(self.als_dir)), "ease": fingerprint(Path(self.ease_dir))}

    def fit(self, train: RatingMatrix) -> None:
        self._load_components()
        if self.ease.n_items != train.X.shape[1]:
            raise ValueError(f"компоненты обучены на {self.ease.n_items} книгах, а в обучающем наборе их "
                             f"{train.X.shape[1]}: переобучите als_neg и ease")

    def configure(self, als_weight: float = 0.5, ease_input=EASE_INPUT) -> None:
        if not 0.0 <= als_weight <= 1.0 or len(ease_input) != 5:
            raise ValueError("als_weight — от 0 до 1, ease_input — пять весов для оценок 1..5")
        self.als_weight, self.ease_input = float(als_weight), tuple(float(v) for v in ease_input)

    def ease_inputs(self, inputs: sp.csr_matrix) -> sp.csr_matrix:
        """Вход EASE: столбцы — 20 000 книг EASE, значение — вес по оценке."""
        weighted = inputs[:, self.ease.top_cols].tocsr()
        weighted.data = np.asarray(self.ease_input, np.float32)[metrics.rounded(weighted.data).astype(int) - 1]
        weighted.eliminate_zeros()  # 3★ с весом 0 — как не поданная книга
        return weighted

    def components(self, inputs: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
        """Баллы ALS и EASE по 20 000 книг EASE (до нормировки)."""
        s_ease = self.ease_inputs(inputs) @ self.ease._B
        s_ease = s_ease.toarray() if sp.issparse(s_ease) else np.asarray(s_ease)
        s_als = self.als.score(inputs)[:, self.ease.top_cols]
        return s_als.astype(np.float64), s_ease.astype(np.float64)

    def score(self, inputs: sp.csr_matrix) -> np.ndarray:
        s_als, s_ease = self.components(inputs)
        out = np.full(inputs.shape, -np.inf, dtype=np.float32)
        out[:, self.ease.top_cols] = self.als_weight * _z(s_als) + (1 - self.als_weight) * _z(s_ease)
        return out

    def save(self, path: Path) -> None:
        write_params(path, {"als_dir": self.als_dir, "ease_dir": self.ease_dir, "als_weight": self.als_weight,
                            "ease_input": list(self.ease_input), "component_fingerprints": self._fps})

    @classmethod
    def load(cls, path: Path) -> "Mix":
        p = read_params(path)
        m = cls(p["als_dir"], p["ease_dir"])
        m._load_components()
        if m._fps != p["component_fingerprints"]:
            raise ValueError(f"{path}: ALS или EASE переобучены после сборки смеси — пересоберите "
                             "(`evaluate mix --stage val`) и пересчитайте `calibrate mix`")
        m.configure(als_weight=p["als_weight"], ease_input=p["ease_input"])
        return m
