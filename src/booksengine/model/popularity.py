"""Baseline: популярность (TODO п. 6). Всем один список минус прочитанное.

Байесовское среднее (Σr + m·μ) / (n + m) не даёт книге с тремя пятёрками обогнать «1984».
"""
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from booksengine.model.base import read_params, write_params
from booksengine.model.matrix import RatingMatrix

FORMULAS = ("count", "bayes", "bayes_log")


class Popularity:
    name = "popularity"

    def __init__(self, formula: str = "bayes_log", m: float = 100.0):
        if formula not in FORMULAS:
            raise ValueError(f"formula ∈ {FORMULAS}")
        self.formula, self.m = formula, float(m)
        self.scores_: np.ndarray | None = None

    def fit(self, train: RatingMatrix) -> None:
        n = train.X.getnnz(axis=0).astype(np.float64)
        s = np.asarray(train.X.sum(axis=0), dtype=np.float64).ravel()
        mu = s.sum() / n.sum()
        bayes = (s + self.m * mu) / np.maximum(n + self.m, 1e-12)
        sc = {"count": n, "bayes": bayes, "bayes_log": bayes * np.log1p(n)}[self.formula]
        self.scores_ = sc.astype(np.float32)

    def configure(self, **score_params) -> None:
        if score_params:
            raise ValueError(f"у популярности нет настроек выдачи: {score_params}")

    def score(self, inputs: sp.csr_matrix) -> np.ndarray:
        return np.tile(self.scores_, (inputs.shape[0], 1))

    def save(self, path: Path) -> None:
        write_params(path, {"formula": self.formula, "m": self.m})
        np.save(path / "scores.npy", self.scores_)

    @classmethod
    def load(cls, path: Path) -> "Popularity":
        p = cls(**read_params(path))
        p.scores_ = np.load(path / "scores.npy")
        return p
