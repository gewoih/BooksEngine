"""ALS на неявном сигнале с уверенностью c = 1 + α·r (TODO п. 7).

Обучение — `implicit`. Fold-in нового человека — своя явная формула (её повторит C#):
x_u = (YᵀY + Yᵀ(C_u − I)Y + λI)⁻¹ · YᵀC_u·p_u, где p_u = 1 на оценённых книгах, C_u − I = diag(α·r).

Негативный сигнал (TODO п. 8) — настройка выдачи, обучение не меняет: низкая оценка по правилу
`neg_rule` получает p = −1 и уверенность c = 1 + β (`neg_weight`) — вектор человека отталкивается
от соседей книги, а не притягивается. Правило «le2» — r ≤ 2 («не понравилось»).
"""
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from booksengine.model.base import read_params, write_params
from booksengine.model.matrix import RatingMatrix
from booksengine.model.split import SEED

NEG_RULES = ("none", "le2")


def negative_mask(r: np.ndarray, rule: str) -> np.ndarray:
    return r <= 2.0 if rule == "le2" else np.zeros(len(r), dtype=bool)


class ALS:
    name = "als"

    def __init__(self, factors: int = 128, regularization: float = 0.1, alpha: float = 10.0,
                 iterations: int = 15, seed: int = SEED):
        self.factors, self.regularization, self.alpha = int(factors), float(regularization), float(alpha)
        self.iterations, self.seed = int(iterations), int(seed)
        self.neg_rule, self.neg_weight = "none", 0.0
        self.item_factors: np.ndarray | None = None
        self._YtY: np.ndarray | None = None
        self._model = None

    def _params(self) -> dict:
        return {"factors": self.factors, "regularization": self.regularization, "alpha": self.alpha,
                "iterations": self.iterations, "seed": self.seed}

    def fit(self, train: RatingMatrix) -> None:
        from implicit.cpu.als import AlternatingLeastSquares
        from threadpoolctl import threadpool_limits
        conf = train.X.astype(np.float32, copy=True)
        conf.data = 1.0 + self.alpha * conf.data  # implicit: значение матрицы и есть уверенность c
        model = AlternatingLeastSquares(factors=self.factors, regularization=self.regularization, alpha=1.0,
                                        iterations=self.iterations, random_state=self.seed)
        with threadpool_limits(1, "blas"):  # implicit параллелит сам; BLAS-потоки ему мешают
            model.fit(conf, show_progress=False)
        self._model = model
        self._set_items(np.asarray(model.item_factors, dtype=np.float32))

    def _set_items(self, Y: np.ndarray) -> None:
        self.item_factors = Y
        Y64 = Y.astype(np.float64)
        self._YtY = Y64.T @ Y64

    def configure(self, neg_rule: str = "none", neg_weight: float = 0.0) -> None:
        if neg_rule not in NEG_RULES:
            raise ValueError(f"neg_rule: {' | '.join(NEG_RULES)}, а не {neg_rule!r}")
        self.neg_rule, self.neg_weight = neg_rule, float(neg_weight)

    def fold_in(self, inputs: sp.csr_matrix) -> np.ndarray:
        Y = self.item_factors.astype(np.float64)
        reg = self.regularization * np.eye(Y.shape[1])
        out = np.zeros((inputs.shape[0], Y.shape[1]))
        for u in range(inputs.shape[0]):
            s, e = inputs.indptr[u], inputs.indptr[u + 1]
            if s == e:
                continue
            Yu = Y[inputs.indices[s:e]]
            r = inputs.data[s:e].astype(np.float64)
            neg = negative_mask(r, self.neg_rule)
            m = np.where(neg, self.neg_weight, self.alpha * r)  # c − 1
            p = np.where(neg, -1.0, 1.0)
            out[u] = np.linalg.solve(self._YtY + (Yu.T * m) @ Yu + reg, Yu.T @ ((1.0 + m) * p))
        return out

    def score(self, inputs: sp.csr_matrix) -> np.ndarray:
        return (self.fold_in(inputs) @ self.item_factors.T.astype(np.float64)).astype(np.float32)

    def save(self, path: Path) -> None:
        write_params(path, {**self._params(), "neg_rule": self.neg_rule, "neg_weight": self.neg_weight})
        np.save(path / "item_factors.npy", self.item_factors)

    @classmethod
    def load(cls, path: Path) -> "ALS":
        p = read_params(path)
        neg = {k: p.pop(k) for k in ("neg_rule", "neg_weight") if k in p}
        m = cls(**p)
        m._set_items(np.load(path / "item_factors.npy"))
        m.configure(**neg)
        return m
