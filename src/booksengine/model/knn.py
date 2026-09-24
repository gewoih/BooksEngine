"""item-kNN на центрированных оценках.

Похожесть книг i, j — косинус векторов (r − μ_u), вклад пользователя взвешен 1/log(1 + n_u)
(понижение веса активных, решение этапа 1), с усадкой n_common/(n_common + β). Хранятся k_max соседей
по модулю похожести со знаком. Выдача: score_i = Σ_j s_ij·(r_uj − μ_u) по книгам входа j
(с normalize — делённое на Σ|s_ij|). Ничьи (например, вход из одних пятёрок) разрываются популярностью.
"""
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from booksengine.model.base import read_params, write_params
from booksengine.model.matrix import RatingMatrix

TIE_EPS = 1e-6


def _center(X: sp.csr_matrix) -> sp.csr_matrix:
    n = np.diff(X.indptr)
    mu = np.asarray(X.sum(axis=1)).ravel() / np.maximum(n, 1)
    C = X.astype(np.float32, copy=True)
    C.data -= np.repeat(mu, n).astype(np.float32)
    return C


class ItemKNN:
    name = "knn"

    def __init__(self, beta: float = 0.0, k_max: int = 200, block: int = 500):
        self.beta, self.k_max, self.block = float(beta), int(k_max), int(block)
        self.k, self.normalize = self.k_max, False
        self.nbr_idx: np.ndarray | None = None
        self.nbr_sim: np.ndarray | None = None
        self.pop_: np.ndarray | None = None
        self._S: sp.csr_matrix | None = None

    def fit(self, train: RatingMatrix) -> None:
        X = train.X
        n_items = X.shape[1]
        k = min(self.k_max, n_items - 1)
        n_u = np.diff(X.indptr)
        w = np.sqrt(1.0 / np.log1p(np.maximum(n_u, 1))).astype(np.float32)
        Z = sp.diags(w) @ _center(X)
        Z = Z.tocsr()
        norms = np.sqrt(np.asarray(Z.multiply(Z).sum(axis=0)).ravel())
        inv = np.where(norms > 0, 1.0 / np.where(norms > 0, norms, 1.0), 0.0).astype(np.float32)
        ZT = Z.T.tocsr()
        if self.beta:
            B = X.copy()
            B.data[:] = 1.0
            BT = B.T.tocsr()
        idx = np.zeros((n_items, k), dtype=np.int32)
        sim = np.zeros((n_items, k), dtype=np.float32)
        for a in range(0, n_items, self.block):
            b = min(a + self.block, n_items)
            D = (ZT[a:b] @ Z).toarray().astype(np.float32, copy=False)
            D *= inv[a:b, None]
            D *= inv[None, :]
            if self.beta:
                Cn = (BT[a:b] @ B).toarray().astype(np.float32, copy=False)
                D *= Cn / (Cn + self.beta)
            D[np.arange(b - a), np.arange(a, b)] = 0.0
            part = np.argpartition(-np.abs(D), k - 1, axis=1)[:, :k]
            vals = np.take_along_axis(D, part, axis=1)
            order = np.lexsort((part, -np.abs(vals)), axis=1)
            idx[a:b] = np.take_along_axis(part, order, axis=1)
            sim[a:b] = np.take_along_axis(vals, order, axis=1)
            print(f"  kNN: {b}/{n_items}", end="\r", flush=True)
        counts = X.getnnz(axis=0).astype(np.float32)
        self.nbr_idx, self.nbr_sim = idx, sim
        self.pop_ = counts / max(float(counts.max()), 1.0)
        self.configure(k=k, normalize=False)

    def configure(self, k: int | None = None, normalize: bool = False) -> None:
        k = self.nbr_idx.shape[1] if k is None else int(k)
        if k > self.nbr_idx.shape[1]:
            raise ValueError(f"k={k} > сохранённых соседей {self.nbr_idx.shape[1]}")
        self.k, self.normalize = k, bool(normalize)
        n = self.nbr_idx.shape[0]
        rows = np.repeat(np.arange(n), k)
        self._S = sp.csr_matrix((self.nbr_sim[:, :k].ravel(), (rows, self.nbr_idx[:, :k].ravel())), shape=(n, n))

    def score(self, inputs: sp.csr_matrix) -> np.ndarray:
        C = _center(inputs)
        num = (C @ self._S.T).toarray()
        if self.normalize:
            Bn = inputs.copy()
            Bn.data[:] = 1.0
            den = (Bn @ abs(self._S).T).toarray()
            num = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        return (num + TIE_EPS * self.pop_).astype(np.float32)

    def save(self, path: Path) -> None:
        write_params(path, {"beta": self.beta, "k_max": self.k_max, "block": self.block,
                            "k": self.k, "normalize": self.normalize})
        np.savez(path / "neighbors.npz", idx=self.nbr_idx, sim=self.nbr_sim, pop=self.pop_)

    @classmethod
    def load(cls, path: Path) -> "ItemKNN":
        p = read_params(path)
        m = cls(beta=p["beta"], k_max=p["k_max"], block=p["block"])
        d = np.load(path / "neighbors.npz")
        m.nbr_idx, m.nbr_sim, m.pop_ = d["idx"], d["sim"], d["pop"]
        m.configure(k=p["k"], normalize=p["normalize"])
        return m
