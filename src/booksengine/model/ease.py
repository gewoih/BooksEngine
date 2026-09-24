"""EASE^R (TODO п. 7): B = I − P·diag(1/diag P), P = (XᵀX + λI)⁻¹, diag B = 0.

Только top-N самых оценённых произведений (матрица N×N в памяти); остальные получают −∞.
Для C# веса урезаются до top-k на столбец — на диск пишется только урезанная версия.
"""
from pathlib import Path

import numpy as np
import scipy.linalg
import scipy.sparse as sp

from booksengine.model.base import read_params, write_params
from booksengine.model.matrix import RatingMatrix


class EASE:
    name = "ease"

    def __init__(self, lam: float = 500.0, n_top: int = 30_000, block: int = 2_000):
        self.lam, self.n_top, self.block = float(lam), int(n_top), int(block)
        self.topk: int | None = None
        self.top_cols: np.ndarray | None = None
        self.n_items = 0
        self.B_full: np.ndarray | None = None
        self._B: np.ndarray | sp.csc_matrix | None = None

    def fit(self, train: RatingMatrix) -> None:
        counts = train.X.getnnz(axis=0)
        n = min(self.n_top, len(counts))
        self.top_cols = np.sort(np.argsort(-counts, kind="stable")[:n])
        self.n_items = train.X.shape[1]
        Xb = train.X[:, self.top_cols].tocsc()
        Xb.data[:] = 1.0
        XbT = Xb.T.tocsr()
        G = np.empty((n, n), dtype=np.float32)
        for a in range(0, n, self.block):  # XᵀX блоками: целиком разреженный результат не влезает в память
            b = min(a + self.block, n)
            G[:, a:b] = (XbT @ Xb[:, a:b]).toarray()
        del Xb, XbT
        G[np.diag_indices(n)] += self.lam
        # Одна плотная копия n × n на всё обучение (TODO п. 32): LAPACK обращает на месте только матрицу, лежащую
        # по столбцам, а построчную G scipy молча копирует. G симметрична точно (целые счётчики во float32),
        # поэтому G.T — та же матрица по столбцам: результат побитово тот же, что inv(G), без копии.
        P = scipy.linalg.inv(G.T, overwrite_a=True, check_finite=False)
        if not np.shares_memory(P, G):
            raise MemoryError("EASE: обращение не на месте — вторая копия матрицы n × n")
        del G
        P /= -np.diag(P)[None, :]            # B = I − P·diag(1/diag P) вне диагонали, деление на месте
        np.fill_diagonal(P, 0.0)
        self.B_full = P
        self.configure(topk=None)

    def configure(self, topk: int | None = None) -> None:
        self.topk = None if topk is None else int(topk)
        if self.topk is None:
            self._B = self.B_full
            return
        n = self.B_full.shape[0]
        k = min(self.topk, n)
        rows, cols, vals = [], [], []
        for a in range(0, n, self.block):
            b = min(a + self.block, n)
            blk = self.B_full[:, a:b]
            part = np.argpartition(-np.abs(blk), k - 1, axis=0)[:k]
            rows.append(part.ravel(order="F"))
            cols.append(np.repeat(np.arange(a, b), k))
            vals.append(np.take_along_axis(blk, part, axis=0).ravel(order="F"))
        self._B = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
        self._B.eliminate_zeros()

    def score(self, inputs: sp.csr_matrix) -> np.ndarray:
        Xin = inputs[:, self.top_cols]
        Xin.data[:] = 1.0
        s = Xin @ self._B
        s = s.toarray() if sp.issparse(s) else np.asarray(s)
        out = np.full((inputs.shape[0], self.n_items), -np.inf, dtype=np.float32)
        out[:, self.top_cols] = s
        return out

    def save(self, path: Path) -> None:
        if self.topk is None:
            raise ValueError("полная матрица EASE на диск не пишется: сначала configure(topk=...)")
        write_params(path, {"lam": self.lam, "n_top": self.n_top, "block": self.block, "topk": self.topk,
                            "n_items": self.n_items})
        np.save(path / "top_cols.npy", self.top_cols)
        sp.save_npz(path / "B.npz", self._B.tocsc())

    @classmethod
    def load(cls, path: Path) -> "EASE":
        p = read_params(path)
        m = cls(lam=p["lam"], n_top=p["n_top"], block=p["block"])
        m.topk, m.n_items = p["topk"], p["n_items"]
        m.top_cols = np.load(path / "top_cols.npy")
        m._B = sp.load_npz(path / "B.npz").tocsc()
        return m


class EASELike(EASE):
    """Толпа, которая целится в оценку (TODO п. 38): вход — звёзды с весами, как при выдаче (1★ −2 … 5★ +2),
    цель — «оценка − 3» у прочитанных, 0 у непрочитанных, то есть ровно «ценность топа», по которой выбирается
    выдача (docs/resheniya.md). EASE учился «прочтёт ли»; этот — «сколько ценности даст».

    B = argmin |Y − Xw·B|² + λ|B|², diag B = 0: B = P·XwᵀY − P·diag(μ), P = (XwᵀXw + λI)⁻¹,
    μ_j = (P·XwᵀY)_jj / P_jj (формула сверена с прямым решением по столбцам, tests/test_ease.py).
    Столбцы B считаются блоками и сразу урезаются до topk соседей: вторая плотная матрица n × n не нужна.
    Формат на диске — как у EASE: смесь (`Mix`) подаёт в него те же взвешенные звёзды.
    """
    name = "ease_like"

    def __init__(self, lam: float = 500.0, n_top: int = 30_000, block: int = 2_000, topk: int = 500,
                 weights=(-2.0, -1.0, 0.0, 1.0, 2.0)):
        super().__init__(lam, n_top, block)
        self.topk_target, self.weights = int(topk), tuple(float(w) for w in weights)

    def fit(self, train: RatingMatrix) -> None:
        from booksengine.model.metrics import rounded
        counts = train.X.getnnz(axis=0)
        n = min(self.n_top, len(counts))
        self.top_cols = np.sort(np.argsort(-counts, kind="stable")[:n])
        self.n_items = train.X.shape[1]
        X = train.X[:, self.top_cols].tocsc()
        r = rounded(X.data).astype(int)
        Xw = X.copy()
        Xw.data = np.asarray(self.weights, dtype=np.float32)[r - 1]
        Xw.eliminate_zeros()
        Y = X.copy()
        Y.data = (r - 3).astype(np.float32)
        Y.eliminate_zeros()
        del X
        XwT = Xw.T.tocsr()
        G = np.empty((n, n), dtype=np.float32)
        for a in range(0, n, self.block):
            b = min(a + self.block, n)
            G[:, a:b] = (XwT @ Xw[:, a:b]).toarray()
        G[np.diag_indices(n)] += self.lam
        P = scipy.linalg.inv(G.T, overwrite_a=True, check_finite=False)  # G симметрична точно — см. EASE.fit
        if not np.shares_memory(P, G):
            raise MemoryError("EASELike: обращение не на месте — вторая копия матрицы n × n")
        del G, Xw
        diag = np.diag(P).copy()
        k = min(self.topk_target, n)
        rows, cols, vals = [], [], []
        for a in range(0, n, self.block):
            b = min(a + self.block, n)
            D = P @ (XwT @ Y[:, a:b]).toarray()
            idx = np.arange(b - a)
            D -= P[:, a:b] * (D[a + idx, idx] / diag[a:b])[None, :]
            D[a + idx, idx] = 0.0
            part = np.argpartition(-np.abs(D), k - 1, axis=0)[:k]
            rows.append(part.ravel(order="F"))
            cols.append(np.repeat(np.arange(a, b), k))
            vals.append(np.take_along_axis(D, part, axis=0).ravel(order="F"))
        self._B = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
        self._B.eliminate_zeros()
        self.topk, self.B_full = k, None

    def configure(self, topk: int | None = None) -> None:
        if topk is not None and topk != self.topk:
            raise ValueError("EASELike урезается при обучении: другой topk — переобучить")

    def save(self, path: Path) -> None:
        super().save(path)
        p = read_params(path)
        write_params(path, p | {"target": "rating-3", "weights": list(self.weights)})
