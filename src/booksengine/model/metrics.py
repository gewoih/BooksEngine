"""Метрики ранжирования (TODO п. 5, спецификация 3a §4).

Выигрыш скрытой книги: 5★ → 2, 4★ → 1, остальное — 0 (решение пользователя, вариант B).
Дробные оценки (среднее по изданиям) округляются «половина вверх».
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp

from booksengine.model.split import BUCKET_ORDER

K = 20
METRICS = ("ndcg20", "ndcg10", "recall20", "map20", "low20")
_DISC = 1.0 / np.log2(np.arange(2, K + 2))


def rounded(r) -> np.ndarray:
    return np.floor(np.asarray(r, dtype=np.float64) + 0.5)


def gains(r) -> np.ndarray:
    rr = rounded(r)
    return np.where(rr >= 5, 2.0, np.where(rr >= 4, 1.0, 0.0))


def top_k(scores: np.ndarray, exclude: sp.csr_matrix, k: int = K) -> np.ndarray:
    """Столбцы топ-k по убыванию балла без произведений входа; при равенстве — меньший столбец."""
    s = np.array(scores, dtype=np.float32, copy=True)
    ex = exclude.tocoo()
    s[ex.row, ex.col] = -np.inf
    part = np.argpartition(-s, k - 1, axis=1)[:, :k]
    vals = np.take_along_axis(s, part, axis=1)
    order = np.lexsort((part, -vals), axis=1)
    top = np.take_along_axis(part, order, axis=1)
    top[~np.isfinite(np.take_along_axis(vals, order, axis=1))] = -1
    return top


def user_metrics(top: np.ndarray, hidden_cols: np.ndarray, hidden_ratings: np.ndarray) -> dict[str, float]:
    g = gains(hidden_ratings)
    gain_of = dict(zip(hidden_cols.tolist(), g.tolist()))
    hit = np.array([gain_of.get(c, 0.0) for c in top.tolist()])
    ideal = np.sort(g)[::-1]
    out: dict[str, float] = {}
    for name, kk in (("ndcg20", 20), ("ndcg10", 10)):
        best = ideal[:kk]
        idcg = float((best * _DISC[:len(best)]).sum())
        out[name] = float((hit[:kk] * _DISC[:kk]).sum()) / idcg if idcg > 0 else np.nan
    n_rel = int((g > 0).sum())
    rel = hit > 0
    if n_rel:
        denom = min(n_rel, len(top))
        out["recall20"] = float(rel.sum()) / denom
        out["map20"] = float((np.cumsum(rel) / np.arange(1, len(top) + 1))[rel].sum()) / denom
    else:
        out["recall20"] = out["map20"] = np.nan
    low = set(hidden_cols[rounded(hidden_ratings) <= 2].tolist())
    out["low20"] = len(low & set(top.tolist())) / len(low) if low else np.nan
    return out


def coverage(tops: np.ndarray, n_items: int) -> float:
    u = np.unique(tops)
    return len(u[u >= 0]) / n_items


def summarize(per_user: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> dict:
    """Среднее по пользователям и 95% бутстреп-интервал; NaN (нет скрытых 4–5★ / 1–2★) не входят."""
    rng = np.random.default_rng(seed)
    parts = [("all", per_user)] + [(b, per_user[per_user["bucket"] == b]) for b in BUCKET_ORDER]
    out: dict = {}
    for name, part in parts:
        out[name] = {}
        for m in METRICS:
            v = part[m].dropna().to_numpy(dtype=np.float64)
            if len(v) == 0:
                out[name][m] = {"mean": None, "lo": None, "hi": None, "n": 0}
                continue
            boots = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
            out[name][m] = {"mean": float(v.mean()), "lo": float(np.quantile(boots, 0.025)),
                            "hi": float(np.quantile(boots, 0.975)), "n": int(len(v))}
    return out
