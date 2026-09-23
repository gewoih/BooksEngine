"""Матрица оценок «пользователь × произведение» и отложенные группы в том же пространстве столбцов.

Столбцы — всё CF-ядро (distinct work_id в ratings.parquet) по возрастанию, в том числе книги, у которых
после изъятия тестовых пользователей не осталось оценок в обучении: они остаются кандидатами.
"""
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass
class RatingMatrix:
    X: sp.csr_matrix       # пользователи × произведения, float32, оценка 1–5
    user_ids: np.ndarray   # строка → user_id
    work_ids: np.ndarray   # столбец → work_id


@dataclass
class Holdout:
    user_ids: np.ndarray
    buckets: np.ndarray
    inputs: sp.csr_matrix          # вход fold-in, те же столбцы, что у обучения
    hidden_cols: list[np.ndarray]  # скрытые произведения пользователя (номера столбцов)
    hidden_ratings: list[np.ndarray]
    exclude: sp.csr_matrix | None = None  # не советовать: вход и продолжения начатых серий; None — только вход


def columns(work_ids: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    cols = np.minimum(np.searchsorted(work_ids, wanted), len(work_ids) - 1)
    if not np.array_equal(work_ids[cols], wanted):
        raise KeyError("произведение вне CF-ядра")
    return cols


def to_csr(user_id, work_id, rating, work_ids: np.ndarray) -> tuple[np.ndarray, sp.csr_matrix]:
    users, rows = np.unique(np.asarray(user_id), return_inverse=True)
    X = sp.csr_matrix((np.asarray(rating, dtype=np.float32), (rows, columns(work_ids, np.asarray(work_id)))),
                      shape=(len(users), len(work_ids)))
    X.sort_indices()
    return users, X


def catalog_works(ratings_path: Path) -> np.ndarray:
    return duckdb.execute("SELECT DISTINCT work_id FROM read_parquet(?) ORDER BY 1",
                          [str(ratings_path)]).fetchnumpy()["work_id"]


def load_train(ratings_path: Path, holdout_path: Path) -> RatingMatrix:
    """Все оценки, кроме пользователей валидации и теста."""
    work_ids = catalog_works(ratings_path)
    d = duckdb.execute("SELECT r.user_id, r.work_id, r.rating FROM read_parquet(?) r "
                       "ANTI JOIN read_parquet(?) h USING (user_id)",
                       [str(ratings_path), str(holdout_path)]).fetchnumpy()
    users, X = to_csr(d["user_id"], d["work_id"], d["rating"], work_ids)
    return RatingMatrix(X, users, work_ids)


def load_holdout(split_dir: Path, group: str, work_ids: np.ndarray) -> Holdout:
    inp = pd.read_parquet(split_dir / f"{group}_input.parquet")
    hid = pd.read_parquet(split_dir / f"{group}_hidden.parquet").sort_values(["user_id", "work_id"])
    users = pd.read_parquet(split_dir / "holdout_users.parquet")
    users = users[users["group"] == group].sort_values("user_id")
    ids, X = to_csr(inp.user_id.to_numpy(), inp.work_id.to_numpy(), inp.rating.to_numpy(), work_ids)
    if not np.array_equal(ids, users.user_id.to_numpy()):
        raise ValueError(f"вход группы {group} и holdout_users.parquet разошлись")
    cols = columns(work_ids, hid.work_id.to_numpy())
    ratings = hid.rating.to_numpy()
    hu = hid.user_id.to_numpy()
    starts, ends = np.searchsorted(hu, ids, side="left"), np.searchsorted(hu, ids, side="right")
    return Holdout(ids, users.bucket.to_numpy(), X,
                   [cols[s:e] for s, e in zip(starts, ends)], [ratings[s:e] for s, e in zip(starts, ends)])
