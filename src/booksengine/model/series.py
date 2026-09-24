"""Серии книг: продолжения начатой серии не советуются и не считаются попаданием (решение 2026-09-23).

Серия берётся из названия Goodreads «Title (Series, #N)» — поле `series` изданий без справочника серий
не группирует (у каждой книги «Гарри Поттера» свой id). Бокс-сет «(Harry Potter, #1-7)» — та же серия:
оценка серии целиком закрывает все её части.
"""
import re
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import duckdb
import numpy as np
import scipy.sparse as sp

from booksengine.model.matrix import Holdout

_PAREN = re.compile(r"\(([^()]*#[^()]*)\)")
_PART = re.compile(r"^\s*(.+?),?\s*#\s*[\d.]+(?:\s*[-–]\s*[\d.]+)?\s*$")


def series_keys(title: str) -> list[str]:
    out = []
    for group in _PAREN.findall(title or ""):
        for part in group.split(";"):
            if m := _PART.match(part):
                out.append(m.group(1).strip().lower())
    return out


class SeriesIndex:
    def __init__(self, titles):
        self._keys = [series_keys(t) for t in titles]
        members = defaultdict(list)
        for col, keys in enumerate(self._keys):
            for k in keys:
                members[k].append(col)
        self._members = {k: np.array(v) for k, v in members.items()}

    @classmethod
    def from_works(cls, works_path: Path, work_ids: np.ndarray) -> "SeriesIndex":
        t = duckdb.execute("SELECT work_id, title FROM read_parquet(?)", [str(works_path)]).df()
        return cls(t.set_index("work_id").title.reindex(work_ids).fillna("").tolist())

    def continuations(self, cols: np.ndarray) -> np.ndarray:
        """Все столбцы серий, в которые входят cols (сами cols из серий — тоже)."""
        keys = {k for c in cols.tolist() for k in self._keys[c]}
        if not keys:
            return np.array([], dtype=np.int64)
        return np.unique(np.concatenate([self._members[k] for k in keys]))


def exclusion(X: sp.csr_matrix, index: SeriesIndex) -> sp.csr_matrix:
    """Что не советовать человеку: его вход и все книги начатых им серий (для `metrics.top_k`)."""
    rows, cols = [], []
    for u in range(X.shape[0]):
        inp = X.indices[X.indptr[u]:X.indptr[u + 1]]
        ex = np.union1d(inp, index.continuations(inp))
        rows.append(np.full(len(ex), u))
        cols.append(ex)
    r, c = np.concatenate(rows), np.concatenate(cols)
    return sp.csr_matrix((np.ones(len(r), dtype=np.float32), (r, c)), shape=X.shape)


def without_started_series(hold: Holdout, index: SeriesIndex) -> Holdout:
    """Продолжения начатых серий вычеркнуть из кандидатов (`exclude`) и из скрытых книг."""
    X = hold.inputs
    hidden_cols, hidden_ratings = [], []
    for u in range(X.shape[0]):
        cont = index.continuations(X.indices[X.indptr[u]:X.indptr[u + 1]])
        keep = ~np.isin(hold.hidden_cols[u], cont)
        hidden_cols.append(hold.hidden_cols[u][keep])
        hidden_ratings.append(hold.hidden_ratings[u][keep])
    return replace(hold, hidden_cols=hidden_cols, hidden_ratings=hidden_ratings, exclude=exclusion(X, index))
