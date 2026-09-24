"""Фильтры выдачи сверх начатых серий (`series.py`, TODO п. 11): не советовать то, что человек по сути уже оценил.

Книга j убирается, если у неё тот же основной автор, что у оценённой книги i, и:
- дубль: тот же ключ названия (`clean.title_key_sql`), номера в серии не различаются — адаптации, пересказы,
  «тени», не слитые при очистке (TODO п. 17);
- j — сборник (`is_collection` или «X / Y» в названии), а название i входит в его название целыми словами:
  «Animal Farm / 1984» при оценённом «1984»;
- наоборот, i — сборник, j — его часть.
Бокс-сеты серий «(Harry Potter, #1-7)» отсекает фильтр серий.
"""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from booksengine.data.clean import series_no_sql, title_key_sql


def work_info(clean_dir: Path, work_ids: np.ndarray) -> pd.DataFrame:
    """Книги ядра по столбцам матрицы: название, основной автор, ключ названия, номер в серии, сборник ли."""
    d = duckdb.execute(f"""
        WITH prim AS (
            SELECT work_id, arg_min(author_id, position) AS author_id
            FROM read_parquet(?) WHERE coalesce(role, '') = '' GROUP BY 1)
        SELECT w.work_id, w.title, a.name AS author, p.author_id,
               {title_key_sql('w.title')} AS key, {series_no_sql('w.best_edition_title')} AS series_no,
               coalesce(w.is_collection, false) OR w.title LIKE '% / %' AS is_collection
        FROM read_parquet(?) w
        LEFT JOIN prim p USING (work_id) LEFT JOIN read_parquet(?) a USING (author_id)
    """, [str(clean_dir / "work_authors.parquet"), str(clean_dir / "works.parquet"),
          str(clean_dir / "authors.parquet")]).df()
    d = d.set_index("work_id").reindex(work_ids)
    if d.title.isna().any():
        raise ValueError("в works.parquet нет части книг ядра")
    return d.reset_index()


def _contains(outer: str, inner: str) -> bool:
    return f" {inner} " in f" {outer} "


class RatedFilter:
    """Проверка кандидатов против оценённых книг одного человека."""

    def __init__(self, info: pd.DataFrame, rated_cols: np.ndarray):
        self._info = info
        self._by_author: dict[int, list[tuple[str, str, bool]]] = {}
        for r in info.iloc[rated_cols].itertuples():
            if pd.notna(r.author_id) and r.key:
                self._by_author.setdefault(int(r.author_id), []).append((r.key, r.series_no, bool(r.is_collection)))

    def is_rated_already(self, col: int) -> bool:
        j = self._info.iloc[col]
        if pd.isna(j.author_id) or not j.key:
            return False
        for key, sno, coll in self._by_author.get(int(j.author_id), []):
            if key == j.key and (sno == j.series_no or not sno or not j.series_no):
                return True
            if (j.is_collection and _contains(j.key, key)) or (coll and _contains(key, j.key)):
                return True
        return False
