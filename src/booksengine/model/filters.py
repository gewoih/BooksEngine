"""Фильтры выдачи сверх начатых серий (`series.py`).

`RatedFilter` — не советовать то, что человек по сути уже оценил. Книга j убирается, если у неё тот же основной
автор, что у оценённой книги i, и:
- дубль: тот же ключ названия (`clean.title_key_sql`), номера в серии не различаются — адаптации, пересказы,
  «тени», не слитые при очистке;
- j — сборник (`is_collection` или «X / Y» в названии), а название i входит в его название целыми словами:
  «Animal Farm / 1984» при оценённом «1984»;
- наоборот, i — сборник, j — его часть.
Бокс-сеты серий «(Harry Potter, #1-7)» отсекает фильтр серий.

`ListPicker` — правила самого списка: сборник не советуется (советуется сама книга), поздний том (#2 и дальше)
неначатой серии заменяется первой книгой серии, книг одного основного автора — не больше одной на каждые 10 мест.
Те же правила — в замере (`layers.evaluate`): судья видит тот же список, что и человек.
"""
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from booksengine.data.clean import series_no_sql, title_key_sql
from booksengine.model.series import SeriesIndex

# почему книга не попала в список
RATED, COLLECTION, LATER, LATER_DROPPED, AUTHOR = "rated", "collection", "later", "later_dropped", "author"
WHY_REMOVED = {RATED: "уже оценено по сути (дубль, сборник или часть оценённого сборника)",
               COLLECTION: "сборник — советуется сама книга",
               LATER: "поздний том неначатой серии — вместо него первая книга серии",
               LATER_DROPPED: "поздний том неначатой серии, а первой книги среди кандидатов нет",
               AUTHOR: "лишняя книга автора (не больше одной на каждые 10 мест)"}


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


@dataclass(frozen=True)
class Books:
    """Поля `work_info` массивами по столбцам: проверки тысяч кандидатов без обращений к строкам pandas."""
    author: np.ndarray       # основной автор, −1 — неизвестен
    key: np.ndarray          # ключ названия
    series_no: np.ndarray    # номер в серии строкой, '' — нет
    collection: np.ndarray   # сборник или бокс-сет

    @classmethod
    def of(cls, info: pd.DataFrame) -> "Books":
        return cls(info.author_id.fillna(-1).to_numpy(dtype=np.int64), info.key.to_numpy(),
                   info.series_no.to_numpy(), info.is_collection.fillna(False).to_numpy(dtype=bool))


def _contains(outer: str, inner: str) -> bool:
    return f" {inner} " in f" {outer} "


class RatedFilter:
    """Проверка кандидатов против оценённых книг одного человека."""

    def __init__(self, info: "pd.DataFrame | Books", rated_cols: np.ndarray):
        self._b = info if isinstance(info, Books) else Books.of(info)
        b = self._b
        self._by_author: dict[int, list[tuple[str, str, bool]]] = {}
        for i in np.asarray(rated_cols).tolist():
            if b.author[i] >= 0 and b.key[i]:
                self._by_author.setdefault(int(b.author[i]), []).append((b.key[i], b.series_no[i], bool(b.collection[i])))

    def is_rated_already(self, col: int) -> bool:
        b = self._b
        key, sno = b.key[col], b.series_no[col]
        if b.author[col] < 0 or not key:
            return False
        for k, s, coll in self._by_author.get(int(b.author[col]), []):
            if k == key and (s == sno or not s or not sno):
                return True
            if (b.collection[col] and _contains(key, k)) or (coll and _contains(k, key)):
                return True
        return False


def author_cap(top: int) -> int:
    """Сколько книг одного основного автора можно в списке из top: одна на каждые 10 мест (20 → 2, 50 → 5), не меньше
    одной. Пять книг одного автора в двадцати — пять ставок на один вкус; остальные его книги — после отклика."""
    return max(1, top // 10)


class ListPicker:
    """Список выдачи из кандидатов по убыванию балла.

    Всегда убирается «уже оценено по сути» (`RatedFilter`). Правила списка (`rules=True`):
    - сборник или бокс-сет не советуется — советуется сама книга;
    - поздний том (#2 и дальше) неначатой серии заменяется первой книгой серии, если она сама кандидат (из
      нескольких первых — с большим баллом); иначе убирается. Начатые серии исключены раньше (`series.exclusion`);
    - книг одного основного автора — не больше `author_cap(top)`, остальные остаются ниже списка."""

    def __init__(self, info: pd.DataFrame, series: SeriesIndex):
        self.books = Books.of(info)
        sno = pd.to_numeric(info.series_no, errors="coerce").to_numpy(dtype=np.float64)
        self.later = np.nan_to_num(sno, nan=0.0) >= 2
        self.first = sno == 1
        self.series = series

    def first_book(self, col: int, score) -> int | None:
        """Первая книга (#1) серии книги col с наибольшим баллом; None — ни одна не кандидат (балл −∞)."""
        firsts = [c for c in self.series.continuations(np.array([col])).tolist() if self.first[c]]
        if not firsts:
            return None
        s = [score(c) for c in firsts]
        best = int(np.argmax(s))
        return firsts[best] if np.isfinite(s[best]) else None

    def pick(self, order, score, top: int, rated: RatedFilter | None = None,
             rules: bool = True) -> tuple[list[int], list[tuple[int, str]]]:
        """order — кандидаты (столбцы ядра) по убыванию балла; score(col) — балл книги, −∞ у не-кандидатов (вход,
        начатые серии, вне модели). Возвращает выбранные столбцы и убранные — (столбец, причина из WHY_REMOVED)."""
        b, cap = self.books, author_cap(top)
        picked, removed, done, per_author = [], [], set(), {}
        for c in order:
            if len(picked) == top:
                break
            c = int(c)
            if c in done:
                continue
            done.add(c)
            if rated is not None and rated.is_rated_already(c):
                removed.append((c, RATED))
                continue
            if not rules:
                picked.append(c)
                continue
            if b.collection[c]:
                removed.append((c, COLLECTION))
                continue
            if self.later[c]:
                f = self.first_book(c, score)
                if (f is None or f in done or b.collection[f]
                        or (rated is not None and rated.is_rated_already(f))):
                    removed.append((c, LATER_DROPPED))
                    continue
                removed.append((c, LATER))
                c = f
                done.add(c)
            a = int(b.author[c])
            if a >= 0 and per_author.get(a, 0) >= cap:
                removed.append((c, AUTHOR))
                continue
            per_author[a] = per_author.get(a, 0) + 1
            picked.append(c)
        return picked, removed


def pick_top(scores: np.ndarray, top_cols: np.ndarray, pos_of: np.ndarray, picker: ListPicker,
             rated: list[RatedFilter | None], k: int, pool: int = 100) -> list[np.ndarray]:
    """Список из k книг для каждой строки scores (баллы по книгам top_cols, −∞ — не кандидат) по правилам `picker`.
    Кандидаты — первые pool по баллу; если правила убрали слишком много — весь порядок. pos_of: столбец ядра →
    место в top_cols (−1 — вне). При равном балле раньше книга с меньшим столбцом, как в `recommend`."""
    n = min(max(pool, k), scores.shape[1])
    part = np.argpartition(-scores, n - 1, axis=1)[:, :n]
    vals = np.take_along_axis(scores, part, axis=1)
    o = np.lexsort((part, -vals), axis=1)
    cand, fin = np.take_along_axis(part, o, axis=1), np.isfinite(np.take_along_axis(vals, o, axis=1))
    out = []
    for i in range(scores.shape[0]):
        row = scores[i]

        def score(c: int, row=row) -> float:
            return float(row[pos_of[c]]) if pos_of[c] >= 0 else -np.inf

        picks, _ = picker.pick(top_cols[cand[i][fin[i]]], score, k, rated[i])
        if len(picks) < k and fin[i].all() and n < scores.shape[1]:   # первых pool не хватило
            full = np.argsort(-row, kind="stable")
            picks, _ = picker.pick(top_cols[full[np.isfinite(row[full])]], score, k, rated[i])
        out.append(np.array(picks, dtype=np.int64))
    return out
