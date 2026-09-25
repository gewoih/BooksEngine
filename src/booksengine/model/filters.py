"""Фильтры выдачи сверх начатых серий (`series.py`).

Одно произведение (`Books.same`) — у книг общий автор (основной автор одной есть среди авторов другой: переводчик
или редактор бывает записан первым), номера в серии не различаются (кроме сборников), и:
- тот же ключ названия (`clean.title_key_sql`) — адаптации, пересказы, «тени», не слитые при очистке;
- сборник (`is_collection` или «X / Y») содержит название другой книги целыми словами;
- слова одного варианта названия (название, оригинальное, лучшее издание) целиком входят в другой — «Alice in
  Wonderland» и «Alice's Adventures in Wonderland & Through the Looking-Glass», «Don Quixote» (оригинал «Don Quijote
  de La Mancha») и «Don Quijote de la Mancha I», «The Lottery» и «The Lottery and Other Stories». Служебные слова
  (`_STOP`) и обороты «and other stories» не считаются. Заодно так отсекаются продолжения без «#» («Madeline's
  Rescue» при оценённой «Madeline») — как и продолжения начатых серий. Изредка склеиваются разные книги одного
  автора («Midnight» и «The Key to Midnight»): одна пропущенная книга дешевле, чем совет уже прочитанной.

`RatedFilter` — не советовать то, что человек по сути уже оценил: одно произведение с оценённой книгой.

`ListPicker` — правила самого списка: сборник не советуется (советуется сама книга), поздний том (#2 и дальше)
неначатой серии заменяется первой книгой серии, второе издание того же произведения не советуется, книг одного
основного автора — не больше одной на каждые 10 мест. Те же правила — в замере (`layers.evaluate`): судья видит тот
же список, что и человек.
"""
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from booksengine.data.clean import series_no_sql, title_key_sql
from booksengine.model.series import SeriesIndex

# почему книга не попала в список
RATED, COLLECTION, LATER, LATER_DROPPED, AUTHOR, DUPLICATE = (
    "rated", "collection", "later", "later_dropped", "author", "duplicate")
WHY_REMOVED = {RATED: "уже оценено по сути (то же произведение под другим названием, сборник с ним или его часть)",
               DUPLICATE: "то же произведение, что книга выше в списке",
               COLLECTION: "сборник — советуется сама книга",
               LATER: "поздний том неначатой серии — вместо него первая книга серии",
               LATER_DROPPED: "поздний том неначатой серии, а первой книги среди кандидатов нет",
               AUTHOR: "лишняя книга автора (не больше одной на каждые 10 мест)"}


# служебные слова названий (английский, испанский, французский, немецкий, итальянский) и пометки издания;
# одиночное «i» — значимое («I, Robot», «Don Quijote I»)
_STOP = frozenset("""
the a an and or of in on at to for with from by s
de la el los las del y e le les du des et un une l d
der die das den dem und ein eine einer il lo gli di
complete collected selected essential illustrated annotated unabridged edition
""".split())
_WITH_OTHERS = re.compile(r"\b(and|&)( \w+)? other (stories|tales|poems|writings|works|plays|essays)\b")


def name_words(title) -> frozenset:
    """Значимые слова названия: без хвостовой скобки серии, регистра, диакритики, служебных слов и «and other stories»."""
    if not isinstance(title, str):
        return frozenset()
    t = unicodedata.normalize("NFKD", re.sub(r"\s*\([^)]*\)\s*$", "", title).lower())
    t = _WITH_OTHERS.sub(" ", "".join(c for c in t if not unicodedata.combining(c)))
    return frozenset(w for w in re.split(r"[\W_]+", t) if w and w not in _STOP)


def work_info(clean_dir: Path, work_ids: np.ndarray) -> pd.DataFrame:
    """Книги ядра по столбцам матрицы: название (и варианты), основной и все авторы, ключ названия, номер в серии,
    сборник ли."""
    d = duckdb.execute(f"""
        WITH prim AS (
            SELECT work_id, arg_min(author_id, position) AS author_id
            FROM read_parquet(?) WHERE coalesce(role, '') = '' GROUP BY 1),
        everyone AS (SELECT work_id, list(DISTINCT author_id) AS authors FROM read_parquet(?) GROUP BY 1)
        SELECT w.work_id, w.title, w.original_title, w.best_edition_title, a.name AS author, p.author_id, e.authors,
               {title_key_sql('w.title')} AS key, {series_no_sql('w.best_edition_title')} AS series_no,
               coalesce(w.is_collection, false) OR w.title LIKE '% / %' AS is_collection
        FROM read_parquet(?) w
        LEFT JOIN prim p USING (work_id) LEFT JOIN everyone e USING (work_id) LEFT JOIN read_parquet(?) a USING (author_id)
    """, [str(clean_dir / "work_authors.parquet"), str(clean_dir / "work_authors.parquet"),
          str(clean_dir / "works.parquet"), str(clean_dir / "authors.parquet")]).df()
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
    authors: np.ndarray      # все авторы (frozenset), с переводчиками и редакторами
    names: np.ndarray        # варианты названия — кортеж непустых наборов значимых слов

    @classmethod
    def of(cls, info: pd.DataFrame) -> "Books":
        author = info.author_id.fillna(-1).to_numpy(dtype=np.int64)
        everyone = info.authors if "authors" in info else pd.Series([None] * len(info))
        authors = np.empty(len(info), dtype=object)
        authors[:] = [frozenset(() if e is None or (np.isscalar(e) and pd.isna(e)) else (int(x) for x in e))
                      | ({int(a)} if a >= 0 else set()) for e, a in zip(everyone, author)]
        cols = [info[c] if c in info else pd.Series([None] * len(info))
                for c in ("title", "original_title", "best_edition_title")]
        names = np.empty(len(info), dtype=object)
        names[:] = [tuple({w for w in map(name_words, v) if w}) for v in zip(*cols)]
        return cls(author, info.key.to_numpy(), info.series_no.to_numpy(),
                   info.is_collection.fillna(False).to_numpy(dtype=bool), authors, names)

    def same(self, i: int, j: int) -> bool:
        """Одно ли произведение книги i и j (правила — в описании модуля)."""
        ai, aj = self.author[i], self.author[j]
        if ai < 0 or aj < 0 or not (ai == aj or ai in self.authors[j] or aj in self.authors[i]):
            return False
        ci, cj = self.collection[i], self.collection[j]
        si, sj = self.series_no[i], self.series_no[j]
        if si and sj and si != sj and not (ci or cj):
            return False
        ki, kj = self.key[i], self.key[j]
        if ki and kj and (ki == kj or (cj and _contains(kj, ki)) or (ci and _contains(ki, kj))):
            return True
        return any(a <= b or b <= a for a in self.names[i] for b in self.names[j])

    def kin(self, col: int) -> frozenset:
        """Авторы, по которым ищутся возможные двойники книги: основной и все остальные."""
        return self.authors[col] if self.author[col] >= 0 else frozenset()


def _contains(outer: str, inner: str) -> bool:
    return f" {inner} " in f" {outer} "


class RatedFilter:
    """Проверка кандидатов против оценённых книг одного человека."""

    def __init__(self, info: "pd.DataFrame | Books", rated_cols: np.ndarray):
        self._b = info if isinstance(info, Books) else Books.of(info)
        self._by_author = _AuthorIndex(self._b)
        for i in np.asarray(rated_cols).tolist():
            self._by_author.add(int(i))

    def is_rated_already(self, col: int) -> bool:
        return self._by_author.has_same(col)


class _AuthorIndex:
    """Книги по авторам: поиск того же произведения только среди книг с общим автором."""

    def __init__(self, books: Books):
        self._b, self._cols = books, {}

    def add(self, col: int) -> None:
        for a in self._b.kin(col):
            self._cols.setdefault(a, []).append(col)

    def has_same(self, col: int) -> bool:
        seen = set()
        for a in self._b.kin(col):
            for c in self._cols.get(a, ()):
                if c not in seen:
                    seen.add(c)
                    if c == col or self._b.same(c, col):
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
    - то же произведение, что книга выше в списке (`Books.same`), не советуется второй раз;
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
        shown = _AuthorIndex(b)
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
            if shown.has_same(c):
                removed.append((c, DUPLICATE))
                continue
            a = int(b.author[c])
            if a >= 0 and per_author.get(a, 0) >= cap:
                removed.append((c, AUTHOR))
                continue
            per_author[a] = per_author.get(a, 0) + 1
            shown.add(c)
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
