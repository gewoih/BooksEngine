"""Очистка: staging-таблицы → чистый каталог и матрица оценок на уровне произведений.

Каждое правило — отдельная функция над таблицами DuckDB-сессии. Правило пишет в журнал, сколько строк /
пользователей / произведений оно убрало и почему. Функции не зависят от путей, поэтому тестируются
на крошечных синтетических таблицах.

Входные таблицы (имена как в staging): interactions, book_id_map, user_id_map, editions, works, authors, genres.
"""
from dataclasses import asdict, dataclass, field
from pathlib import Path

import duckdb

# Если задан, крупные промежуточные таблицы пишутся сюда сжатым Parquet (а не держатся в памяти/спилле).
SPILL_DIR: Path | None = None


@dataclass
class Step:
    rule: str
    reason: str
    rows_before: int
    rows_after: int
    users_before: int | None = None
    users_after: int | None = None
    works_before: int | None = None
    works_after: int | None = None
    detail: dict = field(default_factory=dict)

    @property
    def rows_removed(self) -> int:
        return self.rows_before - self.rows_after


class CleaningLog:
    def __init__(self) -> None:
        self.steps: list[Step] = []

    def add(self, step: Step) -> None:
        self.steps.append(step)
        print(f"  [{step.rule}] {step.rows_before:,} → {step.rows_after:,} (−{step.rows_removed:,})")

    def records(self) -> list[dict]:
        return [asdict(s) | {"rows_removed": s.rows_removed} for s in self.steps]


def _one(con: duckdb.DuckDBPyConnection, sql: str):
    return con.execute(sql).fetchone()[0]


def _materialize(con, name: str, sql: str) -> None:
    if SPILL_DIR is None:
        con.execute(f"CREATE OR REPLACE TEMP TABLE {name} AS {sql}")
        return
    SPILL_DIR.mkdir(parents=True, exist_ok=True)
    path = SPILL_DIR / f"{name}.parquet"
    con.execute(f"COPY ({sql}) TO '{path}' (FORMAT parquet, COMPRESSION zstd)")
    con.execute(f"CREATE OR REPLACE TEMP VIEW {name} AS SELECT * FROM '{path}'")


def _shape(con, table: str, user_col: str | None = "user_id", work_col: str | None = "work_id") -> tuple:
    cols = ["count(*)"]
    cols.append(f"count(DISTINCT {user_col})" if user_col else "NULL")
    cols.append(f"count(DISTINCT {work_col})" if work_col else "NULL")
    return con.execute(f"SELECT {', '.join(cols)} FROM {table}").fetchone()


# --- 1. Структурные правила ------------------------------------------------------------------------------


def structural(con, log: CleaningLog, src: str = "interactions", dst: str = "i_valid") -> None:
    """Null в ключах, значения вне допустимых диапазонов, точные дубли (user, book)."""
    valid = """user_id IS NOT NULL AND book_id IS NOT NULL AND rating IS NOT NULL AND rating BETWEEN 0 AND 5
               AND coalesce(is_read, 0) IN (0, 1) AND coalesce(is_reviewed, 0) IN (0, 1)"""
    before, after_ranges, distinct_pairs = con.execute(f"""
        SELECT count(*), count(*) FILTER ({valid}), count(DISTINCT (user_id, book_id)) FILTER ({valid}) FROM {src}
    """).fetchone()
    log.add(Step("structural_invalid", "null в ключах или значения вне диапазонов (rating 0–5, флаги 0/1)",
                 before, after_ranges))
    if distinct_pairs == after_ranges:
        _materialize(con, dst, f"SELECT * FROM {src} WHERE {valid}")
    else:
        # При дубле оставляем запись с максимальной информацией: явная оценка важнее «на полке».
        _materialize(con, dst, f"""
            SELECT * FROM {src} WHERE {valid}
            QUALIFY row_number() OVER (PARTITION BY user_id, book_id ORDER BY rating DESC, is_read DESC) = 1""")
    log.add(Step("structural_duplicates", "точные дубли пары (user, book)", after_ranges, distinct_pairs))


# --- 2. Каталог и ссылочная целостность ------------------------------------------------------------------


def build_catalog(con, log: CleaningLog, patterns: list[str]) -> None:
    """works_valid: произведения с изданиями и названием; флаг is_collection по шаблонам заголовка."""
    works_total = _one(con, "SELECT count(*) FROM works")
    regex = "|".join(f"({p})" for p in patterns)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE _best AS
        SELECT e.work_id, e.book_id, e.title, e.title_without_series, e.language_code, e.description,
               e.publication_year, e.book_id = w.best_book_id AS is_best
        FROM editions e JOIN works w USING (work_id)
        WHERE e.title IS NOT NULL
        QUALIFY row_number() OVER (PARTITION BY e.work_id
            ORDER BY (e.book_id = w.best_book_id) DESC, e.ratings_count DESC NULLS LAST, e.book_id) = 1
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE works_valid AS
        SELECT w.work_id, coalesce(b.book_id, w.best_book_id) AS best_book_id,
               -- original_title у переводов — транслит оригинала («Prestuplenie i nakazanie»),
               -- поэтому основное название — из лучшего издания, оригинальное хранится отдельно
               coalesce(b.title_without_series, b.title, w.original_title) AS title,
               w.original_title, b.title AS best_edition_title,
               coalesce(w.original_publication_year, b.publication_year) AS publication_year,
               b.language_code, w.media_type, w.books_count, w.ratings_count AS gr_ratings_count,
               w.ratings_sum AS gr_ratings_sum, b.description,
               regexp_matches(lower(coalesce(b.title, '')), '{regex}') AS is_collection
        FROM works w
        LEFT JOIN _best b USING (work_id)
        WHERE w.work_id IS NOT NULL
          AND EXISTS (SELECT 1 FROM editions e WHERE e.work_id = w.work_id)
          AND coalesce(w.original_title, b.title_without_series, b.title) IS NOT NULL
    """)
    valid = _one(con, "SELECT count(*) FROM works_valid")
    best_missing = _one(con, "SELECT count(*) FROM _best WHERE NOT is_best")
    no_editions = _one(con, "SELECT count(*) FROM works w WHERE NOT EXISTS "
                            "(SELECT 1 FROM editions e WHERE e.work_id = w.work_id)")
    log.add(Step("works_without_title_or_editions", "произведение без изданий или без названия "
                 "(ни original_title, ни названия лучшего издания)", works_total, valid,
                 works_before=works_total, works_after=valid,
                 detail={"without_editions": no_editions, "best_book_replaced": best_missing,
                         "collections_flagged": _one(con, "SELECT count(*) FROM works_valid WHERE is_collection")}))
    con.execute("DROP TABLE _best")


def link_to_works(con, log: CleaningLog, src: str = "i_valid", dst: str = "i_linked") -> None:
    """interaction.book_id → book_id_map → edition → work_id → works_valid."""
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW _i_join AS
        SELECT i.*, m.book_id AS gr_book_id, e.book_id AS edition_id, e.work_id,
               wv.work_id AS valid_work_id
        FROM {src} i
        LEFT JOIN book_id_map m ON m.book_id_csv = i.book_id
        LEFT JOIN editions e ON e.book_id = m.book_id
        LEFT JOIN works_valid wv ON wv.work_id = e.work_id
    """)
    d = con.execute("""
        SELECT count(*),
               count(*) FILTER (gr_book_id IS NULL),
               count(*) FILTER (gr_book_id IS NOT NULL AND edition_id IS NULL),
               count(*) FILTER (edition_id IS NOT NULL AND work_id IS NULL),
               count(*) FILTER (work_id IS NOT NULL AND valid_work_id IS NULL)
        FROM _i_join
    """).fetchone()
    _materialize(con, dst, """
        SELECT user_id, gr_book_id AS book_id, work_id, rating, is_read, is_reviewed
        FROM _i_join WHERE valid_work_id IS NOT NULL""")
    log.add(Step("broken_links", "взаимодействие не доводится до валидного произведения", d[0],
                 _one(con, f"SELECT count(*) FROM {dst}"),
                 detail={"no_book_id_map": d[1], "no_edition": d[2], "edition_without_work_id": d[3],
                         "work_invalid": d[4]}))
    con.execute("DROP VIEW _i_join")


# --- 3–4. Тип сигнала и переход к произведениям ----------------------------------------------------------


def to_work_level(con, log: CleaningLog, explicit_min: int, explicit_max: int,
                  src: str = "i_linked") -> None:
    """ratings_w: явные оценки, свёрнутые с изданий на произведение (среднее). rating = 0 отбрасывается."""
    total = _one(con, f"SELECT count(*) FROM {src}")
    explicit = f"rating BETWEEN {explicit_min} AND {explicit_max}"
    n_explicit = _one(con, f"SELECT count(*) FROM {src} WHERE {explicit}")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE ratings_w AS
        SELECT user_id, work_id, avg(rating)::REAL AS rating, count(*)::SMALLINT AS n_editions,
               min(rating) AS rating_min, max(rating) AS rating_max
        FROM {src} WHERE {explicit}
        GROUP BY user_id, work_id
    """)
    multi = con.execute("""
        SELECT count(*) FILTER (n_editions > 1), count(*) FILTER (rating_min <> rating_max),
               count(*) FILTER (rating_max - rating_min >= 2)
        FROM ratings_w
    """).fetchone()
    n_ratings = _one(con, "SELECT count(*) FROM ratings_w")
    log.add(Step("split_explicit_signal",
                 "rating = 0 — это не оценка («прочитано без оценки» или «на полке»): отбрасывается — ни модель, "
                 "ни БД его не используют (решение 2026-09-23; исходные записи остаются в staging)",
                 total, n_explicit, detail={"unrated_rows": total - n_explicit}))
    log.add(Step("editions_to_works",
                 "несколько изданий одного произведения у пользователя сворачиваются в одну оценку (среднее)",
                 n_explicit, n_ratings, detail={"pairs_with_multiple_editions": multi[0],
                                                "pairs_conflicting": multi[1], "pairs_conflict_ge2": multi[2]}))
    con.execute("ALTER TABLE ratings_w DROP COLUMN rating_min")
    con.execute("ALTER TABLE ratings_w DROP COLUMN rating_max")


def drop_collections(con, log: CleaningLog, table: str = "ratings_w") -> None:
    before = _shape(con, table)
    con.execute(f"DELETE FROM {table} WHERE work_id IN (SELECT work_id FROM works_valid WHERE is_collection)")
    after = _shape(con, table)
    log.add(Step("collections", "оценки сборников/box sets дублируют оценки входящих произведений",
                 before[0], after[0], before[1], after[1], before[2], after[2]))


# --- 4a. Не-книги ----------------------------------------------------------------------------------------


def flag_nonbooks(con, patterns: list[str], exceptions: list[int], table: str = "works_valid") -> int:
    """is_nonbook: ноты, раскраски, календари, аудиокурсы — по шаблонам названия."""
    regex = "|".join(f"({p})" for p in patterns)
    # пустой список исключений нельзя подставлять как NOT IN (NULL): это дало бы NULL у всех
    exc = f"AND work_id NOT IN ({', '.join(str(int(w)) for w in exceptions)})" if exceptions else ""
    con.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS is_nonbook BOOLEAN")
    con.execute(f"""
        UPDATE {table} SET is_nonbook = regexp_matches(
            lower(coalesce(title, '') || ' | ' || coalesce(best_edition_title, '')), ?) {exc}""", [regex])
    return _one(con, f"SELECT count(*) FROM {table} WHERE is_nonbook")


def drop_nonbooks(con, log: CleaningLog, table: str = "ratings_w") -> None:
    before = _shape(con, table)
    con.execute(f"DELETE FROM {table} WHERE work_id IN (SELECT work_id FROM works_valid WHERE is_nonbook)")
    after = _shape(con, table)
    log.add(Step("nonbooks", "не книги (ноты, раскраски, календари, аудиокурсы): те же читатели, что у книги, "
                 "и попадают в её соседи", before[0], after[0], before[1], after[1], before[2], after[2]))


# --- 4b. Дубли произведений -------------------------------------------------------------------------------


def author_source(con) -> None:
    """_author_src: издание, из которого берутся авторы произведения, — лучшее, а если в нём нет авторов —
    самое популярное издание с авторами. Общий источник для work_authors и основного автора."""
    con.execute("""
        CREATE OR REPLACE TEMP TABLE _author_src AS
        SELECT e.work_id, e.authors FROM editions e JOIN works_valid w USING (work_id)
        WHERE len(e.authors) > 0
        QUALIFY row_number() OVER (PARTITION BY e.work_id
                                   ORDER BY (e.book_id = w.best_book_id) DESC, e.ratings_count DESC NULLS LAST,
                                            e.book_id) = 1
    """)


def primary_authors(con) -> None:
    """work_primary: основной автор произведения — без роли (не иллюстратор, не переводчик), первый по позиции."""
    author_source(con)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE work_primary AS
        SELECT work_id, arg_min(a.author_id, pos) AS author_id
        FROM (SELECT work_id, unnest(authors) AS a, generate_subscripts(authors, 1) AS pos FROM _author_src)
        WHERE coalesce(a.role, '') = '' AND a.author_id IN (SELECT author_id FROM authors)
        GROUP BY 1""")


def title_key_sql(col: str) -> str:
    """Ключ названия для дублей: без хвостовой скобки («(Series, #1)») и пунктуации, буквы любых алфавитов
    сохраняются. Общий для очистки (`find_duplicates`) и фильтра выдачи (`model/filters.py`)."""
    return (f"trim(regexp_replace(lower(regexp_replace({col}, '\\s*\\([^)]*\\)\\s*$', '')), "
            f"'[^\\p{{L}}\\p{{N}}]+', ' ', 'g'))")


def series_no_sql(col: str) -> str:
    """Номер в серии из хвоста названия издания «… #3)»; '' — номера нет."""
    return f"regexp_extract(coalesce({col}, ''), '#\\s*(\\d+(\\.\\d+)?)\\s*\\)\\s*$', 1)"


def find_duplicates(con, adaptation_patterns: list[str], max_shadow_share: float, table: str = "ratings_w") -> int:
    """dup_map: уверенные «тени» — то же название (без хвостовой скобки и пунктуации, буквы любых алфавитов
    сохраняются) + тот же основной автор + номер в серии не различается, не адаптация и не сборник, оценок
    меньше max_shadow_share от главного (самого оценённого в группе)."""
    regex = "|".join(f"({p})" for p in adaptation_patterns)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _dup_keys AS
        SELECT w.work_id, p.author_id, c.n,
               {title_key_sql('w.title')} AS key, {series_no_sql('w.best_edition_title')} AS series_no
        FROM works_valid w
        JOIN work_primary p USING (work_id)
        JOIN (SELECT work_id, count(*) AS n FROM {table} GROUP BY 1) c USING (work_id)
        WHERE NOT coalesce(w.is_collection, false)
          AND NOT regexp_matches(lower(coalesce(w.title, '') || ' | ' || coalesce(w.best_edition_title, '')), ?)
    """, [regex])
    con.execute("""
        CREATE OR REPLACE TEMP TABLE dup_map AS
        SELECT work_id AS shadow_work_id, main_id AS main_work_id FROM (
            SELECT work_id, n, series_no,
                   first_value(work_id) OVER g AS main_id, first_value(n) OVER g AS main_n,
                   first_value(series_no) OVER g AS main_series
            FROM _dup_keys WHERE key <> ''
            WINDOW g AS (PARTITION BY key, author_id ORDER BY n DESC, work_id))
        -- у теней номера в серии часто нет — это не мешает; номер у тени без такого же у главного — другой том
        -- («Musashi» ← «Musashi #2»)
        WHERE work_id <> main_id AND (series_no = main_series OR series_no = '')
          AND n < ? * main_n
    """, [max_shadow_share])
    con.execute("DROP TABLE _dup_keys")
    return _one(con, "SELECT count(*) FROM dup_map")


def merge_duplicates(con, log: CleaningLog, table: str = "ratings_w") -> None:
    """Оценки тени переносятся на главное; обе у одного человека — среднее, взвешенное по изданиям."""
    before = _shape(con, table)
    both = _one(con, f"""SELECT count(*) FROM {table} s JOIN dup_map d ON d.shadow_work_id = s.work_id
                         JOIN {table} m ON m.user_id = s.user_id AND m.work_id = d.main_work_id""")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _merged AS
        SELECT r.user_id, coalesce(d.main_work_id, r.work_id) AS work_id,
               round(sum(r.rating * r.n_editions) / sum(r.n_editions), 6)::REAL AS rating,
               sum(r.n_editions)::SMALLINT AS n_editions
        FROM {table} r LEFT JOIN dup_map d ON d.shadow_work_id = r.work_id
        GROUP BY 1, 2""")
    con.execute(f"DROP TABLE {table}")
    con.execute(f"ALTER TABLE _merged RENAME TO {table}")
    after = _shape(con, table)
    log.add(Step("duplicates", "«теневое» произведение той же книги сливается в главное (как издания): иначе "
                 "модель советует прочитанное под другим work_id", before[0], after[0], before[1], after[1],
                 before[2], after[2], {"shadows": _one(con, "SELECT count(*) FROM dup_map"),
                                       "users_rated_both": both}))


# --- 5. Аномальные пользователи -------------------------------------------------------------------------


def user_stats(con, table: str = "ratings_w") -> None:
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE user_stats AS
        -- округление: параллельные агрегаты с плавающей точкой отличаются в последних битах от прогона к прогону,
        -- а сравнение с порогом должно быть детерминированным
        SELECT user_id, count(*) AS n, round(avg(rating), 6) AS mean,
               round(coalesce(stddev_pop(rating), 0), 6) AS sd,
               -- доля самого частого значения; дробные (среднее по изданиям) — «половина вверх», как в метриках
               round(greatest(avg((floor(rating + 0.5) = 1)::INT), avg((floor(rating + 0.5) = 2)::INT),
                              avg((floor(rating + 0.5) = 3)::INT), avg((floor(rating + 0.5) = 4)::INT),
                              avg((floor(rating + 0.5) = 5)::INT)), 6) AS mode_share
        FROM {table} GROUP BY user_id
    """)


def _drop_users(con, log, table: str, where: str, rule: str, reason: str, detail: dict | None = None) -> None:
    before = _shape(con, table)
    con.execute(f"DELETE FROM {table} WHERE user_id IN (SELECT user_id FROM user_stats WHERE {where})")
    after = _shape(con, table)
    log.add(Step(rule, reason, before[0], after[0], before[1], after[1], before[2], after[2], detail or {}))


def filter_users(con, log: CleaningLog, max_sd: float, min_ratings_for_sd: int, max_ratings: int,
                 table: str = "ratings_w", max_mode_share: float | None = None) -> None:
    user_stats(con, table)
    _drop_users(con, log, table, f"n >= {min_ratings_for_sd} AND sd < {max_sd}", "users_low_variance",
                f"разброс оценок < {max_sd} при ≥ {min_ratings_for_sd} оценках: все книги оценены одинаково, "
                "предпочтения не различимы",
                {"of_them_all_fives": _one(con, f"SELECT count(*) FROM user_stats WHERE n >= {min_ratings_for_sd} "
                                                f"AND sd < {max_sd} AND mean > 4.9")})
    if max_mode_share is not None:
        _drop_users(con, log, table, f"n >= {min_ratings_for_sd} AND mode_share >= {max_mode_share}",
                    "users_monotone", f"≥ {max_mode_share:.0%} оценок одного значения при ≥ {min_ratings_for_sd} "
                    "оценках: вкуса относительно своей средней не видно")
    _drop_users(con, log, table, f"n > {max_ratings}", "users_hyperactive",
                f"больше {max_ratings} оценённых произведений: каталогизация/импорт, а не личный вкус")


# --- 6. k-core -------------------------------------------------------------------------------------------


def kcore(con, table: str, min_user: int, min_work: int, dst: str | None = None,
          max_iter: int = 50) -> tuple[int, int, int, int]:
    """Итеративно удаляет пользователей с < min_user и произведения с < min_work оценками до сходимости.

    Возвращает (строк, пользователей, произведений, итераций). Если dst не задан — работает на копии.
    """
    work = dst or "_kcore_tmp"
    con.execute(f"CREATE OR REPLACE TEMP TABLE {work} AS SELECT user_id, work_id FROM {table}"
                if dst is None else f"CREATE OR REPLACE TEMP TABLE {work} AS SELECT * FROM {table}")
    it = 0
    while it < max_iter:
        it += 1
        n0 = _one(con, f"SELECT count(*) FROM {work}")
        con.execute(f"""
            DELETE FROM {work} WHERE user_id IN
                (SELECT user_id FROM {work} GROUP BY user_id HAVING count(*) < {min_user})
        """)
        con.execute(f"""
            DELETE FROM {work} WHERE work_id IN
                (SELECT work_id FROM {work} GROUP BY work_id HAVING count(*) < {min_work})
        """)
        if _one(con, f"SELECT count(*) FROM {work}") == n0:
            break
    rows, users, works = _shape(con, work)
    if dst is None:
        con.execute(f"DROP TABLE {work}")
    return rows, users, works, it


def kcore_options(con, options: list[list[int]], table: str = "ratings_w") -> list[dict]:
    res = []
    total = _one(con, f"SELECT count(*) FROM {table}")
    for ku, kw in options:
        rows, users, works, it = kcore(con, table, ku, kw)
        res.append({"min_user": ku, "min_work": kw, "ratings": rows, "users": users, "works": works,
                    "kept_share": rows / total if total else 0,
                    "density": rows / (users * works) if users and works else 0, "iterations": it})
        print(f"  k-core ({ku}, {kw}): {rows:,} оценок, {users:,} польз., {works:,} произв.")
    return res


def apply_kcore(con, log: CleaningLog, min_user: int, min_work: int, table: str = "ratings_w") -> None:
    before = _shape(con, table)
    _, _, _, it = kcore(con, table, min_user, min_work, dst="_ratings_core")
    con.execute(f"DROP TABLE {table}")
    con.execute(f"ALTER TABLE _ratings_core RENAME TO {table}")
    after = _shape(con, table)
    log.add(Step("kcore", f"итеративно: пользователь ≥ {min_user} оценок, произведение ≥ {min_work} оценок "
                 "(меньше — недостаточно статистики для CF и для честной отложенной выборки)",
                 before[0], after[0], before[1], after[1], before[2], after[2], {"iterations": it}))
