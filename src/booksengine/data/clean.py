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
    con.execute(f"""
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
    """ratings_w: явные оценки, свёрнутые с изданий на произведение (среднее); shelf_w: rating = 0."""
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
    shelf_rows = total - n_explicit
    # Если у пары (user, work) есть явная оценка хоть одного издания, «полочная» запись избыточна.
    _materialize(con, "shelf_w", f"""
        SELECT user_id, work_id, max(is_read)::TINYINT AS is_read
        FROM {src} s WHERE rating = 0
          AND NOT EXISTS (SELECT 1 FROM ratings_w r WHERE r.user_id = s.user_id AND r.work_id = s.work_id)
        GROUP BY user_id, work_id""")
    n_ratings = _one(con, "SELECT count(*) FROM ratings_w")
    log.add(Step("split_explicit_signal",
                 "rating = 0 — это не оценка («прочитано без оценки» или «на полке»): уходит в shelf_events",
                 total, n_explicit, detail={"shelf_rows": shelf_rows,
                                            "shelf_pairs_work_level": _one(con, "SELECT count(*) FROM shelf_w")}))
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


# --- 5. Аномальные пользователи -------------------------------------------------------------------------


def user_stats(con, table: str = "ratings_w") -> None:
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE user_stats AS
        -- округление: параллельные агрегаты с плавающей точкой отличаются в последних битах от прогона к прогону,
        -- а сравнение с порогом должно быть детерминированным
        SELECT user_id, count(*) AS n, round(avg(rating), 6) AS mean,
               round(coalesce(stddev_pop(rating), 0), 6) AS sd
        FROM {table} GROUP BY user_id
    """)


def _drop_users(con, log, table: str, where: str, rule: str, reason: str, detail: dict | None = None) -> None:
    before = _shape(con, table)
    con.execute(f"DELETE FROM {table} WHERE user_id IN (SELECT user_id FROM user_stats WHERE {where})")
    after = _shape(con, table)
    log.add(Step(rule, reason, before[0], after[0], before[1], after[1], before[2], after[2], detail or {}))


def filter_users(con, log: CleaningLog, max_sd: float, min_ratings_for_sd: int, max_ratings: int,
                 table: str = "ratings_w") -> None:
    user_stats(con, table)
    _drop_users(con, log, table, f"n >= {min_ratings_for_sd} AND sd < {max_sd}", "users_low_variance",
                f"разброс оценок < {max_sd} при ≥ {min_ratings_for_sd} оценках: все книги оценены одинаково, "
                "предпочтения не различимы",
                {"of_them_all_fives": _one(con, f"SELECT count(*) FROM user_stats WHERE n >= {min_ratings_for_sd} "
                                                f"AND sd < {max_sd} AND mean > 4.9")})
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
