"""Загрузка очищенного каталога (data/clean/*.parquet) в PostgreSQL.

Схемой владеет C# (`dotnet/BooksEngine.Db`, EF Core-миграции); здесь таблицы только наполняются.

Как устроено:
- parquet → временные stg_*-таблицы через COPY (CSV из pyarrow: NULL — пустое поле, '' — "");
- внешние id Goodreads сопоставляются внутренним через external_ids; новым выдаются id
  из последовательности таблицы в порядке Goodreads-id — на пустой БД раскладка id воспроизводима;
- upsert только изменившихся строк (IS DISTINCT FROM), сущности Goodreads, пропавшие из parquet,
  удаляются. Если на пропавшее произведение ссылаются оценки/полки пользователей приложения,
  FK (ON DELETE RESTRICT) роняет загрузку — оценки молча не теряются;
- works.best_edition_id вставляется сразу: FK на editions отложенный (DEFERRABLE в миграции);
- всё в одной транзакции под advisory lock (две загрузки параллельно не идут), в конце сверка числа строк с manifest.json: расхождение → откат.

Идемпотентность: загрузка того же manifest (по sha256) пропускается; с --force выполняется,
но не меняет ни одной строки.
"""
import hashlib
import io
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import duckdb
import psycopg
import pyarrow.csv as pacsv

from booksengine.paths import CLEAN_DIR

SOURCE = "goodreads"


def pg_dsn(dbname: str | None = None) -> str:
    """Строка подключения из POSTGRES_* (.env читается в booksengine.paths)."""
    env = os.environ.get
    return (f"host={env('POSTGRES_HOST', 'localhost')} port={env('POSTGRES_PORT', '5432')} "
            f"dbname={dbname or env('POSTGRES_DB', 'booksengine')} user={env('POSTGRES_USER', 'booksengine')} "
            f"password={env('POSTGRES_PASSWORD', 'booksengine')}")


@dataclass(frozen=True)
class Entity:
    """Сущность каталога с внешним id: parquet-таблица → таблица БД."""
    entity_type: str      # значение external_ids.entity_type
    parquet: str
    table: str
    gr_key: str           # колонка с Goodreads-id в parquet
    columns: dict[str, str]  # колонка (имя в parquet и в БД совпадает) → тип PostgreSQL


# Не переносятся: works.best_edition_title (всегда равен title), works.media_type (размечен
# неверно, фильтровать по нему нельзя), works.books_count (считается по editions),
# editions.author_ids (авторы — в work_authors).
AUTHORS = Entity("author", "authors", "authors", "author_id", {
    "name": "text", "average_rating": "double precision", "ratings_count": "bigint",
})
WORKS = Entity("work", "works", "works", "work_id", {
    "title": "text", "original_title": "text", "publication_year": "integer",
    "language_code": "text", "description": "text", "is_collection": "boolean",
    "in_cf": "boolean", "cf_ratings": "integer", "cf_mean_rating": "double precision",
    "gr_ratings_count": "bigint", "gr_ratings_sum": "bigint",
})
EDITIONS = Entity("edition", "editions", "editions", "book_id", {
    "title": "text", "title_without_series": "text", "isbn": "text", "isbn13": "text",
    "asin": "text", "kindle_asin": "text", "language_code": "text", "country_code": "text",
    "format": "text", "is_ebook": "boolean", "num_pages": "integer",
    "publication_year": "integer", "publisher": "text", "ratings_count": "bigint",
    "average_rating": "double precision", "image_url": "text", "url": "text",
})

# Таблицы manifest.json, которые попадают в БД. users / ratings / shelf_events — нет:
# обучение читает parquet, в БД модель попадёт векторами (решение 2026-09-23).
LOADED_TABLES = ["works", "editions", "authors", "work_authors", "work_genres"]


def manifest_sha256(clean_dir: Path) -> str:
    return hashlib.sha256((clean_dir / "manifest.json").read_bytes()).hexdigest()


def load(clean_dir: Path = CLEAN_DIR, dsn: str | None = None, force: bool = False) -> dict:
    """Загрузить каталог. Возвращает {"skipped": bool, "changes": {...}, "counts": {...}}."""
    manifest = json.loads((clean_dir / "manifest.json").read_text())
    expected = {t: manifest["outputs"][t]["rows"] for t in LOADED_TABLES}
    sha = manifest_sha256(clean_dir)

    with psycopg.connect(dsn or pg_dsn()) as conn:
        source_id = _source_id(conn)
        if not conn.execute("SELECT pg_try_advisory_xact_lock(hashtext('booksengine.load-db'))").fetchone()[0]:
            raise RuntimeError("Другая загрузка каталога уже идёт в эту БД")
        last = conn.execute(
            "SELECT manifest_sha256 FROM source_loads WHERE source_id = %s ORDER BY id DESC LIMIT 1",
            (source_id,)).fetchone()
        if last and last[0] == sha and not force:
            print(f"Каталог из этого manifest уже загружен (sha256 {sha[:12]}); --force — загрузить заново")
            return {"skipped": True, "changes": {}, "counts": {}}

        log = _Timer()
        _stage(conn, duckdb.connect(), clean_dir, log)
        changes = _merge(conn, source_id, log)
        counts = _verify(conn, source_id, expected)
        log("сверка с manifest")
        conn.execute("INSERT INTO source_loads (source_id, manifest_sha256, counts) VALUES (%s, %s, %s)",
                     (source_id, sha, json.dumps(counts)))
        # commit — на выходе из with; любое исключение выше откатывает всю загрузку
    for name, n in changes.items():
        if n:
            print(f"  {name}: {n}")
    print(f"Загружено: {counts}")
    return {"skipped": False, "changes": changes, "counts": counts}


class _Timer:
    def __init__(self) -> None:
        self.t0 = time.monotonic()

    def __call__(self, step: str) -> None:
        print(f"  [{time.monotonic() - self.t0:6.1f} с] {step}", flush=True)


def _source_id(conn: psycopg.Connection) -> int:
    try:
        row = conn.execute("SELECT id FROM sources WHERE code = %s", (SOURCE,)).fetchone()
    except psycopg.errors.UndefinedTable as e:
        raise RuntimeError("Схемы нет: примените миграции "
                           "(cd dotnet/BooksEngine.Db && dotnet ef database update)") from e
    if row is None:
        raise RuntimeError(f"Нет источника '{SOURCE}' в sources — миграции применены не полностью")
    return row[0]


# ---------- staging ----------

def _stage_specs(clean_dir: Path) -> dict[str, tuple[str, str]]:
    """stg-таблица → (DDL колонок, SELECT DuckDB в том же порядке)."""
    def p(name: str) -> str:
        return f"read_parquet('{clean_dir / (name + '.parquet')}')"

    def entity(e: Entity, extra_ddl: str = "", extra_sel: str = "") -> tuple[str, str]:
        ddl = f"gr_id bigint NOT NULL{extra_ddl}, " + ", ".join(
            f"{c} {t}" for c, t in e.columns.items())
        sel = f"SELECT {e.gr_key}{extra_sel}, " + ", ".join(e.columns) + f" FROM {p(e.parquet)}"
        return ddl, sel

    return {
        "stg_authors": entity(AUTHORS),
        "stg_works": entity(WORKS, ", gr_best_edition bigint NOT NULL", ", best_book_id"),
        "stg_editions": entity(EDITIONS, ", gr_work bigint NOT NULL", ", work_id"),
        "stg_work_authors": ("gr_work bigint NOT NULL, gr_author bigint NOT NULL, role text, position smallint NOT NULL",
                             f"SELECT work_id, author_id, role, position FROM {p('work_authors')}"),
        "stg_work_genres": ("gr_work bigint NOT NULL, genre text NOT NULL, votes integer NOT NULL, share double precision NOT NULL",
                            f"SELECT work_id, genre, votes, share FROM {p('work_genres')}"),
    }


def _stage(conn: psycopg.Connection, duck: duckdb.DuckDBPyConnection, clean_dir: Path, log: _Timer) -> None:
    for table, (ddl, select) in _stage_specs(clean_dir).items():
        conn.execute(f"CREATE TEMP TABLE {table} ({ddl}) ON COMMIT DROP")
        reader = duck.execute(select).to_arrow_reader(batch_size=200_000)
        with conn.cursor().copy(f"COPY {table} FROM STDIN (FORMAT csv)") as copy:
            for batch in reader:
                buf = io.BytesIO()
                pacsv.write_csv(batch, buf, pacsv.WriteOptions(include_header=False))
                copy.write(buf.getbuffer())
        conn.execute(f"ANALYZE {table}")
        log(f"COPY {table}")


# ---------- merge ----------

def _map_ids(conn: psycopg.Connection, source_id: int, e: Entity, stg: str) -> dict[str, int]:
    """map_<type>(gr_id → id): существующие из external_ids, новым — nextval в порядке gr_id."""
    m = f"map_{e.entity_type}"
    conn.execute(f"""
        CREATE TEMP TABLE {m} ON COMMIT DROP AS
        SELECT s.gr_id, x.internal_id AS id
        FROM {stg} s LEFT JOIN external_ids x
          ON x.source_id = {source_id} AND x.entity_type = '{e.entity_type}' AND x.external_id = s.gr_id::text
    """)
    conn.execute(f"""
        UPDATE {m} SET id = n.id
        FROM (SELECT gr_id, nextval(pg_get_serial_sequence('{e.table}', 'id')) AS id
              FROM (SELECT gr_id FROM {m} WHERE id IS NULL ORDER BY gr_id) o) n
        WHERE {m}.gr_id = n.gr_id
    """)
    conn.execute(f"ALTER TABLE {m} ADD PRIMARY KEY (gr_id)")
    conn.execute(f"CREATE UNIQUE INDEX ON {m} (id)")
    conn.execute(f"ANALYZE {m}")
    new = conn.execute(f"""
        INSERT INTO external_ids (source_id, entity_type, external_id, internal_id)
        SELECT {source_id}, '{e.entity_type}', gr_id::text, id FROM {m}
        ON CONFLICT DO NOTHING
    """).rowcount
    return {f"{e.table}: новых": new}


def _delete_stale(conn: psycopg.Connection, source_id: int, e: Entity) -> dict[str, int]:
    """Сущности Goodreads, которых больше нет в parquet: удалить вместе с их external_ids."""
    m = f"map_{e.entity_type}"
    conn.execute(f"""
        CREATE TEMP TABLE stale_{e.entity_type} ON COMMIT DROP AS
        SELECT x.external_id, x.internal_id FROM external_ids x
        WHERE x.source_id = {source_id} AND x.entity_type = '{e.entity_type}'
          AND NOT EXISTS (SELECT 1 FROM {m} WHERE {m}.id = x.internal_id)
    """)
    try:
        n = conn.execute(f"DELETE FROM {e.table} t USING stale_{e.entity_type} s WHERE t.id = s.internal_id").rowcount
    except psycopg.errors.ForeignKeyViolation as err:
        raise RuntimeError(f"В новом каталоге нет {e.table}, на которые ссылаются данные пользователей "
                           f"приложения (оценки/полки). Загрузка отменена, БД не изменена.\n{err}") from err
    conn.execute(f"""
        DELETE FROM external_ids x USING stale_{e.entity_type} s
        WHERE x.source_id = {source_id} AND x.entity_type = '{e.entity_type}' AND x.external_id = s.external_id
    """)
    return {f"{e.table}: удалено": n}


def _upsert(conn: psycopg.Connection, table: str, key: list[str], cols: list[str], select: str) -> int:
    """INSERT … ON CONFLICT: обновляет только строки, где что-то изменилось."""
    all_cols = key + cols
    if not cols:
        sql = f"INSERT INTO {table} ({', '.join(all_cols)}) {select} ON CONFLICT DO NOTHING"
    else:
        sets = ", ".join(f"{c} = excluded.{c}" for c in cols)
        cur = ", ".join(f"{table}.{c}" for c in cols)
        exc = ", ".join(f"excluded.{c}" for c in cols)
        sql = (f"INSERT INTO {table} ({', '.join(all_cols)}) {select} "
               f"ON CONFLICT ({', '.join(key)}) DO UPDATE SET {sets} "
               f"WHERE ({cur}) IS DISTINCT FROM ({exc})")
    return conn.execute(sql).rowcount


def _upsert_entity(conn: psycopg.Connection, e: Entity, stg: str, extra: dict[str, str] | None = None,
                   join: str = "") -> dict[str, int]:
    """Вставить/обновить строки сущности; extra — дополнительные колонки (колонка БД → выражение SQL)."""
    extra = extra or {}
    cols = list(e.columns) + list(extra)
    exprs = [f"s.{c}" for c in e.columns] + list(extra.values())
    select = (f"SELECT m.id, {', '.join(exprs)} FROM {stg} s "
              f"JOIN map_{e.entity_type} m ON m.gr_id = s.gr_id {join}")
    return {f"{e.table}: вставлено/изменено": _upsert(conn, e.table, ["id"], cols, select)}


def _replace_links(conn: psycopg.Connection, table: str, key: list[str], cols: list[str],
                   new_select: str) -> dict[str, int]:
    """Связи произведений Goodreads: удалить отсутствующие в новых данных, остальные upsert.

    Связи произведения считаются принадлежащими источнику, из которого оно загружено; при слиянии
    нескольких источников в одно произведение сюда понадобится колонка source_id.
    """
    conn.execute(f"CREATE TEMP TABLE new_{table} ON COMMIT DROP AS {new_select}")
    on = " AND ".join(f"n.{k} = t.{k}" for k in key)
    deleted = conn.execute(f"""
        DELETE FROM {table} t USING map_work mw
        WHERE t.work_id = mw.id AND NOT EXISTS (SELECT 1 FROM new_{table} n WHERE {on})
    """).rowcount
    changed = _upsert(conn, table, key, cols, f"SELECT {', '.join(key + cols)} FROM new_{table}")
    return {f"{table}: удалено": deleted, f"{table}: вставлено/изменено": changed}


def _merge(conn: psycopg.Connection, source_id: int, log: _Timer) -> dict[str, int]:
    changes: dict[str, int] = {}
    for e, stg in [(AUTHORS, "stg_authors"), (WORKS, "stg_works"), (EDITIONS, "stg_editions")]:
        changes |= _map_ids(conn, source_id, e, stg)
    log("сопоставление id")

    # Удаление до вставки: связи и издания удалённых произведений уходят каскадом.
    for e in (EDITIONS, WORKS, AUTHORS):
        changes |= _delete_stale(conn, source_id, e)
    log("удаление пропавших")

    changes |= _upsert_entity(conn, AUTHORS, "stg_authors")
    log("authors")
    # Произведение ссылается на издание, которое вставится следующим шагом: FK проверится при коммите.
    conn.execute("SET CONSTRAINTS fk_works_editions_best_edition_id DEFERRED")
    changes |= _upsert_entity(conn, WORKS, "stg_works", {"best_edition_id": "me.id"},
                              "JOIN map_edition me ON me.gr_id = s.gr_best_edition")
    log("works")
    changes |= _upsert_entity(conn, EDITIONS, "stg_editions", {"work_id": "mw.id"},
                              "JOIN map_work mw ON mw.gr_id = s.gr_work")
    log("editions")

    changes["genres: новых"] = conn.execute(
        "INSERT INTO genres (name) SELECT DISTINCT genre FROM stg_work_genres ORDER BY genre "
        "ON CONFLICT (name) DO NOTHING").rowcount

    changes |= _replace_links(conn, "work_authors", ["work_id", "author_id"], ["role", "position"], """
        SELECT mw.id AS work_id, ma.id AS author_id, s.role, s.position
        FROM stg_work_authors s JOIN map_work mw ON mw.gr_id = s.gr_work JOIN map_author ma ON ma.gr_id = s.gr_author
    """)
    log("work_authors")
    changes |= _replace_links(conn, "work_genres", ["work_id", "genre_id"], ["votes", "share"], """
        SELECT mw.id AS work_id, g.id AS genre_id, s.votes, s.share
        FROM stg_work_genres s JOIN map_work mw ON mw.gr_id = s.gr_work JOIN genres g ON g.name = s.genre
    """)
    log("genres, work_genres")
    return changes


# ---------- сверка ----------

def _verify(conn: psycopg.Connection, source_id: int, expected: dict[str, int]) -> dict[str, int]:
    """Число строк в БД (сущности источника) против manifest.json. Расхождение — исключение и откат."""
    def one(sql: str) -> int:
        return conn.execute(sql).fetchone()[0]

    def by_source(e: Entity) -> int:
        return one(f"SELECT count(*) FROM {e.table} t JOIN external_ids x ON x.internal_id = t.id "
                   f"AND x.source_id = {source_id} AND x.entity_type = '{e.entity_type}'")

    gr_works = f"(SELECT internal_id FROM external_ids WHERE source_id = {source_id} AND entity_type = 'work')"
    actual = {e.table: by_source(e) for e in (WORKS, EDITIONS, AUTHORS)} | {
        "work_authors": one(f"SELECT count(*) FROM work_authors WHERE work_id IN {gr_works}"),
        "work_genres": one(f"SELECT count(*) FROM work_genres WHERE work_id IN {gr_works}"),
    }
    problems = [f"{t}: в БД {actual[t]}, в manifest {expected[t]}" for t in expected if actual[t] != expected[t]]
    for e in (WORKS, EDITIONS, AUTHORS):
        n = one(f"SELECT count(*) FROM external_ids WHERE source_id = {source_id} AND entity_type = '{e.entity_type}'")
        if n != expected[e.table]:
            problems.append(f"external_ids для {e.table}: {n}, в manifest {expected[e.table]}")
    no_best = one(f"SELECT count(*) FROM works WHERE best_edition_id IS NULL AND id IN {gr_works}")
    if no_best:
        problems.append(f"произведений без best_edition_id: {no_best}")
    if problems:
        raise RuntimeError("Сверка с manifest.json не прошла, загрузка отменена:\n  " + "\n  ".join(problems))
    return actual
