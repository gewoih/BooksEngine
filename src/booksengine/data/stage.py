"""Этап staging: сырые файлы датасета → типизированный Parquet (zstd).

Здесь нет никакой фильтрации — только разбор форматов. Битые строки считаются и попадают в статистику,
чтобы очистка и отчёт работали с полной картиной.
"""
import json
import time
from pathlib import Path

import duckdb

from booksengine.db import connect, ensure_free_space
from booksengine.paths import STAGING_DIR, raw_path

PARQUET = "(FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 1000000)"


def _nz(col: str) -> str:
    """Пустая строка в JSON датасета означает отсутствие значения."""
    return f"NULLIF(trim({col}), '')"


def _int(col: str, typ: str = "BIGINT") -> str:
    return f"TRY_CAST({_nz(col)} AS {typ})"


def _copy(con: duckdb.DuckDBPyConnection, query: str, name: str) -> dict:
    ensure_free_space()
    out = STAGING_DIR / f"{name}.parquet"
    t0 = time.time()
    con.execute(f"COPY ({query}) TO '{out}' {PARQUET}")
    rows = con.execute(f"SELECT count(*) FROM '{out}'").fetchone()[0]
    stat = {"rows": rows, "seconds": round(time.time() - t0, 1), "bytes": out.stat().st_size}
    print(f"  {name}: {rows:,} строк, {stat['bytes'] / 1e6:,.0f} МБ, {stat['seconds']} с")
    return stat


def _count_lines(path: Path) -> int:
    n = 0
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            n += chunk.count(b"\n")
    return n


def _json_source(con, path: Path, columns: dict[str, str]) -> str:
    cols = "{" + ", ".join(f"'{k}': '{v}'" for k, v in columns.items()) + "}"
    return (
        f"read_json('{path}', format='newline_delimited', columns={cols}, "
        f"ignore_errors=true, maximum_object_size=67108864)"
    )


def stage_interactions(con) -> dict:
    src = raw_path("interactions")
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW raw_interactions AS
        SELECT * FROM read_csv('{src}', header=true, store_rejects=true,
            columns={{'user_id':'INTEGER','book_id':'INTEGER','is_read':'TINYINT',
                      'rating':'TINYINT','is_reviewed':'TINYINT'}})
    """)
    stat = _copy(con, "SELECT * FROM raw_interactions", "interactions")
    stat["rejected_lines"] = con.execute("SELECT count(*) FROM reject_errors").fetchone()[0]
    stat["raw_lines"] = _count_lines(src) - 1
    return stat


def stage_id_maps(con) -> dict:
    stats = {}
    for name, key, val, typ in [
        ("book_id_map", "book_id_csv", "book_id", "BIGINT"),
        ("user_id_map", "user_id_csv", "user_id", "VARCHAR"),
    ]:
        src = raw_path(name)
        stats[name] = _copy(
            con,
            f"SELECT * FROM read_csv('{src}', header=true, columns={{'{key}':'INTEGER','{val}':'{typ}'}})",
            name,
        )
        stats[name]["raw_lines"] = _count_lines(src) - 1
    return stats


def stage_editions(con) -> dict:
    src = raw_path("books")
    columns = {
        "book_id": "VARCHAR", "work_id": "VARCHAR", "title": "VARCHAR", "title_without_series": "VARCHAR",
        "authors": "STRUCT(author_id VARCHAR, role VARCHAR)[]", "isbn": "VARCHAR", "isbn13": "VARCHAR",
        "asin": "VARCHAR", "kindle_asin": "VARCHAR", "language_code": "VARCHAR", "country_code": "VARCHAR",
        "format": "VARCHAR", "is_ebook": "VARCHAR", "num_pages": "VARCHAR", "publication_year": "VARCHAR",
        "publisher": "VARCHAR", "series": "VARCHAR[]", "ratings_count": "VARCHAR", "average_rating": "VARCHAR",
        "text_reviews_count": "VARCHAR", "popular_shelves": "STRUCT(count VARCHAR, name VARCHAR)[]",
        "similar_books": "VARCHAR[]", "description": "VARCHAR", "image_url": "VARCHAR", "url": "VARCHAR",
    }
    query = f"""
        SELECT
            {_int('book_id')} AS book_id,
            {_int('work_id')} AS work_id,
            book_id AS book_id_raw, work_id AS work_id_raw,
            {_nz('title')} AS title,
            {_nz('title_without_series')} AS title_without_series,
            list_transform(authors, a -> struct_pack(
                author_id := TRY_CAST(NULLIF(trim(a.author_id), '') AS BIGINT),
                role := NULLIF(trim(a.role), ''))) AS authors,
            {_nz('isbn')} AS isbn, {_nz('isbn13')} AS isbn13,
            {_nz('asin')} AS asin, {_nz('kindle_asin')} AS kindle_asin,
            {_nz('language_code')} AS language_code, {_nz('country_code')} AS country_code,
            {_nz('format')} AS format, is_ebook = 'true' AS is_ebook,
            {_int('num_pages', 'INTEGER')} AS num_pages,
            {_int('publication_year', 'INTEGER')} AS publication_year,
            {_nz('publisher')} AS publisher,
            list_transform(series, s -> TRY_CAST(s AS BIGINT)) AS series,
            {_int('ratings_count')} AS ratings_count,
            TRY_CAST({_nz('average_rating')} AS DOUBLE) AS average_rating,
            {_int('text_reviews_count')} AS text_reviews_count,
            list_transform(popular_shelves[1:20], s -> struct_pack(
                name := s.name, count := TRY_CAST(s.count AS INTEGER))) AS popular_shelves,
            len(popular_shelves) AS n_popular_shelves,
            list_transform(similar_books, s -> TRY_CAST(s AS BIGINT)) AS similar_books,
            {_nz('description')} AS description,
            {_nz('image_url')} AS image_url, {_nz('url')} AS url
        FROM {_json_source(con, src, columns)}
    """
    stat = _copy(con, query, "editions")
    stat["raw_lines"] = _count_lines(src)
    return stat


def stage_works(con) -> dict:
    src = raw_path("works")
    columns = {k: "VARCHAR" for k in [
        "work_id", "best_book_id", "books_count", "original_title", "original_publication_year",
        "original_publication_month", "original_publication_day", "original_language_id", "media_type",
        "ratings_count", "ratings_sum", "reviews_count", "text_reviews_count", "rating_dist",
        "default_description_language_code", "default_chaptering_book_id",
    ]}
    query = f"""
        SELECT
            {_int('work_id')} AS work_id, work_id AS work_id_raw,
            {_int('best_book_id')} AS best_book_id,
            {_int('books_count', 'INTEGER')} AS books_count,
            {_nz('original_title')} AS original_title,
            {_int('original_publication_year', 'INTEGER')} AS original_publication_year,
            {_int('original_language_id', 'INTEGER')} AS original_language_id,
            {_nz('media_type')} AS media_type,
            {_int('ratings_count')} AS ratings_count,
            {_int('ratings_sum')} AS ratings_sum,
            {_int('reviews_count')} AS reviews_count,
            {_int('text_reviews_count')} AS text_reviews_count,
            {_nz('rating_dist')} AS rating_dist
        FROM {_json_source(con, src, columns)}
    """
    stat = _copy(con, query, "works")
    stat["raw_lines"] = _count_lines(src)
    return stat


def stage_authors(con) -> dict:
    src = raw_path("authors")
    columns = {k: "VARCHAR" for k in ["author_id", "name", "average_rating", "ratings_count", "text_reviews_count"]}
    query = f"""
        SELECT {_int('author_id')} AS author_id, author_id AS author_id_raw, {_nz('name')} AS name,
               TRY_CAST({_nz('average_rating')} AS DOUBLE) AS average_rating,
               {_int('ratings_count')} AS ratings_count,
               {_int('text_reviews_count')} AS text_reviews_count
        FROM {_json_source(con, src, columns)}
    """
    stat = _copy(con, query, "authors")
    stat["raw_lines"] = _count_lines(src)
    return stat


def stage_genres(con) -> dict:
    src = raw_path("genres")
    columns = {"book_id": "VARCHAR", "genres": "MAP(VARCHAR, INTEGER)"}
    query = f"""
        SELECT {_int('book_id')} AS book_id, g.key AS genre, g.value AS votes
        FROM (SELECT book_id, unnest(map_entries(genres)) AS g FROM {_json_source(con, src, columns)})
    """
    stat = _copy(con, query, "genres")
    stat["raw_lines"] = _count_lines(src)
    return stat


STEPS = {
    "interactions": stage_interactions,
    "id_maps": stage_id_maps,
    "works": stage_works,
    "authors": stage_authors,
    "genres": stage_genres,
    "editions": stage_editions,
}


def run(only: list[str] | None = None, force: bool = False) -> dict:
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    stats_path = STAGING_DIR / "stage_stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    con = connect()
    for name, fn in STEPS.items():
        if only and name not in only:
            continue
        if name in stats and not force:
            print(f"  {name}: уже в staging, пропуск")
            continue
        print(f"[stage] {name}")
        stats[name] = fn(con)
        stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    con.close()
    return stats
