"""Amazon Reviews'23 (McAuley Lab) — второй источник для книжной модели: мост на Goodreads по ISBN/ASIN и
сигнал перевода (Wikidata) для книг без моста. Дизайн: docs/superpowers/specs/2026-09-29-amazon-reviews-bridge-design.md.

Сырьё — RAW_DIR/amazon_reviews_2023/ (Books.csv.gz, Kindle_Store.csv.gz, meta_Books.jsonl.gz,
meta_Kindle_Store.jsonl.gz), см. scripts/fetch_amazon_raw.sh. Этот модуль сеть не трогает вообще, кроме
`apply_translation_signal` (через wikidata.check_translations).
"""
import re
from pathlib import Path

import duckdb
import pandas as pd

MIN_USER = 5
MIN_WORK = 5
CURRENT_YEAR = 2026


def stage_ratings(con: duckdb.DuckDBPyConnection, books_csv: Path, kindle_csv: Path) -> None:
    """_az_ratings: user_id, parent_asin, rating, timestamp, source ('books'|'kindle') — оба rating_only-
    файла как есть, без фильтрации."""
    con.execute("""
        CREATE OR REPLACE TEMP TABLE _az_ratings AS
        SELECT user_id::VARCHAR AS user_id, parent_asin::VARCHAR AS parent_asin, rating::DOUBLE AS rating,
               timestamp::BIGINT AS timestamp, 'books'::VARCHAR AS source
        FROM read_csv_auto(?)
        UNION ALL
        SELECT user_id::VARCHAR AS user_id, parent_asin::VARCHAR AS parent_asin, rating::DOUBLE AS rating,
               timestamp::BIGINT AS timestamp, 'kindle'::VARCHAR AS source
        FROM read_csv_auto(?)
    """, [str(books_csv), str(kindle_csv)])


def stage_meta(con: duckdb.DuckDBPyConnection, books_jsonl: Path, kindle_jsonl: Path) -> None:
    """_az_meta: parent_asin, title, author, isbn10, isbn13, publisher_raw, language, categories, source —
    поля из `details` (карта строка->строка на диске, как popular_shelves у книжных editions) — автор,
    ISBN-10/13, издатель+дата текстом, язык."""
    columns = {
        "parent_asin": "VARCHAR", "title": "VARCHAR", "author": "STRUCT(name VARCHAR)",
        "details": "MAP(VARCHAR, VARCHAR)", "categories": "VARCHAR[]",
    }
    cols_sql = "{" + ", ".join(f"'{k}': '{v}'" for k, v in columns.items()) + "}"

    def _src(path: Path) -> str:
        return (f"read_json('{path}', format='newline_delimited', columns={cols_sql}, "
                f"ignore_errors=true, maximum_object_size=67108864)")

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _az_meta AS
        SELECT parent_asin, title, author.name AS author, details['ISBN 10'] AS isbn10,
               details['ISBN 13'] AS isbn13, details['Publisher'] AS publisher_raw,
               details['Language'] AS language, array_to_string(categories, '|') AS categories,
               'books' AS source
        FROM {_src(books_jsonl)}
        UNION ALL
        SELECT parent_asin, title, author.name AS author, details['ISBN 10'] AS isbn10,
               details['ISBN 13'] AS isbn13, details['Publisher'] AS publisher_raw,
               details['Language'] AS language, array_to_string(categories, '|') AS categories,
               'kindle' AS source
        FROM {_src(kindle_jsonl)}
    """)


def extract_year(publisher_raw: str | None) -> int | None:
    """Год издания — минимальное правдоподобное 4-значное число (1900..CURRENT_YEAR) в тексте `Publisher`
    («Chatto & Windus; First Edition (January 1, 2004)» -> 2004). Минимальное, не максимальное: в тексте
    попадаются шумовые числа (артикулы, вес) — они почти всегда больше настоящей даты издания."""
    if not publisher_raw:
        return None
    years = [int(y) for y in re.findall(r"(?:19|20)\d{2}", publisher_raw)]
    plausible = [y for y in years if 1900 <= y <= CURRENT_YEAR]
    return min(plausible) if plausible else None


def build_bridge(con: duckdb.DuckDBPyConnection, editions_path: Path) -> None:
    """bridge: parent_asin, work_id (NULL — книги без пары в Goodreads). Books — по isbn10/isbn13 из
    _az_meta (только цифры и X, без дефисов) на editions.isbn/isbn13; Kindle — parent_asin напрямую на
    editions.kindle_asin, метаданные не нужны. Покрывает все parent_asin из _az_ratings, даже без записи в
    _az_meta (не распарсилась/нет ISBN) — тогда просто нет сигнала для сопоставления, work_id = NULL.
    GROUP BY + max() в конце — защита от дублей ISBN в editions (не размножает строки моста)."""
    con.execute("""
        CREATE OR REPLACE TEMP TABLE _editions_norm AS
        SELECT work_id, regexp_replace(upper(coalesce(isbn::VARCHAR, '')), '[^0-9X]', '', 'g') AS isbn10_n,
               regexp_replace(coalesce(isbn13::VARCHAR, ''), '[^0-9]', '', 'g') AS isbn13_n,
               kindle_asin::VARCHAR AS kindle_asin
        FROM read_parquet(?)
    """, [str(editions_path)])
    con.execute("""
        CREATE OR REPLACE TEMP TABLE bridge AS
        SELECT parent_asin, max(work_id) AS work_id FROM (
            SELECT i.parent_asin,
                   CASE WHEN i.source = 'kindle' THEN ek.work_id
                        ELSE coalesce(e10.work_id, e13.work_id) END AS work_id
            FROM (SELECT DISTINCT parent_asin, source FROM _az_ratings) i
            LEFT JOIN (SELECT parent_asin,
                              regexp_replace(upper(coalesce(isbn10::VARCHAR, '')), '[^0-9X]', '', 'g') AS isbn10_n,
                              regexp_replace(coalesce(isbn13::VARCHAR, ''), '[^0-9]', '', 'g') AS isbn13_n
                       FROM _az_meta WHERE source = 'books') m ON m.parent_asin = i.parent_asin
            LEFT JOIN _editions_norm e10 ON e10.isbn10_n = m.isbn10_n AND m.isbn10_n != ''
            LEFT JOIN _editions_norm e13 ON e13.isbn13_n = m.isbn13_n AND m.isbn13_n != ''
            LEFT JOIN _editions_norm ek ON ek.kindle_asin = i.parent_asin AND i.source = 'kindle'
        )
        GROUP BY parent_asin
    """)
