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
