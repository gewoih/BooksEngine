"""Amazon Reviews'23 (McAuley Lab) — второй источник для книжной модели: мост на Goodreads по ISBN/ASIN и
сигнал перевода (Wikidata) для книг без моста. Дизайн: docs/superpowers/specs/2026-09-29-amazon-reviews-bridge-design.md.

Сырьё — RAW_DIR/amazon_reviews_2023/ (Books.csv.gz, Kindle_Store.csv.gz, meta_Books.jsonl.gz,
meta_Kindle_Store.jsonl.gz), см. scripts/fetch_amazon_raw.sh. Этот модуль сеть не трогает вообще, кроме
`apply_translation_signal` (через wikidata.check_translations).
"""
from pathlib import Path

import duckdb
import pandas as pd

MIN_USER = 5
MIN_WORK = 5


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
