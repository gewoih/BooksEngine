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
    if not isinstance(publisher_raw, str) or not publisher_raw:
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


def apply_kcore(con: duckdb.DuckDBPyConnection, min_user: int = MIN_USER, min_work: int = MIN_WORK) -> dict:
    """Итеративный k-core на _az_ratings по (user_id, parent_asin). Переиспользует `clean.apply_kcore` —
    внутри него имя колонки произведения жёстко `work_id`, поэтому parent_asin временно переименовывается."""
    from booksengine.data import clean

    con.execute("CREATE OR REPLACE TEMP TABLE ratings_w AS SELECT user_id, parent_asin AS work_id, rating, "
                "timestamp, source FROM _az_ratings")
    log = clean.CleaningLog()
    clean.apply_kcore(con, log, min_user, min_work)
    con.execute("CREATE OR REPLACE TEMP TABLE _az_ratings AS SELECT user_id, work_id AS parent_asin, rating, "
                "timestamp, source FROM ratings_w")
    con.execute("DROP TABLE ratings_w")
    return log.records()[-1]


def export(con: duckdb.DuckDBPyConnection, out_dir: Path) -> dict:
    """ratings/items/bridge.parquet в out_dir. items — от _az_ratings (после k-core), не от _az_meta:
    оценённая книга без meta-строки (JSON не распарсился) не теряется, просто с пустыми полями."""
    out_dir.mkdir(parents=True, exist_ok=True)
    opts = "(FORMAT parquet, COMPRESSION zstd)"
    con.execute(f"COPY (SELECT user_id, parent_asin, rating, timestamp, source FROM _az_ratings "
                f"ORDER BY parent_asin, user_id) TO '{out_dir / 'ratings.parquet'}' {opts}")

    counts = con.execute("SELECT parent_asin, count(*) AS n_ratings FROM _az_ratings GROUP BY 1").df()
    meta = con.execute("SELECT parent_asin, title, author, isbn10, isbn13, publisher_raw, language, "
                       "categories, source FROM _az_meta").df()
    items = counts.merge(meta, on="parent_asin", how="left")
    items["year"] = items["publisher_raw"].map(extract_year)
    items = items[["parent_asin", "title", "author", "isbn10", "isbn13", "year", "categories",
                   "n_ratings", "source"]].set_index("parent_asin").reset_index()
    items.to_parquet(out_dir / "items.parquet", index=False)

    con.execute(f"COPY (SELECT b.parent_asin, b.work_id FROM bridge b "
                f"JOIN (SELECT DISTINCT parent_asin FROM _az_ratings) r USING (parent_asin) "
                f"ORDER BY b.parent_asin) TO '{out_dir / 'bridge.parquet'}' {opts}")

    bridged = con.execute(
        "SELECT count(*) FROM bridge b JOIN (SELECT DISTINCT parent_asin FROM _az_ratings) r "
        "USING (parent_asin) WHERE b.work_id IS NOT NULL").fetchone()[0]
    return {"ratings": int(con.execute("SELECT count(*) FROM _az_ratings").fetchone()[0]),
            "items": len(items), "bridged": bridged, "new": len(items) - bridged}


def apply_translation_signal(clean_dir: Path, cache_path: Path, query=None) -> dict:
    """items.parquet получает колонку ru_translation_known: True только для книг без моста на Goodreads, у
    которых нашёлся русский перевод в Wikidata по ISBN-13. С мостом или без ISBN-13 — False (нет сигнала —
    не советуем, решение принято на этапе дизайна)."""
    from booksengine.data import wikidata

    items = pd.read_parquet(clean_dir / "items.parquet")
    bridge = pd.read_parquet(clean_dir / "bridge.parquet")
    unbridged = set(bridge.loc[bridge.work_id.isna(), "parent_asin"])

    items["ru_translation_known"] = False
    candidates = items[items.parent_asin.isin(unbridged) & items.isbn13.notna()]
    if len(candidates):
        result = wikidata.check_translations(candidates.isbn13.tolist(), cache_path,
                                             query=query or wikidata._query)
        has_ru = candidates.isbn13.map(result).fillna(False)
        items.loc[candidates.index, "ru_translation_known"] = has_ru.values

    items.to_parquet(clean_dir / "items.parquet", index=False)
    return {"candidates": len(candidates), "with_translation": int(items.ru_translation_known.sum())}


def prepare(raw_dir: Path, out_dir: Path, editions_path: Path, cache_path: Path, min_user: int = MIN_USER,
           min_work: int = MIN_WORK, translation_query=None) -> dict:
    """Полный прогон: raw_dir/amazon_reviews_2023/{Books,Kindle_Store}.csv.gz + meta_*.jsonl.gz -> staging ->
    k-core -> мост на Goodreads (editions_path) -> экспорт -> сигнал перевода (Wikidata, кэш в cache_path) ->
    out_dir. `translation_query` — подменяется в тестах, без него — настоящий Wikidata."""
    base = raw_dir / "amazon_reviews_2023"
    con = duckdb.connect()
    stage_ratings(con, base / "Books.csv.gz", base / "Kindle_Store.csv.gz")
    stage_meta(con, base / "meta_Books.jsonl.gz", base / "meta_Kindle_Store.jsonl.gz")
    kcore_step = apply_kcore(con, min_user, min_work)
    build_bridge(con, editions_path)
    counts = export(con, out_dir)
    con.close()
    translation = apply_translation_signal(out_dir, cache_path, query=translation_query)
    return {"kcore": kcore_step, "export": counts, "translation": translation}


def report(manifest: dict, clean_dir: Path) -> str:
    """reports/amazon_bridge.md — люди/книги/оценки по источникам, сила моста, книги после 2017 без моста
    с известным переводом (Wikidata) — главное число для решения о следующем шаге."""
    from booksengine.paths import REPORTS_DIR

    items = pd.read_parquet(clean_dir / "items.parquet")
    bridge = pd.read_parquet(clean_dir / "bridge.parquet")
    ratings = pd.read_parquet(clean_dir / "ratings.parquet")

    lines = ["# Amazon Reviews'23: мост на Goodreads и сигнал перевода", ""]
    kcore = manifest.get("kcore") or {}
    if kcore:
        lines.append(f"- **k-core** ({MIN_USER}/{MIN_WORK}): {kcore.get('rows_before', 0):,} → "
                     f"{kcore.get('rows_after', 0):,} оценок ({kcore.get('users_after', 0):,} человек, "
                     f"{kcore.get('works_after', 0):,} книг)")
    for source in ("books", "kindle"):
        r = ratings[ratings.source == source]
        lines.append(f"- **{source}**: {r.user_id.nunique():,} человек, {r.parent_asin.nunique():,} книг, "
                     f"{len(r):,} оценок (после k-core {MIN_USER}/{MIN_WORK})")

    bridged = int(bridge.work_id.notna().sum())
    lines.append(f"- **Мост на Goodreads**: {bridged:,} из {len(bridge):,} книг "
                 f"({bridged / len(bridge) * 100:.1f}%)" if len(bridge) else "- **Мост на Goodreads**: нет данных")

    unbridged_asins = set(bridge.loc[bridge.work_id.isna(), "parent_asin"])
    post2017 = items[(items.year > 2017) & items.parent_asin.isin(unbridged_asins)]
    with_ru = int(post2017.get("ru_translation_known", pd.Series(dtype=bool)).sum())
    lines.append(f"- **Книг после 2017 без моста**: {len(post2017):,}, из них с известным переводом "
                 f"(Wikidata): {with_ru:,}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "amazon_bridge.md"
    out.write_text("\n".join(lines) + "\n")
    return str(out)
