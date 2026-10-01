"""Amazon Reviews'23 (McAuley Lab) — второй источник для книжной модели: мост на Goodreads по ISBN/ASIN и
сигнал перевода (Wikidata) для книг без моста. Дизайн: docs/superpowers/specs/2026-09-29-amazon-reviews-bridge-design.md.

Сырьё — RAW_DIR/amazon_reviews_2023/ (Books.csv.gz, Kindle_Store.csv.gz, meta_Books.jsonl.gz,
meta_Kindle_Store.jsonl.gz), см. scripts/fetch_amazon_raw.sh. Этот модуль сеть не трогает вообще, кроме
`apply_translation_signal` (через wikidata.check_translations).
"""
import re
import shutil
from pathlib import Path

import duckdb
import pandas as pd

MIN_USER = 5
MIN_WORK = 5
CURRENT_YEAR = 2026
NEW_AFTER_YEAR = 2017   # датасет Goodreads кончается 2017-м


def _connect(spill_dir: Path, memory_limit: str = "8GB") -> duckdb.DuckDBPyConnection:
    """Как booksengine.db.connect (лимит памяти, спилл на диск), но спилл — в папку Amazon: db.connect пишет в
    tmp книжного домена, а без temp_directory DuckDB спиллит в ./.tmp корня репозитория."""
    spill_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET temp_directory='{spill_dir}'")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET enable_progress_bar=false")
    return con


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
    Издания сгруппированы по ключу заранее (дубль ISBN не размножает строки), пустые ключи отсечены там же:
    условие на левую сторону в ON превращает LEFT JOIN в DuckDB во вложенный цикл — час на 2.4 млн изданий."""
    con.execute("""
        CREATE OR REPLACE TEMP TABLE _editions_norm AS
        SELECT work_id, regexp_replace(upper(coalesce(isbn::VARCHAR, '')), '[^0-9X]', '', 'g') AS isbn10_n,
               regexp_replace(coalesce(isbn13::VARCHAR, ''), '[^0-9]', '', 'g') AS isbn13_n,
               kindle_asin::VARCHAR AS kindle_asin
        FROM read_parquet(?)
    """, [str(editions_path)])
    con.execute(f"CREATE OR REPLACE TEMP TABLE bridge AS {BRIDGE_SELECT}")


BRIDGE_SELECT = """
    SELECT parent_asin, max(work_id) AS work_id FROM (
        SELECT i.parent_asin,
               CASE WHEN i.source = 'kindle' THEN ek.work_id
                    ELSE coalesce(e10.work_id, e13.work_id) END AS work_id
        FROM (SELECT DISTINCT parent_asin, source FROM _az_ratings) i
        LEFT JOIN (SELECT parent_asin,
                          regexp_replace(upper(coalesce(isbn10::VARCHAR, '')), '[^0-9X]', '', 'g') AS isbn10_n,
                          regexp_replace(coalesce(isbn13::VARCHAR, ''), '[^0-9]', '', 'g') AS isbn13_n
                   FROM _az_meta WHERE source = 'books') m ON m.parent_asin = i.parent_asin
        LEFT JOIN (SELECT isbn10_n, max(work_id) AS work_id FROM _editions_norm
                   WHERE isbn10_n != '' GROUP BY 1) e10 ON e10.isbn10_n = m.isbn10_n
        LEFT JOIN (SELECT isbn13_n, max(work_id) AS work_id FROM _editions_norm
                   WHERE isbn13_n != '' GROUP BY 1) e13 ON e13.isbn13_n = m.isbn13_n
        LEFT JOIN (SELECT kindle_asin, max(work_id) AS work_id FROM _editions_norm
                   WHERE kindle_asin IS NOT NULL AND kindle_asin != '' GROUP BY 1) ek ON ek.kindle_asin = i.parent_asin
    )
    GROUP BY parent_asin
"""


def apply_kcore(con: duckdb.DuckDBPyConnection, min_user: int = MIN_USER, min_work: int = MIN_WORK) -> dict:
    """Итеративный k-core на _az_ratings по (user_id, parent_asin). Переиспользует `clean.apply_kcore` —
    внутри него имя колонки произведения жёстко `work_id`, поэтому parent_asin временно переименовывается."""
    from booksengine.data import clean

    con.execute("CREATE OR REPLACE TEMP TABLE ratings_w AS SELECT user_id, parent_asin AS work_id, rating, "
                "timestamp, source FROM _az_ratings")
    con.execute("DROP TABLE _az_ratings")   # ~55 млн строк: не держать третью копию, пока k-core строит свою
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

    # join в SQL: в pandas попадают только оценённые книги, а не все ~6 млн строк метаданных; повтор parent_asin
    # в метаданных не размножает книгу
    items = con.execute("""
        SELECT c.parent_asin, m.title, m.author, m.isbn10, m.isbn13, m.publisher_raw, m.categories,
               c.n_ratings, c.source
        FROM (SELECT parent_asin, count(*) AS n_ratings, any_value(source) AS source
              FROM _az_ratings GROUP BY 1) c
        LEFT JOIN (SELECT * FROM _az_meta
                   QUALIFY row_number() OVER (PARTITION BY parent_asin ORDER BY source) = 1) m USING (parent_asin)
    """).df()
    items["year"] = items["publisher_raw"].map(extract_year)
    items = items[["parent_asin", "title", "author", "isbn10", "isbn13", "year", "categories",
                   "n_ratings", "source"]]
    items.to_parquet(out_dir / "items.parquet", index=False)

    con.execute(f"COPY (SELECT b.parent_asin, b.work_id FROM bridge b "
                f"JOIN (SELECT DISTINCT parent_asin FROM _az_ratings) r USING (parent_asin) "
                f"ORDER BY b.parent_asin) TO '{out_dir / 'bridge.parquet'}' {opts}")

    bridged = con.execute(
        "SELECT count(*) FROM bridge b JOIN (SELECT DISTINCT parent_asin FROM _az_ratings) r "
        "USING (parent_asin) WHERE b.work_id IS NOT NULL").fetchone()[0]
    return {"ratings": int(con.execute("SELECT count(*) FROM _az_ratings").fetchone()[0]),
            "items": len(items), "bridged": bridged, "new": len(items) - bridged}


# Шкала звёзд Amazon во входе толпы. raw — как есть; q — по квантилям Goodreads: у Amazon 64% пятёрок против 33%,
# и 4★ Amazon по месту в распределении — это 3★ Goodreads (ниже 37% оценок Amazon — ниже 30% у Goodreads).
AMAZON_SCALES = {"raw": (1, 2, 3, 4, 5), "q": (1, 2, 3, 3, 5)}


def bridged_matrix(clean_dir: Path, work_ids, tag: str):
    """Люди Amazon строками обучения толпы (`EASELike.fit(extra=...)`): оценки книг, у которых есть мост на Goodreads и
    которые входят в ядро (`work_ids` — столбцы матрицы обучения). tag — «<шкала>-<порог>»: шкала из AMAZON_SCALES,
    порог — не меньше стольких книг ядра у человека. Несколько изданий одной книги у человека (Books и Kindle) —
    среднее, округлённое как у Goodreads; 0★ — не оценка."""
    import numpy as np
    import scipy.sparse as sp

    from booksengine.model.matrix import columns
    from booksengine.model.metrics import rounded

    scale, _, threshold = tag.partition("-")
    if scale not in AMAZON_SCALES or not threshold.isdigit():
        raise ValueError(f"тег Amazon «{tag}»: нужно <{'|'.join(AMAZON_SCALES)}>-<порог>, например q-5")
    con = duckdb.connect()
    con.register("core", pd.DataFrame({"work_id": np.asarray(work_ids)}))
    d = con.execute("""
        WITH w AS (
            SELECT r.user_id, b.work_id, avg(r.rating) AS rating
            FROM read_parquet(?) r JOIN read_parquet(?) b USING (parent_asin)
            WHERE b.work_id IS NOT NULL AND r.rating >= 1 AND b.work_id IN (SELECT work_id FROM core)
            GROUP BY 1, 2)
        SELECT user_id, work_id, rating FROM w
        WHERE user_id IN (SELECT user_id FROM w GROUP BY 1 HAVING count(*) >= ?)
        ORDER BY user_id, work_id
    """, [str(clean_dir / "ratings.parquet"), str(clean_dir / "bridge.parquet"), int(threshold)]).fetchnumpy()
    con.close()
    stars = np.asarray(AMAZON_SCALES[scale], dtype=np.float32)[rounded(d["rating"]).astype(int) - 1]
    _, rows = np.unique(d["user_id"], return_inverse=True)
    return sp.csr_matrix((stars, (rows, columns(np.asarray(work_ids), d["work_id"].astype(np.int64)))),
                         shape=(int(rows.max()) + 1 if len(rows) else 0, len(work_ids)))


def apply_translation_signal(clean_dir: Path, cache_path: Path, query=None, **check_kwargs) -> dict:
    """items.parquet получает колонки: isbn13_n — ISBN-13 цифрами (из isbn13, иначе из isbn10); wikidata — статус
    проверки: 'ru' (есть русское издание), 'found' (в Wikidata есть, русского нет), 'absent' (в Wikidata нет),
    пусто — не проверялась (с мостом, не новее NEW_AFTER_YEAR, без валидного ISBN, запрос не прошёл);
    ru_translation_known = (wikidata == 'ru'). Проверяются только новые книги без моста: ради них сигнал и
    нужен (у книг с мостом перевод виден по Goodreads). `check_kwargs` — в wikidata.check_translations (повторы, паузы)."""
    from booksengine.data import wikidata

    items = pd.read_parquet(clean_dir / "items.parquet")
    bridge = pd.read_parquet(clean_dir / "bridge.parquet")
    unbridged = items.parent_asin.isin(bridge.loc[bridge.work_id.isna(), "parent_asin"])

    from13 = items.isbn13.map(wikidata.normalize_isbn13)
    items["isbn13_n"] = from13.where(from13.notna(), items.isbn10.map(wikidata.normalize_isbn13))
    new = pd.to_numeric(items.year, errors="coerce") > NEW_AFTER_YEAR
    is_candidate = unbridged & new & items.isbn13_n.notna()

    status = {}
    if is_candidate.any():
        status = wikidata.check_translations(items.loc[is_candidate, "isbn13_n"].tolist(), cache_path,
                                             query=query or wikidata._query, **check_kwargs)
    items["wikidata"] = items.isbn13_n.map(status).where(is_candidate, None)
    items["ru_translation_known"] = items.wikidata.eq("ru")
    items.to_parquet(clean_dir / "items.parquet", index=False)

    checked = int(items.wikidata.notna().sum())
    return {"candidates": int(is_candidate.sum()), "checked": checked,
            "found": int(items.wikidata.isin(["ru", "found"]).sum()),
            "with_translation": int(items.ru_translation_known.sum()),
            "failed": int(is_candidate.sum()) - checked}


def prepare(raw_dir: Path, out_dir: Path, editions_path: Path, cache_path: Path, min_user: int = MIN_USER,
           min_work: int = MIN_WORK, translation_query=None) -> dict:
    """Полный прогон: raw_dir/amazon_reviews_2023/{Books,Kindle_Store}.csv.gz + meta_*.jsonl.gz -> staging ->
    k-core -> мост на Goodreads (editions_path) -> экспорт -> сигнал перевода (Wikidata, кэш в cache_path) ->
    out_dir. `translation_query` — подменяется в тестах, без него — настоящий Wikidata."""
    base = raw_dir / "amazon_reviews_2023"
    spill_dir = out_dir.parent / "tmp"
    con = _connect(spill_dir)
    stage_ratings(con, base / "Books.csv.gz", base / "Kindle_Store.csv.gz")
    stage_meta(con, base / "meta_Books.jsonl.gz", base / "meta_Kindle_Store.jsonl.gz")
    kcore_step = apply_kcore(con, min_user, min_work)
    build_bridge(con, editions_path)
    counts = export(con, out_dir)
    con.close()
    shutil.rmtree(spill_dir, ignore_errors=True)
    translation = apply_translation_signal(out_dir, cache_path, query=translation_query)
    return {"kcore": kcore_step, "export": counts, "translation": translation}


FUNNEL = [
    ("книг", lambda d: d),
    ("с валидным ISBN", lambda d: d[d.isbn13_n.notna()]),
    ("проверено в Wikidata", lambda d: d[d.wikidata.notna()]),
    ("найдено в Wikidata", lambda d: d[d.wikidata.isin(["ru", "found"])]),
    ("с русским изданием", lambda d: d[d.wikidata.eq("ru")]),
]


def report(manifest: dict, clean_dir: Path, top: int = 20) -> str:
    """reports/amazon_bridge.md — люди/книги/оценки и мост по источникам; воронка новых книг без моста до
    «есть русское издание» (все и с 100+ оценками — метрика проверки); примеры названий — глазами проверить,
    не переиздания ли это старых книг (год — издания Amazon, а не произведения)."""
    from booksengine.paths import REPORTS_DIR

    items = pd.read_parquet(clean_dir / "items.parquet").merge(
        pd.read_parquet(clean_dir / "bridge.parquet"), on="parent_asin", how="left")
    ratings = pd.read_parquet(clean_dir / "ratings.parquet")

    lines = ["# Amazon Reviews'23: мост на Goodreads и сигнал перевода", ""]
    kcore = manifest.get("kcore") or {}
    if kcore:
        lines.append(f"- **k-core** ({MIN_USER}/{MIN_WORK}): {kcore.get('rows_before', 0):,} → "
                     f"{kcore.get('rows_after', 0):,} оценок ({kcore.get('users_after', 0):,} человек, "
                     f"{kcore.get('works_after', 0):,} книг)")
    for source in ("books", "kindle"):
        r, s = ratings[ratings.source == source], items[items.source == source]
        bridged = int(s.work_id.notna().sum())
        share = f" ({bridged / len(s):.1%})" if len(s) else ""
        lines.append(f"- **{source}**: {r.user_id.nunique():,} человек, {len(s):,} книг, {len(r):,} оценок; "
                     f"мост на Goodreads — {bridged:,}{share}")

    new = items[(pd.to_numeric(items.year, errors="coerce") > NEW_AFTER_YEAR) & items.work_id.isna()]
    strong = new[new.n_ratings >= 100]
    lines += ["", f"## Книги после {NEW_AFTER_YEAR} года без пары в Goodreads", "",
              "| | все | 100+ оценок |", "|---|---|---|"]
    lines += [f"| {label} | {len(pick(new)):,} | {len(pick(strong)):,} |" for label, pick in FUNNEL]
    failed = (manifest.get("translation") or {}).get("failed", 0)
    if failed:
        lines += ["", f"Wikidata не ответила для {failed:,} книг — повторный `amazon-bridge` дозапросит их "
                      "(ответы кэшируются)."]

    def titles(d: pd.DataFrame) -> list[str]:
        return [f"- {t.title} — {t.author}, {int(t.year)}, {t.n_ratings:,} оценок"
                for t in d.nlargest(top, "n_ratings").itertuples()]

    lines += ["", "С русским изданием, по числу оценок:", ""] + titles(new[new.wikidata.eq("ru")])
    lines += ["", "Все новые без пары, по числу оценок (не переиздания ли старых книг):", ""] + titles(new)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "amazon_bridge.md"
    out.write_text("\n".join(lines) + "\n")
    return str(out)
