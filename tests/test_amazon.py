import gzip
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from booksengine.data import amazon


@pytest.fixture
def con():
    return duckdb.connect()


def test_stage_ratings_unions_books_and_kindle_with_source_column(tmp_path, con):
    books = tmp_path / "books.csv"
    books.write_text("user_id,parent_asin,rating,timestamp\nU1,B001,5.0,1000\nU2,B002,3.0,1001\n")
    kindle = tmp_path / "kindle.csv"
    kindle.write_text("user_id,parent_asin,rating,timestamp\nU1,K001,4.0,1002\n")

    amazon.stage_ratings(con, books, kindle)

    rows = con.execute(
        "SELECT user_id, parent_asin, rating, timestamp, source FROM _az_ratings ORDER BY parent_asin"
    ).fetchall()
    assert rows == [
        ("U1", "B001", 5.0, 1000, "books"),
        ("U2", "B002", 3.0, 1001, "books"),
        ("U1", "K001", 4.0, 1002, "kindle"),
    ]


def test_stage_meta_extracts_isbn_author_language_from_details(tmp_path, con):
    books = tmp_path / "meta_books.jsonl"
    books.write_text(json.dumps({
        "parent_asin": "0701169850", "title": "Chaucer",
        "author": {"name": "Peter Ackroyd"},
        "details": {"Publisher": "Chatto & Windus; First Edition (January 1, 2004)",
                    "Language": "English", "ISBN 10": "0701169850", "ISBN 13": "978-0701169855"},
        "categories": ["Books", "Literature & Fiction"],
    }) + "\n")
    kindle = tmp_path / "meta_kindle.jsonl"
    kindle.write_text(json.dumps({
        "parent_asin": "B0192CTMWI", "title": "Some Ebook",
        "author": {"name": "Jane Doe"},
        "details": {"Language": "English"},
        "categories": ["Kindle Store"],
    }) + "\n")

    amazon.stage_meta(con, books, kindle)

    rows = con.execute(
        "SELECT parent_asin, title, author, isbn10, isbn13, publisher_raw, language, categories, source "
        "FROM _az_meta ORDER BY source"
    ).fetchall()
    assert rows[0] == ("0701169850", "Chaucer", "Peter Ackroyd", "0701169850", "978-0701169855",
                       "Chatto & Windus; First Edition (January 1, 2004)", "English",
                       "Books|Literature & Fiction", "books")
    assert rows[1][0] == "B0192CTMWI" and rows[1][3] is None and rows[1][8] == "kindle"


def test_stage_meta_handles_book_without_details_or_categories(tmp_path, con):
    books = tmp_path / "meta_books.jsonl"
    books.write_text(json.dumps({"parent_asin": "X1", "title": "No Details Book"}) + "\n")
    kindle = tmp_path / "meta_kindle.jsonl"
    kindle.write_text("")

    amazon.stage_meta(con, books, kindle)

    row = con.execute("SELECT parent_asin, isbn10, categories FROM _az_meta").fetchone()
    assert row == ("X1", None, None)


def test_extract_year_picks_earliest_plausible_four_digit_year():
    assert amazon.extract_year("Chatto & Windus; First Edition (January 1, 2004)") == 2004
    assert amazon.extract_year("Heinemann; First Edition (May 20, 1996)") == 1996


def test_extract_year_ignores_out_of_range_noise_and_prefers_min_over_max():
    # в проверке на реальных данных max иногда цеплял шумовое число (2085-2099) из текста издателя
    assert amazon.extract_year("Some Press (2011); this edition 2018, item code 2094812") == 2011


def test_extract_year_returns_none_without_plausible_year():
    assert amazon.extract_year(None) is None
    assert amazon.extract_year("") is None
    assert amazon.extract_year("No year mentioned here") is None


def test_build_bridge_matches_books_by_isbn_and_kindle_by_asin(tmp_path, con):
    editions = pd.DataFrame([
        {"work_id": 1, "isbn": "0701169850", "isbn13": "9780701169855", "kindle_asin": None},
        {"work_id": 2, "isbn": None, "isbn13": None, "kindle_asin": "B0192CTMWI"},
    ])
    editions_path = tmp_path / "editions.parquet"
    editions.to_parquet(editions_path)

    con.execute("CREATE OR REPLACE TEMP TABLE _az_meta AS SELECT * FROM (VALUES "
                "('0701169850', 'books', '0701169850', '978-0701169855')) "
                "t(parent_asin, source, isbn10, isbn13)")
    con.execute("CREATE OR REPLACE TEMP TABLE _az_ratings AS SELECT * FROM (VALUES "
                "('U1', '0701169850', 5.0, 1000, 'books'), "
                "('U1', 'B0192CTMWI', 5.0, 1001, 'kindle'), "
                "('U1', 'ZZZUNKNOWN', 3.0, 1002, 'kindle')) "
                "t(user_id, parent_asin, rating, timestamp, source)")

    amazon.build_bridge(con, editions_path)

    rows = dict(con.execute("SELECT parent_asin, work_id FROM bridge").fetchall())
    assert rows == {"0701169850": 1, "B0192CTMWI": 2, "ZZZUNKNOWN": None}


def test_build_bridge_treats_missing_isbn_as_no_match(tmp_path, con):
    editions = pd.DataFrame([{"work_id": 1, "isbn": "1111111111", "isbn13": "9781111111111",
                              "kindle_asin": None}])
    editions_path = tmp_path / "editions.parquet"
    editions.to_parquet(editions_path)

    con.execute("CREATE OR REPLACE TEMP TABLE _az_meta AS SELECT * FROM (VALUES "
                "('NOISBN', 'books', NULL, NULL)) t(parent_asin, source, isbn10, isbn13)")
    con.execute("CREATE OR REPLACE TEMP TABLE _az_ratings AS SELECT * FROM (VALUES "
                "('U1', 'NOISBN', 5.0, 1000, 'books')) t(user_id, parent_asin, rating, timestamp, source)")

    amazon.build_bridge(con, editions_path)

    row = con.execute("SELECT work_id FROM bridge WHERE parent_asin = 'NOISBN'").fetchone()
    assert row == (None,)


def test_build_bridge_does_not_fan_out_on_duplicate_isbn_in_editions(tmp_path, con):
    # данные Goodreads не идеальны: два разных work_id с одним и тем же ISBN
    editions = pd.DataFrame([
        {"work_id": 1, "isbn": "2222222222", "isbn13": None, "kindle_asin": None},
        {"work_id": 2, "isbn": "2222222222", "isbn13": None, "kindle_asin": None},
    ])
    editions_path = tmp_path / "editions.parquet"
    editions.to_parquet(editions_path)

    con.execute("CREATE OR REPLACE TEMP TABLE _az_meta AS SELECT * FROM (VALUES "
                "('DUP', 'books', '2222222222', NULL)) t(parent_asin, source, isbn10, isbn13)")
    con.execute("CREATE OR REPLACE TEMP TABLE _az_ratings AS SELECT * FROM (VALUES "
                "('U1', 'DUP', 5.0, 1000, 'books')) t(user_id, parent_asin, rating, timestamp, source)")

    amazon.build_bridge(con, editions_path)

    rows = con.execute("SELECT parent_asin, work_id FROM bridge WHERE parent_asin = 'DUP'").fetchall()
    assert len(rows) == 1   # ровно одна строка мостa, не две — дубль ISBN не размножает bridge
    assert rows[0][1] in (1, 2)


def test_build_bridge_does_not_match_isbn_less_books_to_isbn_less_editions(tmp_path, con):
    # в настоящем editions ~1 млн изданий без ISBN-10 и ~0.8 млн без ISBN-13: пустое не должно совпадать с пустым
    editions = pd.DataFrame([
        {"work_id": 7, "isbn": None, "isbn13": None, "kindle_asin": None},
        {"work_id": 8, "isbn": "", "isbn13": "", "kindle_asin": None},
    ])
    editions_path = tmp_path / "editions.parquet"
    editions.to_parquet(editions_path)

    con.execute("CREATE OR REPLACE TEMP TABLE _az_meta AS SELECT * FROM (VALUES "
                "('NOISBN', 'books', NULL, NULL), ('EMPTY', 'books', '', '')) "
                "t(parent_asin, source, isbn10, isbn13)")
    con.execute("CREATE OR REPLACE TEMP TABLE _az_ratings AS SELECT * FROM (VALUES "
                "('U1', 'NOISBN', 5.0, 1000, 'books'), ('U1', 'EMPTY', 5.0, 1001, 'books'), "
                "('U1', 'K9', 5.0, 1002, 'kindle')) t(user_id, parent_asin, rating, timestamp, source)")

    amazon.build_bridge(con, editions_path)

    rows = dict(con.execute("SELECT parent_asin, work_id FROM bridge").fetchall())
    assert rows == {"NOISBN": None, "EMPTY": None, "K9": None}


def test_bridge_query_uses_hash_joins_not_nested_loops(tmp_path, con):
    # nested-loop join по 2.4 млн изданий — ~час на реальных данных вместо долей секунды
    pd.DataFrame([{"work_id": 1, "isbn": "0701169850", "isbn13": "9780701169855", "kindle_asin": "K1"}]
                 ).to_parquet(tmp_path / "editions.parquet")
    con.execute("CREATE OR REPLACE TEMP TABLE _az_meta AS SELECT * FROM (VALUES "
                "('B1', 'books', '0701169850', '978-0701169855')) t(parent_asin, source, isbn10, isbn13)")
    con.execute("CREATE OR REPLACE TEMP TABLE _az_ratings AS SELECT * FROM (VALUES "
                "('U1', 'B1', 5.0, 1, 'books'), ('U1', 'K1', 5.0, 2, 'kindle')) "
                "t(user_id, parent_asin, rating, timestamp, source)")
    amazon.build_bridge(con, tmp_path / "editions.parquet")

    plan = con.execute(f"EXPLAIN {amazon.BRIDGE_SELECT}").fetchall()[0][1]
    assert "NL_JOIN" not in plan and "NESTED_LOOP" not in plan
    assert dict(con.execute("SELECT parent_asin, work_id FROM bridge").fetchall()) == {"B1": 1, "K1": 1}


def test_export_does_not_duplicate_items_when_meta_repeats_an_asin(tmp_path, con):
    con.execute("CREATE OR REPLACE TEMP TABLE _az_ratings AS SELECT * FROM (VALUES "
                "('U1', 'B001', 5.0, 1000, 'books')) t(user_id, parent_asin, rating, timestamp, source)")
    con.execute("CREATE OR REPLACE TEMP TABLE _az_meta AS SELECT * FROM (VALUES "
                "('B001', 'Title', 'Author', NULL, NULL, 'Pub (2019)', 'English', 'Books', 'books'), "
                "('B001', 'Title', 'Author', NULL, NULL, 'Pub (2019)', 'English', 'Books', 'books')) "
                "t(parent_asin, title, author, isbn10, isbn13, publisher_raw, language, categories, source)")
    con.execute("CREATE OR REPLACE TEMP TABLE bridge AS SELECT * FROM (VALUES ('B001', NULL::BIGINT)) "
                "t(parent_asin, work_id)")

    counts = amazon.export(con, tmp_path)

    assert counts["items"] == 1 and counts["new"] == 1
    assert len(pd.read_parquet(tmp_path / "items.parquet")) == 1


def test_connect_bounds_memory_and_spills_to_given_dir(tmp_path):
    con = amazon._connect(tmp_path / "spill")
    assert con.execute("SELECT current_setting('temp_directory')").fetchone()[0] == str(tmp_path / "spill")
    limit = con.execute("SELECT current_setting('memory_limit')").fetchone()[0]
    assert limit.endswith("GiB") and float(limit.split()[0]) <= 8


def test_apply_kcore_filters_low_volume_users_and_items(con):
    rows = ", ".join(f"('U{u}', 'B{w}', 5.0, 1000, 'books')" for u in range(5) for w in range(5))
    con.execute(f"CREATE OR REPLACE TEMP TABLE _az_ratings AS SELECT * FROM (VALUES {rows}, "
                f"('U9', 'B9', 5.0, 1000, 'books')) "   # 1 оценка у пользователя и книги — не пройдёт k-core
                f"t(user_id, parent_asin, rating, timestamp, source)")

    step = amazon.apply_kcore(con, min_user=5, min_work=5)

    remaining = con.execute("SELECT count(*) FROM _az_ratings").fetchone()[0]
    assert remaining == 25   # плотная матрица 5x5 проходит k-core целиком
    assert step["rule"] == "kcore"
    assert con.execute("SELECT count(*) FROM _az_ratings WHERE parent_asin = 'B9'").fetchone()[0] == 0


def test_export_writes_ratings_items_and_bridge_parquet(tmp_path, con):
    con.execute("CREATE OR REPLACE TEMP TABLE _az_ratings AS SELECT * FROM (VALUES "
                "('U1', 'B001', 5.0, 1000, 'books'), ('U2', 'B001', 4.0, 1001, 'books'), "
                "('U1', 'B002', 3.0, 1002, 'books')) "   # B002 оценена, но без записи в _az_meta
                "t(user_id, parent_asin, rating, timestamp, source)")
    con.execute("CREATE OR REPLACE TEMP TABLE _az_meta AS SELECT * FROM (VALUES "
                "('B001', 'Some Title', 'Some Author', '0701169850', '978-0701169855', "
                "'Publisher (2004)', 'English', 'Books|Fiction', 'books')) "
                "t(parent_asin, title, author, isbn10, isbn13, publisher_raw, language, categories, source)")
    con.execute("CREATE OR REPLACE TEMP TABLE bridge AS SELECT * FROM (VALUES "
                "('B001', 1), ('B002', NULL)) t(parent_asin, work_id)")

    counts = amazon.export(con, tmp_path)
    assert counts == {"ratings": 3, "items": 2, "bridged": 1, "new": 1}

    ratings = pd.read_parquet(tmp_path / "ratings.parquet")
    assert list(ratings.columns) == ["user_id", "parent_asin", "rating", "timestamp", "source"]
    assert len(ratings) == 3

    items = pd.read_parquet(tmp_path / "items.parquet").set_index("parent_asin")
    assert list(items.columns) == ["title", "author", "isbn10", "isbn13", "year", "categories",
                                   "n_ratings", "source"]
    assert items.loc["B001", "year"] == 2004
    assert items.loc["B001", "n_ratings"] == 2
    # B002 оценена, но нет meta-строки для неё — не должна пропасть из items, просто с пустыми полями
    assert items.loc["B002", "n_ratings"] == 1
    assert pd.isna(items.loc["B002", "title"])

    bridge = pd.read_parquet(tmp_path / "bridge.parquet")
    assert list(bridge.columns) == ["parent_asin", "work_id"]
    assert len(bridge) == 2




def _items(rows: list[dict]) -> pd.DataFrame:
    base = {"title": "T", "author": "A", "isbn10": None, "isbn13": None, "year": 2019, "categories": "Books",
            "n_ratings": 10, "source": "books"}
    return pd.DataFrame([base | r for r in rows])


def test_apply_translation_signal_checks_new_unbridged_books_and_records_status(tmp_path):
    _items([
        {"parent_asin": "A1", "isbn13": "978-0000000002"},                 # новая, без моста -> есть перевод
        {"parent_asin": "A2", "isbn13": "978-0000000019", "year": 2020},   # новая, в Wikidata без русского
        {"parent_asin": "A3", "isbn10": "0306406152", "year": 2021},       # только ISBN-10 -> в Wikidata нет
        {"parent_asin": "A4", "isbn13": "978-0000000026"},                 # с мостом — не проверяем
        {"parent_asin": "A5", "isbn13": "978-0000000033", "year": 2015},   # старая — не проверяем
        {"parent_asin": "A6", "isbn13": "978-0000000000"},                 # контрольная цифра не сходится
    ]).to_parquet(tmp_path / "items.parquet")
    pd.DataFrame([{"parent_asin": a, "work_id": 1 if a == "A4" else None}
                  for a in ("A1", "A2", "A3", "A4", "A5", "A6")]).to_parquet(tmp_path / "bridge.parquet")
    queried = []

    def fake_query(batch):
        queried.extend(batch)
        return {"9780000000002": True, "9780000000019": False}

    stats = amazon.apply_translation_signal(tmp_path, tmp_path / "cache.parquet", query=fake_query,
                                            rate_limit_s=0)

    assert sorted(queried) == ["9780000000002", "9780000000019", "9780306406157"]
    assert stats == {"candidates": 3, "checked": 3, "found": 2, "with_translation": 1, "failed": 0}
    items = pd.read_parquet(tmp_path / "items.parquet").set_index("parent_asin")
    assert items.wikidata.fillna("не проверялась").to_dict() == {
        "A1": "ru", "A2": "found", "A3": "absent",
        "A4": "не проверялась", "A5": "не проверялась", "A6": "не проверялась"}
    assert items.ru_translation_known.to_dict() == {"A1": True, "A2": False, "A3": False,
                                                    "A4": False, "A5": False, "A6": False}
    assert items.loc["A3", "isbn13_n"] == "9780306406157"


def test_apply_translation_signal_leaves_failed_requests_unchecked(tmp_path):
    _items([{"parent_asin": "A1", "isbn13": "978-0000000002"}]).to_parquet(tmp_path / "items.parquet")
    pd.DataFrame([{"parent_asin": "A1", "work_id": None}]).to_parquet(tmp_path / "bridge.parquet")

    def down(batch):
        raise OSError("504 Gateway Timeout")

    stats = amazon.apply_translation_signal(tmp_path, tmp_path / "cache.parquet", query=down,
                                            retries=2, backoff_s=0, rate_limit_s=0)

    assert stats == {"candidates": 1, "checked": 0, "found": 0, "with_translation": 0, "failed": 1}
    items = pd.read_parquet(tmp_path / "items.parquet")
    assert items.wikidata.isna().all() and not items.ru_translation_known.any()


def test_apply_translation_signal_is_noop_without_candidates(tmp_path):
    _items([{"parent_asin": "A1"}]).to_parquet(tmp_path / "items.parquet")
    pd.DataFrame([{"parent_asin": "A1", "work_id": None}]).to_parquet(tmp_path / "bridge.parquet")

    calls = []
    stats = amazon.apply_translation_signal(tmp_path, tmp_path / "cache.parquet",
                                            query=lambda b: calls.append(b) or {})
    assert stats == {"candidates": 0, "checked": 0, "found": 0, "with_translation": 0, "failed": 0}
    assert calls == []


def _write_gz(path, text: str) -> None:
    with gzip.open(path, "wt") as f:
        f.write(text)


def test_prepare_end_to_end_with_synthetic_raw_files(tmp_path):
    raw = tmp_path / "raw"
    base = raw / "amazon_reviews_2023"
    base.mkdir(parents=True)

    ratings_rows = "user_id,parent_asin,rating,timestamp\n" + "".join(
        f"U{u},B1,5.0,{1000 + u}\n" for u in range(5))
    _write_gz(base / "Books.csv.gz", ratings_rows)
    _write_gz(base / "Kindle_Store.csv.gz", "user_id,parent_asin,rating,timestamp\n")

    meta_row = json.dumps({
        "parent_asin": "B1", "title": "Test Book", "author": {"name": "A. Uthor"},
        "details": {"ISBN 10": "0306406152", "ISBN 13": "978-0306406157", "Publisher": "Pub (2019)",
                    "Language": "English"},
        "categories": ["Books", "Fiction"],
    })
    _write_gz(base / "meta_Books.jsonl.gz", meta_row + "\n")
    _write_gz(base / "meta_Kindle_Store.jsonl.gz", "")

    # пустой editions — мост не найдётся, B1 останется книгой без пары в Goodreads
    editions_path = tmp_path / "editions.parquet"
    pd.DataFrame(columns=["work_id", "isbn", "isbn13", "kindle_asin"]).astype(
        {"work_id": "int64"}).to_parquet(editions_path)

    out_dir = tmp_path / "clean"
    cache_path = tmp_path / "wikidata_cache.parquet"

    manifest = amazon.prepare(raw, out_dir, editions_path, cache_path, min_user=1, min_work=1,
                              translation_query=lambda batch: {"9780306406157": True})

    assert manifest["kcore"]["rule"] == "kcore"
    assert manifest["export"] == {"ratings": 5, "items": 1, "bridged": 0, "new": 1}
    assert manifest["translation"] == {"candidates": 1, "checked": 1, "found": 1, "with_translation": 1,
                                       "failed": 0}

    items = pd.read_parquet(out_dir / "items.parquet")
    assert items.iloc[0]["ru_translation_known"] == True
    assert items.iloc[0]["year"] == 2019
    assert not (tmp_path / "tmp").exists()   # спилл DuckDB убран за собой


def test_report_shows_bridge_per_source_and_translation_funnel(tmp_path, monkeypatch):
    from booksengine import paths
    monkeypatch.setattr(paths, "REPORTS_DIR", tmp_path)

    pd.DataFrame([
        {"user_id": "U1", "parent_asin": "B1", "rating": 5.0, "timestamp": 1, "source": "books"},
        {"user_id": "U2", "parent_asin": "B2", "rating": 4.0, "timestamp": 1, "source": "books"},
        {"user_id": "U1", "parent_asin": "B3", "rating": 4.0, "timestamp": 1, "source": "books"},
        {"user_id": "U1", "parent_asin": "K1", "rating": 4.0, "timestamp": 2, "source": "kindle"},
    ]).to_parquet(tmp_path / "ratings.parquet")
    items = _items([
        {"parent_asin": "B1", "title": "Educated", "author": "Tara Westover", "n_ratings": 900,
         "isbn13_n": "9780000000002", "wikidata": "ru"},
        {"parent_asin": "B2", "n_ratings": 150, "isbn13_n": "9780000000019", "wikidata": "absent"},
        {"parent_asin": "B3", "n_ratings": 5, "year": 2010},                       # с мостом, старая
        {"parent_asin": "K1", "source": "kindle", "n_ratings": 20, "isbn13_n": None, "wikidata": None},
    ])
    items["ru_translation_known"] = items.wikidata.eq("ru")
    items.to_parquet(tmp_path / "items.parquet")
    pd.DataFrame([{"parent_asin": "B1", "work_id": None}, {"parent_asin": "B2", "work_id": None},
                  {"parent_asin": "B3", "work_id": 7}, {"parent_asin": "K1", "work_id": None}]
                 ).to_parquet(tmp_path / "bridge.parquet")
    manifest = {"kcore": {}, "export": {}, "translation": {"failed": 3}}

    text = Path(amazon.report(manifest, tmp_path)).read_text()

    assert "- **books**: 2 человек, 3 книг, 3 оценок; мост на Goodreads — 1 (33.3%)" in text
    assert "- **kindle**: 1 человек, 1 книг, 1 оценок; мост на Goodreads — 0 (0.0%)" in text
    assert "| книг | 3 | 2 |" in text
    assert "| с валидным ISBN | 2 | 2 |" in text
    assert "| найдено в Wikidata | 1 | 1 |" in text
    assert "| с русским изданием | 1 | 1 |" in text
    assert "Educated — Tara Westover, 2019, 900 оценок" in text
    assert "не ответила для 3 книг" in text


def test_bridged_matrix_puts_amazon_readers_into_core_columns_with_scale_and_threshold(tmp_path):
    """Люди Amazon — строки в столбцах ядра Goodreads: только книги с мостом и из ядра, 0★ отброшены, два издания одной
    книги у человека (Books и Kindle) — одна оценка (среднее), порог — книг ядра у человека, шкала q: 4★ Amazon → 3★."""
    pd.DataFrame([
        {"user_id": "U1", "parent_asin": "B1", "rating": 5.0, "timestamp": 1, "source": "books"},
        {"user_id": "U1", "parent_asin": "K1", "rating": 4.0, "timestamp": 1, "source": "kindle"},  # то же, что B1
        {"user_id": "U1", "parent_asin": "B2", "rating": 4.0, "timestamp": 1, "source": "books"},
        {"user_id": "U1", "parent_asin": "B3", "rating": 3.0, "timestamp": 1, "source": "books"},   # вне ядра
        {"user_id": "U1", "parent_asin": "B4", "rating": 5.0, "timestamp": 1, "source": "books"},   # без моста
        {"user_id": "U2", "parent_asin": "B2", "rating": 5.0, "timestamp": 1, "source": "books"},   # 1 книга ядра
        {"user_id": "U3", "parent_asin": "B1", "rating": 0.0, "timestamp": 1, "source": "books"},   # 0★ — нет оценки
        {"user_id": "U3", "parent_asin": "B2", "rating": 2.0, "timestamp": 1, "source": "books"},
    ]).to_parquet(tmp_path / "ratings.parquet")
    pd.DataFrame([{"parent_asin": "B1", "work_id": 10}, {"parent_asin": "K1", "work_id": 10},
                  {"parent_asin": "B2", "work_id": 20}, {"parent_asin": "B3", "work_id": 99},
                  {"parent_asin": "B4", "work_id": None}]).to_parquet(tmp_path / "bridge.parquet")
    work_ids = np.array([10, 20, 30])

    X = amazon.bridged_matrix(tmp_path, work_ids, "q-2").toarray()
    # U1: книга 10 — среднее 4.5 → 5 → q: 5; книга 20 — 4 → q: 3. U2 и U3 — меньше 2 книг ядра.
    np.testing.assert_array_equal(X, [[5, 3, 0]])

    raw = amazon.bridged_matrix(tmp_path, work_ids, "raw-1").toarray()
    assert raw.shape == (3, 3)
    assert sorted(map(tuple, raw.tolist())) == [(0, 2, 0), (0, 5, 0), (5, 4, 0)]


def test_bridged_matrix_rejects_unknown_tag(tmp_path):
    with pytest.raises(ValueError):
        amazon.bridged_matrix(tmp_path, np.array([1]), "zzz-5")
