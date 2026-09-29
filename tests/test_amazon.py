import json

import duckdb
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
