"""Полный проход clean.* + export дважды на синтетическом наборе — проверка детерминизма."""
import duckdb

from booksengine.data import clean, pipeline

CFG = {
    "signal": {"explicit_min": 1, "explicit_max": 5},
    "collections": {"title_patterns": [r"zznomatchzz"], "drop_from_ratings": False},
    "nonbooks": {"title_patterns": [r"zznomatchzz"], "exceptions": []},
    "duplicates": {"max_shadow_share": 0.1, "adaptation_patterns": [r"zznomatchzz"]},
    "users": {"low_variance_max_sd": 0.2, "low_variance_min_ratings": 100, "max_ratings": 1000,
              "monotone_max_mode_share": 0.9},
    "kcore": {"min_user_ratings": 1, "min_work_ratings": 1},
}


def _build_staging(con) -> None:
    con.execute("""
        CREATE TABLE interactions AS SELECT * FROM (VALUES
            (1, 10, 1, 5, 0), (1, 11, 1, 4, 0), (1, 12, 1, 3, 0),
            (2, 10, 1, 4, 0), (2, 12, 1, 5, 0),
            (3, 12, 1, 2, 0))
        t(user_id, book_id, is_read, rating, is_reviewed)""")
    con.execute("CREATE TABLE book_id_map AS SELECT * FROM (VALUES (10, 10), (11, 11), (12, 12)) "
                "t(book_id_csv, book_id)")
    con.execute("CREATE TABLE user_id_map AS SELECT * FROM (VALUES "
                "(1, 'ext-user-1'), (2, 'ext-user-2'), (3, 'ext-user-3')) t(user_id_csv, user_id)")
    con.execute("""
        CREATE TABLE editions AS
        SELECT 10 AS book_id, 100 AS work_id, 'Dune' AS title, 'Dune' AS title_without_series,
               'desc' AS description, 'eng' AS language_code, 1965 AS publication_year, 50 AS ratings_count,
               [struct_pack(author_id := 1, role := NULL::VARCHAR)] AS authors,
               '9780441013593' AS isbn, '9780441013593' AS isbn13, NULL AS asin, NULL AS kindle_asin,
               'US' AS country_code, 'Paperback' AS format, false AS is_ebook, 412 AS num_pages,
               'Ace Books' AS publisher, 4.2 AS average_rating, NULL AS image_url, NULL AS url
        UNION ALL
        SELECT 11, 100, 'Dune (Special)', 'Dune (Special)', 'desc', 'eng', 1990, 5,
               [struct_pack(author_id := 1, role := NULL::VARCHAR)],
               NULL, NULL, NULL, NULL, 'US', 'Hardcover', false, 500, 'Ace Books', 4.3, NULL, NULL
        UNION ALL
        SELECT 12, 200, 'Emma', 'Emma', 'desc', 'eng', 1815, 30,
               [struct_pack(author_id := 2, role := NULL::VARCHAR)],
               '9780141439587' AS isbn, NULL, NULL, NULL, 'GB', 'Paperback', false, 474, 'Penguin', 4.1, NULL, NULL
    """)
    con.execute("""
        CREATE TABLE works AS SELECT * FROM (VALUES
            (100, 10, 'Dune', 1965, 'book', 2, 5, 20),
            (200, 12, 'Emma', 1815, 'book', 1, 3, 12))
        t(work_id, best_book_id, original_title, original_publication_year, media_type, books_count,
          ratings_count, ratings_sum)""")
    con.execute("CREATE TABLE authors AS SELECT * FROM (VALUES "
                "(1, 'Frank Herbert', 4.2, 1000), (2, 'Jane Austen', 4.1, 2000)) "
                "t(author_id, name, average_rating, ratings_count)")
    con.execute("CREATE TABLE genres (book_id BIGINT, genre VARCHAR, votes INTEGER)")


def _run(clean_dir, monkeypatch) -> dict:
    """Полный проход pipeline.prepare()'s clean-этапа (без staging/profile/validate) + export."""
    monkeypatch.setattr(pipeline, "CLEAN_DIR", clean_dir)
    clean.SPILL_DIR = None  # только TEMP-таблицы в памяти — прогоны друг другу не мешают
    con = duckdb.connect()
    _build_staging(con)
    log = clean.CleaningLog()
    clean.structural(con, log)
    clean.build_catalog(con, log, CFG["collections"]["title_patterns"])
    clean.link_to_works(con, log)
    s = CFG["signal"]
    clean.to_work_level(con, log, s["explicit_min"], s["explicit_max"])
    nb = CFG["nonbooks"]
    clean.flag_nonbooks(con, nb["title_patterns"], nb["exceptions"])
    clean.drop_nonbooks(con, log)
    clean.primary_authors(con)
    d = CFG["duplicates"]
    clean.find_duplicates(con, d["adaptation_patterns"], d["max_shadow_share"])
    clean.merge_duplicates(con, log)
    u = CFG["users"]
    clean.filter_users(con, log, u["low_variance_max_sd"], u["low_variance_min_ratings"], u["max_ratings"],
                        max_mode_share=u["monotone_max_mode_share"])
    k = CFG["kcore"]
    clean.apply_kcore(con, log, k["min_user_ratings"], k["min_work_ratings"])
    clean_dir.mkdir(parents=True, exist_ok=True)
    outputs = pipeline.export(con)
    con.close()
    return outputs


def test_clean_and_export_deterministic_across_runs(tmp_path, monkeypatch):
    out_a = _run(tmp_path / "a", monkeypatch)
    out_b = _run(tmp_path / "b", monkeypatch)
    assert out_a.keys() == out_b.keys()
    for table, a in out_a.items():
        b = out_b[table]
        assert (a["rows"], a["checksum"]) == (b["rows"], b["checksum"]), f"{table}: {a} != {b}"
    # sanity: прогон реально что-то произвёл, сравнение не тривиально по пустым таблицам
    assert out_a["ratings"]["rows"] == 5
    assert out_a["users"]["rows"] == 3
