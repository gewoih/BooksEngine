import duckdb
import pytest

from booksengine.data import clean

PATTERNS = [r'#\s*\d+(\.\d+)?\s*(-|–|—)\s*#?\s*\d+', r'box(ed)?[ -]?set|omnibus']


@pytest.fixture
def con():
    c = duckdb.connect()
    # 3 издания: 10 и 11 — одно произведение 100, 12 — произведение 200 (сборник), 13 — без work_id
    c.execute("CREATE TABLE book_id_map AS SELECT * FROM (VALUES (0, 10), (1, 11), (2, 12), (3, 13), (4, 99)) "
              "t(book_id_csv, book_id)")
    c.execute("""CREATE TABLE editions AS SELECT * FROM (VALUES
        (10, 100, 'Dune', 'Dune', 'eng', 'd', 1965, 50),
        (11, 100, 'Dune (Special)', 'Dune (Special)', 'eng', NULL, 1990, 5),
        (12, 200, 'Saga Boxed Set (Saga, #1-3)', 'Saga Boxed Set', 'eng', NULL, 2000, 3),
        (13, NULL, 'Orphan', 'Orphan', NULL, NULL, NULL, 1),
        (14, 300, NULL, NULL, NULL, NULL, NULL, 0))
        t(book_id, work_id, title, title_without_series, language_code, description, publication_year,
          ratings_count)""")
    c.execute("""CREATE TABLE works AS SELECT * FROM (VALUES
        (100, 10, 'Dune', 1965, 'book', 2, 5, 20),
        (200, 12, NULL, NULL, NULL, 1, 1, 3),
        (300, 14, NULL, NULL, NULL, 1, 0, 0),
        (400, 99, 'No editions', NULL, NULL, 0, 0, 0))
        t(work_id, best_book_id, original_title, original_publication_year, media_type, books_count,
          ratings_count, ratings_sum)""")
    return c


def make_interactions(con, rows):
    con.execute("CREATE OR REPLACE TABLE interactions (user_id INT, book_id INT, is_read TINYINT, rating TINYINT, "
                "is_reviewed TINYINT)")
    if rows:
        con.executemany("INSERT INTO interactions VALUES (?, ?, ?, ?, ?)", rows)


def test_structural_removes_invalid_and_keeps_most_informative_duplicate(con):
    make_interactions(con, [(1, 0, 1, 4, 0), (1, 0, 0, 0, 0), (2, 0, 1, 7, 0), (None, 0, 1, 3, 0), (3, 1, 2, 3, 0)])
    log = clean.CleaningLog()
    clean.structural(con, log)
    rows = con.execute("SELECT user_id, book_id, rating FROM i_valid").fetchall()
    assert rows == [(1, 0, 4)]
    assert [s.rows_removed for s in log.steps] == [3, 1]


def test_catalog_drops_works_without_title_or_editions_and_flags_collections(con):
    log = clean.CleaningLog()
    clean.build_catalog(con, log, PATTERNS)
    works = dict(con.execute("SELECT work_id, is_collection FROM works_valid ORDER BY 1").fetchall())
    assert works == {100: False, 200: True}  # 300 без названия, 400 без изданий
    assert log.steps[0].detail["without_editions"] == 1
    assert con.execute("SELECT title FROM works_valid WHERE work_id = 100").fetchone()[0] == "Dune"


def test_link_to_works_counts_each_broken_link(con):
    make_interactions(con, [(1, 0, 1, 5, 0), (1, 3, 1, 5, 0), (1, 4, 1, 5, 0), (1, 9, 1, 5, 0)])
    log = clean.CleaningLog()
    clean.structural(con, log)
    clean.build_catalog(con, log, PATTERNS)
    clean.link_to_works(con, log)
    d = log.steps[-1].detail
    assert (d["no_book_id_map"], d["no_edition"], d["edition_without_work_id"]) == (1, 1, 1)
    assert con.execute("SELECT work_id FROM i_linked").fetchall() == [(100,)]


def test_editions_collapse_to_work_mean_and_shelf_is_separate(con):
    make_interactions(con, [(1, 0, 1, 5, 0), (1, 1, 1, 2, 0), (2, 0, 1, 0, 0), (3, 0, 0, 0, 0), (3, 1, 1, 4, 0)])
    log = clean.CleaningLog()
    clean.structural(con, log)
    clean.build_catalog(con, log, PATTERNS)
    clean.link_to_works(con, log)
    clean.to_work_level(con, log, 1, 5)
    ratings = con.execute("SELECT user_id, work_id, rating, n_editions FROM ratings_w ORDER BY 1").fetchall()
    assert ratings == [(1, 100, 3.5, 2), (3, 100, 4.0, 1)]
    # у пользователя 3 есть явная оценка другого издания — полочная запись не нужна
    assert con.execute("SELECT user_id, is_read FROM shelf_w").fetchall() == [(2, 1)]
    assert log.steps[-1].detail["pairs_conflict_ge2"] == 1


def test_filter_users_low_variance_and_hyperactive():
    con = duckdb.connect()
    rows = [(1, w, 5.0) for w in range(12)]                          # все пятёрки
    rows += [(2, w, float(1 + w % 5)) for w in range(12)]            # нормальный
    rows += [(3, w, float(1 + w % 3)) for w in range(30)]            # сверхактивный при max=20
    rows += [(4, w, 5.0) for w in range(5)]                          # все пятёрки, но мало оценок
    con.execute("CREATE TABLE ratings_w (user_id INT, work_id INT, rating REAL)")
    con.executemany("INSERT INTO ratings_w VALUES (?, ?, ?)", rows)
    log = clean.CleaningLog()
    clean.filter_users(con, log, max_sd=0.2, min_ratings_for_sd=10, max_ratings=20)
    users = [r[0] for r in con.execute("SELECT DISTINCT user_id FROM ratings_w ORDER BY 1").fetchall()]
    assert users == [2, 4]
    assert log.steps[0].detail["of_them_all_fives"] == 1


def test_kcore_converges_to_fixed_point():
    con = duckdb.connect()
    # users 1..3 оценили works 1..3 (плотное ядро); user 4 оценил works 1 и 9; work 9 только у user 4
    rows = [(u, w) for u in (1, 2, 3) for w in (1, 2, 3)] + [(4, 1), (4, 9)]
    con.execute("CREATE TABLE ratings_w (user_id INT, work_id INT, rating REAL)")
    con.executemany("INSERT INTO ratings_w VALUES (?, ?, 4)", rows)
    rows_, users, works, _ = clean.kcore(con, "ratings_w", 2, 2)
    # work 9 уходит → у user 4 остаётся 1 оценка → уходит и он
    assert (rows_, users, works) == (9, 3, 3)
    log = clean.CleaningLog()
    clean.apply_kcore(con, log, 2, 2)
    assert con.execute("SELECT min(c) FROM (SELECT count(*) c FROM ratings_w GROUP BY user_id)").fetchone()[0] >= 2
    assert con.execute("SELECT min(c) FROM (SELECT count(*) c FROM ratings_w GROUP BY work_id)").fetchone()[0] >= 2
