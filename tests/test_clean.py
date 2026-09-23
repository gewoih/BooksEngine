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


def test_editions_collapse_to_work_mean_and_shelf_dropped(con):
    make_interactions(con, [(1, 0, 1, 5, 0), (1, 1, 1, 2, 0), (2, 0, 1, 0, 0), (3, 0, 0, 0, 0), (3, 1, 1, 4, 0)])
    log = clean.CleaningLog()
    clean.structural(con, log)
    clean.build_catalog(con, log, PATTERNS)
    clean.link_to_works(con, log)
    clean.to_work_level(con, log, 1, 5)
    ratings = con.execute("SELECT user_id, work_id, rating, n_editions FROM ratings_w ORDER BY 1").fetchall()
    assert ratings == [(1, 100, 3.5, 2), (3, 100, 4.0, 1)]
    # записи без оценки (rating = 0) отбрасываются, не хранятся (решение 2026-09-23)
    split_step = log.steps[-2]
    assert split_step.rule == "split_explicit_signal" and split_step.rows_removed == 2
    assert con.execute("SELECT count(*) FROM duckdb_tables() WHERE table_name = 'shelf_w'").fetchone()[0] == 0
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


NONBOOK = [r'sheet music|piano[ /-]vocal', r'\bcolou?ring book\b']


def make_works_valid(con, rows):
    con.execute("CREATE OR REPLACE TABLE works_valid (work_id BIGINT, title VARCHAR, best_edition_title VARCHAR, "
                "is_collection BOOLEAN)")
    con.executemany("INSERT INTO works_valid VALUES (?, ?, ?, false)", rows)


def test_nonbooks_flagged_by_title_with_exceptions_and_dropped(con):
    make_works_valid(con, [(1, "Chamber of Secrets: Sheet Music for Flute", "Chamber of Secrets: Sheet Music"),
                           (2, "Sheet Music: A Rock 'N' Roll Love Story", "Sheet Music"),
                           (3, "ACOTAR Coloring Book", "ACOTAR Coloring Book"),
                           (4, "Dune", "Dune (Dune, #1)")])
    assert clean.flag_nonbooks(con, NONBOOK, exceptions=[2]) == 2
    assert con.execute("SELECT count(*) FROM works_valid WHERE is_nonbook IS NULL").fetchone()[0] == 0
    assert con.execute("SELECT work_id FROM works_valid WHERE is_nonbook ORDER BY 1").fetchall() == [(1,), (3,)]
    con.execute("CREATE OR REPLACE TABLE ratings_w AS SELECT * FROM (VALUES (1, 1, 5.0, 1), (1, 4, 4.0, 1), "
                "(2, 3, 3.0, 1), (2, 2, 4.0, 1)) t(user_id, work_id, rating, n_editions)")
    log = clean.CleaningLog()
    clean.drop_nonbooks(con, log)
    assert con.execute("SELECT work_id FROM ratings_w ORDER BY 1").fetchall() == [(2,), (4,)]
    assert log.steps[0].rule == "nonbooks" and log.steps[0].rows_removed == 2


def test_nonbooks_without_exceptions_flags_everything_matching(con):
    make_works_valid(con, [(1, "Wicked Piano/Vocal", "Wicked"), (2, "Dune", "Dune")])
    assert clean.flag_nonbooks(con, NONBOOK, exceptions=[]) == 1
    assert con.execute("SELECT work_id, is_nonbook FROM works_valid ORDER BY 1").fetchall() == [(1, True), (2, False)]


ADAPT = [r'graphic novel|oxford bookworms', r'\bpart (\d+|one|two)\b']


def dup_fixture(con, works, authors, ratings):
    make_works_valid(con, works)
    con.execute("CREATE OR REPLACE TABLE work_primary (work_id BIGINT, author_id BIGINT)")
    con.executemany("INSERT INTO work_primary VALUES (?, ?)", authors)
    con.execute("CREATE OR REPLACE TABLE ratings_w (user_id INT, work_id BIGINT, rating REAL, n_editions SMALLINT)")
    con.executemany("INSERT INTO ratings_w VALUES (?, ?, ?, ?)", ratings)


def _ratings_for(work_id, n, start=0):
    return [(u, work_id, 4.0, 1) for u in range(start, start + n)]


def test_duplicates_merge_only_confident_shadows(con):
    works = [(1, "The Hobbit", "The Hobbit (Middle-earth, #0)"), (2, "The Hobbit.", "The Hobbit"),
             (3, "The Hobbit: Graphic Novel", "The Hobbit: Graphic Novel"),   # адаптация — не тень
             (4, "Vampire Academy", "Vampire Academy (VA, #1)"), (5, "Vampire Academy", "Vampire Academy (VA, #2)"),
             (6, "Dune", "Dune"), (7, "Dune", "Dune"),                        # другой автор — не тень
             (8, "Emma", "Emma"), (9, "Emma", "Emma"),                        # тень ≥ 10% — не тень
             (10, "...", "..."), (11, "!!!", "!!!"),                          # пустой ключ
             (12, "Анна Каренина", "Анна Каренина"), (13, "Анна Каренина", "Анна Каренина"),
             (14, "Musashi", "Musashi"), (15, "Musashi", "Musashi (Musashi, #2)")]   # номер только у тени — том
    authors = [(1, 1), (2, 1), (3, 1), (4, 2), (5, 2), (6, 3), (7, 4), (8, 5), (9, 5), (10, 6), (11, 6),
               (12, 7), (13, 7), (14, 8), (15, 8)]
    ratings = (_ratings_for(1, 100) + _ratings_for(2, 5, 1000) + _ratings_for(3, 5, 2000)
               + _ratings_for(4, 100) + _ratings_for(5, 5) + _ratings_for(6, 100) + _ratings_for(7, 5)
               + _ratings_for(8, 100) + _ratings_for(9, 20) + _ratings_for(10, 100) + _ratings_for(11, 5)
               + _ratings_for(12, 100) + _ratings_for(13, 5) + _ratings_for(14, 100) + _ratings_for(15, 5))
    dup_fixture(con, works, authors, ratings)
    assert clean.find_duplicates(con, ADAPT, 0.1) == 2
    assert con.execute("SELECT * FROM dup_map ORDER BY 1").fetchall() == [(2, 1), (13, 12)]


def test_merge_duplicates_weighted_average_and_editions_summed(con):
    make_works_valid(con, [(1, "Emma", "Emma"), (2, "Emma", "Emma")])
    con.execute("CREATE OR REPLACE TABLE dup_map AS SELECT 2::BIGINT AS shadow_work_id, 1::BIGINT AS main_work_id")
    con.execute("CREATE OR REPLACE TABLE ratings_w (user_id INT, work_id BIGINT, rating REAL, n_editions SMALLINT)")
    con.executemany("INSERT INTO ratings_w VALUES (?, ?, ?, ?)", [(1, 1, 5.0, 3), (1, 2, 1.0, 1), (2, 2, 3.0, 1)])
    log = clean.CleaningLog()
    clean.merge_duplicates(con, log)
    assert con.execute("SELECT user_id, work_id, rating, n_editions FROM ratings_w ORDER BY 1").fetchall() == \
        [(1, 1, 4.0, 4), (2, 1, 3.0, 1)]
    assert log.steps[0].rule == "duplicates" and log.steps[0].rows_removed == 1
    assert log.steps[0].detail == {"shadows": 1, "users_rated_both": 1}


def test_monotone_users_dropped_by_mode_share_with_half_up_rounding(con):
    rows = ([(1, w, 5.0, 1) for w in range(9)] + [(1, 9, 3.0, 1)]          # 90% пятёрок → удалить
            + [(2, w, 4.5, 1) for w in range(9)] + [(2, 9, 1.0, 1)]        # 4.5 → 5: 90% → удалить
            + [(3, w, 5.0, 1) for w in range(8)] + [(3, 8, 3.0, 1), (3, 9, 1.0, 1)]   # 80% → оставить
            + [(4, w, 5.0, 1) for w in range(9)])                           # 9 оценок < 10 → оставить
    con.execute("CREATE OR REPLACE TABLE ratings_w (user_id INT, work_id BIGINT, rating REAL, n_editions SMALLINT)")
    con.executemany("INSERT INTO ratings_w VALUES (?, ?, ?, ?)", rows)
    log = clean.CleaningLog()
    clean.filter_users(con, log, max_sd=0.2, min_ratings_for_sd=10, max_ratings=3000, max_mode_share=0.9)
    assert [s.rule for s in log.steps] == ["users_low_variance", "users_monotone", "users_hyperactive"]
    assert con.execute("SELECT DISTINCT user_id FROM ratings_w ORDER BY 1").fetchall() == [(3,), (4,)]


def test_filter_users_without_mode_share_keeps_old_steps(con):
    con.execute("CREATE OR REPLACE TABLE ratings_w AS SELECT 1 AS user_id, 1::BIGINT AS work_id, 5.0::REAL AS rating, "
                "1::SMALLINT AS n_editions")
    log = clean.CleaningLog()
    clean.filter_users(con, log, 0.2, 10, 3000)
    assert [s.rule for s in log.steps] == ["users_low_variance", "users_hyperactive"]


def test_primary_author_skips_illustrators_and_works_without_main_author():
    c = duckdb.connect()
    c.execute("""CREATE TABLE editions AS SELECT * FROM (VALUES
        (10, 100, [{'author_id': 7, 'role': 'Illustrator'}, {'author_id': 1, 'role': NULL}], 50),
        (20, 200, [{'author_id': 8, 'role': 'Editor'}], 5))
        t(book_id, work_id, authors, ratings_count)""")
    c.execute("CREATE TABLE works_valid AS SELECT * FROM (VALUES (100, 10), (200, 20)) t(work_id, best_book_id)")
    c.execute("CREATE TABLE authors AS SELECT * FROM (VALUES (1), (7), (8)) t(author_id)")
    clean.primary_authors(c)
    assert c.execute("SELECT * FROM work_primary").fetchall() == [(100, 1)]

