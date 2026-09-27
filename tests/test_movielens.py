import duckdb
import pandas as pd
import pytest

from booksengine.data import movielens


@pytest.fixture
def con():
    return duckdb.connect()


def test_to_ratings_w_rounds_half_stars_up_and_averages_duplicates(con):
    con.execute("""
        CREATE TEMP TABLE _ml_ratings AS SELECT * FROM (VALUES
            (1, 10, 3.0, 100), (2, 10, 0.5, 101), (3, 10, 5.0, 102),
            (4, 10, 3.0, 103), (4, 10, 4.0, 104))
        t(userId, movieId, rating, timestamp)
    """)
    movielens.to_ratings_w(con)
    assert con.execute("SELECT count(*) FROM ratings_w").fetchone()[0] == 4
    rows = dict(((u, w), r) for u, w, r in
                con.execute("SELECT user_id, work_id, rating FROM ratings_w").fetchall())
    assert rows[(1, 10)] == 3   # целая звезда не сдвигается
    assert rows[(2, 10)] == 1   # 0.5 -> 1
    assert rows[(3, 10)] == 5   # 5.0 -> 5
    assert rows[(4, 10)] == 4   # дубль (user, movie): среднее 3.5 -> ceil 4


def test_export_raises_when_ratings_reference_missing_movie(con, tmp_path):
    con.execute("CREATE TEMP TABLE ratings_w AS SELECT * FROM (VALUES (1, 10, 3)) t(user_id, work_id, rating)")
    con.execute("CREATE TEMP TABLE _ml_movies AS SELECT * FROM (VALUES (99, 'Other', 'Drama')) "
                "t(movieId, title, genres)")
    with pytest.raises(ValueError, match="не хватает"):
        movielens.export(con, tmp_path)


def test_export_writes_ratings_users_works_and_trivial_author_tables(con, tmp_path):
    con.execute("CREATE TEMP TABLE ratings_w AS SELECT * FROM (VALUES "
                "(1, 10, 5), (1, 20, 3), (2, 10, 4)) t(user_id, work_id, rating)")
    con.execute("CREATE TEMP TABLE _ml_movies AS SELECT * FROM (VALUES "
                "(10, 'Toy Story (1995)', 'Adventure'), (20, 'Heat (1995)', 'Action')) "
                "t(movieId, title, genres)")
    counts = movielens.export(con, tmp_path)
    assert counts == {"ratings": 3, "users": 2, "works": 2}

    ratings = pd.read_parquet(tmp_path / "ratings.parquet")
    assert list(ratings.columns) == ["user_id", "work_id", "rating"]
    assert len(ratings) == 3

    users = pd.read_parquet(tmp_path / "users.parquet")
    assert sorted(users.user_id) == [1, 2]
    assert (users.user_id == users.external_id).all()

    works = pd.read_parquet(tmp_path / "works.parquet")
    assert sorted(works.title) == ["Heat (1995)", "Toy Story (1995)"]
    assert not works.is_collection.any()
    assert works.original_title.isna().all() and works.best_edition_title.isna().all()

    work_authors = pd.read_parquet(tmp_path / "work_authors.parquet")
    assert list(work_authors.columns) == ["work_id", "author_id", "role", "position"]
    assert len(work_authors) == 0

    authors = pd.read_parquet(tmp_path / "authors.parquet")
    assert list(authors.columns) == ["author_id", "name"]
    assert len(authors) == 0
