from pathlib import Path

import duckdb
import pandas as pd
import pytest

from booksengine.data import movielens


def test_bucket_pool_size_counts_users_per_bucket_from_ratings(tmp_path):
    """split.BUCKET_POOL_SIZE откалиброван под Goodreads (149K/155K/…) — на MovieLens доля отбора в
    проверку/тест (`test_per_bucket / pool`) с этим пулом оказывается втрое меньше задуманной. Свой пул —
    из фактических ratings.parquet домена."""
    rows = []
    wid = 0
    for n_users, n_ratings in ((3, 25), (2, 50), (1, 5000)):   # 25 -> 20-39, 50 -> 40-79, 5000 -> 1000+ (вне пула)
        for u in range(n_users):
            for _ in range(n_ratings):
                wid += 1
                rows.append((f"{n_ratings}-{u}", wid, 5))
    path = tmp_path / "ratings.parquet"
    pd.DataFrame(rows, columns=["user_id", "work_id", "rating"]).to_parquet(path, index=False)

    pool = movielens.bucket_pool_size(path)
    assert pool == {"20-39": 3, "40-79": 2, "80-159": 0, "160-319": 0, "320-999": 0}


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


def _synthetic_ml_csvs(tmp_path) -> Path:
    raw = tmp_path / "raw"
    (raw / "ml-32m").mkdir(parents=True)
    (raw / "ml-32m" / "ratings.csv").write_text(
        "userId,movieId,rating,timestamp\n"
        "1,10,5.0,100\n1,20,4.0,101\n1,30,3.0,102\n"
        "2,10,4.5,110\n2,20,3.5,111\n2,30,2.0,112\n"
        "3,10,5.0,120\n3,20,5.0,121\n3,30,5.0,122\n"
    )
    (raw / "ml-32m" / "movies.csv").write_text(
        "movieId,title,genres\n"
        "10,Toy Story (1995),Adventure\n20,Heat (1995),Action\n30,Se7en (1995),Thriller\n"
    )
    return raw


MOVIES_CFG = {"low_variance_max_sd": 0.2, "low_variance_min_ratings": 10, "monotone_max_mode_share": 0.9,
              "max_ratings": 3000}


def test_prepare_end_to_end_with_synthetic_dataset(tmp_path):
    raw = _synthetic_ml_csvs(tmp_path)
    manifest = movielens.prepare(raw, tmp_path / "clean", MOVIES_CFG, min_user=1, min_work=1)
    assert manifest["outputs"] == {"ratings": 9, "users": 3, "works": 3}
    ratings = pd.read_parquet(tmp_path / "clean" / "ratings.parquet")
    assert set(ratings.rating.unique()) <= {1, 2, 3, 4, 5}
    assert (tmp_path / "clean" / "works.parquet").exists()


def test_prepare_drops_hyperactive_users_per_movies_cfg(tmp_path):
    raw = _synthetic_ml_csvs(tmp_path)
    cfg = MOVIES_CFG | {"max_ratings": 2}   # у всех троих по 3 оценки — «гиперактивны» при пороге 2
    manifest = movielens.prepare(raw, tmp_path / "clean", cfg, min_user=1, min_work=1)
    assert manifest["outputs"]["users"] == 0
    dropped = next(s for s in manifest["cleaning_log"] if s["rule"] == "users_hyperactive")
    assert dropped["users_after"] == 0
