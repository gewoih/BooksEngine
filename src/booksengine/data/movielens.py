"""MovieLens 32M → matrices в той же схеме, что у книг: общий код модели (`split`, `taste`, `layers`, `ease`)
работает с матрицей «человек × произведение × оценка», книжная специфика (сборники, дубли произведений,
не-книги) фильмам не нужна — переиспользует только очистку пользователей и k-core из `booksengine.data.clean`.
Дизайн: docs/superpowers/specs/2026-09-27-movies-crowd-taste-check-design.md.
"""
from pathlib import Path

import duckdb


def load(con: duckdb.DuckDBPyConnection, ratings_csv: Path, movies_csv: Path) -> None:
    """_ml_ratings (userId, movieId, rating, timestamp), _ml_movies (movieId, title, genres) — сырые CSV как есть."""
    con.execute("CREATE OR REPLACE TEMP TABLE _ml_ratings AS SELECT * FROM read_csv_auto(?)", [str(ratings_csv)])
    con.execute("CREATE OR REPLACE TEMP TABLE _ml_movies AS SELECT * FROM read_csv_auto(?)", [str(movies_csv)])


def to_ratings_w(con: duckdb.DuckDBPyConnection) -> None:
    """ratings_w: user_id, work_id, rating — полузвёзды 0.5–5.0 округлены вверх до целых 1–5; повтор пары
    (userId, movieId) — среднее до округления, как повтор издания у книг (`clean.to_work_level`)."""
    con.execute("""
        CREATE OR REPLACE TEMP TABLE ratings_w AS
        SELECT userId AS user_id, movieId AS work_id, ceil(avg(rating))::TINYINT AS rating
        FROM _ml_ratings GROUP BY userId, movieId
    """)


def export(con: duckdb.DuckDBPyConnection, out_dir: Path) -> dict:
    """ratings/users/works/work_authors/authors.parquet — схема, которую читает модель книг
    (`filters.work_info`); авторы и коллекции — пустые/тривиальные, режиссёр появится только с TMDB."""
    out_dir.mkdir(parents=True, exist_ok=True)
    opts = "(FORMAT parquet, COMPRESSION zstd)"
    missing = con.execute("""
        SELECT count(*) FROM (SELECT DISTINCT work_id FROM ratings_w) r
        LEFT JOIN _ml_movies m ON m.movieId = r.work_id WHERE m.movieId IS NULL
    """).fetchone()[0]
    if missing:
        raise ValueError(f"movies.csv: не хватает {missing} строк для movieId из ratings.csv")
    con.execute(f"COPY (SELECT user_id, work_id, rating FROM ratings_w ORDER BY user_id, work_id) "
                f"TO '{out_dir / 'ratings.parquet'}' {opts}")
    con.execute(f"COPY (SELECT DISTINCT user_id, user_id AS external_id FROM ratings_w ORDER BY 1) "
                f"TO '{out_dir / 'users.parquet'}' {opts}")
    con.execute(f"""
        COPY (SELECT r.work_id, m.title, NULL::VARCHAR AS original_title, NULL::VARCHAR AS best_edition_title,
                     false AS is_collection
              FROM (SELECT DISTINCT work_id FROM ratings_w) r JOIN _ml_movies m ON m.movieId = r.work_id
              ORDER BY r.work_id)
        TO '{out_dir / 'works.parquet'}' {opts}
    """)
    con.execute(f"COPY (SELECT NULL::BIGINT AS work_id, NULL::BIGINT AS author_id, NULL::VARCHAR AS role, "
                f"NULL::SMALLINT AS position WHERE false) TO '{out_dir / 'work_authors.parquet'}' {opts}")
    con.execute(f"COPY (SELECT NULL::BIGINT AS author_id, NULL::VARCHAR AS name WHERE false) "
                f"TO '{out_dir / 'authors.parquet'}' {opts}")
    counts = con.execute(
        "SELECT count(*), count(DISTINCT user_id), count(DISTINCT work_id) FROM ratings_w").fetchone()
    return {"ratings": counts[0], "users": counts[1], "works": counts[2]}


def bucket_pool_size(ratings_path: Path) -> dict:
    """Люди по этапам (`split.BUCKET_ORDER`) в самом ratings.parquet — для `split.build(bucket_pool_size=...)`.
    `split.BUCKET_POOL_SIZE` откалиброван по ядру книг (149K/155K/138K/93K/56K); у MovieLens другие размеры
    ядра на этап — с книжным пулом доля отбора в проверку/тест (`test_per_bucket / pool`) оказывается не той,
    что задумана."""
    from booksengine.model import split

    counts = duckdb.execute("SELECT count(*) AS n FROM read_parquet(?) GROUP BY user_id",
                            [str(ratings_path)]).df()["n"].to_numpy()
    buckets = split.bucket_of(counts)
    return {b: int((buckets == b).sum()) for b in split.BUCKET_ORDER}


def prepare(raw_dir: Path, out_dir: Path, cfg: dict, min_user: int = 20, min_work: int = 100) -> dict:
    """Загрузка → округление и дедуп → очистка пользователей (`cfg` — раздел movies: cleaning.yaml) →
    k-core (min_user/min_work — фиксированные значения 20/100, как у книг; не перебираются) → экспорт."""
    from booksengine.data import clean

    con = duckdb.connect()
    load(con, raw_dir / "ml-32m" / "ratings.csv", raw_dir / "ml-32m" / "movies.csv")
    to_ratings_w(con)
    log = clean.CleaningLog()
    clean.filter_users(con, log, cfg["low_variance_max_sd"], cfg["low_variance_min_ratings"], cfg["max_ratings"],
                       max_mode_share=cfg["monotone_max_mode_share"])
    clean.apply_kcore(con, log, min_user, min_work)
    outputs = export(con, out_dir)
    con.close()
    return {"cleaning_log": log.records(), "outputs": outputs}
