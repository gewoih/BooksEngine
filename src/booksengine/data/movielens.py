"""MovieLens 32M → matrices в той же схеме, что у книг (проверка толпы и вкуса на фильмах, шаг 1 плана
«Фильмы» — TODO.md, docs/superpowers/specs/2026-09-27-movies-crowd-taste-check-design.md). Переиспользует
очистку пользователей и k-core из `booksengine.data.clean` без изменений — книжные правила (сборники, дубли
произведений, не-книги) фильмам не нужны.
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
