"""MovieLens 32M → matrices в той же схеме, что у книг: общий код модели (`split`, `taste`, `layers`, `ease`)
работает с матрицей «человек × произведение × оценка», книжная специфика (сборники, дубли произведений,
не-книги) фильмам не нужна — переиспользует только очистку пользователей и k-core из `booksengine.data.clean`.
Дизайн: docs/superpowers/specs/2026-09-27-movies-crowd-taste-check-design.md.

Профиль фильмов: канонический формат `profiles/<имя>_movies.csv` — `imdb_id, rating, title` (imdb_id — как в
IMDb, без «tt» и ведущих нулей). `import_letterboxd` строит его из экспорта Letterboxd. `materialize_profile`
сопоставляет imdb_id -> movieId через `links.csv` MovieLens и пишет CSV, который понимает
`recommend.read_profile(id_col="movie_id")` — так же, как read_profile понимает goodreads_work_id у книг.
"""
import math
from pathlib import Path

import duckdb
import pandas as pd


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
    con.execute(f"COPY (SELECT NULL::BIGINT AS shadow_work_id, NULL::BIGINT AS main_work_id WHERE false) "
                f"TO '{out_dir / 'work_merges.parquet'}' {opts}")
    # «Documentary» в жанрах MovieLens -> 'non-fiction' книжного механизма (filters.nonfiction, NONFICTION_SHARE):
    # тот же порог, тот же второй список (игровое/документальное), без изменений filters.py
    con.execute(f"""
        COPY (SELECT r.work_id, 'non-fiction'::VARCHAR AS genre, 1::INTEGER AS votes, 1.0::DOUBLE AS share
              FROM (SELECT DISTINCT work_id FROM ratings_w) r JOIN _ml_movies m ON m.movieId = r.work_id
              WHERE list_contains(string_split(m.genres, '|'), 'Documentary')
              ORDER BY r.work_id)
        TO '{out_dir / 'work_genres.parquet'}' {opts}
    """)
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


def import_letterboxd(export_csv: Path) -> pd.DataFrame:
    """Экспорт Letterboxd (tmdbID, imdbID, Title, Year, Rating10, ...) -> канонический профиль фильмов
    (imdb_id, rating, title). Rating10 — шаг 1 из 10 — делится пополам и округляется вверх (та же половина-вверх,
    что у самих оценок MovieLens). Без imdbID или без оценки (только «просмотрено» в Letterboxd) — не входит."""
    df = pd.read_csv(export_csv)
    df = df.dropna(subset=["imdbID", "Rating10"]).copy()
    df["imdb_id"] = df.imdbID.str.removeprefix("tt").astype("int64")
    df["rating"] = df.Rating10.apply(lambda r: min(5, math.ceil(r / 2)))
    return df[["imdb_id", "rating", "Title"]].rename(columns={"Title": "title"})


def materialize_profile(profile_csv: Path, links_csv: Path, out_csv: Path) -> dict:
    """Канонический профиль (imdb_id, rating[, title]) -> movie_id через links.csv MovieLens (movieId, imdbId,
    tmdbId) -> CSV для `recommend.read_profile(id_col="movie_id")`, как goodreads_work_id у книг. imdb_id без
    movieId в links.csv (фильм новее среза MovieLens или не входил в него) — не попадает, считается в отчёте."""
    prof = pd.read_csv(profile_csv)
    for col in ("imdb_id", "rating"):
        if col not in prof.columns:
            raise ValueError(f"{profile_csv}: нет колонки {col}")
    links = pd.read_csv(links_csv)
    links["imdbId"] = links.imdbId.astype("int64")
    merged = prof.merge(links[["movieId", "imdbId"]], left_on="imdb_id", right_on="imdbId", how="left")
    matched = merged.dropna(subset=["movieId"]).copy()
    matched["movie_id"] = matched.movieId.astype("int64")
    cols = ["movie_id", "rating"] + (["title"] if "title" in matched.columns else [])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    matched[cols].to_csv(out_csv, index=False)
    return {"total": len(prof), "matched": len(matched), "unmatched": len(prof) - len(matched)}
