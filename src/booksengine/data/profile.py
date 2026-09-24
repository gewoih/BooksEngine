"""Профилирование сырых (staging) данных: объёмы, связи, распределения, качество.

Каждая метрика — именованный SQL-запрос; результат сохраняется как список записей и попадает в отчёт.
"""
import math

import duckdb

BUCKETS_USERS = [1, 5, 10, 20, 50, 200, 1000, 3000, 10000]
BUCKETS_WORKS = [1, 5, 10, 20, 50, 200, 1000, 10000]


def _bucket_sql(col: str, edges: list[int]) -> str:
    parts = []
    for lo, hi in zip(edges, edges[1:] + [None]):
        label = f"{lo}+" if hi is None else f"{lo}–{hi - 1}"
        cond = f"{col} >= {lo}" + ("" if hi is None else f" AND {col} < {hi}")
        parts.append(f"WHEN {cond} THEN '{label}'")
    return f"CASE {' '.join(parts)} END"


# Базовая таблица явных оценок на уровне произведения — нужна для распределений до очистки.
SETUP = """
CREATE OR REPLACE TEMP TABLE _p_rw AS
SELECT i.user_id, e.work_id, avg(i.rating) AS rating
FROM interactions i
JOIN book_id_map m ON m.book_id_csv = i.book_id
JOIN editions e ON e.book_id = m.book_id
WHERE i.rating > 0 AND e.work_id IS NOT NULL
GROUP BY ALL;
CREATE OR REPLACE TEMP TABLE _p_ws AS SELECT work_id, count(*) AS n, avg(rating) AS mean FROM _p_rw GROUP BY 1;
CREATE OR REPLACE TEMP TABLE _p_us AS
SELECT user_id, count(*) AS n, avg(rating) AS mean, coalesce(stddev_pop(rating), 0) AS sd,
       avg((w.n < 20)::INT) AS obscure_share, avg((rating = 5)::INT) AS five_share
FROM _p_rw JOIN _p_ws w USING (work_id) GROUP BY user_id;
CREATE OR REPLACE TEMP TABLE _p_ui AS
SELECT user_id, count(*) AS all_n, avg((rating > 0)::INT) AS rated_share FROM interactions GROUP BY 1;
"""

QUERIES = {
    "files": """
        SELECT 'interactions' AS file, count(*) AS rows FROM interactions UNION ALL
        SELECT 'book_id_map', count(*) FROM book_id_map UNION ALL
        SELECT 'user_id_map', count(*) FROM user_id_map UNION ALL
        SELECT 'editions', count(*) FROM editions UNION ALL
        SELECT 'works', count(*) FROM works UNION ALL
        SELECT 'authors', count(*) FROM authors UNION ALL
        SELECT 'genres (book×genre)', count(*) FROM genres
    """,
    "interactions_overview": """
        SELECT count(*) AS rows, count(DISTINCT user_id) AS users, count(DISTINCT book_id) AS books,
               count(*) FILTER (user_id IS NULL OR book_id IS NULL OR rating IS NULL) AS null_keys,
               count(*) FILTER (rating NOT BETWEEN 0 AND 5) AS rating_out_of_range,
               count(*) - count(DISTINCT (user_id, book_id)) AS duplicate_pairs,
               count(*) FILTER (rating > 0) AS explicit_ratings,
               count(*) FILTER (is_reviewed = 1) AS reviewed
        FROM interactions
    """,
    "rating_x_is_read": """
        SELECT rating, is_read, count(*) AS n, round(100.0 * count(*) / sum(count(*)) OVER (), 2) AS pct
        FROM interactions GROUP BY ALL ORDER BY ALL
    """,
    "id_maps": """
        SELECT 'book_id_map' AS map, count(*) AS rows, count(DISTINCT book_id_csv) AS uniq_csv,
               count(DISTINCT book_id) AS uniq_target FROM book_id_map UNION ALL
        SELECT 'user_id_map', count(*), count(DISTINCT user_id_csv), count(DISTINCT user_id) FROM user_id_map
    """,
    "link_chain": """
        SELECT count(*) AS interactions,
               count(*) FILTER (m.book_id IS NULL) AS no_book_map,
               count(*) FILTER (m.book_id IS NOT NULL AND e.book_id IS NULL) AS no_edition,
               count(*) FILTER (e.book_id IS NOT NULL AND e.work_id IS NULL) AS edition_without_work,
               count(*) FILTER (e.work_id IS NOT NULL AND w.work_id IS NULL) AS work_missing,
               (SELECT count(DISTINCT i2.user_id) FROM interactions i2
                  LEFT JOIN user_id_map u ON u.user_id_csv = i2.user_id WHERE u.user_id IS NULL) AS users_no_map
        FROM interactions i
        LEFT JOIN book_id_map m ON i.book_id = m.book_id_csv
        LEFT JOIN editions e ON e.book_id = m.book_id
        LEFT JOIN works w ON w.work_id = e.work_id
    """,
    "editions_quality": """
        SELECT count(*) AS editions, count(DISTINCT book_id) AS uniq_ids,
               count(*) FILTER (work_id IS NULL) AS no_work_id,
               count(*) FILTER (title IS NULL) AS no_title,
               count(*) FILTER (authors IS NULL OR len(authors) = 0) AS no_authors,
               count(*) FILTER (language_code IS NULL) AS no_language,
               count(*) FILTER (description IS NULL) AS no_description,
               count(*) FILTER (isbn IS NULL AND isbn13 IS NULL) AS no_isbn,
               count(*) FILTER (publication_year IS NULL) AS no_year,
               (SELECT count(*) FROM editions e WHERE NOT EXISTS
                  (SELECT 1 FROM book_id_map m WHERE m.book_id = e.book_id)) AS not_in_interactions_map
        FROM editions
    """,
    "works_quality": """
        SELECT count(*) AS works, count(DISTINCT work_id) AS uniq_ids,
               count(*) FILTER (original_title IS NULL) AS no_original_title,
               count(*) FILTER (NOT EXISTS (SELECT 1 FROM editions e WHERE e.work_id = w.work_id)) AS no_editions,
               count(*) FILTER (NOT EXISTS (SELECT 1 FROM editions e WHERE e.book_id = w.best_book_id))
                   AS best_book_missing
        FROM works w
    """,
    "editions_per_work": f"""
        SELECT {_bucket_sql('n', [1, 2, 5, 10, 50, 200])} AS editions_per_work, count(*) AS works
        FROM (SELECT work_id, count(*) AS n FROM editions WHERE work_id IS NOT NULL GROUP BY 1)
        GROUP BY 1 ORDER BY min(n)
    """,
    "media_type": """
        SELECT coalesce(w.media_type, '(пусто)') AS media_type, count(*) AS works, sum(s.n) AS explicit_ratings
        FROM works w LEFT JOIN _p_ws s USING (work_id) GROUP BY 1 ORDER BY 2 DESC
    """,
    "media_type_not_book_top": """
        SELECT coalesce(w.original_title, e.title) AS title, w.media_type, s.n AS explicit_ratings
        FROM works w JOIN _p_ws s USING (work_id) JOIN editions e ON e.book_id = w.best_book_id
        WHERE w.media_type IN ('not a book', 'periodical', 'article') ORDER BY s.n DESC LIMIT 8
    """,
    "authors_quality": """
        SELECT (SELECT count(*) FROM authors) AS authors, (SELECT count(*) FROM authors WHERE name IS NULL) AS no_name,
               count(*) AS edition_author_links, count(*) FILTER (a.author_id IS NULL) AS link_to_missing_author
        FROM (SELECT unnest(authors).author_id AS aid FROM editions) x LEFT JOIN authors a ON a.author_id = x.aid
    """,
    "genres_overview": """
        SELECT genre, count(DISTINCT book_id) AS editions FROM genres GROUP BY 1 ORDER BY 2 DESC
    """,
    "languages_rated": """
        SELECT coalesce(e.language_code, '(пусто)') AS language_code, count(*) AS explicit_ratings
        FROM interactions i JOIN book_id_map m ON m.book_id_csv = i.book_id JOIN editions e ON e.book_id = m.book_id
        WHERE i.rating > 0 GROUP BY 1 ORDER BY 2 DESC LIMIT 12
    """,
    "user_activity": f"""
        SELECT {_bucket_sql('n', BUCKETS_USERS)} AS explicit_ratings_per_user, count(*) AS users,
               sum(n) AS ratings, round(median(sd), 2) AS median_sd, round(median(mean), 2) AS median_mean,
               round(median(obscure_share), 3) AS median_obscure_share, round(median(five_share), 2) AS median_five_share
        FROM _p_us GROUP BY 1 ORDER BY min(n)
    """,
    "user_quantiles": """
        SELECT count(*) AS users_with_explicit, quantile_disc(n, 0.5) AS p50, quantile_disc(n, 0.9) AS p90,
               quantile_disc(n, 0.99) AS p99, quantile_disc(n, 0.999) AS p999, max(n) AS max,
               (SELECT count(*) FROM _p_ui WHERE rated_share = 0) AS users_without_explicit
        FROM _p_us
    """,
    "user_variance": """
        SELECT CASE WHEN sd = 0 THEN 'a) 0' WHEN sd < 0.2 THEN 'b) (0; 0.2)' WHEN sd < 0.3 THEN 'c) [0.2; 0.3)'
                    WHEN sd < 0.5 THEN 'd) [0.3; 0.5)' ELSE 'e) ≥ 0.5' END AS sd_bucket,
               count(*) AS users, sum(n) AS ratings,
               count(*) FILTER (mean > 4.9) AS all_fives_users
        FROM _p_us WHERE n >= 10 GROUP BY 1 ORDER BY 1
    """,
    "top_users": """
        SELECT user_id, n AS explicit_ratings, round(sd, 2) AS sd, round(mean, 2) AS mean,
               round(obscure_share, 2) AS obscure_share, round(five_share, 2) AS five_share, all_n AS all_interactions
        FROM _p_us JOIN _p_ui USING (user_id) ORDER BY n DESC LIMIT 12
    """,
    "work_popularity": f"""
        SELECT {_bucket_sql('n', BUCKETS_WORKS)} AS explicit_ratings_per_work, count(*) AS works, sum(n) AS ratings
        FROM _p_ws GROUP BY 1 ORDER BY min(n)
    """,
    "work_quantiles": """
        SELECT count(*) AS works_with_explicit, quantile_disc(n, 0.5) AS p50, quantile_disc(n, 0.9) AS p90,
               quantile_disc(n, 0.99) AS p99, max(n) AS max FROM _p_ws
    """,
    "rating_distribution_explicit": """
        SELECT rating, count(*) AS n, round(100.0 * count(*) / sum(count(*)) OVER (), 2) AS pct
        FROM interactions WHERE rating > 0 GROUP BY 1 ORDER BY 1
    """,
}


def _clean(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if hasattr(v, "item"):
        return v.item()
    return v


def run(con: duckdb.DuckDBPyConnection) -> dict[str, list[dict]]:
    print("[profile] подготовка")
    con.execute(SETUP)
    out = {}
    for name, sql in QUERIES.items():
        print(f"[profile] {name}")
        cur = con.execute(sql)
        cols = [d[0] for d in cur.description]
        out[name] = [{c: _clean(v) for c, v in zip(cols, row)} for row in cur.fetchall()]
    for t in ("_p_rw", "_p_ws", "_p_us", "_p_ui"):
        con.execute(f"DROP TABLE {t}")
    return out
