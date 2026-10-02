"""`booksengine export-app`: данные приложения сверх каталога Goodreads (`load-db`) → PostgreSQL.

Выдачу приложению считает сервис `serve` (тот же код, что у `recommend`), модели в БД нет. Здесь — то, что нужно
самому приложению:
- новые книги единой базы (`--domain books-amazon`, после 2017 — их нет в Goodreads): произведения, новые авторы,
  связи и жанр; внешний id — ключ «автор|название» (`merged.new_work_keys`), источник `amazon`; русское название,
  если перевод известен, — в ru_title;
- русские названия книг и имена авторов (`ru-titles` → data/ru/): works.ru_title, authors.ru_name;
  Книга, пропавшая из базы, удаляется; если её оценил пользователь приложения — выгрузка падает, оценки не теряются.
  База без новых книг (книжная) новые книги приложения не трогает;
- обложки книг ядра (самое популярное издание с картинкой) и слияния теней Goodreads (импорт CSV).
Всё одной транзакцией.
"""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import psycopg

from booksengine.data.merged import NEW_AUTHOR_OFFSET, NEW_WORK_OFFSET, author_key_sql, new_work_keys
from booksengine.model.matrix import catalog_works


def covers(goodreads_dir: Path, work_ids: np.ndarray) -> pd.DataFrame:
    """Обложка — самое популярное издание с картинкой: у лучшего издания она есть у 65% ядра, у любого — у 88%."""
    d = duckdb.execute("""
        SELECT DISTINCT ON (work_id) work_id AS gr_work_id, image_url FROM read_parquet(?)
        WHERE image_url IS NOT NULL AND image_url NOT LIKE '%nophoto%'
        ORDER BY work_id, ratings_count DESC NULLS LAST, book_id""", [str(goodreads_dir / "editions.parquet")]).df()
    return d[d.gr_work_id.isin(work_ids)].sort_values("gr_work_id").reset_index(drop=True)


def new_works(clean_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Новые книги базы: (произведения с ключом, их авторы, жанры). Автор — `gr_author` (Goodreads) или `author_key`
    (новый автор Amazon: его id — номер по порядку, ключ имени постоянен)."""
    keys = new_work_keys(clean_dir)
    if keys.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    con = duckdb.connect()
    con.register("k", keys.rename_axis("work_id").reset_index())
    works = con.execute("""
        SELECT k.key, w.title, w.ru_title, w.publication_year, w.language_code, coalesce(w.is_collection, false)
               AS is_collection, w.in_cf, coalesce(w.cf_ratings, 0)::INTEGER AS cf_ratings, w.cf_mean_rating
        FROM read_parquet(?) w JOIN k USING (work_id) ORDER BY k.key""", [str(clean_dir / "works.parquet")]).df()
    authors = con.execute(f"""
        SELECT k.key, wa.role, wa.position, a.name,
               CASE WHEN wa.author_id < {NEW_AUTHOR_OFFSET} THEN wa.author_id END AS gr_author,
               CASE WHEN wa.author_id >= {NEW_AUTHOR_OFFSET} THEN {author_key_sql('a.name')} END AS author_key
        FROM read_parquet(?) wa JOIN k USING (work_id) JOIN read_parquet(?) a USING (author_id)
        ORDER BY k.key, wa.position""", [str(clean_dir / "work_authors.parquet"), str(clean_dir / "authors.parquet")]).df()
    genres = con.execute("""
        SELECT k.key, g.genre, g.votes, g.share FROM read_parquet(?) g JOIN k USING (work_id) ORDER BY 1, 2""",
                         [str(clean_dir / "work_genres.parquet")]).df()
    con.close()
    works["ru_title"] = works.ru_title.where(works.ru_title.notna() & works.ru_title.ne(""))
    return works, authors, genres


def _source(conn: psycopg.Connection, code: str) -> int:
    row = conn.execute("SELECT id FROM sources WHERE code = %s", (code,)).fetchone()
    if row is None:
        raise RuntimeError(f"нет источника '{code}' в sources: примените миграции "
                           "(cd dotnet/BooksEngine.Db && dotnet ef database update)")
    return row[0]


def _temp(conn: psycopg.Connection, name: str, ddl: str, rows) -> None:
    conn.execute(f"CREATE TEMP TABLE {name} ({ddl}) ON COMMIT DROP")
    with conn.cursor().copy(f"COPY {name} FROM STDIN") as cp:
        for r in rows:
            cp.write_row([None if (v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA) else v
                          for v in r])


def _write_new_works(conn: psycopg.Connection, works: pd.DataFrame, authors: pd.DataFrame,
                     genres: pd.DataFrame) -> dict[str, int]:
    gr, az = _source(conn, "goodreads"), _source(conn, "amazon")
    _temp(conn, "nw", "key text, title text, ru_title text, publication_year integer, language_code text, "
          "is_collection boolean, in_cf boolean, cf_ratings integer, cf_mean_rating double precision",
          works[["key", "title", "ru_title", "publication_year", "language_code", "is_collection", "in_cf",
                 "cf_ratings", "cf_mean_rating"]].astype(object).itertuples(index=False))
    _temp(conn, "nwa", "key text, role text, position smallint, name text, gr_author bigint, author_key text",
          authors.astype(object).itertuples(index=False))
    _temp(conn, "nwg", "key text, genre text, votes integer, share double precision",
          genres.astype(object).itertuples(index=False))
    out = {}
    # пропавшие из базы: оценки и полки пользователей держат их (FK RESTRICT) — тогда выгрузка падает целиком
    try:
        out["новых книг удалено"] = conn.execute(f"""
            DELETE FROM works w USING external_ids x
            WHERE x.source_id = {az} AND x.entity_type = 'work' AND x.internal_id = w.id
              AND x.external_id NOT IN (SELECT key FROM nw)""").rowcount
    except psycopg.errors.ForeignKeyViolation as e:
        raise RuntimeError("В базе больше нет новых книг, которые оценили пользователи приложения. Выгрузка "
                           f"отменена, БД не изменена.\n{e}") from e
    conn.execute(f"DELETE FROM external_ids WHERE source_id = {az} AND entity_type = 'work' "
                 "AND external_id NOT IN (SELECT key FROM nw)")
    # новые — id из последовательности works, внешний id — ключ
    conn.execute(f"""
        INSERT INTO external_ids (source_id, entity_type, external_id, internal_id)
        SELECT {az}, 'work', key, nextval(pg_get_serial_sequence('works', 'id')) FROM nw
        WHERE key NOT IN (SELECT external_id FROM external_ids WHERE source_id = {az} AND entity_type = 'work')
        ORDER BY key""")
    conn.execute(f"""
        CREATE TEMP TABLE nw_id ON COMMIT DROP AS
        SELECT nw.*, x.internal_id AS id FROM nw
        JOIN external_ids x ON x.source_id = {az} AND x.entity_type = 'work' AND x.external_id = nw.key""")
    cols = ["title", "ru_title", "publication_year", "language_code", "is_collection", "in_cf", "cf_ratings",
            "cf_mean_rating"]
    out["новых книг вставлено/изменено"] = conn.execute(f"""
        INSERT INTO works (id, {', '.join(cols)}) SELECT id, {', '.join(cols)} FROM nw_id
        ON CONFLICT (id) DO UPDATE SET {', '.join(f'{c} = excluded.{c}' for c in cols)}
        WHERE ({', '.join(f'works.{c}' for c in cols)}) IS DISTINCT FROM ({', '.join(f'excluded.{c}' for c in cols)})
    """).rowcount
    # новые авторы Amazon — по ключу имени; у остальных — автор Goodreads из каталога
    conn.execute(f"""
        INSERT INTO external_ids (source_id, entity_type, external_id, internal_id)
        SELECT {az}, 'author', author_key, nextval(pg_get_serial_sequence('authors', 'id'))
        FROM (SELECT DISTINCT author_key FROM nwa WHERE author_key IS NOT NULL) a
        WHERE author_key NOT IN (SELECT external_id FROM external_ids WHERE source_id = {az} AND entity_type = 'author')
        ORDER BY author_key""")
    conn.execute(f"""
        INSERT INTO authors (id, name)
        SELECT DISTINCT ON (x.internal_id) x.internal_id, nwa.name FROM nwa
        JOIN external_ids x ON x.source_id = {az} AND x.entity_type = 'author' AND x.external_id = nwa.author_key
        ORDER BY x.internal_id, nwa.name
        ON CONFLICT (id) DO UPDATE SET name = excluded.name WHERE authors.name IS DISTINCT FROM excluded.name""")
    conn.execute(f"""
        CREATE TEMP TABLE nwa_id ON COMMIT DROP AS
        SELECT DISTINCT ON (w.id, coalesce(g.internal_id, a.internal_id))
               w.id AS work_id, coalesce(g.internal_id, a.internal_id) AS author_id, nwa.role, nwa.position
        FROM nwa JOIN nw_id w USING (key)
        LEFT JOIN external_ids g ON g.source_id = {gr} AND g.entity_type = 'author' AND g.external_id = nwa.gr_author::text
        LEFT JOIN external_ids a ON a.source_id = {az} AND a.entity_type = 'author' AND a.external_id = nwa.author_key
        WHERE coalesce(g.internal_id, a.internal_id) IS NOT NULL
        ORDER BY 1, 2, nwa.position""")
    conn.execute("DELETE FROM work_authors WHERE work_id IN (SELECT id FROM nw_id)")
    out["новых книг: авторов"] = conn.execute(
        "INSERT INTO work_authors (work_id, author_id, role, position) SELECT * FROM nwa_id").rowcount
    conn.execute("INSERT INTO genres (name) SELECT DISTINCT genre FROM nwg ORDER BY 1 ON CONFLICT (name) DO NOTHING")
    conn.execute("DELETE FROM work_genres WHERE work_id IN (SELECT id FROM nw_id)")
    out["новых книг: жанров"] = conn.execute("""
        INSERT INTO work_genres (work_id, genre_id, votes, share)
        SELECT w.id, g.id, nwg.votes, nwg.share FROM nwg JOIN nw_id w USING (key) JOIN genres g ON g.name = nwg.genre
    """).rowcount
    # авторы Amazon без книг (книги пропали из базы)
    conn.execute(f"""
        DELETE FROM authors a USING external_ids x
        WHERE x.source_id = {az} AND x.entity_type = 'author' AND x.internal_id = a.id
          AND NOT EXISTS (SELECT 1 FROM work_authors wa WHERE wa.author_id = a.id)""")
    conn.execute(f"""
        DELETE FROM external_ids x WHERE x.source_id = {az} AND x.entity_type = 'author'
          AND NOT EXISTS (SELECT 1 FROM authors a WHERE a.id = x.internal_id)""")
    return out


def _write_russian(conn: psycopg.Connection, titles: pd.DataFrame, names: pd.DataFrame) -> dict[str, int]:
    """Русские названия книг Goodreads и имена авторов Goodreads (data/ru/) — по внешним id; прежние, которых больше
    нет, стираются. У новых книг Amazon русское название пишет `_write_new_works`."""
    gr = _source(conn, "goodreads")
    titles = titles[titles.work_id < NEW_WORK_OFFSET]
    _temp(conn, "ru_w", "ext text, ru text", ((str(int(w)), t) for w, t in zip(titles.work_id, titles.ru_title)))
    _temp(conn, "ru_a", "ext text, ru text", ((str(int(a)), n) for a, n in zip(names.author_id, names.ru_name)))
    out = {}
    for table, kind, col, tmp in (("works", "work", "ru_title", "ru_w"), ("authors", "author", "ru_name", "ru_a")):
        conn.execute(f"""
            CREATE TEMP TABLE {tmp}_id ON COMMIT DROP AS
            SELECT x.internal_id AS id, t.ru FROM {tmp} t
            JOIN external_ids x ON x.source_id = {gr} AND x.entity_type = '{kind}' AND x.external_id = t.ext""")
        conn.execute(f"""
            UPDATE {table} t SET {col} = NULL FROM external_ids x
            WHERE x.source_id = {gr} AND x.entity_type = '{kind}' AND x.internal_id = t.id AND t.{col} IS NOT NULL
              AND t.id NOT IN (SELECT id FROM {tmp}_id)""")
        out[f"{table}: по-русски"] = conn.execute(f"""
            UPDATE {table} t SET {col} = r.ru FROM {tmp}_id r
            WHERE t.id = r.id AND t.{col} IS DISTINCT FROM r.ru""").rowcount
    return out


def write(dsn: str, *, works: pd.DataFrame, authors: pd.DataFrame, genres: pd.DataFrame, covers_: pd.DataFrame,
          merges: pd.DataFrame, ru_titles: pd.DataFrame | None = None,
          ru_names: pd.DataFrame | None = None) -> dict[str, int]:
    """Перезаписать данные приложения одной транзакцией: упало — остаётся прежнее целиком. ru_titles/ru_names —
    data/ru/ (`ru-titles`); None — русские названия не трогать."""
    with psycopg.connect(dsn) as conn:
        if not conn.execute("SELECT pg_try_advisory_xact_lock(hashtext('booksengine.export-app'))").fetchone()[0]:
            raise RuntimeError("Другая выгрузка уже идёт в эту БД")
        out = _write_new_works(conn, works, authors, genres) if len(works) else {}
        if ru_titles is not None:
            out |= _write_russian(conn, ru_titles, ru_names)
        gr = _source(conn, "goodreads")
        need = [str(int(g)) for g in pd.concat([covers_.gr_work_id, merges.main_gr]).unique()]
        found = dict(conn.execute(f"""
            SELECT external_id, internal_id FROM external_ids
            WHERE source_id = {gr} AND entity_type = 'work' AND external_id = ANY(%s)""", (need,)).fetchall())
        missing = [g for g in need if g not in found]
        if missing:
            raise ValueError(f"книг нет в каталоге БД: {len(missing)} (например {missing[:5]}); "
                             "сначала `booksengine load-db`")
        conn.execute("TRUNCATE work_covers, work_merges")
        with conn.cursor().copy("COPY work_covers (work_id, image_url) FROM STDIN") as cp:
            for w, u in zip(covers_.gr_work_id, covers_.image_url):
                cp.write_row((found[str(int(w))], u))
        with conn.cursor().copy("COPY work_merges (shadow_external_id, main_work_id) FROM STDIN") as cp:
            for s, w in zip(merges.shadow_gr, merges.main_gr):
                cp.write_row((str(int(s)), found[str(int(w))]))
        out |= {t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in ("work_covers", "work_merges")}
    return out


def run(*, clean_dir: Path, goodreads_dir: Path, ru_dir: Path, dsn: str) -> dict[str, int]:
    """clean_dir — база домена (единая — с новыми книгами), goodreads_dir — очищенный Goodreads (издания),
    ru_dir — русские названия (`ru-titles`; нет — не трогать)."""
    works, authors, genres = new_works(clean_dir)
    merges = (pd.read_parquet(clean_dir / "work_merges.parquet")
              .rename(columns={"shadow_work_id": "shadow_gr", "main_work_id": "main_gr"})[["shadow_gr", "main_gr"]])
    out = write(dsn, works=works, authors=authors, genres=genres,
                covers_=covers(goodreads_dir, catalog_works(clean_dir / "ratings.parquet")), merges=merges,
                **({"ru_titles": pd.read_parquet(ru_dir / "titles.parquet"),
                    "ru_names": pd.read_parquet(ru_dir / "authors.parquet")}
                   if (ru_dir / "titles.parquet").exists() else {}))
    for k, v in out.items():
        print(f"  {k}: {v}")
    return out
