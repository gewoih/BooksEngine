"""load-db на синтетическом каталоге против настоящего PostgreSQL.

Тестовая БД booksengine_test — фикстура `dsn` из conftest.py. Нет Postgres или dotnet — тесты пропускаются.
"""
import json
from pathlib import Path

import psycopg
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from booksengine import db_load


# ---------- синтетический каталог ----------

def _catalog() -> dict[str, list[dict]]:
    """3 произведения, 4 издания, 3 автора. Goodreads-id намеренно не по порядку."""
    works = [
        dict(work_id=300, best_book_id=31, title="Dune", original_title="Dune", publication_year=1965,
             language_code="eng", description="spice", is_collection=False, in_cf=True, cf_ratings=500,
             cf_mean_rating=4.2, gr_ratings_count=1000, gr_ratings_sum=4200),
        dict(work_id=100, best_book_id=11, title="Solaris", original_title="Solaris", publication_year=1961,
             language_code=None, description="", is_collection=False, in_cf=True, cf_ratings=80,
             cf_mean_rating=4.0, gr_ratings_count=300, gr_ratings_sum=1200),
        dict(work_id=200, best_book_id=21, title="Box Set #1-3", original_title=None, publication_year=None,
             language_code="eng", description=None, is_collection=True, in_cf=False, cf_ratings=0,
             cf_mean_rating=None, gr_ratings_count=5, gr_ratings_sum=20),
    ]
    editions = [
        dict(book_id=31, work_id=300, title="Dune", isbn="0441013597", isbn13="9780441013593"),
        dict(book_id=32, work_id=300, title=None, isbn=None, isbn13=None),
        dict(book_id=11, work_id=100, title="Solaris", isbn=None, isbn13="9780156027601"),
        dict(book_id=21, work_id=200, title="Box Set", isbn="", isbn13=None),
    ]
    authors = [dict(author_id=7, name="Frank Herbert", average_rating=4.1, ratings_count=10),
               dict(author_id=5, name="Stanisław Lem", average_rating=4.0, ratings_count=5),
               dict(author_id=9, name=None, average_rating=None, ratings_count=None)]
    work_authors = [dict(work_id=300, author_id=7, role=None, position=0),
                    dict(work_id=100, author_id=5, role=None, position=0),
                    dict(work_id=100, author_id=9, role="Translator", position=1),
                    dict(work_id=200, author_id=7, role=None, position=0)]
    work_genres = [dict(work_id=300, genre="fantasy, paranormal", votes=10, share=0.5),
                   dict(work_id=300, genre="fiction", votes=10, share=0.5),
                   dict(work_id=100, genre="fiction", votes=3, share=1.0)]
    return dict(works=works, editions=editions, authors=authors, work_authors=work_authors,
                work_genres=work_genres)


_SCHEMAS = {
    "works": pa.schema([("work_id", pa.int64()), ("best_book_id", pa.int64()), ("title", pa.string()),
                        ("original_title", pa.string()), ("publication_year", pa.int32()),
                        ("language_code", pa.string()), ("description", pa.string()),
                        ("is_collection", pa.bool_()), ("in_cf", pa.bool_()), ("cf_ratings", pa.int64()),
                        ("cf_mean_rating", pa.float64()), ("gr_ratings_count", pa.int64()),
                        ("gr_ratings_sum", pa.int64())]),
    "editions": pa.schema([("book_id", pa.int64()), ("work_id", pa.int64()), ("title", pa.string()),
                           ("title_without_series", pa.string()), ("isbn", pa.string()), ("isbn13", pa.string()),
                           ("asin", pa.string()), ("kindle_asin", pa.string()), ("language_code", pa.string()),
                           ("country_code", pa.string()), ("format", pa.string()), ("is_ebook", pa.bool_()),
                           ("num_pages", pa.int32()), ("publication_year", pa.int32()), ("publisher", pa.string()),
                           ("ratings_count", pa.int64()), ("average_rating", pa.float64()),
                           ("image_url", pa.string()), ("url", pa.string())]),
    "authors": pa.schema([("author_id", pa.int64()), ("name", pa.string()), ("average_rating", pa.float64()),
                          ("ratings_count", pa.int64())]),
    "work_authors": pa.schema([("work_id", pa.int64()), ("author_id", pa.int64()), ("role", pa.string()),
                               ("position", pa.int16())]),
    "work_genres": pa.schema([("work_id", pa.int64()), ("genre", pa.string()), ("votes", pa.int32()),
                              ("share", pa.float64())]),
}


def _write(clean_dir: Path, catalog: dict[str, list[dict]], manifest_rows: dict[str, int] | None = None) -> Path:
    clean_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in catalog.items():
        schema = _SCHEMAS[name]
        cols = {f.name: [r.get(f.name) for r in rows] for f in schema}
        pq.write_table(pa.table(cols, schema=schema), clean_dir / f"{name}.parquet")
    rows = {name: len(r) for name, r in catalog.items()} | (manifest_rows or {})
    manifest = {"outputs": {name: {"rows": n} for name, n in rows.items()}}
    (clean_dir / "manifest.json").write_text(json.dumps(manifest))
    return clean_dir


def _q(dsn: str, sql: str) -> list[tuple]:
    with psycopg.connect(dsn) as conn:
        return conn.execute(sql).fetchall()


# ---------- тесты ----------

def test_first_load_maps_ids_and_links(dsn, tmp_path):
    result = db_load.load(_write(tmp_path / "c", _catalog()), dsn=dsn)

    assert result["counts"] == {"works": 3, "editions": 4, "authors": 3, "work_authors": 4, "work_genres": 3}
    # Внутренние id выдаются в порядке Goodreads-id — воспроизводимо на пустой БД.
    assert _q(dsn, "SELECT external_id, internal_id FROM external_ids WHERE entity_type = 'work' "
                   "ORDER BY internal_id") == [("100", 1), ("200", 2), ("300", 3)]
    # Связи переведены во внутренние id.
    assert _q(dsn, """
        SELECT w.title, e.isbn13, a.name, wa.role FROM works w
        JOIN editions e ON e.id = w.best_edition_id
        JOIN work_authors wa ON wa.work_id = w.id JOIN authors a ON a.id = wa.author_id
        WHERE w.title = 'Solaris' ORDER BY wa.position""") == [
        ("Solaris", "9780156027601", "Stanisław Lem", None), ("Solaris", "9780156027601", None, "Translator")]
    # NULL и пустая строка различаются после COPY.
    assert _q(dsn, "SELECT title, description FROM works ORDER BY id") == [
        ("Solaris", ""), ("Box Set #1-3", None), ("Dune", "spice")]
    assert _q(dsn, "SELECT count(*) FROM genres") == [(2,)]


def test_same_manifest_is_skipped_and_force_changes_nothing(dsn, tmp_path):
    clean = _write(tmp_path / "c", _catalog())
    db_load.load(clean, dsn=dsn)

    assert db_load.load(clean, dsn=dsn)["skipped"] is True
    forced = db_load.load(clean, dsn=dsn, force=True)
    assert forced["skipped"] is False
    assert all(n == 0 for n in forced["changes"].values()), forced["changes"]
    assert _q(dsn, "SELECT count(*) FROM source_loads") == [(2,)]


def test_reload_updates_changed_and_removes_missing(dsn, tmp_path):
    db_load.load(_write(tmp_path / "v1", _catalog()), dsn=dsn)
    dune_id = _q(dsn, "SELECT id FROM works WHERE title = 'Dune'")[0][0]

    cat = _catalog()
    cat["works"] = [w for w in cat["works"] if w["work_id"] != 200]
    cat["editions"] = [e for e in cat["editions"] if e["work_id"] != 200]
    cat["work_authors"] = [a for a in cat["work_authors"] if a["work_id"] != 200]
    cat["works"][0]["title"] = "Dune (renamed)"
    cat["work_genres"] = [g for g in cat["work_genres"] if g["genre"] != "fiction" or g["work_id"] != 300]
    result = db_load.load(_write(tmp_path / "v2", cat), dsn=dsn)

    assert result["changes"]["works: удалено"] == 1
    assert result["changes"]["works: вставлено/изменено"] == 1
    assert result["changes"]["work_genres: удалено"] == 1
    assert _q(dsn, "SELECT id, title FROM works WHERE id = %d" % dune_id) == [(dune_id, "Dune (renamed)")]
    assert _q(dsn, "SELECT count(*) FROM external_ids WHERE entity_type = 'work'") == [(2,)]
    assert _q(dsn, "SELECT count(*) FROM editions") == [(3,)]


def test_removing_work_with_app_rating_fails_and_rolls_back(dsn, tmp_path):
    db_load.load(_write(tmp_path / "v1", _catalog()), dsn=dsn)
    with psycopg.connect(dsn) as conn:
        conn.execute("INSERT INTO app_users (name) VALUES ('me')")
        conn.execute("INSERT INTO ratings (user_id, work_id, value) "
                     "SELECT u.id, w.id, 5 FROM app_users u, works w WHERE w.title = 'Box Set #1-3'")

    cat = _catalog()
    cat["works"] = [w for w in cat["works"] if w["work_id"] != 200]
    cat["editions"] = [e for e in cat["editions"] if e["work_id"] != 200]
    cat["work_authors"] = [a for a in cat["work_authors"] if a["work_id"] != 200]
    with pytest.raises(RuntimeError, match="данные пользователей"):
        db_load.load(_write(tmp_path / "v2", cat), dsn=dsn)

    assert _q(dsn, "SELECT count(*) FROM works") == [(3,)]
    assert _q(dsn, "SELECT count(*) FROM source_loads") == [(1,)]


def test_manifest_mismatch_fails_and_rolls_back(dsn, tmp_path):
    with pytest.raises(RuntimeError, match="Сверка с manifest.json"):
        db_load.load(_write(tmp_path / "c", _catalog(), manifest_rows={"works": 4}), dsn=dsn)

    assert _q(dsn, "SELECT count(*) FROM works") == [(0,)]
    assert _q(dsn, "SELECT count(*) FROM source_loads") == [(0,)]


def test_concurrent_load_is_refused(dsn, tmp_path):
    clean = _write(tmp_path / "c", _catalog())
    with psycopg.connect(dsn) as other:
        other.execute("SELECT pg_advisory_xact_lock(hashtext('booksengine.load-db'))")
        with pytest.raises(RuntimeError, match="уже идёт"):
            db_load.load(clean, dsn=dsn)
