"""export-app: новые книги единой базы в БД приложения (ключ «автор|название»), обложки, слияния теней."""
import pandas as pd
import psycopg
import pytest

from booksengine import app_export as ax
from booksengine.data import merged
from tests.test_merged import _amazon, _goodreads


@pytest.fixture
def base(tmp_path):
    """Единая база, у новой книги (Educated) автор — новый (его нет в Goodreads), есть русское название."""
    gr, az, out = tmp_path / "books" / "clean", tmp_path / "amazon" / "clean", tmp_path / "books-amazon" / "clean"
    _goodreads(gr)
    _amazon(az)
    a = pd.read_parquet(gr / "authors.parquet")
    a[a.name != "Tara Westover"].to_parquet(gr / "authors.parquet")
    ru = tmp_path / "ru.csv"
    pd.DataFrame([{"author": "Tara Westover", "title": "Educated", "ru_title": "Ученица"}]).to_csv(ru, index=False)
    merged.build(gr, az, out, scale="raw", min_user=2, min_new_ratings=2, ru_titles=ru)
    return out


def test_new_work_key_is_author_and_short_title(base):
    keys = merged.new_work_keys(base)
    assert keys.tolist() == ["tarawestover|educated"] and keys.index[0] >= merged.NEW_WORK_OFFSET


def test_new_works_show_russian_title_and_new_author_by_name_key(base):
    works, authors, genres = ax.new_works(base)
    w = works.iloc[0]
    assert (w.key, w.title, w.original_title) == ("tarawestover|educated", "Ученица / Educated", "Educated")
    a = authors.iloc[0]
    assert pd.isna(a.gr_author) and a.author_key == "tarawestover" and a["name"] == "Tara Westover"


def _new_rows(conn):
    return conn.execute("""
        SELECT x.external_id, w.title, a.name FROM external_ids x JOIN works w ON w.id = x.internal_id
        JOIN work_authors wa ON wa.work_id = w.id JOIN authors a ON a.id = wa.author_id
        WHERE x.source_id = 2 AND x.entity_type = 'work'""").fetchall()


def test_write_is_idempotent_and_refuses_to_drop_rated_books(base, dsn):
    works, authors, genres = ax.new_works(base)
    empty = pd.DataFrame({"gr_work_id": pd.Series(dtype="int64"), "image_url": pd.Series(dtype=str)})
    no_merges = pd.DataFrame({"shadow_gr": pd.Series(dtype="int64"), "main_gr": pd.Series(dtype="int64")})
    kw = dict(covers_=empty, merges=no_merges)
    first = ax.write(dsn, works=works, authors=authors, genres=genres, **kw)
    assert first["новых книг вставлено/изменено"] == 1
    again = ax.write(dsn, works=works, authors=authors, genres=genres, **kw)
    assert again["новых книг вставлено/изменено"] == 0
    with psycopg.connect(dsn) as conn:
        assert _new_rows(conn) == [("tarawestover|educated", "Ученица / Educated", "Tara Westover")]
        wid = conn.execute("SELECT internal_id FROM external_ids WHERE source_id = 2 AND entity_type = 'work'").fetchone()[0]
        uid = conn.execute("INSERT INTO app_users (name) VALUES ('u') RETURNING id").fetchone()[0]
        conn.execute("INSERT INTO ratings (user_id, work_id, value) VALUES (%s, %s, 5)", (uid, wid))
    # база пересобрана: Educated больше нет, вместо неё другая книга
    other = dict(works=works.assign(key="tarawestover|other", title="Other"),
                 authors=authors.assign(key="tarawestover|other"), genres=genres.assign(key="tarawestover|other"))
    with pytest.raises(RuntimeError, match="оценили пользователи"):           # оценённую книгу не удалить молча
        ax.write(dsn, **other, **kw)
    with psycopg.connect(dsn) as conn:
        assert [r[0] for r in _new_rows(conn)] == ["tarawestover|educated"]
        conn.execute("DELETE FROM ratings")
    ax.write(dsn, **other, **kw)
    with psycopg.connect(dsn) as conn:
        assert _new_rows(conn) == [("tarawestover|other", "Other", "Tara Westover")]
    # база без новых книг (книжная) новые книги приложения не трогает
    ax.write(dsn, works=works.iloc[:0], authors=authors.iloc[:0], genres=genres.iloc[:0], **kw)
    with psycopg.connect(dsn) as conn:
        assert len(_new_rows(conn)) == 1
