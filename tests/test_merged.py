import numpy as np
import pandas as pd
import pytest

from booksengine.data import merged


def _goodreads(d):
    d.mkdir(parents=True)
    pd.DataFrame([
        {"user_id": u, "work_id": w, "rating": np.float32(r), "n_editions": np.int16(1)}
        for u, w, r in [(1, 10, 5), (1, 20, 4), (2, 10, 3), (2, 30, 5)]
    ]).to_parquet(d / "ratings.parquet")
    pd.DataFrame({"user_id": [1, 2], "external_id": ["g1", "g2"], "n_ratings": [2, 2],
                  "mean_rating": [4.5, 4.0], "sd_rating": [0.5, 1.0]}).astype({"user_id": "int32"}
                                                                          ).to_parquet(d / "users.parquet")
    pd.DataFrame({"work_id": [10, 20, 30, 40], "best_book_id": [1, 2, 3, 4],
                  "title": ["Pride and Prejudice", "Dune (Dune, #1)", "Old Book", "Not in core"],
                  "original_title": [None] * 4, "best_edition_title": [None] * 4, "publication_year": [1813, 1965, 1990, 2000],
                  "language_code": ["eng"] * 4, "media_type": ["book"] * 4, "books_count": [1] * 4,
                  "gr_ratings_count": [1] * 4, "gr_ratings_sum": [1] * 4, "description": [None] * 4,
                  "is_collection": [False] * 4, "is_nonbook": [False] * 4, "in_cf": [True, True, True, False],
                  "cf_ratings": [2, 1, 1, 0], "cf_mean_rating": [4.0, 4.0, 5.0, None]}).to_parquet(d / "works.parquet")
    pd.DataFrame({"author_id": [100, 200, 300], "name": ["Jane Austen", "Frank Herbert", "Tara Westover"],
                  "average_rating": [4.0, 4.0, 4.0], "ratings_count": [10, 10, 1]}).to_parquet(d / "authors.parquet")
    pd.DataFrame({"work_id": [10, 20, 30], "author_id": [100, 200, 100], "role": [None] * 3,
                  "position": np.array([1, 1, 1], dtype=np.int16)}).to_parquet(d / "work_authors.parquet")
    pd.DataFrame({"work_id": [30], "genre": ["non-fiction"], "votes": [5], "share": [1.0]}).to_parquet(
        d / "work_genres.parquet")
    pd.DataFrame({"shadow_work_id": [99], "main_work_id": [10]}).to_parquet(d / "work_merges.parquet")
    split = d.parent / "model" / "split"
    split.mkdir(parents=True)
    pd.DataFrame({"user_id": [2], "group": ["test"], "n_ratings": [2], "bucket": ["20-39"]}).to_parquet(
        split / "holdout_users.parquet")
    (split / "split.json").write_text("{}")


def _amazon(d):
    d.mkdir(parents=True)
    items = [
        # мост по ISBN уже есть
        {"parent_asin": "P1", "title": "Pride and Prejudice (Penguin Classics)", "author": "Jane Austen", "year": 2019,
         "categories": "Books|Literature & Fiction", "source": "books", "isbn13": None, "wikidata": None},
        # моста нет, но то же произведение — автор и название совпадают с книгой ядра (переиздание)
        {"parent_asin": "P2", "title": "Dune: Deluxe Edition", "author": "Frank  Herbert", "year": 2019,
         "categories": "Books|Science Fiction & Fantasy", "source": "books", "isbn13": None, "wikidata": None},
        # новая книга в двух изданиях (Books и Kindle) — одно произведение
        {"parent_asin": "P3", "title": "Educated: A Memoir", "author": "Tara Westover", "year": 2018,
         "categories": "Books|Biographies & Memoirs", "source": "books", "isbn13": "9780399590504", "wikidata": "ru"},
        {"parent_asin": "K3", "title": "Educated", "author": "Tara Westover", "year": 2019,
         "categories": "Kindle Store|Kindle eBooks|Biographies & Memoirs", "source": "kindle", "isbn13": None,
         "wikidata": None},
        # новая книга, у которой мало читателей, — не входит
        {"parent_asin": "P4", "title": "Tiny Book", "author": "Nobody", "year": 2020,
         "categories": "Books|Romance", "source": "books", "isbn13": None, "wikidata": None},
        # старая книга без моста — не новая, не входит
        {"parent_asin": "P5", "title": "Some 2010 Book", "author": "Somebody", "year": 2010,
         "categories": "Books|Romance", "source": "books", "isbn13": None, "wikidata": None},
    ]
    pd.DataFrame(items).assign(isbn10=None, n_ratings=1).to_parquet(d / "items.parquet")
    pd.DataFrame({"parent_asin": ["P1", "P2", "P3", "K3", "P4", "P5"],
                  "work_id": [10, None, None, None, None, None]}).to_parquet(d / "bridge.parquet")
    rows = []
    for u in range(3):                                   # трое читали всё
        rows += [(f"A{u}", "P1", 5.0), (f"A{u}", "P2", 4.0), (f"A{u}", "P3", 5.0), (f"A{u}", "K3", 4.0),
                 (f"A{u}", "P5", 5.0)]
    rows += [("A0", "P4", 5.0), ("B", "P1", 5.0)]       # у B — одна книга: ниже порога людей
    pd.DataFrame(rows, columns=["user_id", "parent_asin", "rating"]).assign(timestamp=0, source="books").to_parquet(
        d / "ratings.parquet")


@pytest.fixture
def built(tmp_path):
    gr, az, out = tmp_path / "books" / "clean", tmp_path / "amazon" / "clean", tmp_path / "books-amazon" / "clean"
    _goodreads(gr)
    _amazon(az)
    stats = merged.build(gr, az, out, scale="q", min_user=2, min_new_ratings=2)
    return gr, out, stats


def test_build_maps_amazon_books_to_goodreads_works_and_new_works(built):
    gr, out, stats = built
    r = pd.read_parquet(out / "ratings.parquet")
    az = r[r.user_id >= merged.AMAZON_USER_OFFSET]
    works = pd.read_parquet(out / "works.parquet").set_index("work_id")
    new = works[works.source == "amazon"]
    assert len(new) == 1                                                  # Educated — одно произведение
    new_id = new.index[0]
    assert new_id >= merged.NEW_WORK_OFFSET and new.loc[new_id, "ru_known"] == True
    assert new.loc[new_id, "title"] == "Educated"      # поровну оценок — краткое название
    # P1 → 10 по мосту ISBN, P2 → 20 по автору и названию, P3 и K3 → новая книга; P4 (мало читателей), P5 (старая) — нет
    assert sorted(az.work_id.unique().tolist()) == [10, 20, int(new_id)]
    assert az.user_id.nunique() == 3                                      # у B одна книга — ниже порога
    # два издания Educated у человека — одна оценка: среднее 4.5 → 5 по шкале q (5 → 5)
    assert az[az.work_id == new_id].rating.tolist() == [5.0, 5.0, 5.0]
    assert az[az.work_id == 20].rating.tolist() == [3.0, 3.0, 3.0]         # 4★ Amazon → 3★ по шкале q
    key = ["user_id", "work_id"]
    pd.testing.assert_frame_equal(
        r[r.user_id < merged.AMAZON_USER_OFFSET].sort_values(key).reset_index(drop=True),
        pd.read_parquet(gr / "ratings.parquet").sort_values(key).reset_index(drop=True), check_dtype=False)
    assert stats["new_works"] == 1 and stats["amazon_users"] == 3 and stats["by_title"] == 1


def test_build_links_new_work_author_to_goodreads_author_and_marks_nonfiction(built):
    gr, out, _ = built
    works = pd.read_parquet(out / "works.parquet")
    new_id = works.loc[works.source == "amazon", "work_id"].iloc[0]
    wa = pd.read_parquet(out / "work_authors.parquet")
    assert wa[wa.work_id == new_id].author_id.tolist() == [300]           # Tara Westover уже есть в Goodreads
    assert len(pd.read_parquet(out / "authors.parquet")) == 3
    genres = pd.read_parquet(out / "work_genres.parquet")
    assert genres[genres.work_id == new_id].genre.tolist() == ["non-fiction"]
    assert works[works.source == "goodreads"].ru_known.isna().all()      # Goodreads правилом перевода не трогаем


def test_build_copies_goodreads_split_and_users(built, tmp_path):
    gr, out, _ = built
    split = out.parent / "model" / "split"
    assert (split / "holdout_users.parquet").exists() and (split / "split.json").exists()
    users = pd.read_parquet(out / "users.parquet")
    assert users.user_id.nunique() == len(users) == 2 + 3
    assert users[users.user_id >= merged.AMAZON_USER_OFFSET].external_id.str.startswith("amazon:").all()
    pd.testing.assert_frame_equal(pd.read_parquet(out / "work_merges.parquet"),
                                  pd.read_parquet(gr / "work_merges.parquet"), check_dtype=False)


def test_new_author_gets_own_id_when_not_in_goodreads(tmp_path):
    gr, az, out = tmp_path / "books" / "clean", tmp_path / "amazon" / "clean", tmp_path / "m" / "clean"
    _goodreads(gr)
    _amazon(az)
    a = pd.read_parquet(gr / "authors.parquet")
    a[a.name != "Tara Westover"].to_parquet(gr / "authors.parquet")
    merged.build(gr, az, out, scale="raw", min_user=2, min_new_ratings=2)
    authors = pd.read_parquet(out / "authors.parquet")
    new = authors[authors.author_id >= merged.NEW_AUTHOR_OFFSET]
    assert new.name.tolist() == ["Tara Westover"]


def test_profile_row_without_goodreads_id_matches_new_amazon_book(built, tmp_path):
    """Книга профиля, которой нет в Goodreads («Атомные привычки»), — по английскому названию и автору находит новую
    книгу Amazon в единой базе и идёт во вход модели."""
    from booksengine.model.matrix import catalog_works
    from booksengine.recommend import read_profile
    gr, out, _ = built
    prof = tmp_path / "p.csv"
    pd.DataFrame([{"title": "Гордость и предубеждение", "title_en": "Pride and Prejudice", "author": "Jane Austen",
                   "rating": 5, "goodreads_work_id": 10},
                  {"title": "Ученица", "title_en": "Educated", "author": "Tara Westover", "rating": 4,
                   "goodreads_work_id": None},
                  {"title": "Неизвестная", "title_en": "Unknown", "author": "Nobody", "rating": 3,
                   "goodreads_work_id": None}]).to_csv(prof, index=False)
    work_ids = catalog_works(out / "ratings.parquet")
    p = read_profile(prof, work_ids, out)
    new_id = pd.read_parquet(out / "works.parquet").query("source == 'amazon'").work_id.iloc[0]
    got = {int(work_ids[c]) for c in p.x.indices}
    assert got == {10, int(new_id)}
    assert [n for n, _ in p.skipped] == ["Неизвестная"]
    # в базе только Goodreads строка без id так и остаётся без id
    p0 = read_profile(prof, catalog_works(gr / "ratings.parquet"), gr)
    assert sorted(n for n, _ in p0.skipped) == ["Неизвестная", "Ученица"]


def test_curated_russian_titles_mark_translation_and_give_russian_title(tmp_path):
    """Перевод новой книги известен и без Wikidata — по разметке config/amazon_ru_titles.csv (автор и название, как у
    Amazon); оттуда же русское название для выдачи."""
    gr, az, out = tmp_path / "books" / "clean", tmp_path / "amazon" / "clean", tmp_path / "m" / "clean"
    _goodreads(gr)
    _amazon(az)
    items = pd.read_parquet(az / "items.parquet")
    items["wikidata"] = None                                   # Wikidata перевода не знает
    items.to_parquet(az / "items.parquet")
    ru = tmp_path / "ru.csv"
    pd.DataFrame([{"author": "Tara Westover", "title": "Educated", "ru_title": "Ученица"}]).to_csv(ru, index=False)

    merged.build(gr, az, out, scale="q", min_user=2, min_new_ratings=2)
    new = pd.read_parquet(out / "works.parquet").query("source == 'amazon'")
    assert not new.ru_known.iloc[0] and new.ru_title.isna().all()

    stats = merged.build(gr, az, out, scale="q", min_user=2, min_new_ratings=2, ru_titles=ru)
    new = pd.read_parquet(out / "works.parquet").query("source == 'amazon'")
    assert new.ru_known.iloc[0] and new.ru_title.iloc[0] == "Ученица"
    assert stats["new_with_ru"] == 1


def test_goodreads_rows_keeps_columns_and_drops_amazon_people():
    """Вкус в единой базе можно учить только на людях Goodreads: столбцы (книги, в том числе новые) — те же, люди
    Amazon (user_id от AMAZON_USER_OFFSET) — не входят."""
    import scipy.sparse as sp
    from booksengine.model.matrix import RatingMatrix
    X = sp.csr_matrix(np.array([[5, 0, 3], [0, 4, 0], [2, 2, 2]], dtype=np.float32))
    users = np.array([1, 2, merged.AMAZON_USER_OFFSET + 1])
    got = merged.goodreads_rows(RatingMatrix(X, users, np.array([10, 20, merged.NEW_WORK_OFFSET + 1])))
    assert got.user_ids.tolist() == [1, 2] and got.X.shape == (2, 3)
    assert got.work_ids.tolist() == [10, 20, merged.NEW_WORK_OFFSET + 1]
    np.testing.assert_array_equal(got.X.toarray(), [[5, 0, 3], [0, 4, 0]])


def test_taste_rows_keeps_goodreads_and_dense_amazon_readers():
    """Вкус единой базы: люди Goodreads и плотные читатели Amazon (от amazon_min книг) — у новых книг появляются
    координаты вкуса, а люди с парой оценок не сдвигают его."""
    import scipy.sparse as sp
    from booksengine.model.matrix import RatingMatrix
    X = sp.csr_matrix(np.array([[5, 0, 3], [0, 4, 0], [2, 2, 2], [5, 0, 0]], dtype=np.float32))
    users = np.array([1, 2, merged.AMAZON_USER_OFFSET + 1, merged.AMAZON_USER_OFFSET + 2])
    train = RatingMatrix(X, users, np.array([10, 20, merged.NEW_WORK_OFFSET + 1]))
    assert merged.taste_rows(train, amazon_min=3).user_ids.tolist() == [1, 2, merged.AMAZON_USER_OFFSET + 1]
    assert merged.taste_rows(train, amazon_min=None).user_ids.tolist() == [1, 2]          # только Goodreads
    assert merged.taste_rows(train, amazon_min=0).user_ids.tolist() == users.tolist()     # все


def test_new_work_title_has_html_entities_unescaped(tmp_path):
    """В метаданных Amazon встречаются HTML-сущности («Daisy Jones &amp; The Six») — в каталоге их нет."""
    gr, az, out = tmp_path / "books" / "clean", tmp_path / "amazon" / "clean", tmp_path / "m" / "clean"
    _goodreads(gr)
    _amazon(az)
    items = pd.read_parquet(az / "items.parquet")
    items.loc[items.parent_asin.isin(["P3", "K3"]), "title"] = "Educated &amp; Other &quot;Things&quot; Don&#39;t"
    items.to_parquet(az / "items.parquet")
    merged.build(gr, az, out, scale="q", min_user=2, min_new_ratings=2)
    new = pd.read_parquet(out / "works.parquet").query("source == 'amazon'")
    assert new.title.tolist() == ['Educated & Other "Things" Don\'t']


def _holdout_base(d):
    """Единая база: Goodreads-человек 1 (уже отложен), люди Amazon 1_000_000+: у 1_000_000 и 1_000_001 по 3 старых и
    2 новых книги, у 1_000_002 — 1 старая (мало), у 1_000_003 — новых нет."""
    d.mkdir(parents=True)
    old, new = [10, 20, 30], [merged.NEW_WORK_OFFSET + 1, merged.NEW_WORK_OFFSET + 2]
    rows = [(1, 10, 5.0), (1, 20, 4.0)]
    for u in (1_000_000, 1_000_001):
        rows += [(u, w, 4.0) for w in old] + [(u, w, 5.0) for w in new]
    rows += [(1_000_002, 10, 5.0), (1_000_002, new[0], 5.0)] + [(1_000_003, w, 3.0) for w in old]
    pd.DataFrame(rows, columns=["user_id", "work_id", "rating"]).astype({"rating": "float32"}).to_parquet(
        d / "ratings.parquet")
    ids = [1, 1_000_000, 1_000_001, 1_000_002, 1_000_003]
    pd.DataFrame({"user_id": ids, "external_id": ["g1"] + [f"amazon:A{i}" for i in range(4)]}).to_parquet(
        d / "users.parquet")
    split = d.parent / "model" / "split"
    split.mkdir(parents=True)
    pd.DataFrame({"user_id": [1], "group": ["test"], "n_ratings": [2], "bucket": ["20-39"]}).to_parquet(
        split / "holdout_users.parquet")
    return split


def test_amazon_holdout_hides_new_books_of_people_with_enough_old_ones(tmp_path):
    """Отложенные люди Amazon: от min_old старых книг (Goodreads) и от min_new новых; вход — старые, скрыто — все
    новые; люди Goodreads в отложенных остаются, а обучение не видит ни тех, ни других."""
    from booksengine.model.matrix import catalog_works, load_holdout, load_train
    clean = tmp_path / "books-amazon" / "clean"
    split = _holdout_base(clean)
    stats = merged.amazon_holdout(clean, split, n=10, min_old=3, min_new=2)
    assert stats == {"amazon_holdout": 2, "input": 6, "hidden": 4}
    users = pd.read_parquet(split / "holdout_users.parquet")
    assert sorted(users[users.group == "amazon"].user_id) == [1_000_000, 1_000_001]
    assert users[users.group == "test"].user_id.tolist() == [1]
    work_ids = catalog_works(clean / "ratings.parquet")
    hold = load_holdout(split, "amazon", work_ids)
    assert all((work_ids[c] >= merged.NEW_WORK_OFFSET).all() for c in hold.hidden_cols)
    assert (work_ids[hold.inputs.indices] < merged.NEW_WORK_OFFSET).all()
    train = load_train(clean / "ratings.parquet", split / "holdout_users.parquet")
    assert sorted(train.user_ids) == [1_000_002, 1_000_003]


def test_amazon_holdout_is_stable_and_capped(tmp_path):
    """Не больше n человек; выбор — по хэшу внешнего id: тот же при повторе."""
    clean = tmp_path / "books-amazon" / "clean"
    split = _holdout_base(clean)
    merged.amazon_holdout(clean, split, n=1, min_old=3, min_new=2)
    first = pd.read_parquet(split / "amazon_input.parquet").user_id.unique().tolist()
    merged.amazon_holdout(clean, split, n=1, min_old=3, min_new=2)
    users = pd.read_parquet(split / "holdout_users.parquet")
    assert len(first) == 1 and users[users.group == "amazon"].user_id.tolist() == first
