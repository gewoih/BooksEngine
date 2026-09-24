import numpy as np
import pandas as pd
import psycopg
import pytest
import scipy.sparse as sp

from booksengine import db_load
from booksengine import export_model as ex
from booksengine import recommend as rec
from booksengine.model.base import fingerprint
from booksengine.model.chance import Chance
from booksengine.model.filters import RatedFilter, work_info
from booksengine.model.matrix import catalog_works
from booksengine.model.mix import DNF_INPUT, Mix
from booksengine.model.series import SeriesIndex, exclusion
from tests.test_db_load import _catalog, _write
from tests.test_mix import components  # noqa: F401 — фикстура: 40 книг (work_id 100..139), 30 в EASE

# Названия задают серии и «уже оценено»: 100–102 — серия Saga; 103 — сборник «Alpha / Beta» автора 1,
# 104 — «Alpha» того же автора; 105 — «Alpha» другого автора (не дубль).
TITLES = {100: "One (Saga, #1)", 101: "Two (Saga, #2)", 102: "Saga Box (Saga, #1-2)",
          103: "Alpha / Beta", 104: "Alpha", 105: "Alpha"}
AUTHOR = {103: 1, 104: 1, 105: 2}


@pytest.fixture
def world(components, tmp_path):  # noqa: F811
    train, als_dir, ease_dir = components
    clean = tmp_path / "clean"
    clean.mkdir()
    ids = train.work_ids
    pd.DataFrame({"work_id": np.repeat(ids, 2), "user_id": np.tile([1, 2], len(ids)),
                  "rating": 4.0}).to_parquet(clean / "ratings.parquet")
    titles = [TITLES.get(int(w), f"Book {w}") for w in ids]
    pd.DataFrame({"work_id": ids, "title": titles, "best_edition_title": titles,
                  "is_collection": [int(w) == 102 for w in ids]}).to_parquet(clean / "works.parquet")
    pd.DataFrame({"work_id": ids, "author_id": [AUTHOR.get(int(w), 10 + int(w)) for w in ids],
                  "role": None, "position": 0}).to_parquet(clean / "work_authors.parquet")
    pd.DataFrame({"author_id": [1, 2, *[10 + int(w) for w in ids]],
                  "name": ["A1", "A2", *[f"Author {w}" for w in ids]]}).to_parquet(clean / "authors.parquet")
    pd.DataFrame({"book_id": [1, 2, 3, 4], "work_id": [100, 100, 101, 102],
                  "image_url": ["https://x/1m/1.jpg", "https://x/2m/2.jpg",
                                "https://s.gr-assets.com/assets/nophoto/book/111x148.png", None],
                  "ratings_count": [5, 50, 100, 7]}).to_parquet(clean / "editions.parquet")
    pd.DataFrame({"shadow_work_id": [900], "main_work_id": [104]}).to_parquet(clean / "work_merges.parquet")

    models = tmp_path / "models"
    m = Mix(als_dir, ease_dir)
    m.fit(train)
    m.save(models / "mix")
    Chance([-0.9, -0.46, 0.85], 3.0, 0.72, fingerprint(models / "mix")).save(models / "mix" / "chance.json")

    prof = tmp_path / "me.csv"
    pd.DataFrame({"goodreads_work_id": [100, 104, 110, 111, 900, 112], "rating": [5, 4, 2, 3, 5, None],
                  "status": ["read"] * 5 + ["dnf"], "title": ["a", "b", "c", "d", "e", "f"]}).to_csv(prof, index=False)
    return clean, models, {"me": prof}, tmp_path / "tmp"


def test_exclusion_pairs_reproduce_recommend_filters(world):
    clean, *_ = world
    work_ids = catalog_works(clean / "ratings.parquet")
    info = work_info(clean, work_ids)
    pairs = ex.exclusion_pairs(info)
    got = {(int(r.rated), int(r.excluded), r.reason) for r in pairs.itertuples()}
    col = {int(w): c for c, w in enumerate(work_ids)}
    assert (col[100], col[101], ex.SERIES) in got and (col[101], col[102], ex.SERIES) in got
    assert (col[104], col[103], ex.RATED) in got and (col[103], col[104], ex.RATED) in got
    assert not any(r == col[105] or e == col[105] for r, e, _ in got)      # другой автор — не дубль
    # на любых наборах оценённых пары дают ровно то же, что фильтры `recommend`
    idx = SeriesIndex(info.title.tolist())
    rng = np.random.default_rng(0)
    for _ in range(30):
        rated = np.sort(rng.choice(len(work_ids), size=5, replace=False))
        X = sp.csr_matrix((np.ones(5, np.float32), (np.zeros(5, int), rated)), shape=(1, len(work_ids)))
        series_cols = set(exclusion(X, idx).indices.tolist()) - set(rated.tolist())
        f = RatedFilter(info, rated)
        rated_cols = {c for c in range(len(work_ids)) if c not in rated and f.is_rated_already(c)}
        sub = pairs[pairs.rated.isin(rated) & ~pairs.excluded.isin(rated)]
        assert set(sub[sub.reason == ex.SERIES].excluded) == series_cols
        assert set(sub[sub.reason == ex.RATED].excluded) == rated_cols - series_cols


def test_covers_take_most_popular_edition_with_image(world):
    clean, *_ = world
    c = ex.covers(clean, np.arange(100, 140))
    assert c.set_index("gr_work_id").image_url.to_dict() == {100: "https://x/2m/2.jpg"}   # nophoto и NULL — нет


def test_build_matches_model_and_recommend(world):
    clean, models, profiles, tmp = world
    e = ex.build(clean_dir=clean, models_dir=models, profiles=profiles, tmp_dir=tmp, top=5)
    mix = Mix.load(models / "mix")
    assert list(e.items.gr_work_id) == list(range(100, 140)) and list(e.items.col) == list(range(40))
    assert e.items.ease_pos.notna().sum() == len(mix.ease.top_cols) == 30
    np.testing.assert_array_equal(np.stack(e.items.embedding), mix.als.item_factors)
    assert len(e.ease) == mix.ease._B.nnz
    assert e.meta["fingerprint"] == fingerprint(models / "mix")
    assert e.meta["chance_coef"] == [-0.9, -0.46, 0.85] and e.meta["als_neg_rule"] == mix.als.neg_rule
    assert e.meta["dnf_input"] == DNF_INPUT
    assert e.merges.to_dict("records") == [{"shadow_gr": 900, "main_gr": 104}]
    # эталон — ровно выдача `recommend`, книги объяснения — id, а не названия
    want = rec.recommend(profiles["me"], clean_dir=clean, models_dir=models, top=5)
    g = e.golden_recs[e.golden_recs.profile == "me"]
    assert len(g) == 5 and g.because.map(len).sum() > 0
    assert list(g.gr_work_id) == [r.work_id for r in want.recs]
    assert list(g.chance) == [r.chance for r in want.recs]
    assert all(set(b) <= {100, 104, 110, 111, 112} for b in g.because)
    inp = e.golden_inputs[e.golden_inputs.profile == "me"].set_index("gr_work_id")
    assert inp.rating.to_dict() == {100: 5.0, 104: 4.5, 110: 2.0, 111: 3.0, 112: 1.0}   # тень 900 → 104, среднее
    assert inp.dnf.to_dict() == {100: False, 104: False, 110: False, 111: False, 112: True}


# ---------- запись в БД ----------


def _tiny(gr=(100, 300), main_for_shadow=300) -> ex.ModelExport:
    """Две книги каталога test_db_load (Goodreads 100 — Solaris, 300 — Dune), обе в EASE."""
    return ex.ModelExport(
        meta={"fingerprint": "fp1", "als_weight": 0.5},
        items=pd.DataFrame({"gr_work_id": list(gr), "col": [0, 1], "ease_pos": pd.array([0, 1], dtype="Int32"),
                            "embedding": [np.array([0.5, -1.25], np.float32), np.array([1e-8, 3.0], np.float32)]}),
        ease=pd.DataFrame({"from_pos": [0, 1], "to_pos": [1, 0], "weight": np.array([0.25, -0.5], np.float32)}),
        exclusions=pd.DataFrame({"rated_gr": [100], "excluded_gr": [300], "reason": [ex.SERIES]}),
        merges=pd.DataFrame({"shadow_gr": [999], "main_gr": [main_for_shadow]}),
        covers=pd.DataFrame({"gr_work_id": [300], "image_url": ["https://x/1m/1.jpg"]}),
        golden_inputs=pd.DataFrame({"profile": ["me"], "gr_work_id": [100], "rating": [4.5], "dnf": [False]}),
        golden_recs=pd.DataFrame({"profile": ["me"], "rank": [1], "gr_work_id": [300], "score": [1.5],
                                  "chance": [77], "because": [[100]], "despite": pd.array([None], dtype="Int64")}))


def _q(dsn, sql):
    with psycopg.connect(dsn) as c:
        return c.execute(sql).fetchall()


def test_write_maps_goodreads_ids_and_overwrites(dsn, tmp_path):
    db_load.load(_write(tmp_path / "c", _catalog()), dsn=dsn)       # внутренние id: 100 → 1, 300 → 3
    assert ex.write(_tiny(), dsn)["work_embeddings"] == 2
    assert _q(dsn, "SELECT work_id, col, ease_pos, embedding::text FROM work_embeddings ORDER BY col") == [
        (1, 0, 0, "[0.5,-1.25]"), (3, 1, 1, "[1e-08,3]")]
    assert _q(dsn, "SELECT from_pos, to_pos, weight FROM ease_weights ORDER BY 1") == [(0, 1, 0.25), (1, 0, -0.5)]
    assert _q(dsn, "SELECT rated_work_id, excluded_work_id, reason FROM work_exclusions") == [(1, 3, "series")]
    assert _q(dsn, "SELECT shadow_external_id, main_work_id FROM work_merges") == [("999", 3)]
    assert _q(dsn, "SELECT work_id, image_url FROM work_covers") == [(3, "https://x/1m/1.jpg")]
    assert _q(dsn, "SELECT profile, work_id, rating, dnf FROM golden_inputs") == [("me", 1, 4.5, False)]
    assert _q(dsn, "SELECT work_id, because, despite FROM golden_recommendations") == [(3, [1], None)]
    assert _q(dsn, "SELECT fingerprint, params->>'als_weight' FROM model_meta") == [("fp1", "0.5")]
    # повторный экспорт перезаписывает, а не дописывает
    e = _tiny()
    e.meta["fingerprint"] = "fp2"
    e.exclusions = e.exclusions.iloc[0:0]
    ex.write(e, dsn)
    assert _q(dsn, "SELECT fingerprint FROM model_meta") == [("fp2",)]
    assert _q(dsn, "SELECT count(*) FROM work_exclusions") == [(0,)]
    assert _q(dsn, "SELECT count(*) FROM work_embeddings") == [(2,)]


def test_unknown_goodreads_id_fails_and_keeps_previous_model(dsn, tmp_path):
    db_load.load(_write(tmp_path / "c", _catalog()), dsn=dsn)
    ex.write(_tiny(), dsn)
    with pytest.raises(ValueError, match="нет в каталоге БД: 1"):
        ex.write(_tiny(main_for_shadow=777), dsn)                      # 777 — нет в каталоге
    assert _q(dsn, "SELECT fingerprint FROM model_meta") == [("fp1",)]
    assert _q(dsn, "SELECT count(*) FROM work_merges") == [(1,)]
