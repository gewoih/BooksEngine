import json

import numpy as np
import pandas as pd
import pytest

from booksengine.model import layers as ly
from booksengine.model import split
from booksengine.model.als import ALS
from booksengine.model.ease import EASE
from booksengine.model.matrix import load_train
from booksengine.model.mix import Mix
from booksengine.model.taste import Taste
from tests.test_split import synthetic_users


def test_combine_excludes_input_and_cuts_to_crowd_top():
    crowd = np.array([[5.0, 4.0, 3.0, 2.0, 1.0]])
    taste = np.array([[0.0, 0.0, 0.0, 0.0, 10.0]])
    excl = np.array([[True, False, False, False, False]])
    s = ly.Layers.combine(crowd, taste, excl, ly.Variant("mix", 1.0, None))
    assert s[0, 0] == -np.inf and np.argmax(s[0]) == 4           # вкус поднял последнюю книгу толпы
    s = ly.Layers.combine(crowd, taste, excl, ly.Variant("mix", 1.0, 2))
    assert np.isfinite(s[0]).tolist() == [False, True, True, False, False]  # первые 2 толпы без входа


def _data(tmp_path, n_users=90, n_works=40, seed=3):
    """Два вкуса по половинам книг, у каждой книги — свой автор, у книг 30+ — «#2» в названии."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(1, n_users + 1):
        kind = u % 2
        for w in rng.choice(n_works, size=int(rng.integers(20, 31)), replace=False):
            good = (w < n_works // 2) == bool(kind)
            rows.append((u, 100 + int(w), float(rng.choice([4, 5]) if good else rng.choice([1, 2, 3]))))
    r = pd.DataFrame(rows, columns=["user_id", "work_id", "rating"]).astype(
        {"user_id": "int32", "work_id": "int64", "rating": "float32"})
    r.to_parquet(tmp_path / "ratings.parquet", index=False)
    ids = np.arange(100, 100 + n_works)
    title = [f"Book {i} (Saga {i}, #2)" if i >= 130 else f"Book {i}" for i in ids]
    pd.DataFrame({"work_id": ids, "title": title, "best_edition_title": title,
                  "is_collection": False}).to_parquet(tmp_path / "works.parquet")
    pd.DataFrame({"work_id": ids, "author_id": ids % 7, "position": 0, "role": ""}).to_parquet(
        tmp_path / "work_authors.parquet")
    pd.DataFrame({"author_id": np.arange(7), "name": [f"A{i}" for i in range(7)]}).to_parquet(
        tmp_path / "authors.parquet")
    pd.DataFrame({"shadow_work_id": pd.Series([], dtype="int64"),
                  "main_work_id": pd.Series([], dtype="int64")}).to_parquet(tmp_path / "work_merges.parquet")
    synthetic_users(r.user_id.unique()).to_parquet(tmp_path / "users.parquet", index=False)
    return r


@pytest.fixture
def world(tmp_path):
    _data(tmp_path)
    sd, md = tmp_path / "split", tmp_path / "models"
    split.build(tmp_path / "ratings.parquet", tmp_path / "users.parquet", sd, "fp",
                test_per_bucket={"20-49": 15}, val_per_bucket={"20-49": 15}, bucket_pool_size={"20-49": 90},
                seed=11, share=0.2)
    train = load_train(tmp_path / "ratings.parquet", sd / "holdout_users.parquet")
    als = ALS(factors=4, alpha=1.0, iterations=5)
    als.fit(train)
    als.configure("le2", 0.0)
    ease = EASE(lam=10.0)
    ease.fit(train)
    ease.configure(topk=20)
    for name, m in (("als_neg", als), ("ease", ease)):
        (md / name).mkdir(parents=True)
        m.save(md / name)
    mix = Mix(md / "als_neg", md / "ease")
    mix.fit(train)
    (md / "mix").mkdir()
    mix.save(md / "mix")
    taste = Taste(factors=2, reg=0.05, iterations=5)
    taste.fit(train, log=lambda *_: None)
    (md / "taste").mkdir()
    taste.save(md / "taste")
    return tmp_path, sd, md


def test_crowd_read_restores_mix_settings(world):
    _, _, md = world
    L = ly.Layers.from_models(md)
    before = (L.mix.als_weight, L.mix.ease_input, L.mix.als.neg_rule)
    x = load_train(world[0] / "ratings.parquet", world[1] / "holdout_users.parquet").X[:3]
    a = L.crowd("read", x)
    assert (L.mix.als_weight, L.mix.ease_input, L.mix.als.neg_rule) == before
    assert not np.allclose(a, L.crowd("mix", x))


def test_run_val_test_and_profiles(world):
    tp, sd, md = world
    ed = tp / "eval"
    val = ly.run("val", clean_dir=tp, split_dir=sd, models_dir=md, eval_dir=ed)
    assert len(val["summary"]) == len(ly.CROWDS) * len(ly.CUTOFFS) * len(ly.WEIGHTS)
    ref = next(r for r in val["summary"] if r["label"] == ly.Variant(*ly.REFERENCE).label())
    assert ref["groups"]["all"]["ndcg20_diff"]["mean"] == 0.0 and ly.allowed(ref)
    assert ref["groups"]["all"]["value20_diff"]["mean"] == 0.0
    best = max(val["summary"], key=lambda r: r["groups"]["all"]["value20"]["mean"])
    assert val["chosen"] == best["variant"] and set(val["chosen_by_five_value"]) == {"2.0", "3.0"}
    chosen = json.loads((md / "layers" / "params.json").read_text())["variant"]
    assert chosen == val["chosen"]
    assert "← выбран" in ly.report(val)
    test = ly.run("test", clean_dir=tp, split_dir=sd, models_dir=md, eval_dir=ed)
    assert 1 <= len(test["summary"]) <= 2 and (ed / "layers_test.json").exists()
    prof = tp / "profiles"
    prof.mkdir()
    pd.DataFrame({"goodreads_work_id": [100, 101, 102, 125], "rating": [5, 5, 4, 1]}).to_csv(prof / "p.csv", index=False)
    text = ly.profiles(clean_dir=tp, models_dir=md, profiles_dir=prof, top=5)
    assert "## p (4 оценок)" in text and "Совпадает книг" in text


def test_allowed_rejects_significant_ndcg_drop_even_if_small():
    def row(nd_lo, nd_hi, low_lo):
        return {"groups": {"all": {"ndcg20_diff": {"mean": (nd_lo + nd_hi) / 2, "lo": nd_lo, "hi": nd_hi},
                                   "low20_diff": {"mean": low_lo, "lo": low_lo, "hi": low_lo + 0.01}}}}
    assert ly.allowed(row(-0.002, 0.001, -0.02))        # в шуме — допустим
    assert not ly.allowed(row(-0.006, -0.002, -0.02))   # мало, но значимо хуже (вес 1.5 на валидации)
    assert not ly.allowed(row(-0.001, 0.001, 0.001))    # Low@20 значимо выше


def test_saved_layers_score_calibrate_and_refuse_stale_components(world):
    from booksengine.model import chance
    tp, sd, md = world
    ed = tp / "eval"
    ly.run("val", clean_dir=tp, split_dir=sd, models_dir=md, eval_dir=ed)
    L = ly.Layers.load(md / "layers")
    x = load_train(tp / "ratings.parquet", sd / "holdout_users.parquet").X[:4]
    top = L.mix.ease.top_cols
    expect = L.combine(L.crowd(L.variant.crowd, x), L.taste_z(x), np.zeros((4, len(top)), bool), L.variant)
    got = L.score(x)
    np.testing.assert_allclose(got[:, top], expect, rtol=1e-5)
    assert np.isneginf(np.delete(got, top, axis=1)).all()
    out = chance.calibrate("layers", ratings_path=tp / "ratings.parquet", split_dir=sd, models_dir=md, eval_dir=ed)
    assert any("прогноз вкуса" in k for k in out["test"]) and len(out["chance"]["coef"]) == 3
    assert "auc_within_person" in out["test"]["место + щедрость"]
    Taste(factors=2, reg=0.07, iterations=1).save(md / "taste")      # вкус переобучен после выбора веса
    with pytest.raises(ValueError, match="layers val"):
        ly.Layers.load(md / "layers")


def test_value_of_counts_from_three():
    np.testing.assert_array_equal(ly.value_of(np.array([5.0, 4.0, 3.0, 2.0, 1.0, 4.5])), [2, 1, 0, -1, -2, 2])
    np.testing.assert_array_equal(ly.value_of(np.array([5.0, 4.0, 1.0]), five=3.0), [3, 1, -2])
