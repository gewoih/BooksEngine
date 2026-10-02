import numpy as np
import pytest

from booksengine.model import layers as ly
from booksengine.model import metrics
from booksengine.model import rerank as rr
from booksengine.model.ease import EASELike
from booksengine.model.evaluate import load_eval_holdout
from booksengine.model.filters import work_info
from booksengine.model.matrix import load_train
from booksengine.model.mix import Mix
from tests.test_layers import world  # noqa: F401 — фикстура: данные, разбиение и компоненты слоёв


@pytest.fixture
def layered(world):
    """Мир слоёв с толпой «ценность» и выбранным вариантом (models/layers)."""
    tp, sd, md = world
    train = load_train(tp / "ratings.parquet", sd / "holdout_users.parquet")
    like = EASELike(lam=10.0, topk=20)
    like.fit(train)
    like.save(md / "ease_like")
    mix = Mix(md / "als_neg", md / "ease_like")
    mix.fit(train)
    mix.save(md / "mix_like")
    ly.run("val", clean_dir=tp, split_dir=sd, models_dir=md, eval_dir=tp / "eval")
    return tp, sd, md


def _part(tp, sd, md, pool=10):
    L = ly.Layers.load(md / "layers")
    work_ids = load_train(tp / "ratings.parquet", sd / "holdout_users.parquet").work_ids
    info = work_info(tp, work_ids)
    hold = load_eval_holdout(tp / "ratings.parquet", sd, "val", work_ids)
    books, genres = rr.book_table(tp, sd / "holdout_users.parquet", work_ids, info, L.taste.item_bias)
    return L, info, hold, rr.build(L, hold, info, books, genres, rr.value_tastes(md), batch=4, pool=pool)


def test_judge_values_follow_judge_scale():
    r = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 4.6])
    assert rr.judge_values(r, "value").tolist() == [-1.0, -0.5, 0.5, 1.0, 2.0, 2.0]
    assert rr.judge_values(r, "five").tolist() == [0, 0, 0, 0, 1, 1]


def test_key_trades_read_chance_for_value():
    p, v = np.array([0.5, 0.1, 0.4]), np.array([0.5, 2.0, 1.0])
    assert np.argmax(rr.Rerank.key(p, v, -np.inf)) == 0               # осторожно: только «прочтёт ли»
    assert np.argmax(rr.Rerank.key(p, v, 1.5)) == 1                   # смело: ценная, хоть и вряд ли прочтёт
    assert rr.Rerank.key(p, v, 1.5)[0] < 0                            # дешёвая книга опускает среднюю угаданных


def test_build_takes_crowd_top_without_input_and_labels_hidden(layered):
    tp, sd, md = layered
    L, info, hold, part = _part(tp, sd, md)
    n, top = len(hold.user_ids), L.mix.ease.top_cols
    assert part.pos.shape == (n, 10) and part.X.shape == (n * 10, len(rr.FEATURES))
    crowd = part.X[:, rr.FEATURES.index("crowd")].reshape(n, 10)
    assert (np.diff(crowd, axis=1) <= 1e-6).all()                     # по убыванию толпы
    assert (part.X[:, rr.FEATURES.index("crowd_rank")].reshape(n, 10) == np.arange(10)).all()
    ex = hold.exclude.tocsr()
    author = info.author_id.to_numpy()
    for u in range(n):
        cols = top[part.pos[u]]
        assert not np.isin(cols, ex.indices[ex.indptr[u]:ex.indptr[u + 1]]).any()   # ни входа, ни начатых серий
        got = dict(zip(hold.hidden_cols[u].tolist(), metrics.rounded(hold.hidden_ratings[u]).tolist()))
        assert part.rating[u * 10:(u + 1) * 10].tolist() == [got.get(c, 0.0) for c in cols.tolist()]
        x = hold.inputs[u]
        n_auth = [int((author[x.indices] == author[c]).sum()) for c in cols]
        assert part.X[u * 10:(u + 1) * 10, rr.FEATURES.index("author_n")].tolist() == n_auth
    assert np.isnan(part.X[:, rr.FEATURES.index("taste_value")]).all()   # вкусы на шкале судьи не обучены


def test_value_tastes_become_features(layered):
    tp, sd, md = layered
    rr.fit_value_tastes(ratings_path=tp / "ratings.parquet", split_dir=sd, models_dir=md, log=lambda *a: None)
    assert set(rr.value_tastes(md)) == {"taste_value", "taste_five"}
    part = _part(tp, sd, md)[3]
    for name in rr.VALUE_TASTES:
        assert np.isfinite(part.X[:, rr.FEATURES.index(name)]).all()


def test_check_judges_reranked_lists_against_formula(layered):
    tp, sd, md = layered
    res = rr.check(clean_dir=tp, split_dir=sd, models_dir=md, eval_dir=tp / "eval", folds=2, sizes=(3, 6),
                   c_grid=(-np.inf, 0.5), pool=10, log=lambda *a: None)
    assert [(r["set"], r["size"]) for r in res["runs"]] == [("full", 3), ("full", 6), ("scores", 6)]
    for r in res["runs"]:
        assert r["hits"] >= 0 and len(r["points"]) == 2 and r["quality_diff"]["n"] > 0
    assert (tp / "eval" / "rerank_check.json").exists()
    text = rr.report(res)
    assert "Переранжирование" in text and "формула" in text


def test_build_for_training_keeps_read_books_and_weighs_sampled_rest(layered):
    tp, sd, md = layered
    L, info, hold, full = _part(tp, sd, md)
    work_ids = load_train(tp / "ratings.parquet", sd / "holdout_users.parquet").work_ids
    books, genres = rr.book_table(tp, sd / "holdout_users.parquet", work_ids, info, L.taste.item_bias)
    part = rr.build(L, hold, info, books, genres, batch=4, pool=10, neg_rate=0.5)
    assert part.pos is None and len(part.X) == len(part.rating) == len(part.person) == len(part.weight)
    assert (part.rating > 0).sum() == (full.rating > 0).sum()             # прочитанные — все
    assert set(np.unique(part.weight)) <= {1.0, 2.0}
    assert (part.weight[part.rating > 0] == 1.0).all() and len(part.X) < len(full.X)
    m = rr.fit(part, np.arange(len(hold.user_ids)), list(range(len(rr.SETS["full"]))))
    p, v = m.predict(full.X)
    assert ((0 <= p) & (p <= 1)).all() and np.isfinite(v).all()


def test_big_check_learns_on_separate_people_and_judges_test_once(layered):
    tp, sd, md = layered
    rd, rm = tp / "split-rank", tp / "models-rank"
    meta = rr.make_split(ratings_path=tp / "ratings.parquet", users_path=tp / "users.parquet", split_dir=sd,
                         out_dir=rd, per_bucket=20)
    assert meta["rank_users"] == 20
    import pandas as pd
    held = pd.read_parquet(rd / "holdout_users.parquet")
    old = pd.read_parquet(sd / "holdout_users.parquet")
    assert set(old.user_id) < set(held.user_id) and (held.group == "rank").sum() == 20
    assert not set(held[held.group == "rank"].user_id) & set(old.user_id)            # новые люди — не из проверки и теста
    rr.refit(clean_dir=tp, split_dir=rd, models_dir=rm, source_models=md, log=lambda *a: None)
    train = load_train(tp / "ratings.parquet", rd / "holdout_users.parquet")
    assert not set(train.user_ids) & set(held.user_id)                                 # части формулы их не видели
    assert ly.Layers.load(rm / "layers").variant == ly.Layers.load(md / "layers").variant
    res = rr.final(clean_dir=tp, split_dir=rd, models_dir=rm, prod_models=md, eval_dir=tp / "eval-rank",
                   learn=(5,), c_grid=(-np.inf, 0.5), pool=10, log=lambda *a: None)
    assert [x["size"] for x in res["val"]["learn"]] == [5, 20]
    assert res["test"]["n_users"] > 0 and res["test"]["c"] in (-np.inf, 0.5)
    assert "Бустинг" in rr.report_final(res)
