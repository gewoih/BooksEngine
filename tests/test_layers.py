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


def test_combine_excludes_input_and_orders_cutoff_without_ties():
    crowd = np.array([[5.0, 4.0, 3.0, 2.0, 1.0]])
    taste = np.array([[0.0, 0.0, 0.0, 0.0, 10.0]])
    excl = np.array([[True, False, False, False, False]])
    s = ly.Layers.combine(crowd, taste, excl, ly.Variant("mix", 1.0, None))
    assert s[0, 0] == -np.inf and np.argmax(s[0]) == 4           # вкус поднял последнюю книгу толпы
    s = ly.Layers.combine(crowd, taste, excl, ly.Variant("mix", 1.0, 2))
    # вне первых 2 толпы вкус книгу не поднимает; эти книги — сразу после первых, в порядке толпы, без ничьих
    assert s[0, 0] == -np.inf and np.isfinite(s[0, 1:]).all() and np.argsort(-s[0])[:4].tolist() == [1, 2, 3, 4]
    s = ly.Layers.combine(crowd, np.array([[0.0, 0.0, 10.0, 0.0, 0.0]]), excl, ly.Variant("mix", 1.0, 2))
    assert np.argsort(-s[0])[:4].tolist() == [2, 1, 3, 4]        # внутри первых 2 порядок решает вкус


def test_read_first_takes_top_third_and_choose_keeps_hits():
    scores = np.array([0.9, 0.1, 0.8, 0.3, 0.7, 0.2, -np.inf, 0.5, 0.4, 0.6])
    ratings = np.array([5, 3, 5, 4, 4, 2, 5, 3, 4, 3], dtype=float)
    np.testing.assert_array_equal(ly.read_first(scores, ratings), [5, 5, 4])   # 9 кандидатов → первые 3 по баллу
    assert len(ly.read_first(scores[:2], ratings[:2])) == 0                    # меньше трёх — не судим
    assert len(ly.read_first(np.arange(60.0), np.full(60, 4.0))) == ly.TOP_MAX  # треть, но не больше 10

    def row(label, quality, hits):
        return {"label": label, "groups": {"all": {"quality": {"mean": quality}, "hits": {"mean": hits}}}}
    now, sharp, niche = row("нынешняя", 0.40, 2.0), row("лучше", 0.50, 1.9), row("редкие книги", 0.70, 1.0)
    assert ly.choose([now, sharp, niche], now)["label"] == "лучше"             # «редкие» угадывают меньше 90%


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
    assert len(val["summary"]) == len(ly.grid(ly.Layers.from_models(md)))   # без params.json нынешняя — прежняя смесь
    cur = val["summary"][0]
    assert cur["label"] == ly.Variant(*ly.REFERENCE).label() and cur["groups"]["all"]["quality_diff"]["mean"] == 0.0
    g = cur["groups"]["all"]
    assert g["quality"]["n"] > 0 and g["fives_top"]["n"] > 0
    assert abs(g["five_minus_low"]["mean"] - (g["five_share"]["mean"] - g["low_share"]["mean"])) < 1e-9
    assert -1 <= g["quality"]["mean"] <= 2 and val["chosen_by_share"]
    assert all(r["groups"]["all"]["max_author"]["mean"] <= 2 for r in val["summary"])   # топ-20 — по правилам выдачи
    assert val["chosen"] == ly.choose(val["summary"], cur)["variant"]
    saved = json.loads((md / "layers" / "params.json").read_text())
    assert saved["variant"] == val["chosen"] and saved["baseline"] == val["current"]
    # нынешний вариант держится, пока другой не лучше уверенно: равный по качеству его не сменяет
    json.dump(saved | {"variant": val["summary"][1]["variant"]}, (md / "layers" / "params.json").open("w"))
    again = ly.run("val", clean_dir=tp, split_dir=sd, models_dir=md, eval_dir=ed)
    d = next(r for r in again["summary"] if r["variant"] == val["chosen"])["groups"]["all"]["quality_diff"]
    assert (again["chosen"] == val["summary"][1]["variant"]) == (d["lo"] <= 0 or val["chosen"] == val["summary"][1]["variant"])
    json.dump(saved, (md / "layers" / "params.json").open("w"))
    text = ly.report(val)
    assert "← выбран" in text and "Качество списка" in text and "Прежние судьи" in text
    test = ly.run("test", clean_dir=tp, split_dir=sd, models_dir=md, eval_dir=ed)
    assert 1 <= len(test["summary"]) <= 3 and test["current"] == val["current"] and (ed / "layers_test.json").exists()
    assert "Качество списка" in ly.report(test)
    prof = tp / "profiles"
    prof.mkdir()
    pd.DataFrame({"goodreads_work_id": [100, 101, 102, 125], "rating": [5, 5, 4, 1]}).to_csv(prof / "p.csv", index=False)
    text = ly.profiles(clean_dir=tp, models_dir=md, profiles_dir=prof, top=5)
    assert "## p (4 оценок)" in text and "Совпадает книг" in text


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


def test_val_includes_value_crowd_when_trained(world):
    from booksengine.model.ease import EASELike
    tp, sd, md = world
    train = load_train(tp / "ratings.parquet", sd / "holdout_users.parquet")
    like = EASELike(lam=10.0, topk=20)
    like.fit(train)
    like.save(md / "ease_like")
    mix = Mix(md / "als_neg", md / "ease_like")
    mix.fit(train)
    mix.save(md / "mix_like")
    L = ly.Layers.from_models(md)
    assert L.crowds() == ("mix", "like")
    x = train.X[:3]
    top = L.like_mix.ease.top_cols
    np.testing.assert_allclose(L.crowd("like", x, als_weight=0.5), L.like_mix.score(x)[:, top], rtol=1e-5, atol=1e-5)
    assert not np.allclose(L.crowd("like", x, als_weight=0.0), L.crowd("like", x, als_weight=0.5))
    val = ly.run("val", clean_dir=tp, split_dir=sd, models_dir=md, eval_dir=tp / "eval")
    assert len(val["summary"]) == 1 + len(ly.grid(L))            # прежняя смесь + перебор толпы «ценность»
    assert all(v.crowd == "like" for v in ly.grid(L)) and any(v.cutoff for v in ly.grid(L))
    g = val["summary"][0]["groups"]["all"]
    assert g["hits"]["mean"] > 0 and -1 <= g["quality"]["mean"] <= 2 and 0 <= g["five_base"]["mean"] <= 1
    assert "mix_like" in json.loads((md / "layers" / "params.json").read_text())["components"]
    assert ly.Layers.load(md / "layers").like_mix is not None


def test_tune_like_trains_grid_and_saves_best(world):
    from booksengine.model.base import read_params
    tp, sd, md = world
    grid = [(10.0, ly.W0), (5.0, ly.W1)]
    res = ly.tune_like(clean_dir=tp, split_dir=sd, models_dir=md, eval_dir=tp / "eval", grid=grid,
                       fit_kw={"topk": 20, "block": 7})
    assert [(r["lam"], tuple(r["weights"])) for r in res["results"]] == grid
    best = res["best"]
    saved = read_params(md / "ease_like")
    assert saved["lam"] == best["lam"] and saved["weights"] == best["weights"]
    L = ly.Layers.from_models(md)                       # mix_like подаёт в «ценность» те же веса звёзд
    assert list(L.like_mix.ease_input) == best["weights"]
    assert "Выбрано: λ" in ly.report_like(res)
    again = ly.tune_like(clean_dir=tp, split_dir=sd, models_dir=md, eval_dir=tp / "eval",
                         grid=[(best["lam"], tuple(best["weights"]))], fit_kw={"topk": 20})
    assert again["results"][0]["saved"]                 # сохранённая настройка не переобучается
    other = (7.0, ly.W3)
    res3 = ly.tune_like(clean_dir=tp, split_dir=sd, models_dir=md, eval_dir=tp / "eval", grid=[other],
                        fit_kw={"topk": 20})
    assert [(r["lam"], tuple(r["weights"])) for r in res3["results"]] == [(best["lam"], tuple(best["weights"])), other]


def test_profile_check_places_each_hidden_book(world):
    tp, sd, md = world
    ly.run("val", clean_dir=tp, split_dir=sd, models_dir=md, eval_dir=tp / "eval")
    prof = tp / "profiles"
    prof.mkdir()
    # вкус «первая половина книг»: 100–104 хорошие, 125, 126 — плохие
    pd.DataFrame({"goodreads_work_id": [100, 101, 102, 103, 104, 125, 126],
                  "rating": [5, 5, 4, 5, 4, 1, 2]}).to_csv(prof / "p.csv", index=False)
    text = ly.profile_check(clean_dir=tp, models_dir=md, profiles_dir=prof)
    assert "## p: 7 книг из 7" in text and "| 5★ | 3 |" in text and "Взвешенная точность (новая)" in text


def test_tune_like_force_saves_requested_setting(world):
    from booksengine.model.base import read_params
    tp, sd, md = world
    kw = dict(clean_dir=tp, split_dir=sd, models_dir=md, eval_dir=tp / "eval", fit_kw={"topk": 20})
    ly.tune_like(grid=[(10.0, ly.W0)], **kw)
    ly.tune_like(grid=[(7.0, (-2, -1, 0.5, 1, 2))], force=True, **kw)
    saved = read_params(md / "ease_like")
    assert saved["lam"] == 7.0 and saved["weights"] == [-2.0, -1.0, 0.5, 1.0, 2.0]


def test_why_explains_place_of_hidden_book(world):
    from booksengine.model.ease import EASELike
    tp, sd, md = world
    train = load_train(tp / "ratings.parquet", sd / "holdout_users.parquet")
    like = EASELike(lam=10.0, topk=20, weights=ly.W3)
    like.fit(train)
    like.save(md / "ease_like")
    mix = Mix(md / "als_neg", md / "ease_like")
    mix.fit(train)
    mix.configure(als_weight=0.5, ease_input=like.weights)
    mix.save(md / "mix_like")
    ly.run("val", clean_dir=tp, split_dir=sd, models_dir=md, eval_dir=tp / "eval")
    p = json.loads((md / "layers" / "params.json").read_text())
    p["variant"] = {"crowd": "like", "taste_weight": 0.0, "cutoff": None, "als_weight": 0.25}
    (md / "layers" / "params.json").write_text(json.dumps(p))
    prof = tp / "profiles"
    prof.mkdir()
    pd.DataFrame({"goodreads_work_id": [100, 101, 102, 125], "rating": [5, 5, 4, 1]}).to_csv(prof / "p.csv", index=False)
    text = ly.why(clean_dir=tp, models_dir=md, profile_csv=prof / "p.csv", query="Book 101")
    assert "оценена на 5★ и спрятана" in text and "| итог |" in text and "| 125 | 1 |" in text


def test_layers_contributions_sum_to_score(world):
    from booksengine.model import explain
    from booksengine.model.ease import EASELike
    tp, sd, md = world
    train = load_train(tp / "ratings.parquet", sd / "holdout_users.parquet")
    like = EASELike(lam=10.0, topk=20, weights=ly.W3)
    like.fit(train)
    like.save(md / "ease_like")
    mix = Mix(md / "als_neg", md / "ease_like")
    mix.fit(train)
    mix.configure(als_weight=0.5, ease_input=like.weights)
    mix.save(md / "mix_like")
    L = ly.Layers.from_models(md)
    x = train.X[5:6]
    cols = L.mix.ease.top_cols[:7]
    for v in (ly.Variant("like", 0.3, None, 0.25), ly.Variant("mix", 0.5), ly.Variant("like", 0.0, None, 0.0)):
        L.variant = v
        in_cols, c, const = explain.layers_contributions(L, x, cols)
        assert np.array_equal(in_cols, x.indices)
        np.testing.assert_allclose(c.sum(axis=0) + const, L.score(x)[0, cols], atol=1e-4)


def test_recommend_uses_layers_with_chance_and_writes_history(world):
    from booksengine import recommend as rec
    from booksengine.model import chance
    tp, sd, md = world
    ly.run("val", clean_dir=tp, split_dir=sd, models_dir=md, eval_dir=tp / "eval")
    prof = tp / "p.csv"
    pd.DataFrame({"goodreads_work_id": [100, 101, 102, 125], "rating": [5, 5, 4, 1],
                  "title": ["А", "Б", "В", "Г"]}).to_csv(prof, index=False)
    with pytest.raises(FileNotFoundError, match="calibrate layers"):
        rec.recommend(prof, clean_dir=tp, models_dir=md, top=5)
    chance.calibrate("layers", ratings_path=tp / "ratings.parquet", split_dir=sd, models_dir=md, eval_dir=tp / "eval")
    res = rec.recommend(prof, clean_dir=tp, models_dir=md, top=5, history_dir=tp / "history")
    assert len(res.recs) == 5 and all(0 <= r.chance <= 100 for r in res.recs)
    assert not any("Saga" in r.title for r in res.recs)             # поздние тома без первой книги — не в списке
    assert len({r.author for r in res.recs}) == 5                    # список из 5 — одна книга на автора
    assert all(set(r.because) <= {"А", "Б", "В", "Г"} for r in res.recs)
    hist = list((tp / "history").glob("p-*.csv"))
    assert len(hist) == 1 and len(pd.read_csv(hist[0])) == 5
    chance.calibrate("mix", ratings_path=tp / "ratings.parquet", split_dir=sd, models_dir=md, eval_dir=tp / "eval")
    mix_res = rec.recommend(prof, clean_dir=tp, models_dir=md, top=5, model="mix")  # приложение — смесь
    assert len(mix_res.recs) == 5


def test_tune_like_skips_setting_that_guesses_too_little(world, monkeypatch):
    from booksengine.model.base import read_params
    tp, sd, md = world
    kw = dict(clean_dir=tp, split_dir=sd, models_dir=md, eval_dir=tp / "eval", fit_kw={"topk": 20})
    ly.tune_like(grid=[(10.0, ly.W0)], **kw)
    monkeypatch.setattr(ly, "GUARD", 10.0)                # никто не угадывает в 10 раз больше нынешней выдачи
    res = ly.tune_like(grid=[(5.0, ly.W1)], **kw)
    assert [r["best_variant"] for r in res["results"]] == [None, None]
    assert read_params(md / "ease_like")["lam"] == 10.0     # сохранённая толпа осталась
    assert "Не подходят" in ly.report_like(res)


def test_tune_like_reuses_previous_run_and_trains_reused_best_to_save(world, monkeypatch):
    from booksengine.model.base import read_params
    from booksengine.model.ease import EASELike
    tp, sd, md = world
    kw = dict(clean_dir=tp, split_dir=sd, models_dir=md, eval_dir=tp / "eval", fit_kw={"topk": 20})
    ly.tune_like(grid=[(10.0, ly.W0)], **kw)
    monkeypatch.setattr(ly, "GUARD", 10.0)                # (5, W1) посчитана, но не выбрана — не записана
    ly.tune_like(grid=[(5.0, ly.W1)], **kw)
    monkeypatch.setattr(ly, "GUARD", 0.9)
    fits = []
    real_fit = EASELike.fit
    monkeypatch.setattr(EASELike, "fit", lambda self, train: fits.append(self.lam) or real_fit(self, train))
    res = ly.tune_like(grid=[(5.0, ly.W1)], force=True, **kw)
    assert res["results"][1]["reused"] and res["results"][1]["best_variant"] is not None   # решение пересчитано
    assert fits == [5.0]                                  # не в переборе — только чтобы записать выбранную
    assert read_params(md / "ease_like")["lam"] == 5.0
    assert "из прошлого прогона" in ly.report_like(res)


def test_quality_is_mean_star_value_of_guessed_books():
    got = np.array([5, 5, 4, 3, 1], dtype=float)                 # пример пользователя: лучше, чем 5/1/5/5/1
    risky = np.array([5, 1, 5, 5, 1], dtype=float)
    value = lambda r: ly.JUDGE_STARS[r.astype(int) - 1].mean()
    assert value(got) == pytest.approx(0.9) and value(risky) == pytest.approx(0.8)


def test_tune_like_replaces_saved_crowd_only_when_surely_better(world):
    from booksengine.model.base import read_params
    tp, sd, md = world
    kw = dict(clean_dir=tp, split_dir=sd, models_dir=md, eval_dir=tp / "eval", fit_kw={"topk": 20})
    ly.tune_like(grid=[(10.0, ly.W0)], **kw)
    res = ly.tune_like(grid=[(5.0, ly.W1)], **kw)
    saved, new = res["results"]
    assert saved["saved"] and saved["quality_diff"]["mean"] == 0.0
    d = new["quality_diff"]
    assert d["lo"] <= d["mean"] <= d["hi"]
    assert (read_params(md / "ease_like")["lam"] == 5.0) == (d["lo"] > 0)   # сменила — только если уверенно лучше
    assert "разница" in ly.report_like(res)
