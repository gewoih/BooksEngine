import json

import numpy as np
import pandas as pd
import pytest

from booksengine.model import chance, chance_score, evaluate, split
from booksengine.model.base import fingerprint
from tests.test_evaluate import _QUOTA, _write_works
from tests.test_split import synthetic_ratings, synthetic_users


def test_personal_pct_counts_books_above_among_candidates():
    s = np.array([0.9, -np.inf, 0.5, 0.7, 0.1])
    np.testing.assert_allclose(chance.personal_pct(s, np.array([0, 3, 4])), [0.25, 0.5, 1.0])
    assert np.isnan(chance.personal_pct(s, np.array([1])))[0]


def test_fit_recovers_coefficients_and_chooses_prior():
    rng = np.random.default_rng(0)
    n_users, per = 3000, 20
    n_rated = rng.integers(20, 300, n_users)
    true_share = rng.beta(4, 2, n_users)
    k_like = rng.binomial(n_rated, true_share)
    obs = pd.DataFrame({"user_id": np.repeat(np.arange(n_users), per),
                        "pct": 10 ** rng.uniform(-4, 0, n_users * per),
                        "k_like": np.repeat(k_like, per), "n_rated": np.repeat(n_rated, per)})
    truth = chance.Chance([0.2, -0.3, 1.0], prior=10.0, p0=0.67)
    obs["rating"] = np.where(rng.random(len(obs)) < truth.predict(obs.pct, obs.k_like, obs.n_rated), 5.0, 2.0)
    c, losses = chance.fit(obs, p0=0.67, prior_grid=(10.0,))
    np.testing.assert_allclose(c.coef, truth.coef, atol=0.1)
    assert set(losses) == {10.0}


def test_load_rejects_calibration_of_other_model_version(tmp_path):
    c = chance.Chance([0.0, 0.0, 1.0], 3.0, 0.7, model_fp="aaa")
    c.save(tmp_path / "chance.json")
    assert chance.Chance.load(tmp_path / "chance.json", "aaa").coef == c.coef
    with pytest.raises(ValueError, match="calibrate"):
        chance.Chance.load(tmp_path / "chance.json", "bbb")


def test_calibrate_end_to_end(tmp_path):
    rp = tmp_path / "ratings.parquet"
    r = synthetic_ratings(n_users=60)
    r.to_parquet(rp, index=False)
    _write_works(r, tmp_path)
    up = tmp_path / "users.parquet"
    synthetic_users(r.user_id.unique()).to_parquet(up, index=False)
    sd, ed, md = tmp_path / "split", tmp_path / "eval", tmp_path / "models"
    split.build(rp, up, sd, "fp", **_QUOTA, seed=11, share=0.2)
    evaluate.tune("popularity", ratings_path=rp, split_dir=sd, eval_dir=ed, models_dir=md,
                  grid=[({"formula": "count", "m": 0.0}, [{}])])
    out = chance.calibrate("popularity", ratings_path=rp, split_dir=sd, models_dir=md, eval_dir=ed)
    saved = chance.Chance.load(md / "popularity" / "chance.json", fingerprint(md / "popularity"))
    assert saved.coef == out["chance"]["coef"]
    assert "место + щедрость" in json.loads((ed / "chance_popularity.json").read_text())["test"]


def test_relative_score_is_share_of_top_candidate():
    s = np.array([-np.inf, 4.0, 2.0, -1.0])
    np.testing.assert_allclose(chance.relative_score(s, np.array([1, 2, 3])), [1.0, 0.5, -0.25])
    assert np.isnan(chance.relative_score(s, np.array([0])))[0]
    assert np.isnan(chance.relative_score(np.array([-1.0, -2.0]), np.array([0]))).all()   # топ-1 ≤ 0


def test_monotone_rejects_order_breaking_signs():
    names = ("1", "lp", "g", "rel")
    assert chance_score.monotone(names, [0.0, -0.3, 1.0, 0.5], (-1.0, 3.0))
    assert not chance_score.monotone(names, [0.0, -0.3, 1.0, -0.1], (-1.0, 3.0))   # выше балл — ниже шанс
    inter = ("1", "lp", "g", "lp*g")
    assert chance_score.monotone(inter, [0.0, -0.3, 1.0, 0.05], (-1.0, 3.0))
    assert not chance_score.monotone(inter, [0.0, -0.3, 1.0, 0.2], (-1.0, 3.0))    # у щедрых место «наоборот»


def _synthetic_obs(rng, n_users, per, rel_weight):
    n_rated = rng.integers(20, 300, n_users)
    k_like = rng.binomial(n_rated, rng.beta(4, 2, n_users))
    pct = 10 ** rng.uniform(-4, 0, n_users * per)
    rel = 1 - (np.log10(pct) + 4) / 4 + rng.normal(0, 0.3, len(pct))    # связан с местом, но не равен ему
    d = pd.DataFrame({"user_id": np.repeat(np.arange(n_users), per), "bucket": "50-199", "pct": pct, "rel": rel,
                      "n_cand": 20000, "k_like": np.repeat(k_like, per), "n_rated": np.repeat(n_rated, per)})
    own = (d.k_like + 3 * 0.67) / (d.n_rated + 3)
    logit = 0.2 - 0.2 * np.log10(d.pct) + np.log(own / (1 - own)) + rel_weight * d.rel
    d["rating"] = np.where(rng.random(len(d)) < 1 / (1 + np.exp(-logit)), 5.0, 2.0)
    return d


def test_compare_finds_signal_in_score_only_when_it_is_there():
    rng = np.random.default_rng(1)
    with_signal = chance_score.compare(_synthetic_obs(rng, 2000, 20, 1.5), _synthetic_obs(rng, 2000, 20, 1.5),
                                       n_boot=200)
    v = with_signal["variants"]["+ балл"]
    assert v["delta_ci95"][1] < 0 and v["monotone"] and v["coef"][3] > 1.0
    assert with_signal["variants"]["место + щедрость"]["delta_log_loss"] == 0.0

    none = chance_score.compare(_synthetic_obs(rng, 2000, 20, 0.0), _synthetic_obs(rng, 2000, 20, 0.0), n_boot=200)
    lo, hi = none["variants"]["+ балл"]["delta_ci95"]
    assert hi > -0.002 and abs(none["variants"]["+ балл"]["coef"][3]) < 0.2


def test_chance_score_end_to_end(tmp_path):
    rp = tmp_path / "ratings.parquet"
    r = synthetic_ratings(n_users=60)
    r.to_parquet(rp, index=False)
    _write_works(r, tmp_path)
    up = tmp_path / "users.parquet"
    synthetic_users(r.user_id.unique()).to_parquet(up, index=False)
    sd, ed, md = tmp_path / "split", tmp_path / "eval", tmp_path / "models"
    split.build(rp, up, sd, "fp", **_QUOTA, seed=11, share=0.2)
    evaluate.tune("popularity", ratings_path=rp, split_dir=sd, eval_dir=ed, models_dir=md,
                  grid=[({"formula": "count", "m": 0.0}, [{}])])
    out = chance_score.run(ratings_path=rp, split_dir=sd, models_dir=md, eval_dir=ed, clean_dir=tmp_path,
                           profiles=[], name="popularity")
    assert set(out["variants"]) == set(chance_score.VARIANTS)
    assert json.loads((ed / "chance_score_popularity.json").read_text())["n_test"] == out["n_test"]
    assert not (md / "popularity" / "chance.json").exists()     # рабочий шанс не трогает
    assert "+ балл" in chance_score.summary(out)
