import json

import numpy as np
import pandas as pd
import pytest

from booksengine.model import chance, evaluate, split
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
    text = chance.report(out)
    assert "обещано" in text and "← шанс" in text and "Как читать" in text
