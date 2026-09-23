import json

import numpy as np
import pandas as pd
import scipy.sparse as sp

from booksengine.model import evaluate, split
from booksengine.model.matrix import Holdout, RatingMatrix
from booksengine.model.popularity import Popularity
from tests.test_split import synthetic_ratings


def test_run_eval_counts_hidden_book_without_train_ratings_as_miss():
    # 25 книг; у книги 24 нет оценок в обучении — популярность ставит её последней, но она скрыта у пользователя
    X = np.zeros((2, 25), dtype=np.float32)
    X[0, :24] = 4.0
    X[1, :12] = 5.0
    train = RatingMatrix(sp.csr_matrix(X), np.array([1, 2]), np.arange(100, 125))
    model = Popularity(formula="count")
    model.fit(train)
    inp = np.zeros((1, 25), dtype=np.float32)
    inp[0, 0] = 5.0
    hold = Holdout(np.array([7]), np.array(["10-19"]), sp.csr_matrix(inp), [np.array([24])], [np.array([5.0])])
    per_user, cov = evaluate.run_eval(model, hold)
    assert per_user.loc[0, "ndcg20"] == 0.0
    assert per_user.loc[0, "bucket"] == "10-19" and per_user.loc[0, "user_id"] == 7
    assert cov == 0.8  # 20 из 25 книг, прочитанная книга 0 в топ не попала


def test_tune_and_test_end_to_end(tmp_path):
    rp = tmp_path / "ratings.parquet"
    synthetic_ratings(n_users=60).to_parquet(rp, index=False)
    sd = tmp_path / "split"
    split.build(rp, sd, "fp", n_val=5, n_test=10, seed=11, share=0.2)
    grid = [({"formula": "count", "m": 0.0}, [{}]), ({"formula": "bayes", "m": 10.0}, [{}])]
    ed, md = tmp_path / "eval", tmp_path / "models"
    # models_dir обязателен: без него tune пишет в настоящую models/ (было 2026-09-23 — затёрло models/popularity)
    val = evaluate.tune("popularity", ratings_path=rp, split_dir=sd, eval_dir=ed, grid=grid, models_dir=md)
    assert len(val) == 2 and json.loads((ed / "popularity_val.json").read_text()) == val
    res = evaluate.test("popularity", ratings_path=rp, split_dir=sd, eval_dir=ed, models_dir=md)
    assert res["n_users"] == 10 and res["variants"][0]["deployable"]
    assert (md / "popularity" / "scores.npy").exists()
    assert set(res["variants"][0]["summary"]) == {"all", *split.BUCKET_ORDER}


def test_test_reuses_model_saved_by_tune(tmp_path, monkeypatch):
    rp = tmp_path / "ratings.parquet"
    synthetic_ratings(n_users=60).to_parquet(rp, index=False)
    sd = tmp_path / "split"
    split.build(rp, sd, "fp", n_val=5, n_test=10, seed=11, share=0.2)
    ed, md = tmp_path / "eval", tmp_path / "models"
    grid = [({"formula": "count", "m": 0.0}, [{}])]
    evaluate.tune("popularity", ratings_path=rp, split_dir=sd, eval_dir=ed, grid=grid, models_dir=md)

    def no_fit(self, train):
        raise AssertionError("тест не должен переобучать модель, сохранённую при подборе")
    monkeypatch.setattr(Popularity, "fit", no_fit)
    res = evaluate.test("popularity", ratings_path=rp, split_dir=sd, eval_dir=ed, models_dir=md)
    assert res["n_users"] == 10
