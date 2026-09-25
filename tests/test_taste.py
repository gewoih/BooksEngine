import json

import numpy as np
import scipy.sparse as sp

from booksengine.model import split, taste_gap
from booksengine.model.matrix import RatingMatrix
from booksengine.model.taste import Taste, solve_rows, star_stats, tune
from booksengine.model.taste import report as taste_report
from tests.test_evaluate import _QUOTA, _write_works
from tests.test_split import synthetic_ratings, synthetic_users


def taste_matrix(n_users=300, n_items=60, density=0.3, seed=0) -> tuple[RatingMatrix, np.ndarray]:
    """Два вкуса: у людей вкуса A книги первой половины на 5, второй — на 2, у B наоборот; сдвиг человека ±0.5."""
    rng = np.random.default_rng(seed)
    kind = rng.integers(0, 2, n_users)
    first = np.arange(n_items) < n_items // 2
    true = np.where(kind[:, None] == first[None, :].astype(int), 2.0, 5.0)
    true = np.clip(true + rng.choice([-0.5, 0.5], n_users)[:, None], 1, 5)
    mask = rng.random((n_users, n_items)) < density
    X = sp.csr_matrix(np.where(mask, np.round(true), 0).astype(np.float32))
    return RatingMatrix(X, np.arange(n_users), np.arange(100, 100 + n_items)), kind


def test_solve_rows_matches_direct_ridge():
    rng = np.random.default_rng(0)
    F = rng.normal(size=(10, 3)).astype(np.float32)
    off = rng.normal(size=10).astype(np.float32)
    M = sp.csr_matrix(np.array([[5, 0, 3, 0, 1, 0, 0, 4, 0, 0]], dtype=np.float32))
    th, _ = solve_rows(M, F, off, 3.0, 0.1)
    c = np.array([0, 2, 4, 7])
    Z = F[c].astype(np.float64)
    d = M.data - 3.0 - off[c]
    np.testing.assert_allclose(th[0], np.linalg.solve(Z.T @ Z + 0.1 * 4 * np.eye(3), Z.T @ d), rtol=1e-6)


def test_fit_learns_taste_and_orders_new_person():
    train, _ = taste_matrix()
    m = Taste(factors=2, reg=0.05, iterations=8)
    m.fit(train, log=lambda *_: None)
    # новый человек вкуса A: первая половина на 5, пара книг второй — на 2
    x = np.zeros((1, 60), dtype=np.float32)
    x[0, [0, 1, 2, 3]] = 5.0
    x[0, [40, 41]] = 2.0
    s = m.score(sp.csr_matrix(x))[0]
    assert s[4:30].min() > s[42:60].max()          # неоценённые книги «своей» половины выше «чужой»
    assert 3.5 < s[4:30].mean() and s[42:60].mean() < 3.5


def test_prediction_decomposes_into_rated_books_exactly():
    train, _ = taste_matrix()
    m = Taste(factors=3, reg=0.1, iterations=3)
    m.fit(train, log=lambda *_: None)
    cols, r = np.array([1, 5, 33, 40]), np.array([5.0, 4.0, 2.0, 1.0])
    x = sp.csr_matrix((r.astype(np.float32), (np.zeros(4, int), cols)), shape=(1, 60))
    A, Z, d = m.fold_in_system(cols, r)
    j = np.array([7, 50])
    Zj = np.column_stack([np.ones(2), m.item_factors[j]]).astype(np.float64)
    contrib = (Zj @ np.linalg.solve(A, Z.T)) * d[None, :]
    np.testing.assert_allclose(m.score(x)[0, j], m.mu + m.item_bias[j] + contrib.sum(axis=1), rtol=1e-5)


def test_save_load_same_scores(tmp_path):
    train, _ = taste_matrix()
    m = Taste(factors=2, reg=0.05, iterations=2)
    m.fit(train, log=lambda *_: None)
    (tmp_path / "t").mkdir()
    m.save(tmp_path / "t")
    np.testing.assert_array_equal(Taste.load(tmp_path / "t").score(train.X[:5]), m.score(train.X[:5]))


def test_star_stats():
    X = sp.csr_matrix(np.array([[5, 4, 1], [3, 4, 0], [2, 5, 5]], dtype=np.float32))
    s = star_stats(X)
    assert s["star_share"]["5"] == 3 / 8 and s["star_share"]["1"] == 1 / 8
    assert abs(s["users_without_1"] - 2 / 3) < 1e-9 and abs(s["users_without_1_2"] - 1 / 3) < 1e-9


def test_tune_end_to_end(tmp_path):
    rp = tmp_path / "ratings.parquet"
    r = synthetic_ratings(n_users=60)
    r.to_parquet(rp, index=False)
    _write_works(r, tmp_path)
    up = tmp_path / "users.parquet"
    synthetic_users(r.user_id.unique()).to_parquet(up, index=False)
    sd, ed, md = tmp_path / "split", tmp_path / "eval", tmp_path / "models"
    split.build(rp, up, sd, "fp", **_QUOTA, seed=11, share=0.2)
    out = tune(ratings_path=rp, split_dir=sd, models_dir=md, eval_dir=ed, grid=[(2, 0.1), (3, 0.1)], iterations=2)
    assert len(out["results"]) == 2 and (md / "taste" / "items.npz").exists()
    assert json.loads((ed / "taste_val.json").read_text())["stars"]["star_share"]
    assert "| 2 | 0.1 |" in taste_report(out)
    res = taste_gap.run(ratings_path=rp, split_dir=sd, models_dir=md, eval_dir=ed, names=("taste",))
    assert res["reference"] == "taste" and "diff_book_mean" in res["summary"]["taste"]["all"]


def test_tune_extends_previous_run_and_keeps_better_model(tmp_path):
    rp = tmp_path / "ratings.parquet"
    r = synthetic_ratings(n_users=60)
    r.to_parquet(rp, index=False)
    _write_works(r, tmp_path)
    up = tmp_path / "users.parquet"
    synthetic_users(r.user_id.unique()).to_parquet(up, index=False)
    sd, ed, md = tmp_path / "split", tmp_path / "eval", tmp_path / "models"
    split.build(rp, up, sd, "fp", **_QUOTA, seed=11, share=0.2)
    tune(ratings_path=rp, split_dir=sd, models_dir=md, eval_dir=ed, grid=[(2, 0.1)], iterations=2)
    saved = json.loads((md / "taste" / "params.json").read_text())
    # притворяемся, что прежний вариант был недосягаемо хорош: новый не должен затереть модель
    val = json.loads((ed / "taste_val.json").read_text())
    val["results"][0]["personal_auc"]["all"] = 2.0
    (ed / "taste_val.json").write_text(json.dumps(val))
    out = tune(ratings_path=rp, split_dir=sd, models_dir=md, eval_dir=ed, grid=[(2, 0.1), (3, 0.1)], iterations=2)
    assert [(x["factors"], x["reg"]) for x in out["results"]] == [(2, 0.1), (3, 0.1)]   # 2/0.1 не пересчитан
    assert json.loads((md / "taste" / "params.json").read_text()) == saved


def test_report_puts_groups_under_their_headers_whatever_the_stored_order():
    out = {"stars": {"star_share": {str(i): 0.2 for i in range(1, 6)}, "users_without_1": 0.5,
                     "users_without_1_2": 0.2, "user_like_share_quartiles": [0.6, 0.7, 0.8]},
           "results": [{"factors": 32, "reg": 0.05, "fit_seconds": 1, "rmse_hidden": 0.9,
                        "personal_auc": {"all": 0.7, "20-39": 0.1, "80-159": 0.3, "40-79": 0.2}},
                       {"factors": 64, "reg": 0.03, "fit_seconds": 1, "rmse_hidden": 0.9,
                        "personal_auc": {"all": 0.7, "20-39": 0.4, "40-79": 0.5, "80-159": 0.6}}]}
    text = taste_report(out)
    assert "| 20-39 | 40-79 | 80-159 |" in text
    assert "| 0.1000 | 0.2000 | 0.3000 |" in text and "| 0.4000 | 0.5000 | 0.6000 |" in text
