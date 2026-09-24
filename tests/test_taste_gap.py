import json

import numpy as np
import pytest
import scipy.sparse as sp

from booksengine.model import evaluate, split, taste_gap
from booksengine.model.matrix import Holdout
from tests.test_evaluate import _QUOTA, _write_works
from tests.test_split import synthetic_ratings, synthetic_users


def test_personal_auc_counts_pairs_and_ties():
    r = np.array([5.0, 4.0, 3.0, 1.0])
    assert taste_gap.personal_auc(np.array([4.0, 3.0, 2.0, 1.0]), r) == 1.0
    assert taste_gap.personal_auc(np.array([1.0, 2.0, 3.0, 4.0]), r) == 0.0
    assert taste_gap.personal_auc(np.zeros(4), r) == 0.5                       # ничья — половина
    assert taste_gap.personal_auc(np.array([3.0, 1.0, 2.0, 0.0]), r) == 0.75  # 3 пары из 4
    assert np.isnan(taste_gap.personal_auc(np.array([1.0, 2.0]), np.array([4.0, 5.0])))
    assert taste_gap.personal_auc(np.array([2.0, 1.0]), np.array([3.5, 3.4])) == 1.0  # 3.5 → 4: понравилась


def test_book_scores_mean_and_count():
    X = sp.csr_matrix(np.array([[5, 0, 2], [3, 0, 0]], dtype=np.float32))
    s = taste_gap.book_scores(X)
    np.testing.assert_allclose(s["book_mean"].values, [4.0, 10 / 3, 2.0], rtol=1e-6)  # без оценок — средняя по всем
    np.testing.assert_array_equal(s["book_count"].values, [2, 0, 1])


def test_measure_uses_only_allowed_books_and_counts_liked_hits():
    inp = sp.csr_matrix(np.array([[5, 0, 0, 0, 0]], dtype=np.float32))
    hold = Holdout(np.array([1]), np.array(["20-49"]), inp, [np.array([1, 2, 3])], [np.array([5.0, 2.0, 4.0])])
    model = taste_gap.BookScore(np.array([9, 1, 3, 2, 0]))   # 2 (2★) выше 1 (5★) — пара проиграна
    allowed = np.array([True, True, True, False, True])       # книга 3 вне сравнения
    d = taste_gap.measure(model, hold, allowed)
    assert d.loc[0, "auc"] == 0.0 and d.loc[0, "hidden"] == 2
    assert d.loc[0, "hits"] == 2 and d.loc[0, "liked_hits"] == 1   # в топ попали обе разрешённые


def test_run_end_to_end(tmp_path):
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
    res = taste_gap.run(ratings_path=rp, split_dir=sd, models_dir=md, eval_dir=ed, names=("mix", "popularity"))
    assert res["reference"] == "popularity"                     # смеси нет — точка сравнения первая доступная
    assert set(res["summary"]) == {"popularity", "book_mean", "book_count"}
    pop, cnt = res["summary"]["popularity"]["all"], res["summary"]["book_count"]["all"]
    assert pop["auc"]["mean"] == cnt["auc"]["mean"]              # популярность «count» = число оценок книги
    assert pop["diff"]["mean"] == 0.0
    assert json.loads((ed / "taste_gap.json").read_text())["n_users"] == res["n_users"]
    assert "Личная точность" in taste_gap.report(res)


def test_graded_auc_weights_five_over_four():
    r = np.array([5.0, 4.0, 2.0])
    assert taste_gap.graded_auc(np.array([3.0, 2.0, 1.0]), r) == 1.0
    # 4 выше 5, остальное верно: проиграна пара «5 против 4» (вес 1 из 1 + 2 + 1)
    assert taste_gap.graded_auc(np.array([2.0, 3.0, 1.0]), r) == 0.75
    assert taste_gap.graded_auc(np.array([2.0, 3.0, 1.0]), r, five=3.0) == pytest.approx(1 - 2 / (2 + 3 + 1))
    assert taste_gap.graded_auc(np.array([2.0, 3.0, 1.0]), r, five=1.0) == 1.0   # «5 = 4»: пара не считается
    assert np.isnan(taste_gap.graded_auc(np.array([1.0, 2.0]), np.array([5.0, 5.0])))
