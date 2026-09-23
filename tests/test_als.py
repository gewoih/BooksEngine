import numpy as np
import pytest
import scipy.sparse as sp

from booksengine.model.als import ALS
from booksengine.model.matrix import RatingMatrix


def random_matrix(n_users=80, n_items=40, density=0.2, seed=0) -> RatingMatrix:
    rng = np.random.default_rng(seed)
    X = sp.random(n_users, n_items, density=density, format="csr", random_state=rng, dtype=np.float32)
    X.data = rng.integers(1, 6, size=X.nnz).astype(np.float32)
    return RatingMatrix(X, np.arange(n_users), np.arange(100, 100 + n_items))


def test_fold_in_matches_implicit_recalculate_user():
    train = random_matrix()
    m = ALS(factors=8, regularization=0.1, alpha=2.0, iterations=5)
    m.fit(train)
    rows = train.X[:5]
    ours = m.fold_in(rows)
    conf = rows.copy()
    conf.data = 1.0 + conf.data * m.alpha  # implicit ждёт саму уверенность c = 1 + α·r
    theirs = m._model.recalculate_user(np.arange(5), conf)
    np.testing.assert_allclose(ours, theirs, rtol=1e-3, atol=1e-4)


def _identity_model(n=2, **score_params) -> ALS:
    m = ALS(factors=n, regularization=0.0, alpha=1.0)
    m._set_items(np.eye(n, dtype=np.float32))
    m.configure(**score_params)
    return m


def _row(*r) -> sp.csr_matrix:
    return sp.csr_matrix(np.array([r], dtype=np.float32))


def test_fold_in_by_default_treats_every_rated_item_as_liked():
    np.testing.assert_allclose(_identity_model().fold_in(_row(5, 1)), [[1.0, 1.0]], atol=1e-6)


def test_fold_in_low_rating_pulls_away():
    out = _identity_model(neg_rule="le2", neg_weight=3.0).fold_in(_row(5, 1))
    np.testing.assert_allclose(out, [[1.0, -1.0]], atol=1e-6)  # (1 + β)·(−1) / (1 + β)


def test_fold_in_middle_rating_stays_liked():
    out = _identity_model(n=3, neg_rule="le2", neg_weight=3.0).fold_in(_row(5, 3, 4))
    np.testing.assert_allclose(out, [[1.0, 1.0, 1.0]], atol=1e-6)


def test_fold_in_relative_rule_depends_on_own_mean():
    r = _row(5, 5, 2.5)  # средняя 4.17: 2.5 на 1.67 ниже — отрицательная по mu-1.5, но не по le2
    np.testing.assert_allclose(_identity_model(n=3, neg_rule="mu-1.5", neg_weight=1.0).fold_in(r)[0, 2], -1.0,
                               atol=1e-6)
    np.testing.assert_allclose(_identity_model(n=3, neg_rule="le2", neg_weight=1.0).fold_in(r)[0, 2], 1.0,
                               atol=1e-6)


def test_configure_rejects_unknown_rule():
    with pytest.raises(ValueError):
        ALS().configure(neg_rule="sometimes")


def test_neg_settings_survive_save_load(tmp_path):
    m = ALS(factors=8, iterations=3)
    m.fit(random_matrix())
    m.configure(neg_rule="le2", neg_weight=10.0)
    m.save(tmp_path / "als")
    loaded = ALS.load(tmp_path / "als")
    assert (loaded.neg_rule, loaded.neg_weight) == ("le2", 10.0)


def test_score_shape_and_save_load(tmp_path):
    train = random_matrix()
    m = ALS(factors=8, iterations=3)
    m.fit(train)
    x = train.X[:3]
    s = m.score(x)
    assert s.shape == (3, 40) and s.dtype == np.float32
    m.save(tmp_path / "als")
    np.testing.assert_allclose(ALS.load(tmp_path / "als").score(x), s)
