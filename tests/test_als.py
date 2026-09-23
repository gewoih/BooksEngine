import numpy as np
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


def test_score_shape_and_save_load(tmp_path):
    train = random_matrix()
    m = ALS(factors=8, iterations=3)
    m.fit(train)
    x = train.X[:3]
    s = m.score(x)
    assert s.shape == (3, 40) and s.dtype == np.float32
    m.save(tmp_path / "als")
    np.testing.assert_allclose(ALS.load(tmp_path / "als").score(x), s)
