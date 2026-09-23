import numpy as np
import pytest
import scipy.sparse as sp

from booksengine.model.ease import EASE
from booksengine.model.matrix import RatingMatrix


def data(seed=0) -> RatingMatrix:
    rng = np.random.default_rng(seed)
    X = sp.random(60, 12, density=0.3, format="csr", random_state=rng, dtype=np.float32)
    X.data = rng.integers(1, 6, size=X.nnz).astype(np.float32)
    return RatingMatrix(X, np.arange(60), np.arange(12))


def closed_form(X: np.ndarray, lam: float) -> np.ndarray:
    Xb = (X > 0).astype(np.float64)
    P = np.linalg.inv(Xb.T @ Xb + lam * np.eye(X.shape[1]))
    B = np.eye(X.shape[1]) - P / np.diag(P)[None, :]
    np.fill_diagonal(B, 0)
    return B


def test_full_b_matches_closed_form():
    d = data()
    m = EASE(lam=10.0, n_top=12, block=5)
    m.fit(d)
    np.testing.assert_allclose(m.B_full, closed_form(d.X.toarray(), 10.0), atol=1e-4)
    assert np.all(np.diag(m.B_full) == 0)


def test_items_outside_top_n_get_minus_inf_and_topk_sparsifies(tmp_path):
    d = data()
    m = EASE(lam=10.0, n_top=8, block=3)
    m.fit(d)
    outside = np.setdiff1d(np.arange(12), m.top_cols)
    s = m.score(d.X[:2])
    assert np.all(np.isneginf(s[:, outside]))
    m.configure(topk=3)
    assert (m._B != 0).sum(axis=0).max() <= 3
    m.save(tmp_path / "ease")
    np.testing.assert_allclose(EASE.load(tmp_path / "ease").score(d.X[:2]), m.score(d.X[:2]))


def test_save_requires_truncation(tmp_path):
    m = EASE(lam=10.0, n_top=12)
    m.fit(data())
    m.configure(topk=None)
    with pytest.raises(ValueError):
        m.save(tmp_path / "ease")
