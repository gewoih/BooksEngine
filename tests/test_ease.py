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


def test_fit_in_place_is_bitwise_equal_to_copying_version():
    # прежний fit: inv на построчной G (scipy копирует её) и деление с новой матрицей — эталон без копий нынешнего
    import scipy.linalg
    d = data(seed=3)
    m = EASE(lam=7.0, n_top=10, block=4)
    m.fit(d)
    Xb = d.X[:, m.top_cols].tocsc()
    Xb.data[:] = 1.0
    G = (Xb.T @ Xb).toarray().astype(np.float32)
    G[np.diag_indices(10)] += 7.0
    P = scipy.linalg.inv(G, overwrite_a=True, check_finite=False)
    B = P / (-np.diag(P))[None, :]
    np.fill_diagonal(B, 0.0)
    assert m.B_full.dtype == np.float32
    assert np.array_equal(m.B_full, B)


def test_ease_like_matches_direct_ridge_per_column(tmp_path):
    from booksengine.model.ease import EASELike
    rng = np.random.default_rng(5)
    U, N, lam = 120, 12, 3.0
    R = np.where(rng.random((U, N)) < 0.4, rng.integers(1, 6, (U, N)), 0).astype(np.float32)
    train = RatingMatrix(sp.csr_matrix(R), np.arange(U), np.arange(100, 100 + N))
    m = EASELike(lam=lam, n_top=N, block=5, topk=N)
    m.fit(train)
    W = np.array([0, -2, -1, 0, 1, 2], float)
    Xw, Y = W[R.astype(int)], np.where(R > 0, R - 3, 0.0)
    B = np.zeros((N, N))
    for j in range(N):
        keep = np.arange(N) != j
        Xj = Xw[:, keep]
        B[keep, j] = np.linalg.solve(Xj.T @ Xj + lam * np.eye(N - 1), Xj.T @ Y[:, j])
    np.testing.assert_allclose(m._B.toarray(), B, atol=1e-4)
    m.save(tmp_path)
    loaded = EASE.load(tmp_path)                      # смесь читает его как обычный EASE
    np.testing.assert_allclose(loaded._B.toarray(), m._B.toarray())
    small = EASELike(lam=lam, n_top=N, block=5, topk=3)
    small.fit(train)
    assert (small._B.getnnz(axis=0) <= 3).all()


def test_star_weights_are_normalized_to_scale_two():
    from booksengine.model.ease import EASELike, normalize_weights
    assert normalize_weights((1, 2, 4, 8, 16)) == (0.125, 0.25, 0.5, 1.0, 2.0)
    assert normalize_weights((-2, -1, 0, 1, 2)) == (-2.0, -1.0, 0.0, 1.0, 2.0)
    assert EASELike(weights=(2, 4, 8, 16, 32)).weights == (0.125, 0.25, 0.5, 1.0, 2.0)
    with pytest.raises(ValueError):
        normalize_weights((0, 0, 0, 0, 0))
