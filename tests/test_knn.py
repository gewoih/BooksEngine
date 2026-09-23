import numpy as np
import scipy.sparse as sp

from booksengine.model.knn import ItemKNN
from booksengine.model.matrix import RatingMatrix

R = np.array([
    [5, 4, 0, 1, 0],
    [4, 5, 1, 0, 2],
    [1, 2, 5, 4, 0],
    [0, 1, 4, 5, 3],
    [5, 0, 2, 1, 4],
    [2, 1, 5, 0, 5],
], dtype=np.float32)


def brute_sim(X: np.ndarray, beta: float) -> np.ndarray:
    mask = X > 0
    n = mask.sum(1)
    mu = X.sum(1) / n
    A = np.where(mask, X - mu[:, None], 0.0)
    w = 1 / np.log1p(n)
    Z = A * np.sqrt(w)[:, None]
    G = Z.T @ Z
    norms = np.sqrt((Z ** 2).sum(0))
    S = G / np.outer(norms, norms)
    if beta:
        C = mask.T.astype(float) @ mask.astype(float)
        S = S * C / (C + beta)
    np.fill_diagonal(S, 0)
    return S


def model(beta=0.0, block=2) -> ItemKNN:
    m = ItemKNN(beta=beta, k_max=4, block=block)
    m.fit(RatingMatrix(sp.csr_matrix(R), np.arange(6), np.arange(5)))
    return m


def test_similarities_match_brute_force_and_blocks_agree():
    for beta in (0.0, 2.0):
        S = brute_sim(R, beta)
        for block in (1, 2, 5):
            m = model(beta, block)
            for i in range(5):
                got = dict(zip(m.nbr_idx[i].tolist(), m.nbr_sim[i].tolist()))
                for j, s in got.items():
                    assert np.isclose(s, S[i, j], atol=1e-5)
                assert i not in got


def test_score_formula_and_save_load(tmp_path):
    m = model()
    m.configure(k=4, normalize=False)
    x = sp.csr_matrix(np.array([[5, 0, 0, 1, 0]], dtype=np.float32))
    S = brute_sim(R, 0.0)
    dev = np.array([2.0, 0, 0, -2.0, 0])  # средняя входа 3
    expect = S @ dev
    got = m.score(x)[0]
    np.testing.assert_allclose(got, expect + 1e-6 * m.pop_, atol=1e-5)
    m.save(tmp_path / "knn")
    np.testing.assert_allclose(ItemKNN.load(tmp_path / "knn").score(x), m.score(x))


def test_uniform_input_falls_back_to_popularity():
    m = model()
    m.configure(k=4, normalize=True)
    x = sp.csr_matrix(np.array([[5, 5, 0, 0, 0]], dtype=np.float32))  # все оценки одинаковые
    s = m.score(x)[0]
    order = np.argsort(-s[2:], kind="stable") + 2
    assert order.tolist() == (np.argsort(-m.pop_[2:], kind="stable") + 2).tolist()
