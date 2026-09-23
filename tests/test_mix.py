import numpy as np
import pytest
import scipy.sparse as sp

from booksengine.model.als import ALS
from booksengine.model.ease import EASE
from booksengine.model.mix import Mix, _z
from booksengine.model.matrix import RatingMatrix
from tests.test_als import random_matrix

N_TOP = 30  # из 40 книг: 10 вне EASE


@pytest.fixture
def components(tmp_path):
    train = random_matrix()
    als = ALS(factors=8, regularization=0.1, alpha=1.0, iterations=5)
    als.fit(train)
    als.configure(neg_rule="le2")
    als.save(tmp_path / "als")
    ease = EASE(lam=5.0, n_top=N_TOP, block=7)
    ease.fit(train)
    ease.configure(topk=10)
    ease.save(tmp_path / "ease")
    return train, tmp_path / "als", tmp_path / "ease"


def _mix(components, **score) -> Mix:
    train, a, e = components
    m = Mix(a, e)
    m.fit(train)
    m.configure(**score)
    return m


def test_books_outside_ease_are_never_candidates(components):
    m = _mix(components)
    s = m.score(components[0].X[:5])
    outside = np.setdiff1d(np.arange(40), m.ease.top_cols)
    assert len(outside) == 40 - N_TOP
    assert np.isneginf(s[:, outside]).all() and np.isfinite(s[:, m.ease.top_cols]).all()


def test_weight_one_is_normalized_als_and_zero_is_weighted_ease(components):
    x = components[0].X[:5]
    top = _mix(components).ease.top_cols
    s1 = _mix(components, als_weight=1.0).score(x)[:, top]
    np.testing.assert_allclose(s1, _z(_mix(components).als.score(x)[:, top].astype(np.float64)), atol=1e-5)
    m0 = _mix(components, als_weight=0.0)
    w = np.array([-2, -1, 0, 1, 2], np.float32)
    xin = x[:, top].toarray()
    manual = np.where(xin > 0, w[np.clip(np.floor(xin + 0.5).astype(int), 1, 5) - 1], 0.0) @ m0.ease._B.toarray()
    np.testing.assert_allclose(m0.score(x)[:, top], _z(manual), atol=1e-5)


def test_one_star_pushes_ease_neighbours_down(components):
    m = _mix(components, als_weight=0.0)
    top = m.ease.top_cols
    B = m.ease._B.toarray()
    src = int(np.argmax(np.abs(B).sum(axis=1)))  # книга с сильными соседями
    nbr = int(np.argmax(B[src]))
    loved, hated = (sp.csr_matrix(([r], ([0], [top[src]])), shape=(1, 40), dtype=np.float32) for r in (5.0, 1.0))
    assert m.score(loved)[0, top[nbr]] > 0 > m.score(hated)[0, top[nbr]]


def test_save_load_roundtrip_and_stale_component_is_rejected(components, tmp_path):
    train, a, e = components
    m = _mix(components, als_weight=0.6)
    m.save(tmp_path / "mix")
    m2 = Mix.load(tmp_path / "mix")
    assert m2.als_weight == 0.6 and m2.ease_input == (-2.0, -1.0, 0.0, 1.0, 2.0)
    np.testing.assert_array_equal(m2.score(train.X[:4]), m.score(train.X[:4]))
    als = ALS.load(a)
    als.configure(neg_rule="le2", neg_weight=3.0)  # «переобучили» компонент
    als.save(a)
    with pytest.raises(ValueError, match="пересоберите"):
        Mix.load(tmp_path / "mix")


def test_fit_rejects_components_trained_on_other_catalog(components):
    _, a, e = components
    other = RatingMatrix(sp.csr_matrix((3, 7), dtype=np.float32), np.arange(3), np.arange(7))
    with pytest.raises(ValueError, match="переобучите"):
        Mix(a, e).fit(other)
