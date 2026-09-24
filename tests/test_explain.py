import numpy as np
import pytest

from booksengine.model import explain
from tests.test_mix import _mix, components  # noqa: F401 — фикстура


@pytest.mark.parametrize("w", [0.0, 0.5, 1.0])
def test_contributions_sum_to_mix_score(components, w):  # noqa: F811
    m = _mix(components, als_weight=w)
    x = components[0].X[3:4]
    cols = m.ease.top_cols
    in_cols, c = explain.contributions(m, x, cols)
    assert np.array_equal(in_cols, x.indices) and c.shape == (x.nnz, len(cols))
    np.testing.assert_allclose(c.sum(axis=0), m.score(x)[0, cols], atol=1e-4)


def test_books_outside_ease_cannot_be_explained(components):  # noqa: F811
    m = _mix(components)
    outside = np.setdiff1d(np.arange(40), m.ease.top_cols)[:1]
    with pytest.raises(ValueError):
        explain.contributions(m, components[0].X[:1], outside)


def test_reason_names_leaders_and_despite():
    r = explain.reason(np.array([4.0, 0.5, 2.0, -3.0, 1.5]))
    assert r == explain.Reason([0, 2, 4], 3)
    r = explain.reason(np.array([4.0, 0.9, -1.0]))       # 0.9 < 0.25·4 — не называется; −1 < 0.5·4
    assert r == explain.Reason([0], None)
    assert explain.reason(np.array([-1.0, -2.0])) == explain.Reason([], None)
