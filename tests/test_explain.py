import numpy as np
import pytest
import scipy.sparse as sp

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


def test_contributions_with_dnf_sum_to_mix_score(components):  # noqa: F811
    m = _mix(components)
    x = components[0].X[3:4].copy()
    cols = m.ease.top_cols
    k = int(np.flatnonzero(np.isin(x.indices, cols))[0])  # недочитана книга из EASE
    x.data[k] = 1.0
    dnf = sp.csr_matrix(([1.0], ([0], [x.indices[k]])), shape=x.shape, dtype=np.float32)
    assert not np.allclose(m.score(x, dnf), m.score(x))
    _, c = explain.contributions(m, x, cols, dnf)
    np.testing.assert_allclose(c.sum(axis=0), m.score(x, dnf)[0, cols], atol=1e-4)


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


def test_taste_note_names_book_or_typical_rating():
    assert explain.taste_note(np.array([0.1, -0.2]), 0.2) is None                      # сдвиг 0.1 — молчит
    assert explain.taste_note(np.array([0.2, 0.9]), 0.1) == explain.TasteNote(1, 1)
    assert explain.taste_note(np.array([-0.7, 0.1]), 0.0) == explain.TasteNote(-1, 0)
    assert explain.taste_note(np.array([0.2, 0.1]), 0.8) == explain.TasteNote(1, None)   # «обычно ставят»
    assert explain.taste_note(np.array([]), -0.9) == explain.TasteNote(-1, None)


def test_why_text_shows_each_rating():
    from booksengine.recommend import Rec, why_text
    r = Rec(1, "T", "A", 50, ["Тёмная материя", "Марсианин"], "Сумерки",
            {"Тёмная материя": "4★", "Марсианин": "3★", "Сумерки": "не дочитал"}, "против: Марсианин 3★ (у всех 4.1) — похожа")
    assert why_text(r) == ["читатели: Тёмная материя 4★, Марсианин 3★; несмотря на: Сумерки не дочитал",
                           "вкус против: Марсианин 3★ (у всех 4.1) — похожа"]
    assert why_text(Rec(1, "T", "A", 50, [], None)) == ["по профилю в целом"]
