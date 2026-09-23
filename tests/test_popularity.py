import numpy as np
import scipy.sparse as sp

from booksengine.model.matrix import RatingMatrix
from booksengine.model.popularity import Popularity


def three_fives_vs_many_fours() -> RatingMatrix:
    # книга 0: 3 пятёрки; книга 1: 1000 четвёрок; книга 2: 1000 двоек; книга 3: без оценок
    rows, cols, vals = [], [], []
    for u in range(3):
        rows.append(u); cols.append(0); vals.append(5.0)
    for u in range(1000):
        rows += [u, u]; cols += [1, 2]; vals += [4.0, 2.0]
    X = sp.csr_matrix((np.array(vals, dtype=np.float32), (rows, cols)), shape=(1000, 4))
    return RatingMatrix(X, np.arange(1000), np.array([10, 20, 30, 40]))


def test_bayes_smoothing_beats_three_fives():
    p = Popularity(formula="bayes", m=100)
    p.fit(three_fives_vs_many_fours())
    s = p.score(sp.csr_matrix((2, 4), dtype=np.float32))
    assert s.shape == (2, 4)
    assert s[0, 1] > s[0, 0] > s[0, 2]


def test_count_and_save_load(tmp_path):
    p = Popularity(formula="count")
    p.fit(three_fives_vs_many_fours())
    p.save(tmp_path / "pop")
    q = Popularity.load(tmp_path / "pop")
    x = sp.csr_matrix((1, 4), dtype=np.float32)
    assert np.array_equal(p.score(x), q.score(x))
    assert q.score(x)[0].tolist() == [3, 1000, 1000, 0]
