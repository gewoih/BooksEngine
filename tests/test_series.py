import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from booksengine.model import evaluate
from booksengine.model.matrix import Holdout
from booksengine.model.series import SeriesIndex, exclusion, series_keys, without_started_series


@pytest.mark.parametrize("title, keys", [
    ("Harry Potter and the Chamber of Secrets (Harry Potter, #2)", ["harry potter"]),
    ("Harry Potter Boxset (Harry Potter, #1-7)", ["harry potter"]),
    ("Harry Potter: The Prequel (Harry Potter, #0.5)", ["harry potter"]),
    ("Mort (Discworld, #4; Death, #1)", ["discworld", "death"]),
    ("Catch-22 (Catch-22, #1)", ["catch-22"]),
    ("1984", []),
    ("Fruits Basket, Vol. 6", []),
    ("Brave New World / Brave New World Revisited", []),
])
def test_series_keys_from_title(title, keys):
    assert series_keys(title) == keys


# столбцы: 0 HP#1, 1 HP#2, 2 бокс-сет HP, 3 «1984», 4 Mort (Discworld + Death), 5 Discworld #1
TITLES = ["HP 1 (Harry Potter, #1)", "HP 2 (Harry Potter, #2)", "HP Boxset (Harry Potter, #1-7)", "1984",
          "Mort (Discworld, #4; Death, #1)", "The Colour of Magic (Discworld, #1)"]


def test_continuations_cover_whole_series_and_all_series_of_a_book():
    idx = SeriesIndex(TITLES)
    assert idx.continuations(np.array([2])).tolist() == [0, 1, 2]    # оценён бокс-сет — вся серия
    assert idx.continuations(np.array([3])).tolist() == []           # вне серий
    assert idx.continuations(np.array([4])).tolist() == [4, 5]       # Mort тянет весь Discworld


def test_exclusion_is_input_plus_started_series():
    X = sp.csr_matrix(np.array([[0, 0, 0, 5.0, 4.0, 0]], dtype=np.float32))   # «1984» и Mort
    assert exclusion(X, SeriesIndex(TITLES)).toarray()[0].nonzero()[0].tolist() == [3, 4, 5]


def _hold(inputs, hidden_cols, hidden_ratings):
    X = np.zeros((len(inputs), len(TITLES)), dtype=np.float32)
    for u, cols in enumerate(inputs):
        X[u, cols] = 5.0
    return Holdout(np.arange(len(inputs)), np.array(["50-199"] * len(inputs)), sp.csr_matrix(X),
                   [np.array(c) for c in hidden_cols], [np.array(r, dtype=np.float32) for r in hidden_ratings])


def test_without_started_series_drops_continuations_from_hidden_and_candidates():
    hold = without_started_series(_hold([[0], [3]], [[1, 3], [1]], [[5, 4], [5]]), SeriesIndex(TITLES))
    assert hold.hidden_cols[0].tolist() == [3] and hold.hidden_ratings[0].tolist() == [4.0]
    assert hold.hidden_cols[1].tolist() == [1]                       # у второго серия не начата — HP#2 честная цель
    assert hold.exclude.toarray()[0].nonzero()[0].tolist() == [0, 1, 2]
    assert hold.exclude.toarray()[1].nonzero()[0].tolist() == [3]
    assert hold.inputs.toarray()[0].nonzero()[0].tolist() == [0]      # вход модели не меняется


class _Fixed:
    """Выдаёт одни и те же баллы всем: HP#2, бокс-сет, «1984», дальше остальное."""
    def score(self, inputs):
        s = np.linspace(0.0, 0.1, inputs.shape[1], dtype=np.float32)
        s[[1, 2, 3]] = [9.0, 8.0, 7.0]
        return np.repeat(s[None, :], inputs.shape[0], axis=0)


def test_run_eval_never_recommends_continuation_of_started_series():
    titles = TITLES + [f"Filler {i}" for i in range(19)]             # каталог ≥ 20: топ-20 полный
    X = np.zeros((1, len(titles)), dtype=np.float32)
    X[0, 0] = 5.0
    hold = Holdout(np.array([1]), np.array(["50-199"]), sp.csr_matrix(X), [np.array([3])], [np.array([5.0])])
    per_user, _ = evaluate.run_eval(_Fixed(), without_started_series(hold, SeriesIndex(titles)))
    assert per_user.loc[0, "ndcg20"] == 1.0                          # «1984» первой: HP#2 и бокс-сет вычеркнуты


def test_index_from_works_aligns_titles_to_columns(tmp_path):
    p = tmp_path / "works.parquet"
    pd.DataFrame({"work_id": [30, 10, 20], "title": ["1984", "HP 1 (Harry Potter, #1)", "HP 2 (Harry Potter, #2)"]}
                 ).to_parquet(p)
    idx = SeriesIndex.from_works(p, np.array([10, 20, 30, 40]))       # 40 нет в works — вне серий
    assert idx.continuations(np.array([0])).tolist() == [0, 1]
    assert idx.continuations(np.array([3])).tolist() == []
