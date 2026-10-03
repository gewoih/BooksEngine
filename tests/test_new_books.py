import numpy as np
import pandas as pd
import scipy.sparse as sp

from booksengine.model import new_books
from booksengine.model.filters import ListPicker
from booksengine.model.matrix import Holdout


def _info(n):
    return pd.DataFrame({"title": [f"Book {i}" for i in range(n)], "original_title": None, "best_edition_title": None,
                         "author": [f"Author {i}" for i in range(n)], "author_id": np.arange(n, dtype=float),
                         "authors": [[i] for i in range(n)], "key": [f"book {i}" for i in range(n)],
                         "series_no": None, "is_collection": False, "ru_title": None})


def test_model_list_of_new_books_is_judged_against_popular_and_best_rated():
    """Столбцы 0–1 — старые книги (вход), 2–5 — новые. Человек прочёл новые 4 (5★) и 5 (1★). Модель ставит 4 выше
    всех, популярное — 2 (его никто из отложенных не читал), лучшее по оценкам — 5. Список по k=1: модель угадала
    пятёрку, популярное — ничего, лучшее по оценкам — единицу."""
    hold = Holdout(user_ids=np.array([7]), buckets=np.array(["20-39"]),
                   inputs=sp.csr_matrix(np.array([[5.0, 4.0, 0, 0, 0, 0]], dtype=np.float32)),
                   hidden_cols=[np.array([4, 5])], hidden_ratings=[np.array([5.0, 1.0])])
    top = np.arange(6)
    new = np.array([False, False, True, True, True, True])
    model = lambda X, excl: np.array([[9.0, 9.0, 0.1, 0.2, 3.0, 0.3]])
    baselines = {"popular": np.array([0, 0, 100, 50, 10, 5.0]), "rated": np.array([0, 0, 3.0, 3.5, 4.0, 4.9])}
    per = new_books.evaluate(model, baselines, hold, ListPicker(_info(6)), top, new, ks=(1, 2))
    got = per.set_index(["method", "k"])
    assert got.loc[("model", 1), "hits"] == 1 and got.loc[("model", 1), "quality"] == 2.0
    assert got.loc[("popular", 1), "hits"] == 0 and np.isnan(got.loc[("popular", 1), "quality"])
    assert got.loc[("rated", 1), "hits"] == 1 and got.loc[("rated", 1), "quality"] == -1.0
    assert got.loc[("model", 2), "hits"] == 2 and got.loc[("model", 2), "quality"] == 0.5


def test_summary_pairs_model_with_each_baseline():
    per = pd.DataFrame({"user_id": [1, 2, 1, 2], "bucket": "20-39", "method": ["model", "model", "popular", "popular"],
                        "k": 5, "hits": [2.0, 1.0, 1.0, 1.0], "quality": [2.0, 1.0, 1.0, np.nan], "s5": np.nan})
    res = new_books.summarize(per, n_boot=50)
    d = res["5"]["popular"]
    assert d["hits_diff"]["mean"] == 0.5 and d["quality_diff"]["mean"] == 1.0 and d["quality_diff"]["n"] == 1
    assert res["5"]["model"]["hits"]["mean"] == 1.5


def test_paired_compares_model_lists_of_two_model_dirs():
    def per(hits, quality):
        return pd.DataFrame({"user_id": [1, 2, 3], "bucket": "20-39", "method": "model", "k": 5,
                             "hits": hits, "quality": quality, "s5": np.nan})
    a = pd.concat([per([1.0, 0.0, 1.0], [2.0, np.nan, 1.0]),
                   per([9.0, 9.0, 9.0], [2.0, 2.0, 2.0]).assign(method="popular")])   # простые списки не считаются
    b = per([2.0, 1.0, 1.0], [1.0, 2.0, 1.0])
    r = new_books.paired(a, b, n_boot=50)["5"]
    assert r["n_users"] == 3 and r["hits_diff"]["mean"] == 2 / 3
    assert r["quality_diff"]["mean"] == -0.5                  # человек 2 без угаданного у a — в качестве не участвует
    assert "| 5 | 3 |" in new_books.report_paired({"5": r}, "30k", "40k")
