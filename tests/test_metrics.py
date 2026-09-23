import math

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from booksengine.model import metrics


def test_gains_round_half_up():
    r = np.array([1, 2, 2.5, 3, 3.5, 4, 4.5, 5])
    assert metrics.gains(r).tolist() == [0, 0, 0, 0, 1, 1, 2, 2]


def test_top_k_excludes_input_and_breaks_ties_by_column():
    scores = np.array([[5.0, 9.0, 1.0, 5.0, 7.0]], dtype=np.float32)
    exclude = sp.csr_matrix(np.array([[0, 1, 0, 0, 0]], dtype=np.float32))
    assert metrics.top_k(scores, exclude, k=3).tolist() == [[4, 0, 3]]
    assert scores[0, 1] == 9.0  # вход не портится


def test_user_metrics_hand_example():
    top = np.array([7, 3, 9] + list(range(100, 117)))  # 20 позиций
    hidden_cols = np.array([3, 7, 50, 60])
    hidden_ratings = np.array([5.0, 4.0, 5.0, 1.0])   # выигрыши 2, 1, 2, 0; 60 — низкая
    m = metrics.user_metrics(top, hidden_cols, hidden_ratings)
    dcg = 1 / math.log2(2) + 2 / math.log2(3)
    idcg = 2 / math.log2(2) + 2 / math.log2(3) + 1 / math.log2(4)
    assert m["ndcg20"] == pytest.approx(dcg / idcg)
    assert m["ndcg10"] == pytest.approx(dcg / idcg)
    assert m["recall20"] == pytest.approx(2 / 3)
    assert m["map20"] == pytest.approx((1 / 1 + 2 / 2) / 3)
    assert m["low20"] == 0.0


def test_user_metrics_low_hit_and_no_relevant_is_nan():
    top = np.arange(20)
    m = metrics.user_metrics(top, np.array([5, 40]), np.array([2.0, 3.0]))
    assert math.isnan(m["ndcg20"]) and math.isnan(m["recall20"]) and math.isnan(m["map20"])
    assert m["low20"] == 1.0


def test_user_metrics_fractional_low_boundary():
    m = metrics.user_metrics(np.arange(20), np.array([1]), np.array([2.5]))
    assert math.isnan(m["low20"])  # 2.5 → 3: не низкая


def test_top_k_marks_non_finite_as_minus_one():
    scores = np.array([[1.0, -np.inf, 2.0]], dtype=np.float32)
    assert metrics.top_k(scores, sp.csr_matrix((1, 3), dtype=np.float32), k=3).tolist() == [[2, 0, -1]]


def test_coverage_ignores_empty_slots():
    tops = np.array([[0, 1], [1, 2], [2, -1]])
    assert metrics.coverage(tops, n_items=10) == pytest.approx(0.3)


def test_summarize_skips_nan_and_splits_buckets():
    per_user = pd.DataFrame({
        "ndcg20": [1.0, 0.0, np.nan], "ndcg10": [1.0, 0.0, np.nan], "recall20": [1.0, 0.0, np.nan],
        "map20": [1.0, 0.0, np.nan], "low20": [np.nan, np.nan, 0.5], "bucket": ["20-49", "200+", "200+"]})
    s = metrics.summarize(per_user, n_boot=50, seed=0)
    assert s["all"]["ndcg20"]["mean"] == pytest.approx(0.5) and s["all"]["ndcg20"]["n"] == 2
    assert s["20-49"]["ndcg20"]["mean"] == 1.0
    assert s["50-199"]["ndcg20"]["n"] == 0 and s["50-199"]["ndcg20"]["mean"] is None
    assert s["all"]["ndcg20"]["lo"] <= 0.5 <= s["all"]["ndcg20"]["hi"]
