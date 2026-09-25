import numpy as np
import pandas as pd
import pytest

from booksengine.model import matrix, split
from tests.test_split import synthetic_ratings, synthetic_users


def test_to_csr_uses_catalog_columns():
    work_ids = np.array([10, 20, 30, 40])
    users, X = matrix.to_csr(np.array([5, 5, 2]), np.array([30, 10, 40]), np.array([4.0, 5.0, 1.0]), work_ids)
    assert users.tolist() == [2, 5]
    assert X.shape == (2, 4)
    assert X.toarray().tolist() == [[0, 0, 0, 1.0], [5.0, 0, 4.0, 0]]


def test_columns_rejects_unknown_work():
    with pytest.raises(KeyError):
        matrix.columns(np.array([10, 20]), np.array([15]))


def test_train_excludes_holdout_and_holdout_aligns(tmp_path):
    r = synthetic_ratings()
    rp = tmp_path / "ratings.parquet"
    r.to_parquet(rp, index=False)
    up = tmp_path / "users.parquet"
    synthetic_users(r.user_id.unique()).to_parquet(up, index=False)
    out = tmp_path / "split"
    split.build(rp, up, out, "fp", test_per_bucket={"20-39": 16}, val_per_bucket={"20-39": 8},
                bucket_pool_size={"20-39": 40}, seed=11, share=0.2)
    train = matrix.load_train(rp, out / "holdout_users.parquet")
    held = set(pd.read_parquet(out / "holdout_users.parquet").user_id)
    assert not held & set(train.user_ids.tolist())
    assert train.work_ids.tolist() == sorted(r.work_id.unique().tolist())
    assert train.X.nnz == int((~r.user_id.isin(held)).sum())
    n_test = int((pd.read_parquet(out / "holdout_users.parquet")["group"] == "test").sum())
    assert n_test > 0
    h = matrix.load_holdout(out, "test", train.work_ids)
    assert len(h.user_ids) == n_test and h.inputs.shape == (n_test, len(train.work_ids))
    hid = pd.read_parquet(out / "test_hidden.parquet")
    assert sum(len(c) for c in h.hidden_cols) == len(hid)
    u0 = h.user_ids[0]
    expect = sorted(train.work_ids[h.hidden_cols[0]].tolist())
    assert expect == sorted(hid[hid.user_id == u0].work_id.tolist())
