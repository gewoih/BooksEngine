import json

import numpy as np
import pandas as pd
import pytest

from booksengine.model import split


def synthetic_ratings(n_users=40, n_works=60, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(1, n_users + 1):
        n = int(rng.integers(10, 31))
        works = rng.choice(np.arange(100, 100 + n_works), size=n, replace=False)
        for w in works:
            rows.append((u, int(w), float(rng.integers(1, 6))))
    return pd.DataFrame(rows, columns=["user_id", "work_id", "rating"]).astype(
        {"user_id": "int32", "work_id": "int64", "rating": "float32"})


@pytest.fixture
def ratings_path(tmp_path):
    p = tmp_path / "ratings.parquet"
    synthetic_ratings().to_parquet(p, index=False)
    return p


def test_bucket_of_edges():
    assert split.bucket_of(np.array([10, 19, 20, 49, 50, 199, 200, 3000])).tolist() == [
        "10-19", "10-19", "20-49", "20-49", "50-199", "50-199", "200+", "200+"]
    with pytest.raises(ValueError):
        split.bucket_of(np.array([9]))


def test_assign_groups_disjoint_and_deterministic():
    ids = np.arange(1, 101)
    a = split.assign_groups(ids, 10, 20, seed=7)
    b = split.assign_groups(ids[::-1], 10, 20, seed=7)
    assert len(a["val"]) == 10 and len(a["test"]) == 20
    assert not set(a["val"]) & set(a["test"])
    assert a["val"].tolist() == b["val"].tolist() and a["test"].tolist() == b["test"].tolist()


def test_hide_sizes_and_partition():
    r = synthetic_ratings()
    inp, hid = split.hide(r, 0.2, seed=3)
    assert len(inp) + len(hid) == len(r)
    merged = pd.concat([inp, hid]).sort_values(["user_id", "work_id"], ignore_index=True)
    pd.testing.assert_frame_equal(merged, r.sort_values(["user_id", "work_id"], ignore_index=True))
    n = r.groupby("user_id").size()
    h = hid.groupby("user_id").size()
    assert (h == np.maximum(1, np.round(0.2 * n))).all()
    assert not set(zip(inp.user_id, inp.work_id)) & set(zip(hid.user_id, hid.work_id))


def test_build_writes_split_and_excludes_nothing(tmp_path, ratings_path):
    out = tmp_path / "split"
    meta = split.build(ratings_path, out, "fp1", n_val=5, n_test=10, seed=11, share=0.2)
    users = pd.read_parquet(out / "holdout_users.parquet")
    assert users.groupby("group").size().to_dict() == {"test": 10, "val": 5}
    assert users.user_id.is_unique
    r = pd.read_parquet(ratings_path)
    for g in ("val", "test"):
        ids = set(users[users.group == g].user_id)
        inp = pd.read_parquet(out / f"{g}_input.parquet")
        hid = pd.read_parquet(out / f"{g}_hidden.parquet")
        assert set(inp.user_id) == ids and set(hid.user_id) == ids
        assert len(inp) + len(hid) == int(r.user_id.isin(ids).sum())
    assert meta["groups"]["test"]["users"] == 10


def test_build_reuses_and_rebuilds_by_fingerprint(tmp_path, ratings_path):
    out = tmp_path / "split"
    split.build(ratings_path, out, "fp1", n_val=5, n_test=10, seed=11, share=0.2)
    mtime = (out / "test_hidden.parquet").stat().st_mtime_ns
    split.build(ratings_path, out, "fp1", n_val=5, n_test=10, seed=11, share=0.2)
    assert (out / "test_hidden.parquet").stat().st_mtime_ns == mtime
    split.build(ratings_path, out, "fp2", n_val=5, n_test=10, seed=11, share=0.2)
    assert json.loads((out / "split.json").read_text())["fingerprint"].startswith("fp2|")


def test_build_same_seed_same_content(tmp_path, ratings_path):
    a, b = tmp_path / "a", tmp_path / "b"
    split.build(ratings_path, a, "fp", n_val=5, n_test=10, seed=11, share=0.2)
    split.build(ratings_path, b, "fp", n_val=5, n_test=10, seed=11, share=0.2)
    for f in ("holdout_users", "val_input", "val_hidden", "test_input", "test_hidden"):
        pd.testing.assert_frame_equal(pd.read_parquet(a / f"{f}.parquet"), pd.read_parquet(b / f"{f}.parquet"))
