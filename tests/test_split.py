import json

import numpy as np
import pandas as pd
import pytest

from booksengine.model import split


def synthetic_ratings(n_users=40, n_works=60, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(1, n_users + 1):
        n = int(rng.integers(20, 41))
        works = rng.choice(np.arange(100, 100 + n_works), size=n, replace=False)
        for w in works:
            rows.append((u, int(w), float(rng.integers(1, 6))))
    return pd.DataFrame(rows, columns=["user_id", "work_id", "rating"]).astype(
        {"user_id": "int32", "work_id": "int64", "rating": "float32"})


def synthetic_users(user_ids) -> pd.DataFrame:
    """external_id — произвольная строка, как настоящий хэш Goodreads; не совпадает с user_id."""
    return pd.DataFrame({"user_id": user_ids, "external_id": [f"ext-{u}" for u in user_ids]})


@pytest.fixture
def ratings_path(tmp_path):
    p = tmp_path / "ratings.parquet"
    synthetic_ratings().to_parquet(p, index=False)
    return p


@pytest.fixture
def users_path(tmp_path):
    p = tmp_path / "users.parquet"
    synthetic_users(np.arange(1, 41)).to_parquet(p, index=False)
    return p


def test_bucket_of_edges():
    assert split.bucket_of(np.array([20, 49, 50, 199, 200, 3000])).tolist() == [
        "20-49", "20-49", "50-199", "50-199", "200+", "200+"]
    with pytest.raises(ValueError):
        split.bucket_of(np.array([19]))


def test_hash01_deterministic_and_order_independent():
    ids = np.array([f"ext-{i}" for i in range(200)])
    a = split.hash01(ids, salt=7)
    b = split.hash01(ids[::-1], salt=7)
    assert np.array_equal(a, b[::-1])
    assert np.array_equal(a, split.hash01(ids, salt=7))
    assert not np.array_equal(a, split.hash01(ids, salt=8))
    assert (a >= 0).all() and (a < 1).all()


def test_assign_groups_disjoint_and_order_independent():
    ids = np.array([f"ext-{i}" for i in range(2000)])
    bucket = np.full(len(ids), "20-49")
    test_q, val_q = {"20-49": 200}, {"20-49": 100}
    a = split.assign_groups(ids, bucket, seed=7, test_per_bucket=test_q, val_per_bucket=val_q)
    perm = np.random.default_rng(0).permutation(len(ids))
    b = split.assign_groups(ids[perm], bucket[perm], seed=7, test_per_bucket=test_q, val_per_bucket=val_q)
    assert not (a["val"] & a["test"]).any()
    assert set(ids[a["val"]]) == set(ids[perm][b["val"]])
    assert set(ids[a["test"]]) == set(ids[perm][b["test"]])
    # доля близка к запрошенной (test_per_bucket / BUCKET_POOL_SIZE), с допуском на случайность хэша
    assert abs(a["test"].sum() / len(ids) - 200 / split.BUCKET_POOL_SIZE["20-49"]) < 0.01


def test_assign_groups_membership_stable_when_pool_shrinks():
    """Кто-то выпал из ядра (другая очистка) — оставшиеся не меняют группу (TODO п. 26)."""
    ids = np.array([f"ext-{i}" for i in range(3000)])
    bucket = np.full(len(ids), "20-49")
    kw = dict(seed=7, test_per_bucket={"20-49": 300}, val_per_bucket={"20-49": 150})
    full = split.assign_groups(ids, bucket, **kw)
    kept = np.arange(0, len(ids), 2)  # половина пула ушла из ядра
    shrunk = split.assign_groups(ids[kept], bucket[kept], **kw)
    for g in ("val", "test"):
        assert set(ids[kept][shrunk[g]]) == set(ids[full[g]]) & set(ids[kept])


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


def test_build_writes_split_and_excludes_nothing(tmp_path, ratings_path, users_path):
    out = tmp_path / "split"
    test_q, val_q, pool = {"20-49": 16}, {"20-49": 8}, {"20-49": 40}
    meta = split.build(ratings_path, users_path, out, "fp1", test_per_bucket=test_q, val_per_bucket=val_q,
                        bucket_pool_size=pool, seed=11, share=0.2)
    users = pd.read_parquet(out / "holdout_users.parquet")
    assert users.user_id.is_unique
    r = pd.read_parquet(ratings_path)
    for g in ("val", "test"):
        ids = set(users[users.group == g].user_id)
        inp = pd.read_parquet(out / f"{g}_input.parquet")
        hid = pd.read_parquet(out / f"{g}_hidden.parquet")
        assert set(inp.user_id) == ids and set(hid.user_id) == ids
        assert len(inp) + len(hid) == int(r.user_id.isin(ids).sum())
    assert not set(users[users.group == "val"].user_id) & set(users[users.group == "test"].user_id)
    assert meta["groups"]["test"]["users"] == len(set(users[users.group == "test"].user_id))
    assert meta["groups"]["test"]["users"] > 0 and meta["groups"]["val"]["users"] > 0


def test_build_reuses_and_rebuilds_by_fingerprint(tmp_path, ratings_path, users_path):
    out = tmp_path / "split"
    q = dict(test_per_bucket={"20-49": 16}, val_per_bucket={"20-49": 8}, bucket_pool_size={"20-49": 40})
    split.build(ratings_path, users_path, out, "fp1", **q, seed=11, share=0.2)
    mtime = (out / "test_hidden.parquet").stat().st_mtime_ns
    split.build(ratings_path, users_path, out, "fp1", **q, seed=11, share=0.2)
    assert (out / "test_hidden.parquet").stat().st_mtime_ns == mtime
    split.build(ratings_path, users_path, out, "fp2", **q, seed=11, share=0.2)
    assert json.loads((out / "split.json").read_text())["fingerprint"].startswith("fp2|")


def test_build_same_seed_same_content(tmp_path, ratings_path, users_path):
    a, b = tmp_path / "a", tmp_path / "b"
    q = dict(test_per_bucket={"20-49": 16}, val_per_bucket={"20-49": 8}, bucket_pool_size={"20-49": 40})
    split.build(ratings_path, users_path, a, "fp", **q, seed=11, share=0.2)
    split.build(ratings_path, users_path, b, "fp", **q, seed=11, share=0.2)
    for f in ("holdout_users", "val_input", "val_hidden", "test_input", "test_hidden"):
        pd.testing.assert_frame_equal(pd.read_parquet(a / f"{f}.parquet"), pd.read_parquet(b / f"{f}.parquet"))
