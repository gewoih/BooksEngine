"""Отложенная выборка (TODO п. 4, спецификация 3a §3).

Пользователи валидации и теста целиком исключены из обучения: так мерится fold-in — тот же путь,
по которому получат рекомендации реальные пользователи приложения. У каждого скрыто
max(1, round(0.2·n)) случайных оценок, остальное — вход.
"""
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

SEED = 20260923
N_VAL = 5_000
N_TEST = 20_000
HIDDEN_SHARE = 0.2
BUCKETS = ((10, "10-19"), (20, "20-49"), (50, "50-199"), (200, "200+"))
BUCKET_ORDER = [name for _, name in BUCKETS]
GROUPS = ("val", "test")


def bucket_of(n: np.ndarray) -> np.ndarray:
    """Группа активности по полному числу оценок пользователя."""
    n = np.asarray(n)
    if (n < BUCKETS[0][0]).any():
        raise ValueError(f"пользователь с < {BUCKETS[0][0]} оценок вне CF-ядра")
    edges = np.array([lo for lo, _ in BUCKETS])
    return np.array(BUCKET_ORDER)[np.searchsorted(edges, n, side="right") - 1]


def assign_groups(user_ids: np.ndarray, n_val: int, n_test: int, seed: int) -> dict[str, np.ndarray]:
    """Две непересекающиеся случайные группы; результат не зависит от порядка входа."""
    ids = np.sort(np.asarray(user_ids))
    if n_val + n_test > len(ids):
        raise ValueError(f"нужно {n_val + n_test} пользователей, есть {len(ids)}")
    perm = np.random.default_rng(seed).permutation(len(ids))
    return {"val": np.sort(ids[perm[:n_val]]), "test": np.sort(ids[perm[n_val:n_val + n_test]])}


def hide(ratings: pd.DataFrame, share: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Делит оценки каждого пользователя на вход и скрытое. ratings: user_id, work_id, rating."""
    df = ratings.sort_values(["user_id", "work_id"], ignore_index=True)
    key = pd.Series(np.random.default_rng(seed).random(len(df)), index=df.index)
    order = key.groupby(df["user_id"]).rank(method="first")
    n = df.groupby("user_id")["work_id"].transform("size")
    mask = (order <= np.maximum(1, np.round(share * n))).to_numpy()
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)


def fingerprint(data_fp: str, n_val: int, n_test: int, seed: int, share: float) -> str:
    return f"{data_fp}|val={n_val}|test={n_test}|seed={seed}|share={share}"


def build(ratings_path: Path, out_dir: Path, data_fp: str, *, n_val: int = N_VAL, n_test: int = N_TEST,
          seed: int = SEED, share: float = HIDDEN_SHARE, force: bool = False) -> dict:
    """Строит сплит; при том же отпечатке (данные + параметры) переиспользует готовый."""
    fp = fingerprint(data_fp, n_val, n_test, seed, share)
    meta_path = out_dir / "split.json"
    if not force and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["fingerprint"] == fp:
            return meta
    con = duckdb.connect()
    counts = con.execute("SELECT user_id, count(*) AS n FROM read_parquet(?) GROUP BY user_id ORDER BY user_id",
                         [str(ratings_path)]).df()
    groups = assign_groups(counts["user_id"].to_numpy(), n_val, n_test, seed)
    n_of = counts.set_index("user_id")["n"]
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"fingerprint": fp, "seed": seed, "share": share, "groups": {}}
    holdout = []
    for i, g in enumerate(GROUPS):
        ids = groups[g]
        con.register("ids", pd.DataFrame({"user_id": ids}))
        r = con.execute("SELECT r.user_id, r.work_id, r.rating FROM read_parquet(?) r JOIN ids USING (user_id)",
                        [str(ratings_path)]).df()
        con.unregister("ids")
        inp, hid = hide(r, share, seed + 1 + i)
        inp.to_parquet(out_dir / f"{g}_input.parquet", index=False)
        hid.to_parquet(out_dir / f"{g}_hidden.parquet", index=False)
        n = n_of.loc[ids].to_numpy()
        buckets = bucket_of(n)
        holdout.append(pd.DataFrame({"user_id": ids, "group": g, "n_ratings": n, "bucket": buckets}))
        meta["groups"][g] = {
            "users": int(len(ids)), "input": int(len(inp)), "hidden": int(len(hid)),
            "buckets": {b: int((buckets == b).sum()) for b in BUCKET_ORDER},
        }
    pd.concat(holdout, ignore_index=True).to_parquet(out_dir / "holdout_users.parquet", index=False)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1))  # последним: без него сплит неполный
    return meta
