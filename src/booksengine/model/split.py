"""Отложенная выборка: люди валидации и теста.

Пользователи валидации и теста целиком исключены из обучения: так мерится fold-in — тот же путь,
по которому получат рекомендации реальные пользователи приложения. У каждого скрыто
max(1, round(0.2·n)) случайных оценок, остальное — вход.

Тест и валидация набираются поровну по группам активности (`TEST_PER_BUCKET`, `VAL_PER_BUCKET`); размер — с
запасом для самых шумных групп 20-49 и 50-199 (там погрешность метрики в 2-4 раза больше общей, а это профили
пользователя и его друзей). Отбор — по стабильному хэшу внешнего id (`hash01`) с фиксированным порогом на группу,
а не случайной перестановкой всего пула: порог не зависит от того, сколько сейчас всего людей в ядре, поэтому
человек не покидает тест только из-за того, что очистка убрала кого-то другого.
"""
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

SEED = 20260923
HIDDEN_SHARE = 0.2
BUCKETS = ((20, "20-49"), (50, "50-199"), (200, "200+"))
BUCKET_ORDER = [name for _, name in BUCKETS]
GROUPS = ("val", "test")

# Размер группы активности в ядре (20, 100) по замеру 2026-09-23 — калибровка порога хэша.
# Фиксированный, не пересчитывается от текущего прогона: иначе порог, а с ним и состав теста, снова
# зависел бы от того, сколько людей очистка оставила в ядре.
BUCKET_POOL_SIZE = {"20-49": 199_000, "50-199": 278_000, "200+": 118_000}
TEST_PER_BUCKET = {"20-49": 2_000, "50-199": 2_000, "200+": 2_000}
VAL_PER_BUCKET = {"20-49": 1_000, "50-199": 1_000, "200+": 1_000}


def bucket_of(n: np.ndarray) -> np.ndarray:
    """Группа активности по полному числу оценок пользователя."""
    n = np.asarray(n)
    if (n < BUCKETS[0][0]).any():
        raise ValueError(f"пользователь с < {BUCKETS[0][0]} оценок вне CF-ядра")
    edges = np.array([lo for lo, _ in BUCKETS])
    return np.array(BUCKET_ORDER)[np.searchsorted(edges, n, side="right") - 1]


def hash01(external_id: np.ndarray, salt: int) -> np.ndarray:
    """Число в [0, 1), стабильное между прогонами: зависит только от id и salt (blake2b), не от
    порядка или состава входного массива."""
    salt_b = int(salt).to_bytes(8, "little")
    return np.array([int.from_bytes(hashlib.blake2b(salt_b + str(v).encode(), digest_size=8).digest(), "little")
                      / 2 ** 64 for v in external_id])


def assign_groups(external_id: np.ndarray, bucket: np.ndarray, *, seed: int = SEED,
                   test_per_bucket: dict[str, int] = TEST_PER_BUCKET,
                   val_per_bucket: dict[str, int] = VAL_PER_BUCKET,
                   bucket_pool_size: dict[str, int] = BUCKET_POOL_SIZE) -> dict[str, np.ndarray]:
    """Булевы маски val/test, выровненные с входом. Для каждого external_id независимо от остальных:
    попадание решает порог на hash01, откалиброванный по bucket_pool_size."""
    draw = hash01(external_id, seed)
    label = np.full(len(external_id), "", dtype=object)
    for b, pool in bucket_pool_size.items():
        m = np.asarray(bucket) == b
        t = test_per_bucket.get(b, 0) / pool
        v = val_per_bucket.get(b, 0) / pool
        label[m & (draw < t)] = "test"
        label[m & (draw >= t) & (draw < t + v)] = "val"
    return {"val": label == "val", "test": label == "test"}


def hide(ratings: pd.DataFrame, share: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Делит оценки каждого пользователя на вход и скрытое. ratings: user_id, work_id, rating."""
    df = ratings.sort_values(["user_id", "work_id"], ignore_index=True)
    key = pd.Series(np.random.default_rng(seed).random(len(df)), index=df.index)
    order = key.groupby(df["user_id"]).rank(method="first")
    n = df.groupby("user_id")["work_id"].transform("size")
    mask = (order <= np.maximum(1, np.round(share * n))).to_numpy()
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)


def fingerprint(data_fp: str, test_per_bucket: dict[str, int], val_per_bucket: dict[str, int],
                 bucket_pool_size: dict[str, int], seed: int, share: float) -> str:
    tb = tuple(sorted(test_per_bucket.items()))
    vb = tuple(sorted(val_per_bucket.items()))
    pb = tuple(sorted(bucket_pool_size.items()))
    return f"{data_fp}|test={tb}|val={vb}|pool={pb}|seed={seed}|share={share}"


def build(ratings_path: Path, users_path: Path, out_dir: Path, data_fp: str, *,
          test_per_bucket: dict[str, int] = TEST_PER_BUCKET, val_per_bucket: dict[str, int] = VAL_PER_BUCKET,
          bucket_pool_size: dict[str, int] = BUCKET_POOL_SIZE,
          seed: int = SEED, share: float = HIDDEN_SHARE, force: bool = False) -> dict:
    """Строит сплит; при том же отпечатке (данные + параметры) переиспользует готовый."""
    fp = fingerprint(data_fp, test_per_bucket, val_per_bucket, bucket_pool_size, seed, share)
    meta_path = out_dir / "split.json"
    if not force and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["fingerprint"] == fp:
            return meta
    con = duckdb.connect()
    counts = con.execute("SELECT user_id, count(*) AS n FROM read_parquet(?) GROUP BY user_id ORDER BY user_id",
                         [str(ratings_path)]).df()
    users = con.execute("SELECT user_id, external_id FROM read_parquet(?)", [str(users_path)]).df()
    counts = counts.merge(users, on="user_id", how="left")
    if counts["external_id"].isna().any():
        raise ValueError("в users.parquet не нашёлся external_id для части user_id из ratings.parquet")
    buckets_all = bucket_of(counts["n"].to_numpy())
    masks = assign_groups(counts["external_id"].to_numpy(), buckets_all, seed=seed,
                           test_per_bucket=test_per_bucket, val_per_bucket=val_per_bucket,
                           bucket_pool_size=bucket_pool_size)
    groups = {g: np.sort(counts.loc[masks[g], "user_id"].to_numpy()) for g in GROUPS}
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
