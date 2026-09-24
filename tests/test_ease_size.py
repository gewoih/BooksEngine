import numpy as np
import pandas as pd

from booksengine.model import ease_size as es
from booksengine.model.ease import EASE
from booksengine.model.matrix import load_train


def test_ranks_repeat_ease_choice_and_skip_holdout(tmp_path):
    # 5 книг; пользователь 9 — отложен: его оценки не считаются. Равенство — меньший work_id выше.
    r = pd.DataFrame({"user_id": [1, 1, 1, 2, 2, 3, 3, 9, 9],
                      "work_id": [10, 20, 30, 20, 30, 30, 40, 50, 40],
                      "rating": [5.0] * 9})
    rp, hp = tmp_path / "ratings.parquet", tmp_path / "holdout.parquet"
    r.to_parquet(rp, index=False)
    pd.DataFrame({"user_id": [9]}).to_parquet(hp, index=False)
    ranks = es.train_ranks(rp, hp)
    assert ranks.work_ids.tolist() == [10, 20, 30, 40, 50]
    assert ranks.counts.tolist() == [1, 2, 3, 1, 0]          # 50 — только у отложенного
    assert ranks.rank.tolist() == [3, 2, 1, 4, 5]            # 10 и 40 по одной оценке — 10 выше
    for n in (1, 2, 3, 4):
        m = EASE(lam=1.0, n_top=n)
        m.fit(load_train(rp, hp))
        assert m.top_cols.tolist() == sorted(np.flatnonzero(ranks.rank <= n).tolist())


def test_bands_split_on_sizes():
    assert es.band(np.array([1, 30_000, 30_001, 40_000, 55_000, 60_000, 60_001])).tolist() == [0, 0, 1, 1, 3, 3, 4]
    s = es.shares(np.array([5, 35_000, 35_001, 70_000]))
    assert s == {"≤ 30K": 0.25, "30–40K": 0.5, "40–50K": 0.0, "50–60K": 0.0, "> 60K": 0.25}


def test_run_on_synthetic_world(tmp_path, monkeypatch):
    from booksengine.model import split
    from booksengine.model.als import ALS
    from tests.test_split import synthetic_ratings, synthetic_users

    monkeypatch.setattr(es, "SIZES", (10, 20, 30, 40))            # 60 книг — диапазоны помельче
    clean, models, profiles = tmp_path / "clean", tmp_path / "models", tmp_path / "profiles"
    for d in (clean, models, profiles):
        d.mkdir()
    r = synthetic_ratings()
    r.to_parquet(clean / "ratings.parquet", index=False)
    ids = np.sort(r.work_id.unique())
    pd.DataFrame({"work_id": ids, "title": [f"Book {w}" for w in ids]}).to_parquet(clean / "works.parquet")
    pd.DataFrame({"shadow_work_id": [999], "main_work_id": [ids[0]]}).to_parquet(clean / "work_merges.parquet")
    up = tmp_path / "users.parquet"
    synthetic_users(r.user_id.unique()).to_parquet(up, index=False)
    out = tmp_path / "split"
    split.build(clean / "ratings.parquet", up, out, "fp", test_per_bucket={"20-49": 16}, val_per_bucket={"20-49": 8},
                bucket_pool_size={"20-49": 40}, seed=11, share=0.2)
    train = load_train(clean / "ratings.parquet", out / "holdout_users.parquet")
    als = ALS(factors=4, iterations=3)
    als.fit(train)
    als.configure(neg_rule="le2")
    als.save(models / "als_neg")
    ease = EASE(lam=5.0, n_top=10)
    ease.fit(train)
    ease.configure(topk=5)
    ease.save(models / "ease")
    pd.DataFrame({"goodreads_work_id": [*ids[:3], 12345], "rating": [5, 4, 1, 3],
                  "title": ["a", "b", "c", "d"]}).to_csv(profiles / "me.csv", index=False)

    text = es.run(clean_dir=clean, split_dir=out, models_dir=models, profiles_dir=profiles)
    assert "первые 10 мест совпадают" in text
    assert "**me.csv**" in text and "вне ядра: 1" in text
    assert "## Тест" in text and "## Топ-20 ALS" in text
