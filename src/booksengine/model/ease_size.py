"""`booksengine ease-size` (TODO п. 32): что потеряно из-за того, что EASE видит только 20 000 книг ядра.

Замер до переобучения: книги ранжируются по числу оценок в обучении — так же, как `EASE.fit` выбирает
top-N, — и раскладываются по диапазонам мест (≤ 20K, 20–30K, …). Смотрим, сколько в каждом диапазоне:
- оценок на профилях из profiles/ (вход: вне EASE книга доходит до смеси только через ALS);
- входа и скрытых книг теста по группам (скрытая книга вне EASE не может быть угадана смесью вовсе);
- рекомендаций ALS (он видит всё ядро): советовал бы он книги 20–50K, если бы смесь их пускала.
Плюс порог оценок на границе, память на копию матрицы и оценка времени обращения.
"""
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from booksengine.model import metrics
from booksengine.model.matrix import catalog_works

SIZES = (20_000, 30_000, 40_000, 50_000)
BANDS = ["≤ 20K", "20–30K", "30–40K", "40–50K", "> 50K"]
FIT_SECONDS_20K = 100.0   # обращение 20 000 × 20 000 сейчас (TODO п. 32); растёт кубом


@dataclass
class Ranks:
    work_ids: np.ndarray   # столбцы ядра
    counts: np.ndarray     # оценок в обучении по столбцам
    rank: np.ndarray       # место столбца по числу оценок, с 1


def train_ranks(ratings_path: Path, holdout_path: Path) -> Ranks:
    """Места книг как в `EASE.fit`: по убыванию оценок в обучении, при равенстве — меньший столбец."""
    work_ids = catalog_works(ratings_path)
    d = duckdb.execute("SELECT r.work_id, count(*) AS n FROM read_parquet(?) r "
                       "ANTI JOIN read_parquet(?) h USING (user_id) GROUP BY 1",
                       [str(ratings_path), str(holdout_path)]).df()
    counts = np.zeros(len(work_ids), dtype=np.int64)
    counts[np.searchsorted(work_ids, d.work_id.to_numpy())] = d.n.to_numpy()
    rank = np.empty(len(work_ids), dtype=np.int64)
    rank[np.argsort(-counts, kind="stable")] = np.arange(1, len(work_ids) + 1)
    return Ranks(work_ids, counts, rank)


def band(rank: np.ndarray) -> np.ndarray:
    """Номер диапазона BANDS по месту книги."""
    return np.searchsorted(np.array(SIZES), np.asarray(rank), side="left")


def shares(rank: np.ndarray) -> dict[str, float]:
    b = np.bincount(band(rank), minlength=len(BANDS))
    return {name: float(v / max(b.sum(), 1)) for name, v in zip(BANDS, b)}


def thresholds(r: Ranks) -> pd.DataFrame:
    order = np.argsort(r.rank)
    return pd.DataFrame([{"книг в EASE": n, "оценок у последней": int(r.counts[order[n - 1]]),
                          "ГБ на копию (float32)": round(n * n * 4 / 1e9, 1),
                          "обращение, мин (оценка)": round(FIT_SECONDS_20K * (n / 20_000) ** 3 / 60, 1)}
                         for n in SIZES if n <= len(order)])


def profile_table(path: Path, r: Ranks, clean_dir: Path) -> tuple[dict[str, int], pd.DataFrame, int]:
    """Книги профиля по диапазонам; список книг за 20K; сколько книг вне ядра (их размер EASE не касается)."""
    from booksengine.recommend import read_profile
    prof = read_profile(path, r.work_ids, clean_dir)
    cols = prof.x.indices
    counts = np.bincount(band(r.rank[cols]), minlength=len(BANDS))
    far = [{"книга": prof.names[c], "оценка": float(v), "место": int(r.rank[c]), "оценок": int(r.counts[c])}
           for c, v in zip(cols.tolist(), prof.x.data) if r.rank[c] > SIZES[0]]
    return dict(zip(BANDS, counts.tolist())), pd.DataFrame(far).sort_values("место") if far else pd.DataFrame(), \
        len(prof.skipped)


def holdout_table(hold, r: Ranks) -> pd.DataFrame:
    """Доли по диапазонам: оценки входа, скрытые книги, понравившиеся (4–5★) скрытые — по группам."""
    rows = []
    for b in [*sorted(set(hold.buckets.tolist())), "все"]:
        idx = np.arange(len(hold.user_ids)) if b == "все" else np.flatnonzero(hold.buckets == b)
        X = hold.inputs[idx]
        hid = np.concatenate([hold.hidden_cols[i] for i in idx])
        liked = np.concatenate([hold.hidden_cols[i][metrics.rounded(hold.hidden_ratings[i]) >= 4] for i in idx])
        for what, cols in (("вход", X.indices), ("скрытые", hid), ("скрытые 4–5★", liked)):
            rows.append({"группа": b, "что": what, **shares(r.rank[cols])})
    return pd.DataFrame(rows)


def als_table(als, hold, r: Ranks, batch: int = 500) -> pd.DataFrame:
    """Топ-20 ALS по всему ядру (вход и начатые серии исключены): доли рекомендаций по диапазонам."""
    tops = []
    exclude = hold.inputs if hold.exclude is None else hold.exclude
    for s in range(0, len(hold.user_ids), batch):
        tops.append(metrics.top_k(als.score(hold.inputs[s:s + batch]), exclude[s:s + batch], metrics.K))
    top = np.concatenate(tops)
    rows = []
    for b in [*sorted(set(hold.buckets.tolist())), "все"]:
        t = top if b == "все" else top[hold.buckets == b]
        t = t[t >= 0]
        rows.append({"группа": b, **shares(r.rank[t])})
    return pd.DataFrame(rows)


def _md(d: pd.DataFrame) -> str:
    rows = [list(map(str, d.columns))] + [[str(v) for v in row] for row in d.itertuples(index=False)]
    return "\n".join("| " + " | ".join(r) + " |" for r in rows[:1] + [["---"] * len(d.columns)] + rows[1:])


def _pct(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d[BANDS] = (d[BANDS] * 100).round(1)
    return d


def run(*, clean_dir: Path, split_dir: Path, models_dir: Path, profiles_dir: Path) -> str:
    from booksengine.model.evaluate import load_eval_holdout

    ratings = clean_dir / "ratings.parquet"
    r = train_ranks(ratings, split_dir / "holdout_users.parquet")
    out = [f"# Размер EASE (TODO п. 32)\n\nКниг в ядре: {len(r.work_ids)}. Места — по оценкам в обучении, как в `EASE.fit`.\n"]

    ease_cols = models_dir / "ease" / "top_cols.npy"
    if ease_cols.exists():
        top = np.load(ease_cols)
        same = np.array_equal(np.sort(top), np.sort(np.flatnonzero(r.rank <= len(top))))
        out.append(f"Сверка с `models/ease`: первые {len(top)} мест {'совпадают' if same else '**НЕ совпадают**'} "
                   "с книгами EASE.\n")

    out += ["## Порог и память\n", _md(thresholds(r)), ""]

    out.append("## Профили\n")
    for p in sorted(profiles_dir.glob("*.csv")):
        counts, far, skipped = profile_table(p, r, clean_dir)
        out.append(f"**{p.name}** — в ядре по диапазонам: "
                   + ", ".join(f"{k}: {v}" for k, v in counts.items()) + f"; вне ядра: {skipped}\n")
        if len(far):
            out += [_md(far), ""]

    hold = load_eval_holdout(ratings, split_dir, "test", r.work_ids)
    out += ["## Тест: доли по диапазонам, %\n", "Скрытые — без продолжений начатых серий, как в основной метрике.\n",
            _md(_pct(holdout_table(hold, r))), ""]

    if (models_dir / "als_neg").exists():
        from booksengine.model.als import ALS
        out += ["## Топ-20 ALS (`models/als_neg`) по всему ядру на тесте, %\n",
                "Смесь сейчас не пускает книги за 20K; здесь видно, советовал бы их ALS.\n",
                _md(_pct(als_table(ALS.load(models_dir / "als_neg"), hold, r))), ""]
    return "\n".join(out)
