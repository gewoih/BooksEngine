"""Замер новых книг (после 2017, только в единой базе `books-amazon`) на отложенных людях Amazon.

Судья на людях Goodreads новых книг не видит — их нет в их оценках. Здесь — люди Amazon с 20+ старыми книгами и 3+
новыми (`data.merged.amazon_holdout`): вход — старые книги, скрыты все новые. Список из k новых книг (будущая строка
«новинки» в выдаче) по выбранному варианту слоёв против двух простых списков — самые популярные новые книги и лучшие
по оценкам (среднее со сглаживанием к общему). Числа и средние — по обучению, без отложенных людей. Кандидаты у всех —
новые книги в поле зрения толпы (30 000 книг EASE): за ним модель балла не даёт. Правила списка те же, что в выдаче.
Мерка — как у судьи слоёв: угадано (книг списка, которые человек прочёл) и качество — средняя ценность угаданных
(5★ = 2 … 1★ = −1; звёзды Amazon — по шкале базы, 4★ Amazon → 3★). Парная разность «модель − простой список» с 95%
интервалом: модель полезна, если интервал целиком выше нуля.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from booksengine.model import metrics
from booksengine.model.layers import JUDGE_STARS
from booksengine.model.split import BUCKET_ORDER

KS = (5, 20)
GROUP = "amazon"
BASELINES = {"popular": "самые популярные", "rated": "лучшие по оценкам"}
PRIOR = 50          # сглаживание среднего: столько «голосов» общего среднего добавляется к оценкам книги


def evaluate(model, baselines: dict[str, np.ndarray], hold, picker, top_cols: np.ndarray, new: np.ndarray,
             ks=KS, batch: int = 500) -> pd.DataFrame:
    """По человеку, способу и k: угадано, качество, доля пятёрок среди угаданных. model(X, exclude) — баллы по книгам
    top_cols; baselines — балл каждого простого списка по всем книгам ядра; new — новая ли книга (по столбцам ядра)."""
    from booksengine.model.filters import RatedFilter, pick_top
    cand = new[top_cols]
    exclude = (hold.inputs if hold.exclude is None else hold.exclude).tocsr()
    rows = []
    for s in range(0, len(hold.user_ids), batch):
        X = hold.inputs[s:s + batch]
        excl = exclude[s:s + batch]
        off = (excl[:, top_cols].toarray() != 0) | ~cand
        scores = {"model": np.asarray(model(X, excl), dtype=np.float64)}
        for name, b in baselines.items():
            scores[name] = np.broadcast_to(b[top_cols].astype(np.float64), off.shape).copy()
        rated = [RatedFilter(picker.books, X.indices[X.indptr[i]:X.indptr[i + 1]]) for i in range(X.shape[0])]
        for name, sc in scores.items():
            sc = np.where(off, -np.inf, sc)
            for k in ks:
                lists = pick_top(sc, top_cols, picker, rated, k)
                for i, t in enumerate(lists):
                    u = s + i
                    hc, hr = hold.hidden_cols[u], hold.hidden_ratings[u]
                    got = np.clip(metrics.rounded(hr[np.isin(hc, t)]), 1, 5).astype(int)
                    rows.append({"user_id": hold.user_ids[u], "bucket": hold.buckets[u], "method": name, "k": k,
                                 "hits": float(len(got)),
                                 "quality": float(JUDGE_STARS[got - 1].mean()) if len(got) else np.nan,
                                 "s5": float((got == 5).mean()) if len(got) else np.nan})
    return pd.DataFrame(rows)


def summarize(per: pd.DataFrame, n_boot: int = 1000) -> dict:
    """{k: {способ: среднее и интервал угаданного и качества; у простых списков — парная разность модели с ними}}."""
    rng = lambda: np.random.default_rng(0)
    out = {}
    for k, d in per.groupby("k"):
        by = {m: g.set_index("user_id").sort_index() for m, g in d.groupby("method")}
        res = {}
        for m, g in by.items():
            r = {x: metrics.bootstrap(g[x].to_numpy(dtype=np.float64), rng(), n_boot) for x in ("hits", "quality")}
            r["s5"] = {"mean": float(g.s5.mean()) if g.s5.notna().any() else None}
            if m != "model":
                for x in ("hits", "quality"):
                    r[f"{x}_diff"] = metrics.bootstrap((by["model"][x] - g[x]).to_numpy(dtype=np.float64), rng(),
                                                       n_boot)
                r["buckets"] = {b: metrics.bootstrap(
                    (by["model"][x] - g[x])[g.bucket == b].to_numpy(dtype=np.float64), rng(), n_boot)
                    for x in ("hits",) for b in BUCKET_ORDER if (g.bucket == b).any()}
            res[m] = r
        out[str(k)] = res
    return out


def train_stats(ratings_path: Path, holdout_path: Path, work_ids: np.ndarray) -> dict[str, np.ndarray]:
    """Балл простых списков по столбцам ядра — по оценкам обучения (без отложенных людей): число оценок и среднее,
    сглаженное к общему (`PRIOR`)."""
    import duckdb
    d = duckdb.execute("""
        SELECT work_id, count(*) AS n, avg(rating) AS mean FROM read_parquet(?)
        WHERE user_id NOT IN (SELECT user_id FROM read_parquet(?)) GROUP BY 1""",
                       [str(ratings_path), str(holdout_path)]).df().set_index("work_id").reindex(work_ids)
    n, mean = d.n.fillna(0).to_numpy(dtype=np.float64), d["mean"].fillna(0).to_numpy(dtype=np.float64)
    mu = (n * mean).sum() / n.sum()
    return {"popular": n, "rated": (n * mean + PRIOR * mu) / (n + PRIOR)}


def run(*, clean_dir: Path, split_dir: Path, models_dir: Path, eval_dir: Path, ks=KS) -> dict:
    from booksengine.data.merged import NEW_WORK_OFFSET
    from booksengine.model.evaluate import load_eval_holdout
    from booksengine.model.filters import ListPicker, work_info
    from booksengine.model.layers import Layers
    from booksengine.model.matrix import catalog_works
    ratings_path = clean_dir / "ratings.parquet"
    work_ids = catalog_works(ratings_path)
    layers = Layers.load(models_dir / "layers")
    top = layers.mix.ease.top_cols
    hold = load_eval_holdout(ratings_path, split_dir, GROUP, work_ids)
    new = work_ids >= NEW_WORK_OFFSET
    model = lambda X, excl: layers.score(X, exclude=excl)[:, top]
    per = evaluate(model, train_stats(ratings_path, split_dir / "holdout_users.parquet", work_ids), hold,
                   ListPicker(work_info(clean_dir, work_ids)), top, new, ks)
    res = {"n_users": int(len(hold.user_ids)), "candidates": int(new[top].sum()),
           "hidden": int(sum(len(h) for h in hold.hidden_cols)),
           "hidden_in_view": int(sum(np.isin(h, top).sum() for h in hold.hidden_cols)),
           "variant": layers.variant.label(), "summary": summarize(per)}
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "new_books.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return res


def _f(x: dict, signed=False) -> str:
    if x["mean"] is None:
        return "—"
    v = f"{x['mean']:+.3f}" if signed else f"{x['mean']:.3f}"
    return v + (f" [{x['lo']:+.3f}; {x['hi']:+.3f}]" if signed and x.get("lo") is not None else "")


def report(res: dict) -> str:
    lines = ["# Новые книги на людях Amazon", "",
             f"{res['n_users']:,} отложенных людей Amazon (от 20 старых книг и от 3 новых): вход — старые книги, "
             f"скрыты все новые ({res['hidden']:,} оценок, из них в поле зрения толпы {res['hidden_in_view']:,}). "
             f"Кандидаты — {res['candidates']:,} новых книг в поле зрения толпы. Модель — {res['variant']}.", ""]
    for k, r in res["summary"].items():
        lines += [f"## Список из {k} новых книг", "",
                  "| список | угадано | качество | 5★ среди угаданных | модель − список: угадано | модель − список: качество |",
                  "|---|---|---|---|---|---|"]
        for m in ("model", *BASELINES):
            x = r[m]
            name = "модель" if m == "model" else BASELINES[m]
            s5 = "—" if x["s5"]["mean"] is None else f"{x['s5']['mean']:.0%}"
            diff = ("| | |" if m == "model" else f"| {_f(x['hits_diff'], True)} | {_f(x['quality_diff'], True)} |")
            lines.append(f"| {name} | {_f(x['hits'])} | {_f(x['quality'])} | {s5} {diff}")
        lines += ["", "Угадано, модель − самые популярные, по этапам (старых книг у человека): " + ", ".join(
            f"{b} {_f(v, True)}" for b, v in r["popular"]["buckets"].items()), ""]
    lines += ["**Как читать.** Угадано — книг списка, которые человек прочёл (из скрытых новых). Качество — средняя "
              "ценность угаданных: 5★ = 2, 4★ = 1, 3★ = 0.5, 2★ = −0.5, 1★ = −1 (у Amazon 4★ считается 3★ — шкала "
              "базы). Разница — парная, в скобках 95% интервал: модель лучше простого списка, если интервал целиком "
              "выше нуля."]
    return "\n".join(lines) + "\n"
