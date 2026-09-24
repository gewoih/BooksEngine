"""`booksengine taste-gap`: понимает ли модель, что понравится именно этому человеку (TODO п. 37, шаг 0).

NDCG@20 в основном мерит «угадал, что человек прочтёт». Здесь — только «понравится ли»: у человека валидации
берутся скрытые книги (их он точно прочёл), и считается **личная точность** — доля пар «понравившаяся (4–5★) и
непонравившаяся (1–3★)», где модель поставила понравившуюся выше (ничья — половина; 0.5 — монетка). Среднее по
людям, у которых есть скрытые книги обоих видов.

Сравниваются сохранённые модели (смесь, EASE, ALS, kNN — какие есть в models/) и подсказки без модели:
средняя оценка книги у толпы (обучение) и число её оценок (популярность, «голос толпы» без личного).
Все — на одних и тех же скрытых книгах: только книги EASE (смесь не оценивает остальные), без продолжений
начатых серий. Если средняя оценка книги не хуже смеси — у системы нет слоя «вкуса», разрыв настоящий.

Рядом — доля понравившихся среди угаданных в топ-20: из скрытых книг, попавших в топ, сколько на 4–5★.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import rankdata

from booksengine.model import metrics
from booksengine.model.evaluate import MODELS, load_eval_holdout
from booksengine.model.matrix import Holdout, load_train
from booksengine.model.split import BUCKET_ORDER

MODEL_NAMES = ("mix", "taste", "ease", "als_neg", "knn")
LABELS = {"mix": "смесь (основная)", "taste": "модель вкуса (п. 37)", "ease": "EASE (вход «прочитал»)",
          "als_neg": "ALS", "knn": "item-kNN",
          "book_mean": "средняя оценка книги у толпы", "book_count": "число оценок книги (популярность)"}


def label(name: str) -> str:
    return LABELS.get(name, name)


class BookScore:
    """Один балл на книгу для всех людей — подсказка без личного."""

    def __init__(self, values: np.ndarray):
        self.values = np.asarray(values, dtype=np.float32)

    def score(self, inputs: sp.csr_matrix) -> np.ndarray:
        return np.tile(self.values, (inputs.shape[0], 1))


def book_scores(X: sp.csr_matrix) -> dict[str, BookScore]:
    """Средняя оценка и число оценок книги по обучению; книга без оценок — средняя по всем."""
    n = X.getnnz(axis=0).astype(np.float64)
    s = np.asarray(X.sum(axis=0), dtype=np.float64).ravel()
    mean = np.where(n > 0, s / np.maximum(n, 1), s.sum() / max(n.sum(), 1))
    return {"book_mean": BookScore(np.round(mean, 6)), "book_count": BookScore(n)}


def personal_auc(scores: np.ndarray, ratings: np.ndarray) -> float:
    """Доля пар «понравилась / нет», где понравившаяся выше; ничья — ½. NaN — нет книг одного из видов."""
    liked = metrics.rounded(ratings) >= 4
    n_pos, n_neg = int(liked.sum()), int((~liked).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    r = rankdata(scores)
    return float((r[liked].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def measure(model, hold: Holdout, allowed: np.ndarray, batch: int = 500) -> pd.DataFrame:
    """По человеку: личная точность на скрытых книгах из allowed (маска столбцов), угаданные в топ-20."""
    rows = []
    exclude = hold.inputs if hold.exclude is None else hold.exclude
    k = min(metrics.K, hold.inputs.shape[1])
    for s in range(0, len(hold.user_ids), batch):
        sc = model.score(hold.inputs[s:s + batch])
        top = metrics.top_k(sc, exclude[s:s + batch], k)
        for i in range(sc.shape[0]):
            u = s + i
            cols, r = hold.hidden_cols[u], hold.hidden_ratings[u]
            keep = allowed[cols]
            cols, r = cols[keep], r[keep]
            hit = np.isin(cols, top[i])
            rows.append({"user_id": hold.user_ids[u], "bucket": hold.buckets[u],
                         "auc": personal_auc(sc[i, cols].astype(np.float64), r),
                         "hits": int(hit.sum()), "liked_hits": int((metrics.rounded(r[hit]) >= 4).sum()),
                         "hidden": len(r), "liked_hidden": int((metrics.rounded(r) >= 4).sum())})
    return pd.DataFrame(rows)


def _boot(v: np.ndarray, rng, n_boot: int) -> dict:
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    b = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return {"mean": float(v.mean()), "lo": float(np.quantile(b, 0.025)), "hi": float(np.quantile(b, 0.975)),
            "n": int(len(v))}


def summarize(per_user: dict[str, pd.DataFrame], reference: str, n_boot: int = 1000, seed: int = 0) -> dict:
    """Личная точность с 95% бутстреп-интервалом, парная разница с reference (те же люди), угаданные в топ-20."""
    out = {}
    ref = per_user[reference].set_index("user_id")
    mean = per_user["book_mean"].set_index("user_id") if "book_mean" in per_user else ref
    for name, d in per_user.items():
        d = d.set_index("user_id")
        diff = d.auc - ref.auc.reindex(d.index)
        diff_mean = d.auc - mean.auc.reindex(d.index)
        parts = [("all", d.index)] + [(b, d.index[d.bucket == b]) for b in BUCKET_ORDER if (d.bucket == b).any()]
        out[name] = {}
        for part, idx in parts:
            rng = np.random.default_rng(seed)  # одни и те же выборки людей у всех моделей
            hits = int(d.loc[idx, "hits"].sum())
            out[name][part] = {"auc": _boot(d.loc[idx, "auc"].to_numpy(dtype=np.float64), rng, n_boot),
                               "diff": _boot(diff.loc[idx].to_numpy(dtype=np.float64), rng, n_boot),
                               "diff_book_mean": _boot(diff_mean.loc[idx].to_numpy(dtype=np.float64), rng, n_boot),
                               "hits_per_user": hits / max(len(idx), 1),
                               "liked_share_of_hits": d.loc[idx, "liked_hits"].sum() / hits if hits else None,
                               "liked_share_of_hidden": float(d.loc[idx, "liked_hidden"].sum()
                                                              / max(d.loc[idx, "hidden"].sum(), 1))}
    return out


def _fmt(x: dict, signed: bool = False) -> str:
    if x["mean"] is None:
        return "—"
    f = "{:+.3f}" if signed else "{:.3f}"
    return f"{f.format(x['mean'])} [{f.format(x['lo'])}; {f.format(x['hi'])}]"


def report(res: dict) -> str:
    s, ref = res["summary"], res["reference"]
    groups = ["all"] + [b for b in BUCKET_ORDER if b in s[ref]]
    lines = ["# Личная точность: понимает ли модель, что понравится (TODO п. 37)", "",
             f"Валидация, {res['n_users']} человек, скрытые книги в EASE без продолжений начатых серий. "
             f"Личная точность — доля пар «понравилась (4–5★) / нет (1–3★)» среди скрытых книг человека, где "
             f"понравившаяся стоит выше; 0.5 — монетка. В скобках — 95% интервал; разница — с «{label(ref)}» "
             f"на тех же людях.", "",
             "| модель | " + " | ".join(groups)
             + f" | разница с «{label(ref)}», все | разница со средней оценкой книги, все |",
             "|---|" + "---|" * (len(groups) + 2)]
    for name, v in s.items():
        cells = [_fmt(v[g]["auc"]) for g in groups]
        lines.append(f"| {label(name)} | " + " | ".join(cells)
                     + f" | {'—' if name == ref else _fmt(v['all']['diff'], signed=True)}"
                     + f" | {'—' if name == 'book_mean' else _fmt(v['all']['diff_book_mean'], signed=True)} |")
    base = s[ref]["all"]["liked_share_of_hidden"]
    lines += ["", f"Угаданные в топ-20 (все): сколько скрытых книг попало в топ и сколько из них на 4–5★. "
                  f"Среди всех скрытых книг на 4–5★ — {base:.0%}.", "",
              "| модель | угадано на человека | из них понравились |", "|---|---|---|"]
    for name, v in s.items():
        a = v["all"]
        share = "—" if a["liked_share_of_hits"] is None else f"{a['liked_share_of_hits']:.0%}"
        lines.append(f"| {label(name)} | {a['hits_per_user']:.2f} | {share} |")
    lines += ["", "Как читать (план шага 0): если «средняя оценка книги у толпы» не хуже смеси по личной точности — "
                  "у системы нет слоя вкуса, "
                  "переходим к шагу 1. Шаг 1 пройден, если модель вкуса выше и смеси, и средней (обе разницы в её "
                  "строке больше нуля, интервалы нуля не касаются). Если смесь заметно "
                  "лучше средней — разрыв меньше, чем предполагалось, план пересматривается."]
    return "\n".join(lines) + "\n"


def run(*, ratings_path: Path, split_dir: Path, models_dir: Path, eval_dir: Path, names=MODEL_NAMES,
        stage: str = "val") -> dict:
    """Все доступные модели из names (первая доступная — точка сравнения) и подсказки по книгам."""
    train = load_train(ratings_path, split_dir / "holdout_users.parquet")
    hold = load_eval_holdout(ratings_path, split_dir, stage, train.work_ids)
    from booksengine.model.taste import Taste
    classes = {n: c for n, (c, _) in MODELS.items()} | {"taste": Taste}
    scorers = {n: classes[n].load(models_dir / n) for n in names if (models_dir / n / "params.json").exists()}
    scorers.update(book_scores(train.X))
    del train
    allowed = np.ones(hold.inputs.shape[1], dtype=bool)
    if "mix" in scorers:  # смесь оценивает только книги EASE — сравниваем всех на них
        allowed[:] = False
        allowed[scorers["mix"].ease.top_cols] = True
    per_user = {}
    for name, m in scorers.items():
        print(f"taste-gap: {label(name)}…", flush=True)
        per_user[name] = measure(m, hold, allowed)
    reference = next(iter(scorers))
    res = {"stage": stage, "reference": reference, "n_users": int(len(hold.user_ids)),
           "summary": summarize(per_user, reference)}
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "taste_gap.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return res
