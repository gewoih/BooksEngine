"""Прогон моделей: перебор настроек на валидации, финальный замер лучшей на тесте (спецификация 3a §5–6)."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from booksengine.model import metrics
from booksengine.model.base import read_params
from booksengine.model.als import ALS
from booksengine.model.ease import EASE
from booksengine.model.knn import ItemKNN
from booksengine.model.matrix import Holdout, load_holdout, load_train
from booksengine.model.mix import Mix
from booksengine.model.popularity import Popularity
from booksengine.model.series import SeriesIndex, without_started_series
from booksengine.paths import CLEAN_DIR, EVAL_DIR, MODELS_DIR, SPLIT_DIR

RATINGS = CLEAN_DIR / "ratings.parquet"

# имя → (класс, [(параметры обучения, [настройки выдачи, ...]), ...])
MODELS: dict[str, tuple[type, list[tuple[dict, list[dict]]]]] = {
    "popularity": (Popularity, [({"formula": "count", "m": 0.0}, [{}])]
                   + [({"formula": f, "m": m}, [{}]) for f in ("bayes", "bayes_log") for m in (10.0, 100.0, 1000.0)]),
    "als": (ALS, [({"factors": f, "regularization": r, "alpha": a}, [{}])
                  for f in (64, 128, 256) for r in (0.01, 0.1) for a in (1.0, 10.0)]),
    # п. 8, 21: 128 координат (256 — та же точность вдвое дороже, выдача гуще по автору), λ = 0.1, α = 1;
    # перебор веса негативного сигнала «≤ 2» на fold-in, «none» — контроль
    "als_neg": (ALS, [({"factors": 128, "regularization": 0.1, "alpha": 1.0},
                       [{"neg_rule": "none"}] + [{"neg_rule": "le2", "neg_weight": b} for b in (0.0, 1.0, 3.0, 10.0)])]),
    # normalize=True снят: на валидации NDCG@20 0.003–0.014 против 0.24 (2026-09-23)
    "knn": (ItemKNN, [({"beta": b, "k_max": 200}, [{"k": k, "normalize": False} for k in (3, 5, 7, 10, 20, 50)])
                      for b in (0.0, 50.0)]),
    # п. 24: смесь готовых als_neg и ease (вход EASE по оценке −2/−1/0/1/2); 0 — чистый EASE, 1 — чистый ALS
    "mix": (Mix, [({"als_dir": str(MODELS_DIR / "als_neg"), "ease_dir": str(MODELS_DIR / "ease")},
                   [{"als_weight": w} for w in (0.0, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 1.0)])]),
    # п. 32: 30 000 книг вместо 20 000 — все книги профилей из 20–50K лежат до 30K (`booksengine ease-size`)
    "ease": (EASE, [({"lam": lam, "n_top": 30_000}, [{"topk": t} for t in (None, 100, 500)])
                    for lam in (100.0, 500.0, 2000.0)]),
}


def deployable(name: str, score_params: dict) -> bool:
    """Можно ли выбрать настройку: полная матрица EASE в БД не ложится; ALS без негативного сигнала —
    контроль для сверки, правило «≤ 2» зафиксировано (docs/resheniya.md, «негативный сигнал»)."""
    if name == "ease" and score_params.get("topk") is None:
        return False
    return not (name == "als_neg" and score_params.get("neg_rule") == "none")


def run_eval(model, hold: Holdout, batch: int = 500) -> tuple[pd.DataFrame, float]:
    """Топ-K для каждого пользователя группы пачками (≤ 500 × 322K float32 в памяти) и метрики."""
    n_items = hold.inputs.shape[1]
    k = min(metrics.K, n_items)
    rows, tops = [], []
    exclude = hold.inputs if hold.exclude is None else hold.exclude
    for s in range(0, len(hold.user_ids), batch):
        X = hold.inputs[s:s + batch]
        top = metrics.top_k(model.score(X), exclude[s:s + batch], k)
        tops.append(top)
        for i, t in enumerate(top):
            rows.append(metrics.user_metrics(t, hold.hidden_cols[s + i], hold.hidden_ratings[s + i]))
    per_user = pd.DataFrame(rows, columns=list(metrics.METRICS))
    per_user["bucket"] = hold.buckets
    per_user["user_id"] = hold.user_ids
    return per_user, metrics.coverage(np.concatenate(tops), n_items)


def load_eval_holdout(ratings_path: Path, split_dir: Path, group: str, work_ids: np.ndarray) -> Holdout:
    """Отложенная группа без продолжений начатых серий (works.parquet лежит рядом с ratings.parquet)."""
    hold = load_holdout(split_dir, group, work_ids)
    return without_started_series(hold, SeriesIndex.from_works(ratings_path.parent / "works.parquet", work_ids))


def _best(val: list[dict], name: str) -> dict:
    ok = [r for r in val if deployable(name, r["score_params"])]
    return max(ok, key=lambda r: r["summary"]["all"]["ndcg20"]["mean"] or -1.0)


def _needs_refit(name: str, variants: list[dict]) -> bool:
    """Вариант, которого нет в сохранённой модели: полная матрица EASE на диск не пишется."""
    return name == "ease" and any(v.get("topk") is None for v in variants)


def _saved_matches(path: Path, fit_params: dict) -> bool:
    """Модель в path обучена с этими параметрами; настройки выдачи задаются потом через `configure`."""
    if not (path / "params.json").exists():
        return False
    saved = read_params(path)
    return all(saved.get(k) == v for k, v in fit_params.items())


def tune(name: str, *, ratings_path: Path = RATINGS, split_dir: Path = SPLIT_DIR, eval_dir: Path = EVAL_DIR,
         grid: list[tuple[dict, list[dict]]] | None = None, models_dir: Path = MODELS_DIR) -> list[dict]:
    """Перебор на валидации. Лучшая переносимая модель сохраняется в models_dir/<name>/, чтобы `test`
    не обучал её заново (обучающие данные те же); результаты пишутся после каждой настройки."""
    cls, default_grid = MODELS[name]
    eval_dir.mkdir(parents=True, exist_ok=True)
    best_ndcg = -1.0
    train = load_train(ratings_path, split_dir / "holdout_users.parquet")
    hold = load_eval_holdout(ratings_path, split_dir, "val", train.work_ids)
    results = []
    for fit_params, score_grid in grid or default_grid:
        model = cls(**fit_params)
        t0 = time.perf_counter()
        model.fit(train)
        fit_s = round(time.perf_counter() - t0, 1)
        for sp_ in score_grid:
            model.configure(**sp_)
            per_user, cov = run_eval(model, hold)
            summ = metrics.summarize(per_user, n_boot=200)
            results.append({"fit_params": fit_params, "score_params": sp_, "fit_seconds": fit_s,
                            "coverage": cov, "summary": summ})
            print(f"{name} {fit_params} {sp_}: NDCG@20 = {summ['all']['ndcg20']['mean']:.4f}, "
                  f"coverage = {cov:.3f}, обучение {fit_s} с", flush=True)
            (eval_dir / f"{name}_val.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
            ndcg = summ["all"]["ndcg20"]["mean"] or -1.0
            if deployable(name, sp_) and ndcg > best_ndcg:
                best_ndcg = ndcg
                model.save(models_dir / name)
    return results


def test(name: str, *, ratings_path: Path = RATINGS, split_dir: Path = SPLIT_DIR, eval_dir: Path = EVAL_DIR,
         models_dir: Path = MODELS_DIR) -> dict:
    """Обучает лучшую по валидации настройку, сохраняет модель, меряет на тесте.

    Если лучшая настройка вообще (без требования переносимости) отличается от лучшей переносимой
    при тех же параметрах обучения — меряет и её, для сравнения в отчёте (EASE полная против урезанной).
    """
    cls, _ = MODELS[name]
    val = json.loads((eval_dir / f"{name}_val.json").read_text())
    best = _best(val, name)
    train = load_train(ratings_path, split_dir / "holdout_users.parquet")
    hold = load_eval_holdout(ratings_path, split_dir, "test", train.work_ids)
    same_fit = [r for r in val if r["fit_params"] == best["fit_params"]]
    top_any = max(same_fit, key=lambda r: r["summary"]["all"]["ndcg20"]["mean"] or -1.0)
    variants = [best["score_params"]] + ([top_any["score_params"]] if top_any is not best else [])
    if not _needs_refit(name, variants) and _saved_matches(models_dir / name, best["fit_params"]):
        model, fit_s = cls.load(models_dir / name), best["fit_seconds"]  # сохранена при подборе
    else:  # нужен вариант, которого нет на диске, — обучаем заново
        model = cls(**best["fit_params"])
        t0 = time.perf_counter()
        model.fit(train)
        fit_s = round(time.perf_counter() - t0, 1)
    out = {"model": name, "fit_params": best["fit_params"], "fit_seconds": fit_s,
           "n_users": int(len(hold.user_ids)), "variants": []}
    for sp_ in variants:
        model.configure(**sp_)
        per_user, cov = run_eval(model, hold)
        out["variants"].append({
            "label": ", ".join(f"{k}={v}" for k, v in sp_.items()) or "—", "score_params": sp_,
            "deployable": deployable(name, sp_), "coverage": cov, "summary": metrics.summarize(per_user)})
    model.configure(**best["score_params"])
    model.save(models_dir / name)
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / f"{name}_test.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return out
