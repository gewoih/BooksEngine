"""Замер п. 34: добавляет ли балл книги (относительно личного топ-1) сигнал к шансу «понравится».

Временный код эксперимента: принят признак — переходит в `chance.Chance` (и в C#), отклонён — модуль удаляется,
итог в docs/resheniya.md. Рабочий `chance.json` не трогает.

Варианты — логистическая регрессия на валидации, сравнение на тесте:
- «место + щедрость» — текущий шанс (п. 29);
- «+ балл» — ещё r = балл / балл топ-1 (1 — первая книга, 0 — средняя книга смеси);
- «балл вместо места» — r вместо log10(места);
- «+ место × щедрость» — у щедрых место может значить другое.
Сила подтягивания щедрости a — от текущего шанса, у всех вариантов одна.

Критерий (TODO п. 34): log-loss на тесте ниже текущего, 95%-й интервал бутстрэпа по людям не касается нуля;
калибровка по корзинам — в пределах 2 п.п.; шанс не противоречит порядку выдачи (монотонен по баллу у любого человека).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from booksengine.model.chance import _auc, _logistic, _logit, fit, log_loss, personal_pct, relative_score, reliability

VARIANTS: dict[str, tuple[str, ...]] = {
    "место + щедрость": ("1", "lp", "g"),
    "+ балл": ("1", "lp", "g", "rel"),
    "балл вместо места": ("1", "rel", "g"),
    "+ место × щедрость": ("1", "lp", "g", "lp*g"),
}
BASE = "место + щедрость"
TOP = 20              # «внутри топа»: место ≤ 20 среди кандидатов
MIN_BIN = 1000        # корзина калибровки меньше — шум, в максимум расхождения не входит
N_BOOT = 1000


def features(d: pd.DataFrame, names: tuple[str, ...], prior: float, p0: float) -> np.ndarray:
    own = (d.k_like.to_numpy(np.float64) + prior * p0) / (d.n_rated.to_numpy(np.float64) + prior)
    cols = {"1": np.ones(len(d)), "lp": np.log10(d.pct.to_numpy(np.float64)), "g": _logit(own),
            "rel": d.rel.to_numpy(np.float64)}
    cols["lp*g"] = cols["lp"] * cols["g"]
    return np.column_stack([cols[n] for n in names])


def predict(d: pd.DataFrame, names: tuple[str, ...], coef, prior: float, p0: float) -> np.ndarray:
    return 1 / (1 + np.exp(-features(d, names, prior, p0) @ np.asarray(coef)))


def monotone(names: tuple[str, ...], coef, g_range: tuple[float, float]) -> bool:
    """Шанс не убывает с баллом у любого человека: место падает с баллом, r растёт; щедрость у человека одна."""
    c = dict(zip(names, coef))
    lp_ok = all(c.get("lp", 0.0) + c.get("lp*g", 0.0) * g <= 0 for g in g_range)
    return lp_ok and c.get("rel", 0.0) >= 0


def _losses(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def bootstrap_delta(users: np.ndarray, diff: np.ndarray, n_boot: int = N_BOOT, seed: int = 0) -> list[float]:
    """95%-й интервал средней разницы log-loss при пересэмплировании людей (их книги — вместе)."""
    codes, uniq = pd.factorize(users)
    sums = np.bincount(codes, diff)
    cnt = np.bincount(codes).astype(np.float64)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        deltas[b] = (w * sums).sum() / (w * cnt).sum()
    return [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))]


def max_deviation(rel: list[dict]) -> float:
    return max((abs(b["promised"] - b["actual"]) for b in rel if b["n"] >= MIN_BIN), default=float("nan"))


def compare(val: pd.DataFrame, test: pd.DataFrame, *, n_boot: int = N_BOOT) -> dict:
    """Варианты на валидации → метрики на тесте. Строки без r (лучший балл ≤ 0) выброшены у всех вариантов."""
    dropped = {"val": int(val.rel.isna().sum()), "test": int(test.rel.isna().sum())}
    val, test = val[val.rel.notna()].reset_index(drop=True), test[test.rel.notna()].reset_index(drop=True)
    users = val.drop_duplicates("user_id")
    p0 = float((users.k_like / users.n_rated).mean())
    prior = fit(val, p0)[0].prior
    y_val = (val.rating >= 4).to_numpy(np.float64)
    y = (test.rating >= 4).to_numpy(np.float64)
    g = features(test, ("g",), prior, p0)[:, 0]
    top = (test.pct * test.n_cand).round().to_numpy() <= TOP

    fitted, preds = {}, {}
    for name, cols in VARIANTS.items():
        coef = _logistic(features(val, cols, prior, p0), y_val).tolist()
        fitted[name] = coef
        preds[name] = predict(test, cols, coef, prior, p0)
    base_loss = _losses(y, preds[BASE])

    out = {"prior": prior, "p0": p0, "n_val": len(val), "n_test": len(test), "dropped_no_rel": dropped,
           "n_test_top": int(top.sum()), "variants": {}}
    for name, cols in VARIANTS.items():
        p = preds[name]
        rel = reliability(y, p)
        by_bucket = {b: reliability(y[test.bucket == b], p[test.bucket == b]) for b in sorted(test.bucket.unique())}
        diff = _losses(y, p) - base_loss
        out["variants"][name] = {
            "features": list(cols), "coef": fitted[name],
            "monotone": monotone(cols, fitted[name], (float(g.min()), float(g.max()))),
            "log_loss": log_loss(y, p), "brier": float(((p - y) ** 2).mean()),
            "auc": _auc(p[y == 1], p[y == 0]),
            "auc_top": _auc(p[top & (y == 1)], p[top & (y == 0)]),
            "delta_log_loss": float(diff.mean()),
            "delta_ci95": [0.0, 0.0] if name == BASE else bootstrap_delta(test.user_id.to_numpy(), diff, n_boot),
            "max_calibration_gap": max_deviation(rel),
            "max_calibration_gap_by_bucket": {b: max_deviation(r) for b, r in by_bucket.items()},
            "reliability": rel, "by_bucket": by_bucket,
        }
    return out


def profile_chances(csv: Path, cmp: dict, *, clean_dir: Path, models_dir: Path, top: int = TOP) -> dict:
    """Топ выдачи смеси по CSV-профилю: шанс каждого варианта и балл относительно топ-1."""
    from booksengine import recommend as rec
    from booksengine.model import metrics
    from booksengine.model.filters import work_info
    from booksengine.model.matrix import catalog_works
    from booksengine.model.mix import Mix
    work_ids = catalog_works(clean_dir / "ratings.parquet")
    prof = rec.read_profile(csv, work_ids, clean_dir)
    info = work_info(clean_dir, work_ids)
    sc, picked, _ = rec.ranked(prof, Mix.load(models_dir / "mix"), info, top)
    r = metrics.rounded(prof.x.data)
    d = pd.DataFrame({"pct": personal_pct(sc, picked), "rel": relative_score(sc, picked),
                      "k_like": int((r >= 4).sum()), "n_rated": prof.x.nnz})
    out = {"titles": [info.title[c] for c in picked], "score": sc[picked].round(2).tolist(),
           "rel": d.rel.round(3).tolist(), "k_like": int((r >= 4).sum()), "n_rated": int(prof.x.nnz), "chance": {}}
    for name, cols in VARIANTS.items():
        p = predict(d, cols, cmp["variants"][name]["coef"], cmp["prior"], cmp["p0"])
        out["chance"][name] = np.round(p * 100).astype(int).tolist()
    return out


def run(*, ratings_path: Path, split_dir: Path, models_dir: Path, eval_dir: Path, clean_dir: Path,
        profiles: list[Path], name: str = "mix") -> dict:
    """Наблюдения по сохранённой модели → сравнение вариантов → eval_dir/chance_score_<name>.json."""
    from booksengine.model.chance import observations
    from booksengine.model.evaluate import MODELS, load_eval_holdout
    from booksengine.model.matrix import catalog_works
    model = MODELS[name][0].load(models_dir / name)
    work_ids = catalog_works(ratings_path)
    val = observations(model, load_eval_holdout(ratings_path, split_dir, "val", work_ids))
    test = observations(model, load_eval_holdout(ratings_path, split_dir, "test", work_ids))
    out = {"model": name, **compare(val, test)}
    out["profiles"] = {p.stem: profile_chances(p, out, clean_dir=clean_dir, models_dir=models_dir) for p in profiles}
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / f"chance_score_{name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return out


def summary(out: dict) -> str:
    lines = [f"Тест: {out['n_test']} книг, из них в топ-{TOP} — {out['n_test_top']}; без балла выброшено "
             f"{out['dropped_no_rel']}", "",
             f"{'вариант':<22}{'log-loss':>9}{'Δ':>9}{'  95% ДИ Δ':<22}{'AUC':>6}{'AUC топ':>8}"
             f"{'калибр.':>8}  монотонен  коэффициенты"]
    for name, v in out["variants"].items():
        ci = "" if name == BASE else f"  [{v['delta_ci95'][0]:+.5f}, {v['delta_ci95'][1]:+.5f}]"
        coef = ", ".join(f"{f}={c:+.3f}" for f, c in zip(v["features"], v["coef"]))
        lines.append(f"{name:<22}{v['log_loss']:>9.4f}{v['delta_log_loss']:>+9.5f}{ci:<22}{v['auc']:>6.3f}"
                     f"{v['auc_top']:>8.3f}{v['max_calibration_gap'] * 100:>7.1f}п  {'да' if v['monotone'] else 'НЕТ':<9}  {coef}")
    for pname, p in out.get("profiles", {}).items():
        lines += ["", f"Профиль {pname}: {p['k_like']} из {p['n_rated']} на 4–5★, балл топ-1 {p['score'][0]}, "
                      f"топ-{len(p['score'])} {p['score'][-1]}"]
        for name, ch in p["chance"].items():
            lines.append(f"  {name:<22} {min(ch)}–{max(ch)}%  " + " ".join(map(str, ch)))
    return "\n".join(lines)
