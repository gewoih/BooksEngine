"""Шанс, что книга понравится (4–5★), — «насколько книга мне подходит» в процентах (TODO п. 29).

Два признака, логистическая регрессия:
- место книги в личном рейтинге модели: pct = (книг выше + 1) / число кандидатов, признак log10(pct);
- щедрость человека: доля 4–5★ в его оценках, подтянутая к доле по толпе p0 тем сильнее, чем меньше
  оценок: (k + a·p0) / (n + a), признак — её логит. Щедрость объясняет больше места в рейтинге: книга из
  топ-20 нравится придирчивым (< 50% своих 4–5★) в 54% случаев, щедрым (≥ 90%) — в 93% (валидация, ALS).

Учится на скрытых прочитанных книгах валидации, проверяется на тесте: это шанс «понравится, если прочтёте».
Коэффициенты — три числа и a, p0 (переносятся в C#); файл привязан к отпечатку весов модели.
"""
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from booksengine.model import metrics
from booksengine.model.base import fingerprint
from booksengine.model.matrix import Holdout

PRIOR_GRID = (1.0, 3.0, 10.0, 30.0, 100.0)
_EPS = 1e-6


def personal_pct(scores: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Место столбцов cols в личном рейтинге строки scores (−∞ — не кандидат): (выше + 1) / кандидатов.
    Столбец с −∞ (вход, начатая серия, вне модели) — NaN."""
    valid = np.isfinite(scores)
    srt = np.sort(scores[valid])
    s = scores[cols]
    above = len(srt) - np.searchsorted(srt, s, side="right")
    return np.where(np.isfinite(s), (above + 1) / max(len(srt), 1), np.nan)


def observations(model, hold: Holdout, batch: int = 500) -> pd.DataFrame:
    """Скрытая книга → место в рейтинге человека, его 4–5★ во входе (k из n), её оценка."""
    parts = []
    exclude = hold.inputs if hold.exclude is None else hold.exclude
    for s in range(0, len(hold.user_ids), batch):
        X = hold.inputs[s:s + batch]
        sc = model.score(X).astype(np.float64)
        tp = model.taste_prediction(X) if hasattr(model, "taste_prediction") else None
        ex = exclude[s:s + batch].tocoo()
        sc[ex.row, ex.col] = -np.inf
        for i in range(X.shape[0]):
            u = s + i
            h = hold.hidden_cols[u]
            if len(h) == 0:
                continue
            inp = metrics.rounded(X.data[X.indptr[i]:X.indptr[i + 1]])
            d = pd.DataFrame({"user_id": hold.user_ids[u], "bucket": hold.buckets[u],
                              "pct": personal_pct(sc[i], h), "k_like": int((inp >= 4).sum()),
                              "n_rated": len(inp), "rating": metrics.rounded(hold.hidden_ratings[u])})
            if tp is not None:
                d["taste"] = tp[i, h]
            parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    return d[d.pct.notna()].reset_index(drop=True)


def _logit(p):
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _logistic(X: np.ndarray, y: np.ndarray, iters: int = 50) -> np.ndarray:
    """Ньютон (IRLS) с крошечной L2 ради устойчивости; X уже со свободным членом."""
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        H = (X.T * (p * (1 - p))) @ X + 1e-6 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p) - 1e-6 * w)
        w += step
        if np.abs(step).max() < 1e-9:
            break
    return w


@dataclass
class Chance:
    coef: list[float]   # свободный член, log10(pct), логит щедрости
    prior: float        # a: сила подтягивания щедрости к толпе, в «оценках»
    p0: float           # доля 4–5★ по толпе (вход обучающих людей)
    model_fp: str = ""  # отпечаток весов модели, на которой откалибровано

    def _features(self, pct, k_like, n_rated) -> np.ndarray:
        own = (np.asarray(k_like, dtype=np.float64) + self.prior * self.p0) / (np.asarray(n_rated) + self.prior)
        pct = np.asarray(pct, dtype=np.float64)
        return np.column_stack([np.ones(len(pct)), np.log10(pct), _logit(own)])

    def predict(self, pct, k_like, n_rated) -> np.ndarray:
        pct = np.atleast_1d(np.asarray(pct, dtype=np.float64))
        k = np.broadcast_to(k_like, pct.shape)
        n = np.broadcast_to(n_rated, pct.shape)
        return 1 / (1 + np.exp(-self._features(pct, k, n) @ np.asarray(self.coef)))

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=1))

    @classmethod
    def load(cls, path: Path, model_fp: str | None = None) -> "Chance":
        c = cls(**json.loads(path.read_text()))
        if model_fp is not None and c.model_fp != model_fp:
            raise ValueError(f"{path}: откалибровано на другой версии модели — пересчитайте `booksengine calibrate`")
        return c


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, _EPS, 1 - _EPS)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def fit(obs: pd.DataFrame, p0: float, prior_grid=PRIOR_GRID) -> tuple[Chance, dict]:
    """Коэффициенты при каждой силе подтягивания a; берётся a с наименьшим log-loss."""
    y = (obs.rating >= 4).to_numpy(dtype=np.float64)
    losses, best = {}, None
    for a in prior_grid:
        c = Chance([0.0, 0.0, 0.0], a, p0)
        X = c._features(obs.pct, obs.k_like, obs.n_rated)
        c.coef = _logistic(X, y).tolist()
        losses[a] = log_loss(y, c.predict(obs.pct, obs.k_like, obs.n_rated))
        if best is None or losses[a] < losses[best.prior]:
            best = c
    return best, losses


def rank_only(obs_fit: pd.DataFrame, obs: pd.DataFrame) -> np.ndarray:
    """Сравнение: шанс только по месту в рейтинге (без щедрости человека)."""
    X = lambda d: np.column_stack([np.ones(len(d)), np.log10(d.pct.to_numpy())])
    w = _logistic(X(obs_fit), (obs_fit.rating >= 4).to_numpy(dtype=np.float64))
    return 1 / (1 + np.exp(-X(obs) @ w))


def person_only(obs_fit: pd.DataFrame, obs: pd.DataFrame, prior: float, p0: float) -> np.ndarray:
    """Сравнение: шанс только по щедрости человека (без места книги)."""
    def X(d):
        own = (d.k_like.to_numpy() + prior * p0) / (d.n_rated.to_numpy() + prior)
        return np.column_stack([np.ones(len(d)), _logit(own)])
    w = _logistic(X(obs_fit), (obs_fit.rating >= 4).to_numpy(dtype=np.float64))
    return 1 / (1 + np.exp(-X(obs) @ w))


def reliability(y: np.ndarray, p: np.ndarray, bins=(0, .3, .4, .5, .6, .7, .8, .9, 1.0001)) -> list[dict]:
    """Обещанный шанс против доли понравившихся по корзинам обещания."""
    b = pd.cut(p, list(bins), right=False)
    t = pd.DataFrame({"b": b, "y": y, "p": p}).groupby("b", observed=True).agg(
        n=("y", "size"), promised=("p", "mean"), actual=("y", "mean"))
    return [{"bin": str(i), "n": int(r.n), "promised": float(r.promised), "actual": float(r.actual)}
            for i, r in t.iterrows()]


def with_taste(obs_fit: pd.DataFrame, obs: pd.DataFrame, prior: float, p0: float) -> np.ndarray:
    """Сравнение (только замер): место + щедрость + прогноз оценки модели вкуса. В шанс не входит — книга ниже
    в списке могла бы получить больший процент."""
    def X(d):
        own = (d.k_like.to_numpy() + prior * p0) / (d.n_rated.to_numpy() + prior)
        return np.column_stack([np.ones(len(d)), np.log10(d.pct.to_numpy()), _logit(own), d.taste.to_numpy() - 3.0])
    w = _logistic(X(obs_fit), (obs_fit.rating >= 4).to_numpy(dtype=np.float64))
    return 1 / (1 + np.exp(-X(obs) @ w))


def within_person_auc(obs: pd.DataFrame, p: np.ndarray) -> float:
    """Среднее по людям: понравившаяся скрытая книга получила обещание выше непонравившейся (0.5 — монетка)."""
    d = obs.assign(p=p, y=obs.rating >= 4)
    vals = [_auc(g.p[g.y].to_numpy(), g.p[~g.y].to_numpy()) for _, g in d.groupby("user_id") if 0 < g.y.sum() < len(g)]
    return float(np.mean(vals)) if vals else float("nan")


def model_class(name: str):
    """Класс сохранённой модели по имени папки: модели стенда и слои п. 37."""
    if name == "layers":
        from booksengine.model.layers import Layers
        return Layers
    from booksengine.model.evaluate import MODELS
    return MODELS[name][0]


def calibrate(name: str, *, ratings_path: Path, split_dir: Path, models_dir: Path, eval_dir: Path) -> dict:
    """Учим шанс на валидации для сохранённой модели models_dir/<name>, проверяем на тесте.
    Пишет models_dir/<name>/chance.json и eval_dir/chance_<name>.json."""
    from booksengine.model.evaluate import load_eval_holdout
    from booksengine.model.matrix import catalog_works
    model_dir = models_dir / name
    model = model_class(name).load(model_dir)
    work_ids = catalog_works(ratings_path)
    val = observations(model, load_eval_holdout(ratings_path, split_dir, "val", work_ids))
    test = observations(model, load_eval_holdout(ratings_path, split_dir, "test", work_ids))
    users = val.drop_duplicates("user_id")
    p0 = float((users.k_like / users.n_rated).mean())
    c, losses = fit(val, p0)
    c.model_fp = fingerprint(model_dir)
    c.save(model_dir / "chance.json")
    y = (test.rating >= 4).to_numpy(dtype=np.float64)
    p = c.predict(test.pct, test.k_like, test.n_rated)
    variants = {"константа p(4–5★) валидации": np.full(len(y), (val.rating >= 4).mean()),
                "только место в рейтинге": rank_only(val, test),
                "только щедрость человека": person_only(val, test, c.prior, p0),
                "место + щедрость": p}
    if "taste" in val.columns:
        variants["место + щедрость + прогноз вкуса (только замер)"] = with_taste(val, test, c.prior, p0)
    out = {"model": name, "chance": asdict(c), "prior_log_loss_val": {str(k): v for k, v in losses.items()},
           "n_val": len(val), "n_test": len(test), "test_like_share": float(y.mean()),
           "test": {k: {"log_loss": log_loss(y, v), "brier": float(((v - y) ** 2).mean()),
                        "auc": _auc(v[y == 1], v[y == 0]), "auc_within_person": within_person_auc(test, v)}
                    for k, v in variants.items()},
           "reliability": reliability(y, p),
           "by_bucket": {b: reliability(y[test.bucket == b], p[test.bucket == b]) for b in sorted(test.bucket.unique())}}
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / f"chance_{name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return out


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Шанс, что понравившаяся книга получила обещание выше непонравившейся (0.5 — монетка)."""
    r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
