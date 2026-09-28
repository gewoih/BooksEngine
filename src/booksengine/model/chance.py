"""Шанс, что книга получит не меньше `stars` звёзд, — «насколько книга мне подходит» в процентах.

`stars` — порог: у выдачи (`layers`) — 5, шанс пятёрки: цель — книги, которые человек оценит на 5, а 4★ для него
«заметно слабее» (у книг списка «4–5★» — 72–93%, коридор узкий; пятёрка — 15–70%, различает книги лучше); у смеси
приложения — 4, «понравится» (C# считает так). Честная вероятность, не растянутая шкала: «40%» — из десяти таких книг
пятёрку получат четыре, это проверяет журнал выдач.

Два признака, логистическая регрессия:
- место книги в личном рейтинге модели: pct = (книг выше + 1) / число кандидатов, признак log10(pct);
- щедрость человека: доля оценок ≥ stars в его оценках, подтянутая к доле по толпе p0 тем сильнее, чем меньше
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
STARS = {"layers": 5}          # порог шанса по модели; остальные — 4 («понравится»)
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
    """Скрытая книга → место в рейтинге человека, его 4–5★ и 5★ во входе (k_like, k_five из n), её оценка."""
    parts = []
    exclude = hold.inputs if hold.exclude is None else hold.exclude
    for s in range(0, len(hold.user_ids), batch):
        X = hold.inputs[s:s + batch]
        tp = model.taste_prediction(X) if hasattr(model, "taste_prediction") else None
        # у слоёв исключённые не занимают мест в отсечении вкуса — как в выдаче и замере
        sc = (model.score(X) if tp is None else model.score(X, exclude=exclude[s:s + batch])).astype(np.float64)
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
                              "k_five": int((inp >= 5).sum()),
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
    p0: float           # доля оценок ≥ stars по толпе (вход обучающих людей)
    model_fp: str = ""  # отпечаток весов модели, на которой откалибровано
    stars: int = 4      # шанс оценки не ниже stars: 4 — «понравится», 5 — пятёрка

    def label(self) -> str:
        return "5★" if self.stars == 5 else f"{self.stars}–5★"

    def _features(self, pct, k_like, n_rated) -> np.ndarray:
        own = (np.asarray(k_like, dtype=np.float64) + self.prior * self.p0) / (np.asarray(n_rated) + self.prior)
        pct = np.asarray(pct, dtype=np.float64)
        return np.column_stack([np.ones(len(pct)), np.log10(pct), _logit(own)])

    def predict(self, pct, k_like, n_rated) -> np.ndarray:
        """k_like — сколько оценок человека не ниже stars, n_rated — всего оценок."""
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


def _k(obs: pd.DataFrame, stars: int) -> pd.Series:
    """Сколько оценок человека не ниже stars (столбец `observations`)."""
    return obs.k_five if stars == 5 else obs.k_like


def fit(obs: pd.DataFrame, p0: float, prior_grid=PRIOR_GRID, stars: int = 4) -> tuple[Chance, dict]:
    """Коэффициенты при каждой силе подтягивания a; берётся a с наименьшим log-loss."""
    y = (obs.rating >= stars).to_numpy(dtype=np.float64)
    k = _k(obs, stars)
    losses, best = {}, None
    for a in prior_grid:
        c = Chance([0.0, 0.0, 0.0], a, p0, stars=stars)
        X = c._features(obs.pct, k, obs.n_rated)
        c.coef = _logistic(X, y).tolist()
        losses[a] = log_loss(y, c.predict(obs.pct, k, obs.n_rated))
        if best is None or losses[a] < losses[best.prior]:
            best = c
    return best, losses


def rank_only(obs_fit: pd.DataFrame, obs: pd.DataFrame, stars: int = 4) -> np.ndarray:
    """Сравнение: шанс только по месту в рейтинге (без щедрости человека)."""
    X = lambda d: np.column_stack([np.ones(len(d)), np.log10(d.pct.to_numpy())])
    w = _logistic(X(obs_fit), (obs_fit.rating >= stars).to_numpy(dtype=np.float64))
    return 1 / (1 + np.exp(-X(obs) @ w))


def person_only(obs_fit: pd.DataFrame, obs: pd.DataFrame, prior: float, p0: float, stars: int = 4) -> np.ndarray:
    """Сравнение: шанс только по щедрости человека (без места книги)."""
    def X(d):
        own = (_k(d, stars).to_numpy() + prior * p0) / (d.n_rated.to_numpy() + prior)
        return np.column_stack([np.ones(len(d)), _logit(own)])
    w = _logistic(X(obs_fit), (obs_fit.rating >= stars).to_numpy(dtype=np.float64))
    return 1 / (1 + np.exp(-X(obs) @ w))


def reliability(y: np.ndarray, p: np.ndarray, bins=(0, .3, .4, .5, .6, .7, .8, .9, 1.0001)) -> list[dict]:
    """Обещанный шанс против доли понравившихся по корзинам обещания."""
    b = pd.cut(p, list(bins), right=False)
    t = pd.DataFrame({"b": b, "y": y, "p": p}).groupby("b", observed=True).agg(
        n=("y", "size"), promised=("p", "mean"), actual=("y", "mean"))
    return [{"bin": str(i), "n": int(r.n), "promised": float(r.promised), "actual": float(r.actual)}
            for i, r in t.iterrows()]


def with_taste(obs_fit: pd.DataFrame, obs: pd.DataFrame, prior: float, p0: float, stars: int = 4) -> np.ndarray:
    """Сравнение (только замер): место + щедрость + прогноз оценки модели вкуса. В шанс не входит — книга ниже
    в списке могла бы получить больший процент."""
    def X(d):
        own = (_k(d, stars).to_numpy() + prior * p0) / (d.n_rated.to_numpy() + prior)
        return np.column_stack([np.ones(len(d)), np.log10(d.pct.to_numpy()), _logit(own), d.taste.to_numpy() - 3.0])
    w = _logistic(X(obs_fit), (obs_fit.rating >= stars).to_numpy(dtype=np.float64))
    return 1 / (1 + np.exp(-X(obs) @ w))


def within_person_auc(obs: pd.DataFrame, p: np.ndarray, stars: int = 4) -> float:
    """Среднее по людям: скрытая книга с оценкой ≥ stars получила обещание выше остальных (0.5 — монетка)."""
    d = obs.assign(p=p, y=obs.rating >= stars)
    vals = [_auc(g.p[g.y].to_numpy(), g.p[~g.y].to_numpy()) for _, g in d.groupby("user_id") if 0 < g.y.sum() < len(g)]
    return float(np.mean(vals)) if vals else float("nan")


def model_class(name: str):
    """Класс сохранённой модели по имени папки: модели стенда (`evaluate.MODELS`) и слои (`layers`)."""
    if name == "layers":
        from booksengine.model.layers import Layers
        return Layers
    from booksengine.model.evaluate import MODELS
    return MODELS[name][0]


def calibrate(name: str, *, ratings_path: Path, split_dir: Path, models_dir: Path, eval_dir: Path,
              stars: int | None = None) -> dict:
    """Учим шанс на валидации для сохранённой модели models_dir/<name>, проверяем на тесте.
    Пишет models_dir/<name>/chance.json и eval_dir/chance_<name>.json. stars — порог (по умолчанию `STARS`)."""
    stars = STARS.get(name, 4) if stars is None else int(stars)
    from booksengine.model.evaluate import load_eval_holdout
    from booksengine.model.matrix import catalog_works
    model_dir = models_dir / name
    model = model_class(name).load(model_dir)
    work_ids = catalog_works(ratings_path)
    val = observations(model, load_eval_holdout(ratings_path, split_dir, "val", work_ids))
    test = observations(model, load_eval_holdout(ratings_path, split_dir, "test", work_ids))
    users = val.drop_duplicates("user_id")
    p0 = float((_k(users, stars) / users.n_rated).mean())
    c, losses = fit(val, p0, stars=stars)
    c.model_fp = fingerprint(model_dir)
    c.save(model_dir / "chance.json")
    y = (test.rating >= stars).to_numpy(dtype=np.float64)
    p = c.predict(test.pct, _k(test, stars), test.n_rated)
    variants = {f"константа p({c.label()}) валидации": np.full(len(y), (val.rating >= stars).mean()),
                "только место в рейтинге": rank_only(val, test, stars),
                "только щедрость человека": person_only(val, test, c.prior, p0, stars),
                "место + щедрость": p}
    if "taste" in val.columns:
        variants["место + щедрость + прогноз вкуса (только замер)"] = with_taste(val, test, c.prior, p0, stars)
    out = {"model": name, "stars": stars, "chance": asdict(c),
           "prior_log_loss_val": {str(k): v for k, v in losses.items()},
           "n_val": len(val), "n_test": len(test), "test_like_share": float(y.mean()),
           "test": {k: {"log_loss": log_loss(y, v), "brier": float(((v - y) ** 2).mean()),
                        "auc": _auc(v[y == 1], v[y == 0]), "auc_within_person": within_person_auc(test, v, stars)}
                    for k, v in variants.items()},
           "reliability": reliability(y, p),
           "by_bucket": {b: reliability(y[test.bucket == b], p[test.bucket == b]) for b in sorted(test.bucket.unique())}}
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / f"chance_{name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return out


def _bin_name(b: str) -> str:
    lo, hi = (float(x) for x in b.strip("[)").split(","))
    return f"{lo:.0%}–{min(hi, 1.0):.0%}"


def report(out: dict) -> str:
    """Отчёт калибровки: насколько шанс совпадает с долей понравившихся и насколько он различает книги."""
    t = out["test"]
    main = t["место + щедрость"]
    n = f"{out['n_test']:,}".replace(",", " ")
    stars = out.get("stars", 4)
    what = "пятёрки (5★)" if stars == 5 else "«понравится» (4–5★)"
    got = "5★ на деле" if stars == 5 else "понравилось на деле"
    lines = [f"# Шанс {what}: models/{out['model']} — тест, {n} скрытых оценок", "",
             f"**Шанс различает книги с AUC {main['auc']:.2f}** (внутри одного человека — "
             f"{main['auc_within_person']:.2f}); обещанное расходится с реальным не больше чем на "
             f"{max(abs(r['promised'] - r['actual']) for r in out['reliability']):.0%}.", "",
             f"| обещано | {got} | оценок |", "|---|---|---|"]
    lines += [f"| {_bin_name(r['bin'])} (в среднем {r['promised']:.0%}) | {r['actual']:.0%} | {r['n']:,} |".replace(",", " ")
              for r in out["reliability"]]
    lines += ["", "Из чего шанс и что даёт каждая часть:", "", "| что учитывает | AUC | внутри человека |", "|---|---|---|"]
    lines += [f"| {k}{' ← шанс' if k == 'место + щедрость' else ''} | {v['auc']:.3f} | {v['auc_within_person']:.3f} |"
              for k, v in t.items()]
    mark = "5★" if stars == 5 else "4–5★"
    lines += ["", f"**Как читать.** Шанс — вероятность, что человек поставит книге {mark}, по месту книги в его рейтинге "
              f"и его щедрости (какую долю прочитанного он оценивает на {mark}). Хорошо откалиброван, если «обещано» ≈ "
              f"«{got}». AUC — как часто книга с оценкой {mark} получает шанс выше остальных: 0.5 — наугад, 1 — "
              "всегда; «внутри человека» — только между книгами одного человека, без вклада его щедрости."]
    return "\n".join(lines) + "\n"


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Шанс, что понравившаяся книга получила обещание выше непонравившейся (0.5 — монетка)."""
    r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
