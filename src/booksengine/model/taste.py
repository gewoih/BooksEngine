"""Модель вкуса (TODO п. 37, шаг 1): предсказывает саму оценку 1–5, а не факт «прочитал».

Оценка ≈ μ + b_книги + b_человека + x_человека·y_книги (разложение матрицы оценок со сдвигами, Koren, Bell,
Volinsky 2009). Учится только на поставленных оценках — «не читал» не значит «плохо». Обучение — чередующиеся
гребневые регрессии с регуляризацией λ·n (ALS-WR, Zhou et al. 2008): n — число оценок у человека или книги.

Новый человек (fold-in) — та же регрессия по его оценкам при готовых книгах: θ = [b_человека, x] решает
(ZᵀZ + λ·n·I)·θ = Zᵀ·d, Z = [1, y_i] оценённых книг, d_i = r_i − μ − b_i — «насколько выше или ниже, чем книгу
обычно оценивают». Личная шкала — b_человека. Прогноз линеен по d: разбор на вклады оценённых книг точный.
"""
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from booksengine.model import metrics
from booksengine.model.base import read_params, write_params
from booksengine.model.matrix import RatingMatrix, load_train
from booksengine.model.split import BUCKET_ORDER, SEED

GRID = [(32, 0.05), (32, 0.15), (64, 0.05), (64, 0.15)]   # (factors, reg)


def solve_rows(M: sp.csr_matrix, F: np.ndarray, off: np.ndarray, mu: float, reg: float) -> tuple[np.ndarray, float]:
    """Для каждой строки M: θ = argmin Σ (r − μ − off_c − F_c·θ)² + reg·n·|θ|². Возвращает θ и сумму квадратов
    остатков (для контроля обучения). Строка без оценок — нули."""
    K = F.shape[1]
    out = np.zeros((M.shape[0], K))
    eye = np.eye(K)
    sse = 0.0
    for u in range(M.shape[0]):
        s, e = M.indptr[u], M.indptr[u + 1]
        if s == e:
            continue
        c = M.indices[s:e]
        Z = F[c].astype(np.float64)
        d = M.data[s:e].astype(np.float64) - mu - off[c]
        th = np.linalg.solve(Z.T @ Z + reg * (e - s) * eye, Z.T @ d)
        out[u] = th
        sse += float(((d - Z @ th) ** 2).sum())
    return out, sse


class Taste:
    name = "taste"

    def __init__(self, factors: int = 64, reg: float = 0.05, iterations: int = 10, seed: int = SEED):
        self.factors, self.reg, self.iterations, self.seed = int(factors), float(reg), int(iterations), int(seed)
        self.mu = 0.0
        self.item_factors: np.ndarray | None = None   # книги × factors
        self.item_bias: np.ndarray | None = None

    def _F(self) -> np.ndarray:
        return np.column_stack([np.ones(len(self.item_bias)), self.item_factors]).astype(np.float32)

    def fit(self, train: RatingMatrix, log=print) -> None:
        from threadpoolctl import threadpool_limits
        X = train.X.tocsr()
        XT = X.T.tocsr()
        rng = np.random.default_rng(self.seed)
        self.mu = float(np.round(X.data.astype(np.float64).mean(), 6))
        self.item_factors = rng.normal(0, 0.1 / np.sqrt(self.factors), (X.shape[1], self.factors)).astype(np.float32)
        self.item_bias = np.zeros(X.shape[1], dtype=np.float32)
        with threadpool_limits(1, "blas"):  # тысячи маленьких систем: потоки BLAS только мешают
            for it in range(self.iterations):
                t0 = time.perf_counter()
                U, sse = solve_rows(X, self._F(), self.item_bias, self.mu, self.reg)
                Fu = np.column_stack([np.ones(X.shape[0]), U[:, 1:]]).astype(np.float32)
                V, _ = solve_rows(XT, Fu, U[:, 0], self.mu, self.reg)
                self.item_bias, self.item_factors = V[:, 0].astype(np.float32), V[:, 1:].astype(np.float32)
                log(f"  вкус {self.factors}/{self.reg}: итерация {it + 1}, RMSE обучения до шага книг "
                    f"{np.sqrt(sse / X.nnz):.4f}, {time.perf_counter() - t0:.0f} с")

    def configure(self, **score_params) -> None:
        if score_params:
            raise ValueError(f"у модели вкуса нет настроек выдачи: {score_params}")

    def fold_in(self, inputs: sp.csr_matrix) -> np.ndarray:
        """θ = [b_человека, x] по оценкам входа."""
        return solve_rows(inputs.tocsr(), self._F(), self.item_bias, self.mu, self.reg)[0]

    def fold_in_system(self, cols: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Система одного человека A·θ = Zᵀ·d (для объяснения): A, Z, d."""
        Z = self._F()[cols].astype(np.float64)
        d = np.asarray(r, dtype=np.float64) - self.mu - self.item_bias[cols]
        return Z.T @ Z + self.reg * len(cols) * np.eye(Z.shape[1]), Z, d

    def score(self, inputs: sp.csr_matrix) -> np.ndarray:
        """Предсказанная оценка каждой книги ядра."""
        theta = self.fold_in(inputs)
        return (self.mu + self.item_bias[None, :] + theta @ self._F().T.astype(np.float64)).astype(np.float32)

    def save(self, path: Path) -> None:
        write_params(path, {"factors": self.factors, "reg": self.reg, "iterations": self.iterations,
                            "seed": self.seed, "mu": self.mu})
        np.savez(path / "items.npz", factors=self.item_factors, bias=self.item_bias)

    @classmethod
    def load(cls, path: Path) -> "Taste":
        p = read_params(path)
        m = cls(p["factors"], p["reg"], p["iterations"], p["seed"])
        m.mu = p["mu"]
        d = np.load(path / "items.npz")
        m.item_factors, m.item_bias = d["factors"], d["bias"]
        return m


def star_stats(X: sp.csr_matrix) -> dict:
    """Как толпа пользуется шкалой: доля каждой звезды, доля людей без 1★ и без 1–2★, средняя доля 4–5★ у человека."""
    r = metrics.rounded(X.data).astype(int)
    shares = np.bincount(r, minlength=6)[1:] / len(r)
    rows = np.repeat(np.arange(X.shape[0]), np.diff(X.indptr))
    has1 = np.bincount(rows[r == 1], minlength=X.shape[0]) > 0
    has12 = np.bincount(rows[r <= 2], minlength=X.shape[0]) > 0
    n = np.maximum(np.diff(X.indptr), 1)
    like = np.bincount(rows[r >= 4], minlength=X.shape[0]) / n
    return {"star_share": {str(s): float(v) for s, v in zip(range(1, 6), shares)},
            "users_without_1": float(1 - has1.mean()), "users_without_1_2": float(1 - has12.mean()),
            "user_like_share_quartiles": [float(q) for q in np.quantile(like, [0.25, 0.5, 0.75])]}


def _rmse(model, hold) -> float:
    err, n = 0.0, 0
    for s in range(0, len(hold.user_ids), 500):
        sc = model.score(hold.inputs[s:s + 500])
        for i in range(sc.shape[0]):
            c, r = hold.hidden_cols[s + i], hold.hidden_ratings[s + i]
            err += float(((sc[i, c] - r) ** 2).sum())
            n += len(c)
    return float(np.sqrt(err / max(n, 1)))


def tune(*, ratings_path: Path, split_dir: Path, models_dir: Path, eval_dir: Path, grid=GRID,
         iterations: int = 10) -> dict:
    """Перебор на валидации по личной точности (TODO п. 37: вкус судится ею, не NDCG); лучшая → models_dir/taste.

    Дополняет прежний перебор (eval_dir/taste_val.json): уже посчитанные варианты не повторяются, а сохранённая
    модель заменяется, только если новый вариант лучше всех прежних."""
    from booksengine.model import taste_gap
    from booksengine.model.evaluate import load_eval_holdout
    train = load_train(ratings_path, split_dir / "holdout_users.parquet")
    hold = load_eval_holdout(ratings_path, split_dir, "val", train.work_ids)
    allowed = np.ones(train.X.shape[1], dtype=bool)
    ease_cols = models_dir / "ease" / "top_cols.npy"
    if ease_cols.exists():  # как в taste-gap: сравнение на книгах EASE
        allowed[:] = False
        allowed[np.load(ease_cols)] = True
    path = eval_dir / "taste_val.json"
    prev = json.loads(path.read_text())["results"] if path.exists() and (models_dir / "taste").exists() else []
    out = {"stars": star_stats(train.X), "results": prev}
    eval_dir.mkdir(parents=True, exist_ok=True)
    best = max((r["personal_auc"]["all"] for r in prev), default=-1.0)
    done = {(r["factors"], r["reg"], r["iterations"]) for r in prev}
    for factors, reg in grid:
        if (factors, reg, iterations) in done:
            print(f"вкус {factors}/{reg}: уже посчитан, пропуск", flush=True)
            continue
        m = Taste(factors, reg, iterations)
        t0 = time.perf_counter()
        m.fit(train)
        fit_s = round(time.perf_counter() - t0, 1)
        d = taste_gap.measure(m, hold, allowed)
        auc = {"all": float(d.auc.mean())} | {b: float(d.auc[d.bucket == b].mean())
                                              for b in BUCKET_ORDER if (d.bucket == b).any()}
        res = {"factors": factors, "reg": reg, "iterations": iterations, "fit_seconds": fit_s,
               "personal_auc": auc, "rmse_hidden": _rmse(m, hold)}
        out["results"].append(res)
        print(f"вкус {factors}/{reg}: личная точность {auc['all']:.4f}, RMSE скрытых {res['rmse_hidden']:.4f}, "
              f"обучение {fit_s} с", flush=True)
        path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        if auc["all"] > best:
            best = auc["all"]
            m.save(models_dir / "taste")
    return out


def report(out: dict) -> str:
    s = out["stars"]
    q = s["user_like_share_quartiles"]
    lines = ["# Модель вкуса: перебор на валидации (TODO п. 37, шаг 1)", "",
             "Как толпа пользуется шкалой (обучение): " + ", ".join(f"{k}★ — {v:.1%}" for k, v in s["star_share"].items())
             + f". Ни разу не ставили 1★ — {s['users_without_1']:.0%} людей, ни 1★, ни 2★ — "
               f"{s['users_without_1_2']:.0%}. Доля 4–5★ у человека: четверть людей ниже {q[0]:.0%}, "
               f"половина ниже {q[1]:.0%}, три четверти ниже {q[2]:.0%}.", "",
             "| размер | регуляризация | личная точность, все | " + " | ".join(
                 k for k in out["results"][0]["personal_auc"] if k != "all") + " | RMSE скрытых | обучение, с |",
             "|---|---|---|" + "---|" * (len(out["results"][0]["personal_auc"]) - 1) + "---|---|"]
    for r in out["results"]:
        a = r["personal_auc"]
        lines.append(f"| {r['factors']} | {r['reg']} | {a['all']:.4f} | "
                     + " | ".join(f"{v:.4f}" for k, v in a.items() if k != "all")
                     + f" | {r['rmse_hidden']:.4f} | {r['fit_seconds']:.0f} |")
    lines += ["", "Лучшая по личной точности сохранена в models/taste. Сравнение со смесью и средней оценкой книги — "
                  "`booksengine taste-gap` (шаг 1 пройден, если вкус выше обеих)."]
    return "\n".join(lines) + "\n"
