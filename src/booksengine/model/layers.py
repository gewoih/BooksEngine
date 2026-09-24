"""Шаг 2 п. 37: толпа выбирает, вкус упорядочивает (docs/resheniya.md, «слой вкуса»).

Балл = z(толпа) + g·z(вкус) по 30 000 книг EASE; z — нормировка по книгам человека, как в смеси. Толпа —
нынешняя смесь (`mix`: вход EASE по оценке, ALS с негативом «≤ 2») или «только прочитал» (`read`: вход EASE 1
у любой оценки, ALS без негатива) — тогда звёзды толкует один слой вкуса. `cutoff` — книги только из первых N
толпы (вкус лишь переставляет их), None — вкус может поднять любую книгу EASE.

Выбор (правило журнала решений): наибольшая личная точность среди вариантов, у которых NDCG@20 не хуже
нынешнего и Low@20 не выше — парная разница с 95% интервалом, касающимся нуля или лучше. Допуск «падение ≤ 0.005»
убран 2026-09-24: вес 1.5 прошёл по нему валидацию (−0.004) и не прошёл тест (−0.006 [−0.007; −0.005]).
Рядом — страховки глазами из урока п. 23 и серий: доля поздних томов (#2 и дальше) неначатых серий в топ-20
и сколько книг одного автора в топ-20.
"""
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from booksengine.model import metrics
from booksengine.model.base import fingerprint, read_params
from booksengine.model.mix import Mix, _z
from booksengine.model.split import BUCKET_ORDER
from booksengine.model.taste import Taste
from booksengine.model.taste_gap import personal_auc

# «прочитал» и отсечение 300 замерены 2026-09-24 и отклонены (docs/resheniya.md, «слой вкуса, шаг 2»):
# толпа без звёзд — Low@20 8.7% → 11–14%, отсечение не даёт личной точности. Остаются в коде для повторного замера.
CROWDS = ("mix",)
WEIGHTS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5)
CUTOFFS = (None,)
REFERENCE = ("mix", 0.0, None)     # нынешняя выдача
CHECKED = ("ndcg20", "low20", "auc")


@dataclass(frozen=True)
class Variant:
    crowd: str = "mix"
    taste_weight: float = 0.0
    cutoff: int | None = None

    def label(self) -> str:
        c = "толпа по оценкам" if self.crowd == "mix" else "толпа «прочитал»"
        return f"{c}, вкус {self.taste_weight:g}" + (f", из первых {self.cutoff}" if self.cutoff else "")


class Layers:
    """Толпа (готовая смесь) + вкус (готовая модель вкуса); своего обучения нет.

    Как модель выдачи (`load` из models/layers, `score`) — выбранный вариант из params.json. Там же отпечатки
    смеси и вкуса: если компонент переобучен, загрузка падает — иначе вес вкуса и шанс молча стали бы чужими."""
    name = "layers"

    def __init__(self, mix: Mix, taste: Taste, variant: Variant | None = None):
        self.mix, self.taste = mix, taste
        self.variant = variant or Variant(*REFERENCE)
        self._mix_cfg = (mix.als_weight, mix.ease_input, mix.als.neg_rule, mix.als.neg_weight)

    @classmethod
    def from_models(cls, models_dir: Path) -> "Layers":
        return cls(Mix.load(models_dir / "mix"), Taste.load(models_dir / "taste"))

    @staticmethod
    def component_fingerprints(models_dir: Path) -> dict[str, str]:
        return {n: fingerprint(models_dir / n) for n in ("mix", "taste")}

    @classmethod
    def load(cls, path: Path) -> "Layers":
        """models/layers → выбранный вариант; смесь и вкус — из соседних папок."""
        p = read_params(path)
        if p.get("components") != cls.component_fingerprints(path.parent):
            raise ValueError(f"{path}: смесь или вкус переобучены после выбора веса — пересоберите "
                             "`booksengine layers val` и `booksengine calibrate layers`")
        m = cls.from_models(path.parent)
        m.variant = Variant(**p["variant"])
        return m

    def configure(self, **score_params) -> None:
        if score_params:
            raise ValueError(f"у слоёв нет настроек выдачи: {score_params}")

    def score(self, inputs: sp.csr_matrix, dnf: sp.csr_matrix | None = None) -> np.ndarray:
        """Балл выбранного варианта по всем книгам ядра; вне EASE — −∞ (как у смеси). Без исключения входа."""
        top = self.mix.ease.top_cols
        excl = np.zeros((inputs.shape[0], len(top)), dtype=bool)
        s = self.combine(self.crowd(self.variant.crowd, inputs, dnf), self.taste_z(inputs), excl, self.variant)
        out = np.full(inputs.shape, -np.inf, dtype=np.float32)
        out[:, top] = s
        return out

    def taste_prediction(self, inputs: sp.csr_matrix) -> np.ndarray:
        """Прогноз оценки 1–5 модели вкуса по всем книгам ядра — признак шанса «понравится»."""
        return self.taste.score(inputs)

    def crowd(self, kind: str, inputs: sp.csr_matrix, dnf: sp.csr_matrix | None = None) -> np.ndarray:
        """z-балл толпы по книгам EASE (строки × 30 000)."""
        w, ein, rule, beta = self._mix_cfg
        if kind == "read":
            self.mix.configure(als_weight=w, ease_input=(1.0,) * 5)
            self.mix.als.configure("none", 0.0)
        try:
            return self.mix.score(inputs, dnf)[:, self.mix.ease.top_cols].astype(np.float64)
        finally:
            self.mix.configure(als_weight=w, ease_input=ein)
            self.mix.als.configure(rule, beta)

    def taste_z(self, inputs: sp.csr_matrix) -> np.ndarray:
        return _z(self.taste.score(inputs)[:, self.mix.ease.top_cols].astype(np.float64))

    @staticmethod
    def combine(crowd: np.ndarray, taste: np.ndarray, excl: np.ndarray, v: Variant) -> np.ndarray:
        """Итоговый балл по книгам EASE; excl — булева маска «не советовать» (вход, начатые серии)."""
        s = crowd + v.taste_weight * taste
        s[excl] = -np.inf
        if v.cutoff:
            c = np.where(excl, -np.inf, crowd)
            k = min(v.cutoff, c.shape[1]) - 1
            thr = np.take_along_axis(c, np.argpartition(-c, k, axis=1)[:, k:k + 1], axis=1)
            s[c < thr] = -np.inf
        return s


def book_marks(info: pd.DataFrame, top_cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """По книгам EASE: поздний том серии (#2 и дальше) и основной автор (−1 — неизвестен)."""
    sno = pd.to_numeric(info.series_no.iloc[top_cols], errors="coerce").to_numpy()
    later = np.nan_to_num(sno, nan=0.0) >= 2
    author = info.author_id.iloc[top_cols].fillna(-1).to_numpy(dtype=np.int64)
    return later, author


def evaluate(layers: Layers, hold, info: pd.DataFrame, variants: list[Variant], batch: int = 500) -> dict:
    """Метрики по людям для каждого варианта (баллы толпы и вкуса считаются один раз на пачку)."""
    top = layers.mix.ease.top_cols
    pos_of = np.full(hold.inputs.shape[1], -1)
    pos_of[top] = np.arange(len(top))
    later, author = book_marks(info, top)
    exclude = (hold.inputs if hold.exclude is None else hold.exclude).tocsr()
    rows = {v: [] for v in variants}
    for s in range(0, len(hold.user_ids), batch):
        X = hold.inputs[s:s + batch]
        crowds = {k: layers.crowd(k, X) for k in {v.crowd for v in variants}}
        taste = layers.taste_z(X)
        excl = (exclude[s:s + batch][:, top].toarray() != 0)
        for v in variants:
            sc = layers.combine(crowds[v.crowd], taste, excl, v)
            k = min(metrics.K, sc.shape[1])
            part = np.argpartition(-sc, k - 1, axis=1)[:, :k]
            vals = np.take_along_axis(sc, part, axis=1)
            order = np.lexsort((part, -vals), axis=1)
            tp = np.take_along_axis(part, order, axis=1)
            ok = np.isfinite(np.take_along_axis(vals, order, axis=1))
            for i in range(sc.shape[0]):
                u = s + i
                t = tp[i][ok[i]]
                m = metrics.user_metrics(np.where(ok[i], top[tp[i]], -1), hold.hidden_cols[u], hold.hidden_ratings[u])
                hp = pos_of[hold.hidden_cols[u]]
                keep = hp >= 0
                a = author[t]
                a = a[a >= 0]
                m.update(auc=personal_auc(sc[i, hp[keep]], hold.hidden_ratings[u][keep]),
                         later=float(later[t].mean()) if len(t) else np.nan,
                         max_author=float(np.bincount(np.unique(a, return_inverse=True)[1]).max()) if len(a) else 0.0,
                         user_id=hold.user_ids[u], bucket=hold.buckets[u])
                rows[v].append(m)
    return {v: pd.DataFrame(r) for v, r in rows.items()}


def _boot(x: np.ndarray, rng, n_boot: int) -> dict:
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    b = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return {"mean": float(x.mean()), "lo": float(np.quantile(b, 0.025)), "hi": float(np.quantile(b, 0.975)),
            "n": int(len(x))}


def summarize(per_user: dict[Variant, pd.DataFrame], reference: Variant, n_boot: int = 1000) -> list[dict]:
    ref = per_user[reference].set_index("user_id")
    out = []
    for v, d in per_user.items():
        d = d.set_index("user_id")
        parts = [("all", d.index)] + [(b, d.index[d.bucket == b]) for b in BUCKET_ORDER if (d.bucket == b).any()]
        row = {"variant": asdict(v), "label": v.label(), "groups": {}}
        for name, idx in parts:
            g = {}
            for m in CHECKED + ("recall20", "later", "max_author"):
                g[m] = _boot(d.loc[idx, m].to_numpy(dtype=np.float64), np.random.default_rng(0), n_boot)
            for m in CHECKED:
                diff = (d.loc[idx, m] - ref.loc[idx, m]).to_numpy(dtype=np.float64)
                g[f"{m}_diff"] = _boot(diff, np.random.default_rng(0), n_boot)
            row["groups"][name] = g
        out.append(row)
    return out


def allowed(row: dict) -> bool:
    """NDCG@20 не значимо хуже нынешнего и Low@20 не значимо выше: 95% интервал парной разницы касается нуля
    или лежит на лучшей стороне (журнал решений, «слой вкуса»)."""
    g = row["groups"]["all"]
    return g["ndcg20_diff"]["hi"] >= 0 and g["low20_diff"]["lo"] <= 0


def choose(summary: list[dict]) -> dict:
    return max((r for r in summary if allowed(r)), key=lambda r: r["groups"]["all"]["auc"]["mean"])


def _hold(ratings_path: Path, split_dir: Path, stage: str, work_ids: np.ndarray):
    from booksengine.model.evaluate import load_eval_holdout
    return load_eval_holdout(ratings_path, split_dir, stage, work_ids)


def run(stage: str, *, clean_dir: Path, split_dir: Path, models_dir: Path, eval_dir: Path) -> dict:
    """val — перебор и выбор (→ models_dir/layers/params.json); test — один замер выбранного против нынешнего."""
    from booksengine.model.filters import work_info
    from booksengine.model.matrix import catalog_works
    ratings_path = clean_dir / "ratings.parquet"
    work_ids = catalog_works(ratings_path)
    layers = Layers.from_models(models_dir)
    info = work_info(clean_dir, work_ids)
    hold = _hold(ratings_path, split_dir, stage, work_ids)
    ref = Variant(*REFERENCE)
    if stage == "val":
        variants = [Variant(c, w, n) for c in CROWDS for n in CUTOFFS for w in WEIGHTS]
    else:
        chosen = Variant(**json.loads((models_dir / "layers" / "params.json").read_text())["variant"])
        variants = list(dict.fromkeys([ref, chosen]))
    summary = summarize(evaluate(layers, hold, info, variants), ref)
    res = {"stage": stage, "n_users": int(len(hold.user_ids)), "summary": summary}
    if stage == "val":
        best = choose(summary)
        res["chosen"] = best["variant"]
        (models_dir / "layers").mkdir(parents=True, exist_ok=True)
        (models_dir / "layers" / "params.json").write_text(json.dumps(
            {"variant": best["variant"], "label": best["label"],
             "components": Layers.component_fingerprints(models_dir)}, ensure_ascii=False, indent=1))
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / f"layers_{stage}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return res


def _f(x: dict, signed=False, pct=False) -> str:
    if x["mean"] is None:
        return "—"
    if pct:
        return f"{x['mean']:+.1%}" if signed else f"{x['mean']:.1%}"
    return f"{x['mean']:+.3f}" if signed else f"{x['mean']:.3f}"


def _ci(x: dict) -> str:
    return "" if x["mean"] is None else f" [{x['lo']:+.3f}; {x['hi']:+.3f}]"


def report(res: dict) -> str:
    chosen = res.get("chosen")
    lines = [f"# Слои «толпа + вкус» ({'валидация' if res['stage'] == 'val' else 'тест'}, TODO п. 37, шаг 2)", "",
             f"{res['n_users']} человек. Разницы — с нынешней выдачей («толпа по оценкам, вкус 0») на тех же людях, "
             f"в скобках 95% интервал. Допустим — NDCG@20 не значимо хуже и Low@20 не значимо выше (интервал разницы "
             f"касается нуля или лучше); "
             f"из допустимых выбирается наибольшая личная точность.", "",
             "| вариант | личная точность | разница | NDCG@20 | разница | Low@20 | разница | поздние тома | "
             "книг одного автора (макс.) | допустим |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["summary"]:
        g = r["groups"]["all"]
        mark = " **← выбран**" if chosen and r["variant"] == chosen else ""
        lines.append(
            f"| {r['label']}{mark} | {_f(g['auc'])} | {_f(g['auc_diff'], True)}{_ci(g['auc_diff'])} | "
            f"{_f(g['ndcg20'])} | {_f(g['ndcg20_diff'], True)}{_ci(g['ndcg20_diff'])} | {_f(g['low20'], pct=True)} | "
            f"{_f(g['low20_diff'], True, pct=True)} | {_f(g['later'], pct=True)} | {g['max_author']['mean']:.2f} | "
            f"{'да' if allowed(r) else 'нет'} |")
    groups = [b for b in BUCKET_ORDER if b in res["summary"][0]["groups"]]
    lines += ["", "По группам (личная точность / NDCG@20 / Low@20):", "",
              "| вариант | " + " | ".join(groups) + " |", "|---|" + "---|" * len(groups)]
    for r in res["summary"]:
        if res["stage"] == "val" and not (r["variant"] == chosen or tuple(r["variant"].values()) == REFERENCE):
            continue
        cells = [f"{_f(r['groups'][b]['auc'])} / {_f(r['groups'][b]['ndcg20'])} / "
                 f"{_f(r['groups'][b]['low20'], pct=True)}" for b in groups]
        lines.append(f"| {r['label']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def profiles(*, clean_dir: Path, models_dir: Path, profiles_dir: Path, top: int = 20) -> str:
    """Топ-20 каждого профиля profiles/*.csv: нынешняя выдача и выбранный вариант рядом (правило CLAUDE.md)."""
    from booksengine.model.filters import RatedFilter, work_info
    from booksengine.model.matrix import catalog_works
    from booksengine.model.series import SeriesIndex, exclusion
    from booksengine.recommend import read_profile
    work_ids = catalog_works(clean_dir / "ratings.parquet")
    layers = Layers.from_models(models_dir)
    info = work_info(clean_dir, work_ids)
    chosen = Variant(**json.loads((models_dir / "layers" / "params.json").read_text())["variant"])
    ref = Variant(*REFERENCE)
    top_cols = layers.mix.ease.top_cols
    series = SeriesIndex(info.title.tolist())
    later, _ = book_marks(info, top_cols)
    out = [f"# Топ-{top} на профилях: нынешняя выдача и «{chosen.label()}» (TODO п. 37, шаг 2)", "",
           "★ — поздний том (#2 и дальше) неначатой серии. Без шанса и объяснения — они будут на шаге 3."]
    for csv in sorted(profiles_dir.glob("*.csv")):
        prof = read_profile(csv, work_ids, clean_dir)
        if prof.x.nnz == 0:
            continue
        excl = exclusion(prof.x, series)[:, top_cols].toarray() != 0
        crowds = {k: layers.crowd(k, prof.x, prof.dnf) for k in {ref.crowd, chosen.crowd}}
        taste = layers.taste_z(prof.x)
        rated = RatedFilter(info, prof.x.indices)
        lists = {}
        for v in (ref, chosen):
            sc = layers.combine(crowds[v.crowd], taste, excl, v)[0]
            order = [p for p in np.argsort(-sc, kind="stable") if np.isfinite(sc[p])]
            lists[v] = [p for p in order if not rated.is_rated_already(int(top_cols[p]))][:top]
        before = set(lists[ref])
        out += ["", f"## {csv.stem} ({prof.x.nnz} оценок)", "",
                f"| # | нынешняя | {chosen.label()} |", "|---|---|---|"]
        for i in range(top):
            cells = []
            for v in (ref, chosen):
                if i >= len(lists[v]):
                    cells.append("")
                    continue
                p = lists[v][i]
                c = int(top_cols[p])
                new = " (новая)" if v == chosen and p not in before else ""
                cells.append(f"{info.title[c]} — {info.author[c] or '?'}{' ★' if later[p] else ''}{new}")
            out.append(f"| {i + 1} | {cells[0]} | {cells[1]} |")
        out.append(f"\nСовпадает книг: {len(before & set(lists[chosen]))} из {top}.")
    return "\n".join(out) + "\n"
