"""Шаг 2 п. 37: толпа выбирает, вкус упорядочивает (docs/resheniya.md, «слой вкуса»).

Балл = z(толпа) + g·z(вкус) по 30 000 книг EASE; z — нормировка по книгам человека, как в смеси. Толпа —
нынешняя смесь (`mix`: вход EASE по оценке, ALS с негативом «≤ 2») или «только прочитал» (`read`: вход EASE 1
у любой оценки, ALS без негатива) — тогда звёзды толкует один слой вкуса. `cutoff` — книги только из первых N
толпы (вкус лишь переставляет их), None — вкус может поднять любую книгу EASE.

Выбор (журнал решений, «ценность топа»): наибольшая ценность топа — сумма (оценка − 3) скрытых книг в топ-20,
5★ = +2 … 1★ = −2. Пользователь: цель — средняя оценка рекомендаций; сумма, а не среднее, — иначе выиграл бы
топ с одной осторожной угаданной книгой. Раньше выбирали по личной точности при страховках NDCG и Low@20 —
они остались в отчёте для сведения; рядом — выбор при 5★ = +3 (ценность пятёрки — договорённость).
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
from booksengine.model.taste_gap import graded_auc, personal_auc

# «прочитал» и отсечение 300 замерены 2026-09-24 и отклонены (docs/resheniya.md, «слой вкуса, шаг 2»):
# толпа без звёзд — Low@20 8.7% → 11–14%, отсечение не даёт личной точности. Остаются в коде для повторного замера.
CROWDS = ("mix",)
WEIGHTS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0)
CUTOFFS = (None,)
REFERENCE = ("mix", 0.0, None)     # нынешняя выдача
CHECKED = ("value20", "ndcg20", "low20", "auc", "gauc")
# ценность пятёрки против четвёрки (4★ = 1): 1 — «5 = 4» (обычная личная точность), 2 — как в NDCG, 3 — втрое.
# Выбор идёт по 2; отчёт показывает, какой вес вкуса выбрали бы при каждой — чувствительность к договорённости.
FIVE_VALUES = {"value20": 2.0, "value20_5": 3.0}
LIKE_ALS = (0.0, 0.25, 0.5, 0.75)   # вес ALS в толпе «ценность»: ALS находит прочитанное, «ценность» — отсеивает плохое


@dataclass(frozen=True)
class Variant:
    crowd: str = "mix"
    taste_weight: float = 0.0
    cutoff: int | None = None
    als_weight: float = 0.5      # только для толпы «ценность»: вес ALS в её смеси (у «mix» — сохранённый, 0.5)

    def label(self) -> str:
        if self.crowd == "like":
            c = f"толпа «ценность», ALS {self.als_weight:g}"
        else:
            c = {"mix": "толпа по оценкам", "read": "толпа «прочитал»"}[self.crowd]
        return f"{c}, вкус {self.taste_weight:g}" + (f", из первых {self.cutoff}" if self.cutoff else "")


class Layers:
    """Толпа (готовая смесь) + вкус (готовая модель вкуса); своего обучения нет.

    Как модель выдачи (`load` из models/layers, `score`) — выбранный вариант из params.json. Там же отпечатки
    смеси и вкуса: если компонент переобучен, загрузка падает — иначе вес вкуса и шанс молча стали бы чужими."""
    name = "layers"

    def __init__(self, mix: Mix, taste: Taste, variant: Variant | None = None, like_mix: Mix | None = None):
        self.mix, self.taste, self.like_mix = mix, taste, like_mix
        self.variant = variant or Variant(*REFERENCE)
        self._mix_cfg = (mix.als_weight, mix.ease_input, mix.als.neg_rule, mix.als.neg_weight)
        if like_mix is not None and not np.array_equal(like_mix.ease.top_cols, mix.ease.top_cols):
            raise ValueError("EASE «ценность» обучен на других 30 000 книгах, чем EASE смеси")

    @classmethod
    def from_models(cls, models_dir: Path) -> "Layers":
        like = models_dir / "mix_like"
        return cls(Mix.load(models_dir / "mix"), Taste.load(models_dir / "taste"),
                   like_mix=Mix.load(like) if (like / "params.json").exists() else None)

    def crowds(self) -> tuple[str, ...]:
        """Толпы для перебора: нынешняя и, если обучена (TODO п. 38), «ценность» со смесью ALS и без."""
        return CROWDS + (("like",) if self.like_mix is not None else ())

    @staticmethod
    def component_fingerprints(models_dir: Path) -> dict[str, str]:
        names = ("mix", "taste") + (("mix_like",) if (models_dir / "mix_like" / "params.json").exists() else ())
        return {n: fingerprint(models_dir / n) for n in names}

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
        c = self.crowd(self.variant.crowd, inputs, dnf, self.variant.als_weight)
        s = self.combine(c, self.taste_z(inputs), excl, self.variant)
        out = np.full(inputs.shape, -np.inf, dtype=np.float32)
        out[:, top] = s
        return out

    def taste_prediction(self, inputs: sp.csr_matrix) -> np.ndarray:
        """Прогноз оценки 1–5 модели вкуса по всем книгам ядра — признак шанса «понравится»."""
        return self.taste.score(inputs)

    def like_parts(self, inputs: sp.csr_matrix, dnf: sp.csr_matrix | None = None) -> tuple[np.ndarray, np.ndarray]:
        """z(ALS) и z(EASE «ценность») по книгам EASE — толпа «ценность» при любом весе ALS без пересчёта."""
        a, e = self.like_mix.components(inputs, dnf)
        return _z(a), _z(e)

    def crowd(self, kind: str, inputs: sp.csr_matrix, dnf: sp.csr_matrix | None = None,
              als_weight: float = 0.5) -> np.ndarray:
        """z-балл толпы по книгам EASE (строки × 30 000); als_weight — только для «ценности»."""
        if kind == "like":
            za, ze = self.like_parts(inputs, dnf)
            return als_weight * za + (1 - als_weight) * ze
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
        crowds = {k: layers.crowd(k, X) for k in {v.crowd for v in variants} - {"like"}}
        if any(v.crowd == "like" for v in variants):
            za, ze = layers.like_parts(X)
            for w in {v.als_weight for v in variants if v.crowd == "like"}:
                crowds[("like", w)] = w * za + (1 - w) * ze
        taste = layers.taste_z(X)
        excl = (exclude[s:s + batch][:, top].toarray() != 0)
        for v in variants:
            sc = layers.combine(crowds[("like", v.als_weight) if v.crowd == "like" else v.crowd], taste, excl, v)
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
                hr = hold.hidden_ratings[u]
                in_top = np.isin(hold.hidden_cols[u], top[t])
                m.update(auc=personal_auc(sc[i, hp[keep]], hr[keep]), gauc=graded_auc(sc[i, hp[keep]], hr[keep]),
                         gauc3=graded_auc(sc[i, hp[keep]], hr[keep], five=3.0),
                         fives=float((metrics.rounded(hr[in_top]) >= 5).sum()),
                         value20=float((metrics.rounded(hr[in_top]) - 3).sum()),
                         value20_5=float(value_of(hr[in_top], five=3.0).sum()),
                         hits=float(in_top.sum()), hit_stars=float(metrics.rounded(hr[in_top]).sum()),
                         fives_ideal=float(min(metrics.K, int((metrics.rounded(hr) >= 5).sum()))),
                         value_ideal=float(np.clip(np.sort(metrics.rounded(hr) - 3)[::-1][:metrics.K], 0, None).sum()),
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
            for m in CHECKED + ("hits", "fives_ideal", "value_ideal", "value20_5", "gauc3", "recall20", "later", "max_author", "fives"):
                g[m] = _boot(d.loc[idx, m].to_numpy(dtype=np.float64), np.random.default_rng(0), n_boot)
            for m in CHECKED:
                diff = (d.loc[idx, m] - ref.loc[idx, m]).to_numpy(dtype=np.float64)
                g[f"{m}_diff"] = _boot(diff, np.random.default_rng(0), n_boot)
            hits = float(d.loc[idx, "hits"].sum())
            g["mean_hit_stars"] = float(d.loc[idx, "hit_stars"].sum() / hits) if hits else None
            row["groups"][name] = g
        out.append(row)
    return out


def value_of(ratings: np.ndarray, five: float = 2.0) -> np.ndarray:
    """Ценность прочитанной книги: оценка − 3 (3★ — «никак не откликнулось», ноль), 5★ — five."""
    r = metrics.rounded(ratings)
    return np.where(r >= 5, five, r - 3)


def allowed(row: dict) -> bool:
    """NDCG@20 не значимо хуже нынешнего и Low@20 не значимо выше: 95% интервал парной разницы касается нуля
    или лежит на лучшей стороне (журнал решений, «слой вкуса»)."""
    g = row["groups"]["all"]
    return g["ndcg20_diff"]["hi"] >= 0 and g["low20_diff"]["lo"] <= 0


def choose(summary: list[dict], metric: str = "value20") -> dict:
    """Наибольшая ценность топа (docs/resheniya.md, «ценность топа»): сумма (оценка − 3) скрытых книг в топ-20.
    Плохие книги в ней — минус, поэтому отдельные страховки NDCG и Low@20 только показываются."""
    return max(summary, key=lambda r: r["groups"]["all"][metric]["mean"])


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
        variants = [Variant(c, w, n, a) for c in layers.crowds() for a in (LIKE_ALS if c == "like" else (0.5,))
                    for n in CUTOFFS for w in WEIGHTS]
    else:
        chosen = Variant(**json.loads((models_dir / "layers" / "params.json").read_text())["variant"])
        variants = list(dict.fromkeys([ref, chosen]))
    summary = summarize(evaluate(layers, hold, info, variants), ref)
    res = {"stage": stage, "n_users": int(len(hold.user_ids)), "summary": summary}
    if stage == "val":
        best = choose(summary)
        res["chosen"] = best["variant"]
        res["chosen_by_five_value"] = {str(v): choose(summary, m)["label"] for m, v in FIVE_VALUES.items()}
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
             f"{res['n_users']} человек. **Ценность топа** — сумма (оценка − 3) скрытых книг, попавших в топ-20: "
             f"5★ = +2, 4★ = +1, 3★ = 0, 2★ = −1, 1★ = −2 (docs/resheniya.md, «ценность топа»); выбирается наибольшая. "
             f"Разницы — с нынешней выдачей («толпа по оценкам, вкус 0») на тех же людях, в скобках 95% интервал. "
             f"NDCG@20 и Low@20 — для сведения («не хуже» — интервал разницы касается нуля или лучше).", "",
             "| вариант | ценность топа | разница | угадано на человека | средняя оценка угаданных | "
             "пятёрок на человека | NDCG@20 | разница | Low@20 | разница | NDCG и Low не хуже | "
             "взвешенная точность | поздние тома | книг одного автора (макс.) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["summary"]:
        g = r["groups"]["all"]
        mark = " **← выбран**" if chosen and r["variant"] == chosen else ""
        ms = "—" if g["mean_hit_stars"] is None else f"{g['mean_hit_stars']:.2f}"
        lines.append(
            f"| {r['label']}{mark} | {_f(g['value20'])} | {_f(g['value20_diff'], True)}{_ci(g['value20_diff'])} | "
            f"{g['hits']['mean']:.3f} | {ms} | {g['fives']['mean']:.3f} | "
            f"{_f(g['ndcg20'])} | {_f(g['ndcg20_diff'], True)}{_ci(g['ndcg20_diff'])} | "
            f"{_f(g['low20'], pct=True)} | {_f(g['low20_diff'], True, pct=True)} | {'да' if allowed(r) else 'нет'} | "
            f"{_f(g['gauc'])} | {_f(g['later'], pct=True)} | {g['max_author']['mean']:.2f} |")
    ref_all = res["summary"][0]["groups"]["all"]
    lines += ["", f"Потолок — топ из лучших скрытых книг человека (больше угадать нельзя: остальное он не читал или "
                  f"не оценил): ценность {ref_all['value_ideal']['mean']:.2f}, пятёрок {ref_all['fives_ideal']['mean']:.2f} "
                  f"на человека."]
    if "chosen_by_five_value" in res:
        lines += ["", "Выбор при разной ценности пятёрки (4★ = +1): "
                  + "; ".join(f"5★ = +{k.rstrip('0').rstrip('.')} → {v}" for k, v in res["chosen_by_five_value"].items())
                  + ". Одинаковый выбор — точная шкала на решение не влияет."]
    groups = [b for b in BUCKET_ORDER if b in res["summary"][0]["groups"]]
    lines += ["", "По группам (ценность топа / NDCG@20 / Low@20):", "",
              "| вариант | " + " | ".join(groups) + " |", "|---|" + "---|" * len(groups)]
    for r in res["summary"]:
        if res["stage"] == "val" and not (r["variant"] == chosen or Variant(**r["variant"]) == Variant(*REFERENCE)):
            continue
        cells = [f"{_f(r['groups'][b]['value20'])} / {_f(r['groups'][b]['ndcg20'])} / "
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
        crowds = {v: layers.crowd(v.crowd, prof.x, prof.dnf, v.als_weight) for v in (ref, chosen)}
        taste = layers.taste_z(prof.x)
        rated = RatedFilter(info, prof.x.indices)
        lists = {}
        for v in (ref, chosen):
            sc = layers.combine(crowds[v], taste, excl, v)[0]
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


# TODO п. 38, идея 1: подбор толпы «ценность» — регуляризация λ и веса звёзд на входе (1★ … 5★).
# Сохранённый models/ease_like всегда в сравнении и не переобучается. W1: 3★ — слабый плюс «прочитал»; W2: пятёрка
# весит больше; W3 (предложение пользователя): всё положительно и удваивается — пятёрка в 16 раз весомее единицы,
# сигнала «не понравилось» на входе нет. Веса приводятся к масштабу «максимум 2» (`ease.normalize_weights`).
W0, W1, W2 = (-2.0, -1.0, 0.0, 1.0, 2.0), (-2.0, -1.0, 0.5, 1.0, 2.0), (-2.0, -1.0, 0.0, 1.0, 3.0)
W3 = (0.125, 0.25, 0.5, 1.0, 2.0)   # 2/4/8/16/32 после приведения масштаба (normalize_weights)
LIKE_GRID = [(500.0, W0), (250.0, W0), (1000.0, W0), (500.0, W1), (500.0, W2), (500.0, W3)]
LIKE_JUDGED = (Variant("like", 0.1, None, 0.25), Variant("like", 0.0, None, 0.25))   # выбранный вариант и без вкуса


def tune_like(*, clean_dir: Path, split_dir: Path, models_dir: Path, eval_dir: Path, grid=LIKE_GRID,
              fit_kw: dict | None = None, force: bool = False, min_user: int = 20) -> dict:
    """Обучает толпу «ценность» с каждой настройкой, меряет на валидации вариантами LIKE_JUDGED по ценности топа;
    лучшая настройка (по «ценность, ALS 0.25, вкус 0.1») записывается в models/ease_like и models/mix_like;
    force — записать последнюю настройку сетки, даже если по ценности она в пределах шума хуже (решение пользователя);
    min_user — обучать новые настройки только на людях с ≥ min_user оценок (сохранённая — со своим порогом).
    После — `layers val` и `layers test` заново (отпечатки компонентов меняются)."""
    import time

    from booksengine.model.ease import EASE, EASELike
    from booksengine.model.filters import work_info
    from booksengine.model.matrix import load_train
    ratings_path = clean_dir / "ratings.parquet"
    train = load_train(ratings_path, split_dir / "holdout_users.parquet")
    info = work_info(clean_dir, train.work_ids)
    hold = _hold(ratings_path, split_dir, "val", train.work_ids)
    base = Layers.from_models(models_dir)
    ref = Variant(*REFERENCE)
    from booksengine.model.ease import normalize_weights
    grid = [(float(lam), normalize_weights(w), int(min_user)) for lam, w in grid]
    saved = read_params(models_dir / "ease_like") if (models_dir / "ease_like" / "params.json").exists() else {}
    if saved:  # сохранённая толпа — всегда точка сравнения: без неё прогон из одной настройки записал бы худшую
        cur = (float(saved["lam"]), tuple(saved.get("weights", W0)), int(saved.get("min_user", 20)))
        grid = [cur] + [g for g in grid if g != cur]
    results, best, best_model = [], None, None
    eval_dir.mkdir(parents=True, exist_ok=True)
    for lam, weights, mu in grid:
        t0 = time.perf_counter()
        is_saved = bool(saved) and cur == (lam, weights, mu)
        if is_saved:
            ease = EASE.load(models_dir / "ease_like")
        else:
            ease = EASELike(lam=lam, weights=weights, min_user=mu, **(fit_kw or {}))
            ease.fit(train)
        fit_s = round(time.perf_counter() - t0, 1)
        like = Mix(str(models_dir / "als_neg"), str(models_dir / "ease_like"))
        like.als, like.ease = base.mix.als, ease
        like.configure(als_weight=0.5, ease_input=weights)
        layers = Layers(base.mix, base.taste, like_mix=like)
        summ = summarize(evaluate(layers, hold, info, [ref, *LIKE_JUDGED]), ref)
        rows = {r["label"]: r["groups"]["all"] for r in summ}
        by_group = {r["label"]: {b: g["value20"]["mean"] for b, g in r["groups"].items()} for r in summ}
        row = {"lam": lam, "weights": list(weights), "min_user": mu, "fit_seconds": fit_s, "saved": is_saved,
               "value_by_group": by_group[LIKE_JUDGED[1].label()],
               "variants": {LIKE_JUDGED[i].label(): {k: rows[LIKE_JUDGED[i].label()][k] for k in
                                                      ("value20", "value20_diff", "ndcg20", "low20", "fives", "hits")}
                            for i in range(len(LIKE_JUDGED))}}
        results.append(row)
        v = row["variants"][LIKE_JUDGED[0].label()]["value20"]["mean"]
        print(f"«ценность» λ = {lam:g}, веса {weights}, люди с ≥ {mu} оценок: ценность топа {v:.3f}, "
              f"обучение {fit_s} с", flush=True)
        (eval_dir / "ease_like_tune.json").write_text(json.dumps({"results": results}, ensure_ascii=False, indent=1))
        last = (lam, weights, mu) == grid[-1]
        if (force and last) or (not force and (best is None or v > best)):
            best, best_model = v, (lam, weights, mu, ease)
        else:
            del ease
    lam, weights, mu, ease = best_model
    if not (saved and cur == (lam, weights, mu)):
        ease.save(models_dir / "ease_like")
        mix = Mix(models_dir / "als_neg", models_dir / "ease_like")
        mix.fit(train)
        mix.configure(als_weight=0.5, ease_input=weights)
        mix.save(models_dir / "mix_like")
    return {"results": results, "best": {"lam": lam, "weights": list(weights), "min_user": mu}}


def report_like(res: dict) -> str:
    labels = [v.label() for v in LIKE_JUDGED]
    lines = ["# Подбор толпы «ценность» (валидация, TODO п. 38)", "",
             "Разницы — с нынешней выдачей («толпа по оценкам, вкус 0»), в скобках 95% интервал. Веса звёзд — "
             "вход 1★ … 5★. Лучшая по ценности топа (первый вариант) записана в models/ease_like и models/mix_like.", "",
             "| λ | веса звёзд | обучение: людей с ≥ N оценок | вариант | ценность топа | разница | угадано | пятёрок | "
             "NDCG@20 | Low@20 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["results"]:
        for lab in labels:
            g = r["variants"][lab]
            lines.append(f"| {r['lam']:g} | {'/'.join(f'{w:g}' for w in r['weights'])} | {r.get('min_user', 20)} | {lab} | "
                         f"{_f(g['value20'])} | {_f(g['value20_diff'], True)}{_ci(g['value20_diff'])} | "
                         f"{g['hits']['mean']:.2f} | {g['fives']['mean']:.3f} | {_f(g['ndcg20'])} | "
                         f"{_f(g['low20'], pct=True)} |")
    groups = [g for g in BUCKET_ORDER if g in res["results"][0].get("value_by_group", {})]
    if groups:
        lines += ["", f"Ценность топа по группам (вариант «{labels[1]}»):", "",
                  "| λ | веса звёзд | людей с ≥ N | " + " | ".join(groups) + " |", "|---|---|---|" + "---|" * len(groups)]
        for r in res["results"]:
            lines.append(f"| {r['lam']:g} | {'/'.join(f'{w:g}' for w in r['weights'])} | {r.get('min_user', 20)} | "
                         + " | ".join(f"{r['value_by_group'][g]:.3f}" for g in groups) + " |")
    b = res["best"]
    lines += ["", f"Выбрано: λ = {b['lam']:g}, веса {'/'.join(f'{w:g}' for w in b['weights'])}, "
                  f"люди с ≥ {b.get('min_user', 20)} оценок. Дальше — "
                  "`booksengine layers val` и `layers test` (вес ALS и вкуса подбираются заново под новую толпу)."]
    return "\n".join(lines) + "\n"


def profile_check(*, clean_dir: Path, models_dir: Path, profiles_dir: Path) -> str:
    """Проверка на своих оценках (leave-one-out): каждая книга профиля по очереди прячется, выдача считается по
    остальным — на каком месте оказалась спрятанная, у нынешней выдачи и у выбранного варианта (models/layers).
    Хорошо, если пятёрки выше четвёрок, а четвёрки выше низких. Мерит только прочитанное — книги, выбранные самим
    человеком; оценённые — мало (десятки), поэтому это проверка «нет ли явной ошибки», а не выбор настроек."""
    from booksengine.model.filters import work_info
    from booksengine.model.matrix import catalog_works
    from booksengine.model.series import SeriesIndex, exclusion
    from booksengine.recommend import read_profile
    work_ids = catalog_works(clean_dir / "ratings.parquet")
    layers = Layers.load(models_dir / "layers")
    info = work_info(clean_dir, work_ids)
    series = SeriesIndex(info.title.tolist())
    top = layers.mix.ease.top_cols
    pos_of = np.full(len(work_ids), -1)
    pos_of[top] = np.arange(len(top))
    variants = {"нынешняя": Variant(*REFERENCE), "новая": layers.variant}
    out = [f"# Проверка на своих оценках: нынешняя выдача и «{layers.variant.label()}»", "",
           "Каждая оценённая книга по очереди спрятана, выдача посчитана по остальным. Место — среди всех кандидатов "
           "(около 30 000 книг, без оценённых и продолжений начатых серий); меньше — лучше. Хорошо, когда пятёрки "
           "выше четвёрок, четвёрки выше низких. Проверяется только прочитанное — книги, которые человек выбрал сам."]
    for csv in sorted(profiles_dir.glob("*.csv")):
        prof = read_profile(csv, work_ids, clean_dir)
        cols = prof.x.indices
        ok = cols[pos_of[cols] >= 0]
        if len(ok) == 0:
            continue
        # строка на каждую спрятанную книгу: профиль без неё
        rows, dnf_rows = [], []
        for c in ok:
            keep = cols != c
            rows.append(sp.csr_matrix((prof.x.data[keep], (np.zeros(keep.sum(), int), cols[keep])), shape=prof.x.shape))
            d = prof.dnf.toarray()[0]
            d[c] = 0
            dnf_rows.append(sp.csr_matrix(d[None, :]))
        X, D = sp.vstack(rows).tocsr(), sp.vstack(dnf_rows).tocsr()
        excl = exclusion(X, series)[:, top].toarray() != 0
        taste = layers.taste_z(X)
        places = {}
        for name, v in variants.items():
            sc = layers.combine(layers.crowd(v.crowd, X, D, v.als_weight), taste, excl, v)
            p = []
            for i, c in enumerate(ok):
                s_i = sc[i, pos_of[c]]
                p.append(np.nan if not np.isfinite(s_i) else int((sc[i] > s_i).sum()) + 1)
            places[name] = np.array(p, dtype=np.float64)
        r = metrics.rounded(prof.x.data[np.searchsorted(cols, ok)])
        n_cand = int(np.isfinite(layers.combine(layers.crowd("mix", X[:1]), taste[:1], excl[:1], Variant())).sum())
        out += ["", f"## {csv.stem}: {len(ok)} книг из {len(cols)} (остальные вне 30 000 книг EASE — их модель не "
                    f"советует), кандидатов ~{n_cand}", "",
                "| оценка | книг | медиана места: нынешняя | новая | в топ-100: нынешняя | новая |", "|---|---|---|---|---|---|"]
        for lo, hi, lab in ((5, 5, "5★"), (4, 4, "4★"), (1, 3, "1–3★")):
            m = (r >= lo) & (r <= hi)
            if not m.any():
                continue
            cells = [f"{np.nanmedian(places[n][m]):.0f}" for n in variants]
            tops = [f"{int(np.nansum(places[n][m] <= 100))}" for n in variants]
            out.append(f"| {lab} | {int(m.sum())} | {cells[0]} | {cells[1]} | {tops[0]} | {tops[1]} |")
        for n in variants:
            g = graded_auc(-np.nan_to_num(places[n], nan=1e9), r)
            out.append("")
            out.append(f"Взвешенная точность ({n}): {g:.3f} — доля пар «ценнее / менее ценна», где ценная книга "
                       f"стоит выше (0.5 — монетка).")
        out += ["", "| книга | оценка | место: нынешняя | новая |", "|---|---|---|---|"]
        order = np.lexsort((places["новая"], -r))
        for i in order:
            c = ok[i]
            pl = [("—" if np.isnan(places[n][i]) else f"{places[n][i]:.0f}") for n in variants]
            out.append(f"| {prof.names.get(int(c), info.title[c])} | {r[i]:.0f} | {pl[0]} | {pl[1]} |")
    return "\n".join(out) + "\n"


def why(*, clean_dir: Path, models_dir: Path, profile_csv: Path, query: str, top: int = 8) -> str:
    """TODO п. 39: почему книга стоит там, где стоит, у выбранного варианта (models/layers). Книга ищется по
    goodreads_work_id или части названия (сначала среди оценённых); оценённая — прячется, как в profile-check.
    Разбор: место по каждой части (ALS, EASE «ценность», вкус) и вклады книг входа в балл толпы — точно, `explain`."""
    from booksengine.model import explain
    from booksengine.model.filters import work_info
    from booksengine.model.matrix import catalog_works
    from booksengine.model.series import SeriesIndex, exclusion
    from booksengine.recommend import read_profile
    work_ids = catalog_works(clean_dir / "ratings.parquet")
    layers = Layers.load(models_dir / "layers")
    v = layers.variant
    if v.crowd != "like":
        raise ValueError("разбор сделан для толпы «ценность» (п. 38): выбранный вариант другой")
    info = work_info(clean_dir, work_ids)
    prof = read_profile(profile_csv, work_ids, clean_dir)
    if query.isdigit() and int(query) in set(work_ids.tolist()):
        col = int(np.searchsorted(work_ids, int(query)))
    else:
        q = query.lower()
        rated = [c for c, n in prof.names.items() if q in str(n).lower() or q in str(info.title[c]).lower()]
        found = rated or [int(c) for c in np.flatnonzero(info.title.str.lower().str.contains(q, regex=False).to_numpy())]
        if not found:
            raise ValueError(f"книга «{query}» не найдена")
        col = found[0]
    top_cols = layers.mix.ease.top_cols
    pos = int(np.searchsorted(top_cols, col))
    if pos >= len(top_cols) or top_cols[pos] != col:
        return f"«{info.title[col]}» вне 30 000 книг EASE — модель её не советует вовсе.\n"
    keep = prof.x.indices != col
    rating = prof.x.data[~keep]
    x = sp.csr_matrix((prof.x.data[keep], (np.zeros(keep.sum(), int), prof.x.indices[keep])), shape=prof.x.shape)
    d = prof.dnf.toarray()[0]
    d[col] = 0
    dnf = sp.csr_matrix(d[None, :])
    series = SeriesIndex(info.title.tolist())
    excl = exclusion(x, series)[:, top_cols].toarray()[0] != 0
    za, ze = layers.like_parts(x, dnf)
    tz = layers.taste_z(x)
    parts = {"ALS": za[0], "EASE «ценность»": ze[0], "вкус": tz[0],
             "итог": layers.combine(v.als_weight * za + (1 - v.als_weight) * ze, tz, excl[None, :], v)[0]}

    def place(s: np.ndarray) -> str:
        s = np.where(excl, -np.inf, s)
        return "исключена" if not np.isfinite(s[pos]) else f"{int((s > s[pos]).sum()) + 1}"

    lines = [f"# Почему «{info.title[col]}» — {info.author[col] or '?'} стоит там, где стоит", "",
             f"Профиль {profile_csv.stem}" + (f", книга оценена на {metrics.rounded(rating)[0]:.0f}★ и спрятана" if len(rating)
                                               else ", книга не оценена") + f"; вариант «{v.label()}».", "",
             "| часть | z-балл книги | место по одной этой части |", "|---|---|---|"]
    lines += [f"| {k} | {s[pos]:+.2f} | {place(s)} |" for k, s in parts.items()]
    m = layers.like_mix
    w0 = m.als_weight
    try:
        m.configure(als_weight=1.0, ease_input=m.ease_input)
        in_cols, c_als = explain.contributions(m, x, np.array([col]), dnf)
        m.configure(als_weight=0.0, ease_input=m.ease_input)
        _, c_ease = explain.contributions(m, x, np.array([col]), dnf)
    finally:
        m.configure(als_weight=w0, ease_input=m.ease_input)
    c_als, c_ease = c_als[:, 0], c_ease[:, 0]
    total = v.als_weight * c_als + (1 - v.als_weight) * c_ease
    r_in = metrics.rounded(x.data)
    lines += ["", f"Вклады книг входа в балл толпы (ALS × {v.als_weight:g} + EASE × {1 - v.als_weight:g}); "
                  "сумма по книгам = z-балл толпы. Самые тянущие вниз и вверх:", "",
              "| книга входа | оценка | вклад | ALS | EASE |", "|---|---|---|---|---|"]
    order = np.argsort(total)
    for i in dict.fromkeys(list(order[:top]) + list(order[::-1][:top])):
        c = int(in_cols[i])
        lines.append(f"| {prof.names.get(c, info.title[c])} | {r_in[i]:.0f} | {total[i]:+.3f} | {c_als[i]:+.3f} | "
                     f"{c_ease[i]:+.3f} |")
    lines += ["", f"Книг серии этой книги в ядре: {series.continuations(np.array([col])).size} "
                  "(продолжения начатых серий исключаются)."]
    return "\n".join(lines) + "\n"
