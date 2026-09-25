"""Слои «толпа + вкус» (models/layers): толпа выбирает, вкус упорядочивает.

Балл = z(толпа) + g·z(вкус) по 30 000 книг EASE; z — нормировка по книгам человека, как в смеси. Толпы:
- `mix` — прежняя смесь (вход EASE по оценке, ALS с негативом «≤ 2»); с весом вкуса 0 — точка сравнения;
- `like` — толпа «ценность» (models/mix_like): ALS + EASE с целью «оценка − 3» (`ease.EASELike`), вес ALS —
  `Variant.als_weight`;
- `read` — «только прочитал» (вход EASE 1 у любой оценки, ALS без негатива): звёзды толкует один слой вкуса.
`cutoff` — вкус переставляет только первые N толпы, остальные книги идут после них в порядке толпы.

Судья выбора — **качество списка** из 20 книг: человек выбирает из двадцати по настроению, поэтому порядок внутри
них не важен, важен состав — как можно больше пятёрок и ни одной 1–2★. Проверить можно только книги списка, которые
человек прочёл сам (скрытые оценки, «угаданные»): качество — средняя ценность угаданных (`JUDGE_STARS`: 5★ = 2,
4★ = 1, 3★ = 0.5, 2★ = −0.5, 1★ = −1 — смысл оценок пользователя), сначала по человеку, потом по людям. Учитывает
все оценки: четвёрка лучше тройки, одна плохая книга не перечёркивает отличную.
Ограничение (`GUARD`) — угадано не меньше 80% того, что угадывает толпа без вкуса (`anchor`): иначе список уходит в книги,
которые такие люди не читают, и проверить его нечем. Для сравнения в отчёте — прежние судьи: доля 5★ минус доля 1–2★
(четвёрку приравнивала к тройке) и сумма (оценка − 3) угаданных (награждала «угадал, что человек и так прочтёт» —
вкус при ней получал вес 0).
Топ-20 в замере собирается правилами выдачи (`filters.ListPicker`): судья видит тот же список, что и человек.
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
from booksengine.model.taste_gap import graded_auc

# толпа «прочитал» замерена 2026-09-24 и отклонена (Low@20 8.7% → 11–14%); остаётся в коде для повторного замера
CROWDS = ("mix",)
WEIGHTS = (0.0, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 8.0)
CUTOFFS = (None, 100, 300, 500, 1000)
REFERENCE = ("mix", 0.0, None)     # прежняя смесь — для сведения
LIKE_ALS = (0.0, 0.1, 0.25, 0.5, 0.75)   # вес ALS в толпе «ценность»: ALS находит прочитанное, «ценность» — отсеивает плохое
# угадано в топ-20 — не меньше этой доли от опоры (`anchor`). 90% было выбрано без замера и держало вкус на 1.5;
# 80% — решение пользователя 2026-09-25 (вкус 2 угадывает 88%, 3 — 78%); проверяется журналом выдач
GUARD = 0.8
JUDGE_STARS = np.array([-1.0, -0.5, 0.5, 1.0, 2.0])   # ценность угаданной книги с оценкой 1★ … 5★
STARS = tuple(f"s{k}" for k in range(1, 6))           # доля угаданных с оценкой k★
BASE = tuple(f"b{k}" for k in range(1, 6))            # то же среди всего прочитанного — «случайные из прочитанного»
CHECKED = ("quality", "hits")                          # среднее с интервалом и парная разность с нынешней
# только среднее: распределение по звёздам, прежние судьи, сколько книг списка общие с нынешней выдачей
MEANS = STARS + BASE + ("base_quality", "five_minus_low", "value20", "same", "known")


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


# Опора ограничения — толпа «ценность» без вкуса: порог не зависит от выбранного прежде варианта. От нынешней выдачи
# он сползал: выбор упирается в порог, и каждый прогон разрешал терять ещё 10% угаданного.
ANCHOR_LIKE = Variant("like", 0.0, None, 0.25)


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
        """Толпы: прежняя смесь и, если обучена (models/mix_like), «ценность»; перебор берёт последнюю (`grid`)."""
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
        """Итоговый балл по книгам EASE; excl — булева маска «не советовать» (вход, начатые серии).
        С cutoff вкус переставляет только первые N толпы, остальные книги — сразу после них, в порядке толпы: список
        берётся из первых N, а порядок всех книг полный — замер по прочитанным книгам не видит ничьих."""
        s = crowd + v.taste_weight * taste
        if v.cutoff:
            c = np.where(excl, -np.inf, crowd)
            k = min(v.cutoff, c.shape[1]) - 1
            thr = np.take_along_axis(c, np.argpartition(-c, k, axis=1)[:, k:k + 1], axis=1)
            out = (c < thr) & ~excl
            floor = np.where(out | excl, np.inf, s).min(axis=1, keepdims=True)     # худший из первых N
            best_out = np.where(out, crowd, -np.inf).max(axis=1, keepdims=True)     # лучший вне первых N
            ok = np.isfinite(floor) & np.isfinite(best_out)
            s = np.where(out, crowd + np.where(ok, floor - best_out - 1.0, 0.0), s)
        s[excl] = -np.inf
        return s


def popularity(ratings_path: Path, work_ids: np.ndarray) -> np.ndarray:
    """Число оценок каждой книги ядра (по столбцам матрицы) — насколько книга известна."""
    import duckdb
    c = duckdb.execute("SELECT work_id, count(*) AS n FROM read_parquet(?) GROUP BY 1", [str(ratings_path)]).df()
    return c.set_index("work_id").n.reindex(work_ids).fillna(0).to_numpy(dtype=np.float64)


def evaluate(layers: Layers, hold, info: pd.DataFrame, variants: list[Variant], batch: int = 500,
             pop: np.ndarray | None = None) -> dict:
    """Метрики по людям для каждого варианта (баллы толпы и вкуса считаются один раз на пачку). Топ-20 — по правилам
    выдачи (`filters.ListPicker`), «уже оценено по сути» — по входу человека. `same` — сколько книг списка общие
    со списком первого варианта (нынешней выдачи); `known` — медиана числа оценок книг списка (pop — `popularity`):
    растёт ли доля менее известных книг с профилем."""
    from booksengine.model.filters import ListPicker, RatedFilter, pick_top
    from booksengine.model.series import SeriesIndex
    top = layers.mix.ease.top_cols
    pos_of = np.full(hold.inputs.shape[1], -1)
    pos_of[top] = np.arange(len(top))
    picker = ListPicker(info, SeriesIndex(info.title.tolist()))
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
        rated = [RatedFilter(picker.books, X.indices[X.indptr[i]:X.indptr[i + 1]]) for i in range(X.shape[0])]
        first = None
        for v in variants:
            sc = layers.combine(crowds[("like", v.als_weight) if v.crowd == "like" else v.crowd], taste, excl, v)
            lists = pick_top(sc, top, pos_of, picker, rated, metrics.K)
            first = lists if first is None else first
            for i in range(sc.shape[0]):
                u = s + i
                t = lists[i]
                hc, hr = hold.hidden_cols[u], hold.hidden_ratings[u]
                got = np.clip(metrics.rounded(hr[np.isin(hc, t)]), 1, 5).astype(int)
                base = np.clip(metrics.rounded(hr[pos_of[hc] >= 0]), 1, 5).astype(int)
                seen = len(got) > 0          # в списке есть книги, которые человек прочёл сам
                row = {"user_id": hold.user_ids[u], "bucket": hold.buckets[u], "hits": float(len(got)),
                       "quality": float(JUDGE_STARS[got - 1].mean()) if seen else np.nan,
                       "five_minus_low": float((got == 5).mean() - (got <= 2).mean()) if seen else np.nan,
                       "base_quality": float(JUDGE_STARS[base - 1].mean()) if len(base) else np.nan,
                       "value20": float((got - 3).sum()),
                       "same": float(len(np.intersect1d(t, first[i]))),
                       "known": float(np.median(pop[t])) if pop is not None and len(t) else np.nan}
                for k in range(1, 6):
                    row[f"s{k}"] = float((got == k).mean()) if seen else np.nan
                    row[f"b{k}"] = float((base == k).mean()) if len(base) else np.nan
                rows[v].append(row)
    return {v: pd.DataFrame(r) for v, r in rows.items()}


def summarize(per_user: dict[Variant, pd.DataFrame], reference: Variant, n_boot: int = 1000) -> list[dict]:
    ref = per_user[reference].set_index("user_id")
    out = []
    for v, d in per_user.items():
        d = d.set_index("user_id")
        parts = [("all", d.index)] + [(b, d.index[d.bucket == b]) for b in BUCKET_ORDER if (d.bucket == b).any()]
        row = {"variant": asdict(v), "label": v.label(), "groups": {}}
        for name, idx in parts:
            g = {}
            for m in CHECKED:
                g[m] = metrics.bootstrap(d.loc[idx, m].to_numpy(dtype=np.float64), np.random.default_rng(0), n_boot)
            for m in MEANS:
                x = d.loc[idx, m].to_numpy(dtype=np.float64)
                x = x[~np.isnan(x)]
                g[m] = {"mean": float(x.mean()) if len(x) else None, "n": int(len(x))}
            for m in CHECKED:
                diff = (d.loc[idx, m] - ref.loc[idx, m]).to_numpy(dtype=np.float64)
                g[f"{m}_diff"] = metrics.bootstrap(diff, np.random.default_rng(0), n_boot)
            row["groups"][name] = g
        out.append(row)
    return out


def _mean(row: dict, metric: str) -> float:
    m = row["groups"]["all"][metric]["mean"]
    return -np.inf if m is None else m


def anchor(summary: list[dict]) -> dict:
    """Строка опоры ограничения: толпа «ценность» без вкуса, если она в переборе, иначе прежняя смесь без вкуса."""
    by = {r["label"]: r for r in summary}
    return by.get(ANCHOR_LIKE.label()) or by[Variant(*REFERENCE).label()]


def choose(summary: list[dict], baseline: dict, metric: str = "quality") -> dict | None:
    """Наибольшее качество списка (или другой metric — прежние судьи для сравнения) среди вариантов, которые угадывают
    в топ-20 не меньше GUARD от baseline — строки опоры (`anchor`). None — ни один не угадывает столько: так бывает
    у чужой толпы в `tune_like`; в `layers val` опора сама в summary и проходит всегда."""
    floor = GUARD * _mean(baseline, "hits")
    ok = [r for r in summary if _mean(r, "hits") >= floor]
    return max(ok, key=lambda r: _mean(r, metric)) if ok else None


def grid(layers: Layers) -> list[Variant]:
    """Перебор `layers val`: лучшая обученная толпа («ценность» с разным весом ALS, если есть) × вес вкуса × отсечение.
    Отсечение — только с вкусом: без него порядок тот же."""
    crowd = layers.crowds()[-1]
    als = LIKE_ALS if crowd == "like" else (0.5,)
    return [Variant(crowd, w, n, a) for a in als for w in WEIGHTS for n in CUTOFFS if w or n is None]


def _hold(ratings_path: Path, split_dir: Path, stage: str, work_ids: np.ndarray):
    from booksengine.model.evaluate import load_eval_holdout
    return load_eval_holdout(ratings_path, split_dir, stage, work_ids)


def run(stage: str, *, clean_dir: Path, split_dir: Path, models_dir: Path, eval_dir: Path) -> dict:
    """val — перебор и выбор (`choose`) против нынешней выдачи (models_dir/layers/params.json, без него — прежняя смесь);
    выбранный вариант пишется туда же, а нынешний — рядом как «baseline». test — один замер выбранного против baseline
    и прежней смеси."""
    from booksengine.model.filters import work_info
    from booksengine.model.matrix import catalog_works
    ratings_path = clean_dir / "ratings.parquet"
    work_ids = catalog_works(ratings_path)
    layers = Layers.from_models(models_dir)
    info = work_info(clean_dir, work_ids)
    hold = _hold(ratings_path, split_dir, stage, work_ids)
    ref = Variant(*REFERENCE)
    params = models_dir / "layers" / "params.json"
    saved = json.loads(params.read_text()) if params.exists() else {}
    if stage == "val":
        current = Variant(**saved["variant"]) if saved else ref
        extra = [ANCHOR_LIKE] if layers.like_mix is not None else []
        variants = list(dict.fromkeys([current, ref, *extra, *grid(layers)]))
    else:
        current = Variant(**saved.get("baseline", asdict(ref)))
        variants = list(dict.fromkeys([current, ref, Variant(**saved["variant"])]))
    summary = summarize(evaluate(layers, hold, info, variants, pop=popularity(ratings_path, work_ids)), current)
    res = {"stage": stage, "n_users": int(len(hold.user_ids)), "current": asdict(current), "summary": summary}
    if stage == "val":
        a = anchor(summary)
        res["anchor"] = a["label"]
        best = choose(summary, a)
        # нынешний вариант меняется, только если новый лучше уверенно (парная разность выше нуля по всему интервалу):
        # иначе выбор идёт по шуму — так «из первых 500» сменил вариант без отсечения при разнице 0.000 и испортил шанс
        d = best["groups"]["all"]["quality_diff"]
        if saved and _mean(summary[0], "hits") >= GUARD * _mean(a, "hits") and not (d["lo"] is not None and d["lo"] > 0):
            best = summary[0]
        res["chosen"] = best["variant"]
        res["chosen_by_value"] = max(summary, key=lambda r: _mean(r, "value20"))["label"]
        res["chosen_by_share"] = choose(summary, a, "five_minus_low")["label"]
        params.parent.mkdir(parents=True, exist_ok=True)
        params.write_text(json.dumps(
            {"variant": best["variant"], "label": best["label"], "baseline": asdict(current),
             "components": Layers.component_fingerprints(models_dir)}, ensure_ascii=False, indent=1))
    else:
        res["chosen"] = saved["variant"]
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / f"layers_{stage}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return res


def _f(x: dict, signed=False, pct=False) -> str:
    if x["mean"] is None:
        return "—"
    if pct:
        return f"{x['mean']:+.1%}" if signed else f"{x['mean']:.1%}"
    return f"{x['mean']:+.3f}" if signed else f"{x['mean']:.3f}"


def _ci(x: dict, pct=False) -> str:
    if x["mean"] is None:
        return ""
    return f" [{x['lo']:+.1%}; {x['hi']:+.1%}]" if pct else f" [{x['lo']:+.3f}; {x['hi']:+.3f}]"


SHOWN = 12   # сколько лучших вариантов показывать в отчёте `layers val`
HOW_TO_READ = ("Судится список из 20 книг по тем его книгам, которые человек прочёл сам (**угаданные**); порядок "
               "внутри двадцати не важен. **Качество** — средняя ценность угаданных: 5★ = 2, 4★ = 1, 3★ = 0.5, "
               "2★ = −0.5, 1★ = −1 (сначала по человеку, потом по людям). **5★ … 1★** — как угаданные распределены по "
               "оценкам. **Разница** — с нынешним на тех же людях, в скобках 95% интервал: новое сменяет нынешнее, только "
               "если интервал целиком выше нуля. **Угадано** — книг из списка, прочитанных человеком; ограничение — не "
               f"меньше {GUARD:.0%} от толпы без вкуса.")


def short(v: Variant) -> str:
    """Короткое имя варианта для таблиц: у толпы «ценность» — только веса, остальные толпы — словом."""
    parts = ([f"ALS {v.als_weight:g}"] if v.crowd == "like"
             else [{"mix": "старая смесь", "read": "толпа «прочитал»"}[v.crowd]])
    parts.append(f"вкус {v.taste_weight:g}")
    if v.cutoff:
        parts.append(f"первые {v.cutoff}")
    return " · ".join(parts)


def _stars(g: dict, keys=STARS) -> str:
    return " | ".join(_f(g[k], pct=True) for k in reversed(keys))


def report(res: dict) -> str:
    rows, stage = res["summary"], res["stage"]
    current, chosen = Variant(**res["current"]), Variant(**res["chosen"])
    by_variant = {Variant(**r["variant"]): r for r in rows}
    gc, gn = by_variant[current]["groups"]["all"], by_variant[chosen]["groups"]["all"]
    was = "нынешняя" if stage == "val" else "прежняя"      # в тесте выбранный вариант уже стал выдачей
    floor = GUARD * anchor(rows)["groups"]["all"]["hits"]["mean"] if stage == "val" else None
    if chosen == current:
        verdict = f"**Оставлен {was[:-2]}ий вариант: {short(chosen)}** — качество {_f(gn['quality'])}."
    else:
        verdict = (f"**{'Выбран' if stage == 'val' else 'Выбранный'}: {short(chosen)}** — качество {_f(gn['quality'])} "
                   f"против {_f(gc['quality'])} у {was[:-2]}его ({short(current)}), разница "
                   f"{_f(gn['quality_diff'], True)}{_ci(gn['quality_diff'])}.")
    verdict += f" Угадано {gn['hits']['mean']:.2f} книги на человека" + (
        f" (порог {floor:.2f})." if floor is not None else f" ({gc['hits']['mean']:.2f} у {was[:-2]}его).")
    lines = [f"# Слои «толпа + вкус» — {'валидация' if stage == 'val' else 'тест'}, {res['n_users']} человек", "",
             verdict]
    if stage == "val":
        lines.append(f"Прежние судьи выбрали бы: «доля 5★ − доля 1–2★» — {short(Variant(**_by_label(rows, res.get('chosen_by_share'))))}; "
                     f"сумма (оценка − 3) угаданных — {short(Variant(**_by_label(rows, res['chosen_by_value'])))}.")
    lines += ["", "| вариант | качество | разница | 5★ | 4★ | 3★ | 2★ | 1★ | угадано | общих книг | известность |",
              "|---|---|---|---|---|---|---|---|---|---|---|",
              f"| *случайные из прочитанного* | {_f(gc['base_quality'])} | | {_stars(gc, BASE)} | | | |"]
    by_quality = sorted(rows, key=lambda r: -_mean(r, "quality"))
    if stage == "val":
        best = {r["label"] for r in [r for r in by_quality if _mean(r, "hits") >= floor][:SHOWN]}
        must = {current.label(), chosen.label(), Variant(*REFERENCE).label(), anchor(rows)["label"]}
        shown = [r for r in by_quality if r["label"] in best | must]
    else:
        shown = by_quality
    for r in shown:
        g, v = r["groups"]["all"], Variant(**r["variant"])
        mark = (" **← выбран**" if v == chosen else "") + (f" ({was})" if v == current else "")
        if floor is not None and g["hits"]["mean"] < floor:
            mark += " (мало угадывает)"
        lines.append(f"| {short(v)}{mark} | {_f(g['quality'])} | {_f(g['quality_diff'], True)}{_ci(g['quality_diff'])} | "
                     f"{_stars(g)} | {g['hits']['mean']:.2f} | {g['same']['mean']:.1f} | {_known(g)} |")
    if len(shown) < len(rows):
        lines += ["", f"Остальные варианты ({len(rows) - len(shown)}) — в models/eval/layers_{stage}.json."]
    groups = [b for b in BUCKET_ORDER if b in rows[0]["groups"]]
    changed = chosen != current
    lines += ["", f"По этапам — числу оценок у человека ({short(chosen)}):", "",
              "| оценок | людей | качество" + (" | разница" if changed else "") + " | угадано | известность |",
              "|---|---|---|" + ("---|" if changed else "") + "---|---|"]
    for b in groups:
        g = by_variant[chosen]["groups"][b]
        diff = f" | {_f(g['quality_diff'], True)}{_ci(g['quality_diff'])}" if changed else ""
        lines.append(f"| {b} | {g['hits']['n']} | {_f(g['quality'])}{diff} | {g['hits']['mean']:.2f} | {_known(g)} |")
    lines += ["", "**Как читать.** " + HOW_TO_READ + " Варианты толпы «ценность»: ALS — вес ALS в толпе, вкус — вес "
              "модели вкуса, первые N — вкус переставляет только первые N книг толпы. **Общих книг** — сколько книг из "
              f"20 совпадает со списком {was[:-2]}ей выдачи. **Известность** — медиана числа оценок у книг списка "
              "(в ядре, среднее по людям): чем меньше, тем менее известные книги советуются."]
    return "\n".join(lines) + "\n"


def _known(g: dict) -> str:
    k = g.get("known", {}).get("mean")
    return "—" if k is None else f"{k:,.0f}".replace(",", " ")


def _by_label(rows: list[dict], label: str | None) -> dict:
    return next(r["variant"] for r in rows if r["label"] == label)


def profiles(*, clean_dir: Path, models_dir: Path, profiles_dir: Path, top: int = 20) -> str:
    """Топ-N каждого профиля profiles/*.csv: прежняя выдача (до последнего `layers val`) и выбранная рядом — поиск
    дефектов глазами."""
    from booksengine.model.filters import ListPicker, RatedFilter, pick_top, work_info
    from booksengine.model.matrix import catalog_works
    from booksengine.model.series import SeriesIndex, exclusion
    from booksengine.recommend import read_profile
    work_ids = catalog_works(clean_dir / "ratings.parquet")
    layers = Layers.from_models(models_dir)
    info = work_info(clean_dir, work_ids)
    saved = json.loads((models_dir / "layers" / "params.json").read_text())
    chosen = Variant(**saved["variant"])
    ref = Variant(**saved.get("baseline", asdict(Variant(*REFERENCE))))   # прежняя выдача — до последнего `layers val`
    top_cols = layers.mix.ease.top_cols
    pos_of = np.full(len(work_ids), -1)
    pos_of[top_cols] = np.arange(len(top_cols))
    series = SeriesIndex(info.title.tolist())
    picker = ListPicker(info, series)
    out = [f"# Топ-{top} на профилях: прежняя выдача «{ref.label()}» и выбранная «{chosen.label()}»", "",
           "Списки собраны правилами выдачи (без сборников, поздний том — первой книгой серии, книг одного автора — "
           "не больше одной на каждые 10 мест). Без шанса и объяснения."]
    for csv in sorted(profiles_dir.glob("*.csv")):
        prof = read_profile(csv, work_ids, clean_dir)
        if prof.x.nnz == 0:
            continue
        excl = exclusion(prof.x, series)[:, top_cols].toarray() != 0
        crowds = {v: layers.crowd(v.crowd, prof.x, prof.dnf, v.als_weight) for v in (ref, chosen)}
        taste = layers.taste_z(prof.x)
        rated = RatedFilter(picker.books, prof.x.indices)
        lists = {v: pick_top(layers.combine(crowds[v], taste, excl, v), top_cols, pos_of, picker, [rated], top)[0].tolist()
                 for v in (ref, chosen)}
        before = set(lists[ref])
        out += ["", f"## {csv.stem} ({prof.x.nnz} оценок)", "",
                "| # | прежняя | выбранная |", "|---|---|---|"]
        for i in range(top):
            cells = []
            for v in (ref, chosen):
                if i >= len(lists[v]):
                    cells.append("")
                    continue
                c = lists[v][i]
                new = " (новая)" if v == chosen and c not in before else ""
                cells.append(f"{info.title[c]} — {info.author[c] or '?'}{new}")
            out.append(f"| {i + 1} | {cells[0]} | {cells[1]} |")
        out.append(f"\nСовпадает книг: {len(before & set(lists[chosen]))} из {top}.")
    return "\n".join(out) + "\n"


# Подбор толпы «ценность» (`tune_like`): регуляризация λ и веса звёзд на входе (1★ … 5★).
# Сохранённый models/ease_like всегда в сравнении и не переобучается. W1: 3★ — слабый плюс «прочитал»; W2: пятёрка
# весит больше; W3 (предложение пользователя): всё положительно и удваивается — пятёрка в 16 раз весомее единицы,
# сигнала «не понравилось» на входе нет. Веса приводятся к масштабу «максимум 2» (`ease.normalize_weights`).
W0, W1, W2 = (-2.0, -1.0, 0.0, 1.0, 2.0), (-2.0, -1.0, 0.5, 1.0, 2.0), (-2.0, -1.0, 0.0, 1.0, 3.0)
W3 = (0.125, 0.25, 0.5, 1.0, 2.0)   # 1/2/4/8/16 после приведения масштаба (normalize_weights)
# W4 — мягкие минусы: одна двойка не топит автора, как у W0/W1; W5 — низкие оценки не тянут ни вверх, ни вниз
W4, W5 = (-1.0, -0.5, 0.5, 1.0, 2.0), (0.0, 0.0, 0.5, 1.0, 2.0)
# Веса звёзд выбраны по смыслу оценок (решение пользователя 2026-09-25): W4 — наравне с лучшим по замеру (0.470 против
# 0.473 у W1, разница в пределах шума). Подбирается только λ — иначе ночь вернула бы W1 по шуму.
LIKE_GRID = [(lam, W4) for lam in (250.0, 500.0)]
# Каждая настройка судится этими вариантами тем же судьёй, что `layers val` (`choose`): без вкуса, со вкусом, со вкусом
# внутри первых 300 толпы. Весь перебор веса вкуса и ALS — потом, `layers val` на выбранной толпе.
LIKE_JUDGED = (Variant("like", 0.0, None, 0.25), Variant("like", 1.0, None, 0.25), Variant("like", 1.0, 300, 0.25))


def _setting(row: dict) -> tuple:
    return float(row["lam"]), tuple(float(w) for w in row["weights"]), int(row.get("min_user", 20))


def _same_numbers(old: dict, new: dict) -> bool:
    """Прошлый прогон считал на тех же данных, компонентах и судье: у сохранённой толпы те же цифры всех вариантов."""
    try:
        return all(np.isclose(old["variants"][lab][k]["mean"], g[k]["mean"], rtol=1e-9, atol=1e-12)
                   for lab, g in new["variants"].items() for k in ("quality", "hits"))
    except (KeyError, TypeError):
        return False


def tune_like(*, clean_dir: Path, split_dir: Path, models_dir: Path, eval_dir: Path, grid=LIKE_GRID,
              fit_kw: dict | None = None, force: bool = False, min_user: int = 20) -> dict:
    """Обучает толпу «ценность» с каждой настройкой и судит её вариантами LIKE_JUDGED тем же судьёй, что `layers val`:
    качество списка у лучшего варианта из тех, что угадывают не меньше GUARD от сохранённой толпы без вкуса; если не угадывает
    ни один — настройка не подходит. Нынешняя — сохранённая толпа (первая в переборе) с вариантом из models/layers,
    без него — первый из LIKE_JUDGED. Лучшая настройка записывается в models/ease_like и models/mix_like; force —
    записать последнюю настройку сетки, даже если она хуже (решение пользователя); min_user — обучать новые настройки
    только на людях с ≥ min_user оценок (сохранённая — со своим порогом).

    Сохранённую толпу новая настройка сменяет, только если лучше неё уверенно: парная разность качества на тех же людях
    (лучший вариант настройки против лучшего варианта сохранённой) выше нуля по всему 95% интервалу — иначе выбор
    шёл бы по шуму. Для этого качество каждого человека хранится в ease_like_tune.json.

    Посчитанное в прошлый прогон (eval_dir/ease_like_tune.json) не пересчитывается, если сохранённая толпа дала те же
    цифры, что тогда: данные, компоненты и судья те же. Лучшая из таких настроек обучается заново один раз — чтобы
    записать её. После — `layers val` и `layers test` заново (отпечатки компонентов меняются)."""
    import time

    from booksengine.model.ease import EASE, EASELike, normalize_weights
    from booksengine.model.filters import work_info
    from booksengine.model.matrix import load_train
    ratings_path = clean_dir / "ratings.parquet"
    train = load_train(ratings_path, split_dir / "holdout_users.parquet")
    info = work_info(clean_dir, train.work_ids)
    hold = _hold(ratings_path, split_dir, "val", train.work_ids)
    base = Layers.from_models(models_dir)
    chosen = read_params(models_dir / "layers").get("variant") if (models_dir / "layers" / "params.json").exists() else None
    now = Variant(**chosen) if chosen and chosen["crowd"] == "like" else LIKE_JUDGED[0]
    judged = list(dict.fromkeys([now, *LIKE_JUDGED, ANCHOR_LIKE]))
    grid = [(float(lam), normalize_weights(w), int(min_user)) for lam, w in grid]
    saved = read_params(models_dir / "ease_like") if (models_dir / "ease_like" / "params.json").exists() else {}
    if saved:  # сохранённая толпа — всегда точка сравнения: без неё прогон из одной настройки записал бы худшую
        cur = (float(saved["lam"]), tuple(saved.get("weights", W0)), int(saved.get("min_user", 20)))
        grid = [cur] + [g for g in grid if g != cur]
    path = eval_dir / "ease_like_tune.json"
    users = [int(u) for u in hold.user_ids]
    prev_run = json.loads(path.read_text()) if path.exists() else {}
    # без качества по людям парно не сравнить, без распределения по звёздам — не показать
    prev = ({_setting(r): r for r in prev_run.get("results", [])
             if "per_user" in r and all("s5" in g for g in r["variants"].values())}
            if prev_run.get("user_ids") == users else {})
    names = {v.label(): short(v) for v in judged}
    pop = popularity(ratings_path, train.work_ids)
    reuse = False
    results, best, best_model, baseline, ref_q = [], None, None, None, None
    eval_dir.mkdir(parents=True, exist_ok=True)
    for lam, weights, mu in grid:
        is_saved = bool(saved) and cur == (lam, weights, mu)
        old = prev.get((lam, weights, mu))
        if reuse and old is not None and not is_saved:
            row, ease = old | {"saved": False, "reused": True}, None
        else:
            t0 = time.perf_counter()
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
            per = evaluate(layers, hold, info, judged, pop=pop)
            summ = summarize(per, judged[0])
            baseline = baseline or anchor(summ)       # опора — сохранённая толпа (первая настройка) без вкуса
            row = {"lam": lam, "weights": list(weights), "min_user": mu, "fit_seconds": fit_s, "saved": is_saved,
                   "variants": {r["label"]: {k: r["groups"]["all"][k] for k in
                                             ("quality", "hits", "five_minus_low", *STARS)} for r in summ},
                   "per_user": {v.label(): [None if np.isnan(x) else round(float(x), 5) for x in per[v]["quality"]]
                                for v in judged}}
            if is_saved:
                reuse = old is not None and _same_numbers(old, row)
        top = choose([{"label": lab, "groups": {"all": g}} for lab, g in row["variants"].items()], baseline)
        row["best_variant"] = None if top is None else top["label"]
        row["quality"] = None if top is None else row["variants"][top["label"]]["quality"]
        row["best_short"] = None if top is None else names.get(top["label"], top["label"])
        q = None if top is None else np.array([np.nan if x is None else x for x in row["per_user"][top["label"]]])
        if is_saved:
            ref_q = q
        row["quality_diff"] = (None if q is None or ref_q is None
                               else metrics.bootstrap(q - ref_q, np.random.default_rng(0), 1000))
        d = row["quality_diff"]
        sure = is_saved or ref_q is None or (d is not None and d["lo"] is not None and d["lo"] > 0)
        results.append(row)
        v = -np.inf if top is None else _mean(top, "quality")
        how = ("уже посчитано в прошлый прогон" if row.get("reused")
               else f"обучение {row['fit_seconds']} с")
        print(f"«ценность» λ = {lam:g}, веса {weights}, люди с ≥ {mu} оценок: "
              + (f"качество списка {v:.3f} ({row['best_short']})" if top is not None
                 else f"не подходит — ни один вариант не угадывает {GUARD:.0%} от сохранённой толпы без вкуса")
              + ("" if d is None or is_saved else f", разница с сохранённой {_f(d, True)}{_ci(d)}")
              + f", {how}", flush=True)
        path.write_text(json.dumps({"user_ids": users, "results": results}, ensure_ascii=False))
        last = (lam, weights, mu) == grid[-1]
        if (force and last) or (not force and (best is None or (sure and v > best))):
            best, best_model = v, (lam, weights, mu, ease)
        else:
            del ease
    lam, weights, mu, ease = best_model
    if not (saved and cur == (lam, weights, mu)):
        if ease is None:   # лучшая посчитана в прошлый прогон — обучить заново, чтобы записать
            print(f"«ценность» λ = {lam:g}, веса {weights}: обучаю заново, чтобы записать", flush=True)
            ease = EASELike(lam=lam, weights=weights, min_user=mu, **(fit_kw or {}))
            ease.fit(train)
        ease.save(models_dir / "ease_like")
        mix = Mix(models_dir / "als_neg", models_dir / "ease_like")
        mix.fit(train)
        mix.configure(als_weight=0.5, ease_input=weights)
        mix.save(models_dir / "mix_like")
    return {"results": results, "floor": GUARD * _mean(baseline, "hits"),
            "best": {"lam": lam, "weights": list(weights), "min_user": mu},
            "kept": bool(saved) and cur == (lam, weights, mu)}


def _setting_name(r: dict) -> str:
    who = "" if r.get("min_user", 20) == 20 else f", люди с ≥ {r['min_user']} оценок"
    return f"λ {r['lam']:g}, веса {'/'.join(f'{w:g}' for w in r['weights'])}{who}"


def report_like(res: dict) -> str:
    b = res["best"]
    best = next(r for r in res["results"] if _setting(r) == (b["lam"], tuple(b["weights"]), b.get("min_user", 20)))
    if res.get("kept", best.get("saved")):
        verdict = f"**Сохранённая толпа осталась: {_setting_name(best)}** — ни одна настройка не лучше уверенно."
    else:
        d = best.get("quality_diff")
        verdict = (f"**Выбрана: {_setting_name(best)}**" + (f" — разница с сохранённой {_f(d, True)}{_ci(d)}."
                                                          if d else "."))
    lines = ["# Подбор толпы «ценность» — валидация", "", verdict,
             "Дальше — `booksengine layers val` и `layers test`: вес ALS и вкуса подбираются заново под толпу.", "",
             "| настройка | лучший вариант | качество | разница с сохранённой | 5★ | 4★ | 3★ | 2★ | 1★ | угадано |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["results"]:
        name = _setting_name(r) + (" (сохранённая)" if r.get("saved") else "") + (
            " (из прошлого прогона)" if r.get("reused") else "")
        if r["best_variant"] is None:
            lines.append(f"| {name} | не подходит — мало угадывает | | | | | | | | |")
            continue
        g, d = r["variants"][r["best_variant"]], r.get("quality_diff")
        diff = "" if r.get("saved") or d is None else f"{_f(d, True)}{_ci(d)}"
        lines.append(f"| {name} | {r.get('best_short') or r['best_variant']} | {_f(g['quality'])} | {diff} | "
                     f"{_stars(g)} | {g['hits']['mean']:.2f} |")
    lines += ["", "**Как читать.** " + HOW_TO_READ + f" Порог угаданного здесь — {res['floor']:.2f} книги на человека. "
              "У каждой настройки (λ — регуляризация, веса звёзд 1★ … 5★ на входе) показан лучший из нескольких "
              "вариантов слоёв; полный перебор веса вкуса и ALS — в `layers val`."]
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
    baseline = read_params(models_dir / "layers").get("baseline", asdict(Variant(*REFERENCE)))
    variants = {"прежняя": Variant(**baseline), "новая": layers.variant}
    out = [f"# Проверка на своих оценках: прежняя выдача «{variants['прежняя'].label()}» и «{layers.variant.label()}»", "",
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
                "| оценка | книг | медиана места: прежняя | новая | в топ-100: прежняя | новая |", "|---|---|---|---|---|---|"]
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
        out += ["", "| книга | оценка | место: прежняя | новая |", "|---|---|---|---|"]
        order = np.lexsort((places["новая"], -r))
        for i in order:
            c = ok[i]
            pl = [("—" if np.isnan(places[n][i]) else f"{places[n][i]:.0f}") for n in variants]
            out.append(f"| {prof.names.get(int(c), info.title[c])} | {r[i]:.0f} | {pl[0]} | {pl[1]} |")
    return "\n".join(out) + "\n"


def why(*, clean_dir: Path, models_dir: Path, profile_csv: Path, query: str, top: int = 8) -> str:
    """Почему книга стоит там, где стоит, у выбранного варианта (models/layers). Книга ищется по
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
        raise ValueError("разбор сделан для толпы «ценность»: выбранный вариант другой")
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
