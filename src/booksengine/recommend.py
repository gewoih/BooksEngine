"""`booksengine recommend`: оценки человека из CSV → топ книг с шансом и объяснением; выдача пишется в журнал.

Модель — слои «толпа + вкус» (models/layers), если выбраны, иначе смесь (models/mix); `model="mix"` — смесь явно
(ею считает приложение). Список собирают правила `filters.ListPicker`: без сборников, поздний том неначатой серии —
первой книгой, не больше одной книги автора на каждые 10 мест.

CSV: `goodreads_work_id`, `rating` 1–5, необязательно `status` (`dnf` без оценки = 1; во входе
EASE недочитанная книга весит 0 — `mix.DNF_INPUT`) и `title` (так книга
называется в объяснении). Книгу с произведением сопоставляет нейросеть заранее, своего поиска нет.
Тень из `work_merges.parquet` заменяется главным произведением,
несколько строк одного произведения — средней оценкой (как издания при очистке).
"""
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import scipy.sparse as sp

from booksengine import journal
from booksengine.model import explain, metrics
from booksengine.model.base import fingerprint
from booksengine.model.chance import Chance, personal_pct
from booksengine.model.filters import WHY_REMOVED, ListPicker, RatedFilter, work_info
from booksengine.model.matrix import catalog_works
from booksengine.model.mix import Mix
from booksengine.model.series import SeriesIndex, exclusion

# почему книга из CSV не попала в модель
NO_ID, NOT_IN_CATALOG, NOT_IN_CORE = "нет goodreads_work_id", "нет в каталоге", "вне CF-ядра (мало оценок)"


@dataclass
class Profile:
    x: sp.csr_matrix              # 1 × книги ядра, оценка 1–5
    dnf: sp.csr_matrix            # 1 × книги ядра, 1 — недочитана (все строки книги в CSV — dnf)
    names: dict[int, str]         # столбец входа → как книга названа у человека
    skipped: list[tuple[str, str]]  # (книга, почему не учтена)
    outside: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))  # оценённые книги каталога вне ядра


# «Смелая» выдача для журнала: вкус 4 при том же отсечении, что у выбранного варианта, — смелее порога
# `layers.GUARD`. Не показывается, пишется рядом — прочитанное из неё покажет, не слишком ли осторожен порог.
BOLD_TASTE = 4.0


@dataclass
class Rec:
    work_id: int
    title: str
    author: str
    chance: int
    because: list[str]            # книги профиля, чьи читатели ценят эту (толпа), — названия
    despite: str | None           # книга профиля, чьи читатели её не ценят
    rated: dict[str, str] = field(default_factory=dict)   # название книги профиля → оценка («5★», «не дочитал»)
    taste: str | None = None      # что говорит вкус («за: Сиддхартха 5★ (у всех 3.9) — похожа»), если сдвигает заметно
    debug: dict = field(default_factory=dict)             # для журнала: известность, место у толпы, поднял ли вкус


@dataclass
class Result:
    recs: list[Rec]
    removed: list[tuple[str, str]] = field(default_factory=list)  # (книга, почему не в списке) — `filters.ListPicker`
    skipped: list[tuple[str, str]] = field(default_factory=list)
    n_used: int = 0


def read_profile(path: Path, work_ids: np.ndarray, clean_dir: Path) -> Profile:
    p = pd.read_csv(path, dtype={"goodreads_work_id": "Int64"})
    for col in ("goodreads_work_id", "rating"):
        if col not in p.columns:
            raise ValueError(f"{path}: нет колонки {col}")
    if "title" not in p.columns:
        p["title"] = None
    dnf = p.get("status", pd.Series("", index=p.index)).eq("dnf")
    p["rating"] = p.rating.where(p.rating.notna() | ~dnf, 1)
    bad = p[p.rating.isna() | ~p.rating.isin([1, 2, 3, 4, 5])]
    if len(bad):
        raise ValueError(f"{path}: оценка — целое 1–5 (строки {', '.join(str(i + 2) for i in bad.index)})")

    merges = pd.read_parquet(clean_dir / "work_merges.parquet").set_index("shadow_work_id").main_work_id
    p["work_id"] = p.goodreads_work_id.map(lambda w: merges.get(w, w) if pd.notna(w) else w).astype("Int64")
    catalog = set(duckdb.execute("SELECT work_id FROM read_parquet(?)",
                                 [str(clean_dir / "works.parquet")]).fetchnumpy()["work_id"].tolist())
    name = p.title.fillna(p.goodreads_work_id.astype(str))
    why = np.select([p.work_id.isna(), ~p.work_id.isin(catalog), ~p.work_id.isin(work_ids)],
                    [NO_ID, NOT_IN_CATALOG, NOT_IN_CORE], "")
    skipped = [(n, w) for n, w in zip(name, why) if w]

    used = p[why == ""].assign(name=name[why == ""])
    g = used.assign(dnf=dnf[why == ""]).groupby("work_id", sort=True).agg(
        rating=("rating", "mean"), name=("name", "first"), dnf=("dnf", "all"))
    cols = np.searchsorted(work_ids, g.index.to_numpy(dtype=np.int64))
    x, d = (sp.csr_matrix((v, (np.zeros(len(cols), dtype=int), cols)), shape=(1, len(work_ids)))
            for v in (g.rating.to_numpy(np.float32), g.dnf.to_numpy(np.float32)))
    d.eliminate_zeros()
    outside = np.unique(p.work_id[why == NOT_IN_CORE].to_numpy(dtype=np.int64))
    return Profile(x, d, dict(zip(cols.tolist(), g.name)), skipped, outside)


def load_model(models_dir: Path, model: str | None = None):
    """Модель выдачи и её шанс: лучшая — слои «толпа + вкус» (models/layers), если выбрана; иначе смесь (models/mix).
    model="mix" — смесь явно (приложение и его эталон пока считают ею)."""
    from booksengine.model.layers import Layers
    layers_dir = models_dir / "layers"
    if model != "mix" and (layers_dir / "params.json").exists():
        model, d = Layers.load(layers_dir), layers_dir
    else:
        model, d = Mix.load(models_dir / "mix"), models_dir / "mix"
    if not (d / "chance.json").exists():
        raise FileNotFoundError(f"нет {d / 'chance.json'}: запустите `booksengine calibrate {d.name}`")
    return model, Chance.load(d / "chance.json", model_fp=fingerprint(d)), fingerprint(d)


def recommend(ratings_csv: Path, *, clean_dir: Path, models_dir: Path, top: int = 20,
              history_dir: Path | None = None, model: str | None = None, rules: bool = True) -> Result:
    """rules=False — без правил списка (сборники, поздние тома, книги автора), только «уже оценено»: так считает
    приложение, и его эталон строится без них."""
    work_ids = catalog_works(clean_dir / "ratings.parquet")
    prof = read_profile(ratings_csv, work_ids, clean_dir)
    if prof.x.nnz == 0:
        return Result([], skipped=prof.skipped)
    model, chance, model_fp = load_model(models_dir, model)
    # оценённые книги вне ядра — строками после ядра: в модель не входят, но их издания в ядре — «уже оценено»
    info = work_info(clean_dir, np.concatenate([work_ids, prof.outside]))
    n = len(work_ids)

    # как при калибровке шанса: вход и начатые серии — не кандидаты
    series = SeriesIndex(info.title[:n].tolist())
    sc = model.score(prof.x, prof.dnf)[0].astype(np.float64)
    ex = exclusion(prof.x, series)
    sc[ex.indices] = -np.inf
    order = np.argsort(-sc, kind="stable")
    order = order[np.isfinite(sc[order])]

    picker = ListPicker(info, series)
    rated = RatedFilter(picker.books, np.concatenate([prof.x.indices, np.arange(n, len(info))]))
    picked, removed = picker.pick(order, lambda c: sc[c], top, rated, rules=rules)
    picked = np.array(picked, dtype=np.int64)

    r = metrics.rounded(prof.x.data)
    pct = chance.predict(personal_pct(sc, picked), int((r >= 4).sum()), prof.x.nnz)
    if isinstance(model, Mix):
        in_cols, contrib = explain.contributions(model, prof.x, picked, prof.dnf)
        taste, const = np.zeros_like(contrib), np.zeros(len(picked))
    else:
        in_cols, contrib, taste, const = explain.layers_parts(model, prof.x, picked, prof.dnf)
    names = [prof.names[c] for c in in_cols.tolist()]
    dnf = set(prof.dnf.indices.tolist())
    stars = [("не дочитал" if c in dnf else f"{v:g}★") for c, v in zip(in_cols.tolist(), prof.x.data.tolist())]
    recs = []
    for k, c in enumerate(picked):
        why = explain.reason(contrib[:, k])
        shown = why.because + ([] if why.despite is None else [why.despite])
        note = explain.taste_note(taste[:, k], const[k])
        recs.append(Rec(int(work_ids[c]), info.title[c], info.author[c] or "", int(round(pct[k] * 100)),
                        [names[i] for i in why.because], None if why.despite is None else names[why.despite],
                        {names[i]: stars[i] for i in shown},
                        None if note is None else _taste_text(note, model.taste, c, in_cols, names, prof.x.data, stars)))
    bold = []
    if not isinstance(model, Mix):
        bold = _debug(model, prof, recs, picked, ex, picker, rated, top, rules, clean_dir, work_ids)
    res = Result(recs, [(f"{info.title[c]} — {info.author[c] or '?'}", WHY_REMOVED[w]) for c, w in removed],
                 prof.skipped, prof.x.nnz)
    if history_dir is not None:
        journal.save(res.recs, ratings_csv.stem, model_fp, history_dir,
                     bold=[(int(work_ids[c]), info.title[c], info.author[c] or "") for c in bold])
    return res


def _taste_text(note: explain.TasteNote, taste, col: int, in_cols: np.ndarray, names: list[str], x: np.ndarray,
                stars: list[str]) -> str:
    """«за: Сиддхартха 5★ (у всех 3.9) — похожа» — книга профиля, оценённая выше или ниже обычного, и похожа ли на
    неё советуемая по вкусу; без книги — «эту книгу обычно ставят 4.3». Обычная оценка — по модели вкуса (μ + b)."""
    side = "за" if note.sign > 0 else "против"
    if note.book is None:
        return f"{side}: эту книгу обычно ставят {taste.mu + taste.item_bias[col]:.1f}"
    i = note.book
    usual = taste.mu + taste.item_bias[in_cols[i]]
    similar = (note.sign > 0) == (x[i] > usual)       # выше обычного и тянет вверх — похожа; ниже и тянет вверх — нет
    return f"{side}: {names[i]} {stars[i]} (у всех {usual:.1f}) — {'похожа' if similar else 'не похожа'}"


def _debug(layers, prof: Profile, recs: list[Rec], picked: np.ndarray, ex: sp.csr_matrix, picker: ListPicker,
           rated: RatedFilter, top: int, rules: bool, clean_dir: Path, work_ids: np.ndarray) -> list[int]:
    """Отладка слоёв для журнала (пишет в `Rec.debug`): известность книги, её место у толпы без вкуса, поднял ли её
    вкус в список (списка толпы без вкуса она бы не попала), баллы толпы и вкуса. Возвращает «смелую» выдачу
    (вкус BOLD_TASTE) — столбцы ядра."""
    from dataclasses import replace

    from booksengine.model.layers import popularity
    top_cols = layers.mix.ease.top_cols
    v = layers.variant
    crowd = layers.crowd(v.crowd, prof.x, prof.dnf, v.als_weight)[0]
    taste = layers.taste_z(prof.x, prof.dnf)[0]

    def full(s: np.ndarray) -> np.ndarray:
        out = np.full(len(work_ids), -np.inf)
        out[top_cols] = s
        out[ex.indices] = -np.inf
        return out

    def pick(s: np.ndarray) -> list[int]:
        order = np.argsort(-s, kind="stable")
        return picker.pick(order[np.isfinite(s[order])], lambda c: s[c], top, rated, rules=rules)[0]

    c_full = full(crowd)
    by_crowd = set(pick(c_full))
    bold = pick(full(layers.combine(crowd[None, :], taste[None, :], np.zeros((1, len(top_cols)), bool),
                                    replace(v, taste_weight=BOLD_TASTE))[0]))
    place = np.argsort(np.argsort(-c_full, kind="stable"), kind="stable") + 1
    pop = popularity(clean_dir / "ratings.parquet", work_ids)
    pos = np.searchsorted(top_cols, picked)
    for rec, c, p in zip(recs, picked.tolist(), pos.tolist()):
        rec.debug = {"n_ratings": int(prof.x.nnz), "known": int(pop[c]), "crowd_place": int(place[c]),
                     "by_taste": int(c not in by_crowd), "crowd_z": round(float(crowd[p]), 3),
                     "taste_z": round(float(v.taste_weight * taste[p]), 3)}
    return bold


LEGEND = ("Как читать: «читатели» — книги из твоего профиля (рядом твоя оценка), чьи читатели ценят и эту; "
          "«несмотря на» — чьи читатели её не ценят. «Вкус» сравнивает твою оценку со средней у всех: книга похожа "
          "на ту, что ты оценил выше других, или не похожа на ту, что ниже; «обычно ставят» — книгу ценят все.")


def _book(name: str, rec: Rec) -> str:
    return f"{name} {rec.rated[name]}" if name in rec.rated else name


def why_text(r: Rec) -> list[str]:
    """Подпись к книге: строка толпы и, если вкус сдвигает заметно, строка вкуса."""
    line = ("читатели: " + ", ".join(_book(b, r) for b in r.because)) if r.because else "по профилю в целом"
    if r.despite:
        line += f"; несмотря на: {_book(r.despite, r)}"
    return [line] + ([f"вкус {r.taste}"] if r.taste else [])


def format_result(res: Result) -> str:
    out = [f"Учтено оценок: {res.n_used}", "", LEGEND, ""]
    for i, r in enumerate(res.recs, 1):
        out.append(f"{i:3}. {r.title} — {r.author}  [{r.chance}%]")
        out += [f"      {w}" for w in why_text(r)]
    if res.removed:
        out += ["", "Убраны из списка:"] + [f"  - {t}: {w}" for t, w in res.removed]
    if res.skipped:
        out += ["", "Не учтены:"] + [f"  - {n}: {w}" for n, w in res.skipped]
    return "\n".join(out)
