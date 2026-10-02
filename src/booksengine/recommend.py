"""`booksengine recommend`: оценки человека из CSV → топ книг с шансом и объяснением; выдача пишется в журнал.

Модель — слои «толпа + вкус» (models/layers), если выбраны, иначе смесь (models/mix); `model="mix"` — смесь явно.
Приложение берёт ту же выдачу (`Engine` в памяти, `serve`). Список собирают правила `filters.ListPicker`: без сборников и поздних томов неначатых серий,
не больше одной книги автора на каждые 10 мест.

CSV: `goodreads_work_id`, `rating` 1–5, необязательно `status` (`dnf` без оценки = 1; во входе
EASE недочитанная книга весит 0 — `mix.DNF_INPUT`) и `title` (так книга
называется в объяснении). Книги без оценки: `status` = `want` — «хочу прочитать», иначе пустая оценка — «прочитано,
оценку не помню». Обе не советуются (и то же произведение под другим названием; у прочитанной — и продолжения серии) и
идут во вход толпы слабым плюсом (`mix.WANT_INPUT`, `mix.READ_INPUT`), во вкус — нет: оценки нет. Книгу с произведением сопоставляет нейросеть заранее, своего поиска нет.
Тень из `work_merges.parquet` заменяется главным произведением,
несколько строк одного произведения — средней оценкой (как издания при очистке).
"""
import hashlib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import scipy.sparse as sp

from booksengine import journal
from booksengine.model import explain, metrics
from booksengine.model.base import fingerprint
from booksengine.model.chance import Chance, personal_pct
from booksengine.model.filters import WHY_REMOVED, Books, ListPicker, RatedFilter, nonfiction, work_info
from booksengine.model.matrix import catalog_works
from booksengine.model.mix import READ_INPUT, WANT_INPUT, Mix
from booksengine.model.series import SeriesIndex, exclusion

# почему книга из CSV не попала в модель
NO_ID, NOT_IN_CATALOG, NOT_IN_CORE = "нет goodreads_work_id", "нет в каталоге", "вне CF-ядра (мало оценок)"


@dataclass
class Profile:
    x: sp.csr_matrix              # 1 × книги ядра, оценка 1–5
    dnf: sp.csr_matrix            # 1 × книги ядра, 1 — недочитана (все строки книги в CSV — dnf)
    names: dict[int, str]         # столбец входа → как книга названа у человека
    skipped: list[tuple[str, str]]  # (книга, почему не учтена)
    outside: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))  # книги профиля каталога вне ядра
    read: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))     # прочитаны без оценки (столбцы)
    want: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))     # «хочу прочитать» (столбцы)

    def _row(self, cols: np.ndarray, values: np.ndarray) -> sp.csr_matrix:
        return sp.csr_matrix((values.astype(np.float32), (np.zeros(len(cols), dtype=int), cols)), shape=self.x.shape)

    def seen(self) -> sp.csr_matrix:
        """1 × книги ядра: всё прочитанное — оценённое, недочитанное и без оценки (для исключения и серий)."""
        cols = np.union1d(self.x.indices, self.read)
        return self._row(cols, np.ones(len(cols)))

    def shelf(self) -> sp.csr_matrix:
        """1 × книги ядра: книги без оценки с весом во входе толпы («хочу прочитать», «прочитано без оценки»)."""
        cols = np.concatenate([self.want, self.read])
        o = np.argsort(cols)
        return self._row(cols[o], np.r_[np.full(len(self.want), WANT_INPUT), np.full(len(self.read), READ_INPUT)][o])

    def not_advised(self) -> np.ndarray:
        """Столбцы, которые не советуются по сути (`RatedFilter`): оценённое, прочитанное без оценки, полка."""
        return np.union1d(np.union1d(self.x.indices, self.read), self.want)


# Код, от которого зависит состав выдачи: его отпечаток пишется в журнал рядом с отпечатком модели (файлы models/) —
# правка кода меняет выдачу при тех же моделях (как исправление отсечения вкуса 2026-09-29)
CODE = ("recommend.py", "model/layers.py", "model/filters.py", "model/series.py", "model/mix.py", "model/als.py",
        "model/taste.py", "model/ease.py", "model/chance.py")


def code_fingerprint(root: Path = Path(__file__).parent, files: tuple[str, ...] = CODE) -> str:
    h = hashlib.blake2b(digest_size=8)
    for f in files:
        h.update(f.encode())
        h.update((root / f).read_bytes())
    return h.hexdigest()


# «Смелая» выдача для журнала: вкус 4 при том же отсечении, что у выбранного варианта, — смелее порога
# `layers.GUARD`. Не показывается, пишется рядом — прочитанное из неё покажет, не слишком ли осторожен порог.
BOLD_TASTE = 4.0
# два списка: художественная литература и нон-фикшн (`filters.nonfiction`) — у читателя двух областей одна не
# вытесняет другую, список выбирается под настроение
FICTION, NONFICTION = "Художественная литература", "Нон-фикшн"
# новые книги единой базы (после 2017, `data.merged`) — своим списком: в общем модель их занижает (люди Goodreads не
# могли их прочесть, а для неё это «не взяли»), а среди новых выбирает хорошо — на людях Amazon 5 её новинок угаданы
# вдвое чаще 5 самых популярных (`model.new_books`)
NEW, NEW_TOP = "Новинки (после 2017)", 10
NEW_NOTE = ("Шанс у новинок примерный: он настроен на людях Goodreads, а новинок они не читали. Место в списке — "
            "по той же модели, что и выше.")


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
    section: str = ""             # список: FICTION / NONFICTION, "" — один общий


@dataclass
class Result:
    recs: list[Rec]
    removed: list[tuple[str, str]] = field(default_factory=list)  # (книга, почему не в списке) — `filters.ListPicker`
    skipped: list[tuple[str, str]] = field(default_factory=list)
    n_used: int = 0
    n_read: int = 0               # прочитано без оценки: не советуется, во входе толпы — слабый плюс
    n_want: int = 0               # «хочу прочитать»: то же
    chance: str = "4–5★"          # какой шанс в скобках (`Chance.label`)
    bold: list[tuple[int, str, str, str]] = field(default_factory=list)  # «смелая» выдача для журнала: id, название, автор, список
    ranks: dict[int, list[tuple[str, int]]] = field(default_factory=dict)  # размер → (список, work_id) по порядку


def display_title(info: pd.DataFrame, c: int) -> str:
    """Название книги в выдаче: у новой книги Amazon с известным переводом — «русское / английское»."""
    ru = info.ru_title[c] if "ru_title" in info else None
    return f"{ru} / {info.title[c]}" if isinstance(ru, str) and ru else info.title[c]


def read_profile(path: Path, work_ids: np.ndarray, clean_dir: Path, id_col: str = "goodreads_work_id") -> Profile:
    """id_col — колонка с id в пространстве work_id (для книг это сам goodreads_work_id; для фильмов —
    предварительно сопоставленный movieId, не imdb_id: канонический профиль фильмов хранит imdb_id, но
    read_profile работает с уже сопоставленным id, как и для книг — `movielens.materialize_profile`)."""
    return profile_from_frame(pd.read_csv(path, dtype={id_col: "Int64"}), work_ids, clean_dir, id_col, where=str(path))


def profile_from_frame(p: pd.DataFrame, work_ids: np.ndarray, clean_dir: Path, id_col: str = "goodreads_work_id",
                       where: str = "профиль") -> Profile:
    """То же, что `read_profile`, из таблицы в памяти (приложение: `serve`); where — как назвать источник в ошибке."""
    p = p.copy()
    for col in (id_col, "rating"):
        if col not in p.columns:
            raise ValueError(f"{where}: нет колонки {col}")
    p[id_col] = p[id_col].astype("Int64")
    if "title" not in p.columns:
        p["title"] = None
    status = p.get("status", pd.Series("", index=p.index)).fillna("")
    dnf = status.eq("dnf")
    p["rating"] = p.rating.where(p.rating.notna() | ~dnf, 1)
    unrated = p.rating.isna().to_numpy()      # прочитано, оценку не помню, или «хочу прочитать»
    wanted = unrated & status.eq("want").to_numpy()
    bad = p[~unrated & ~p.rating.isin([1, 2, 3, 4, 5])]
    if len(bad):
        raise ValueError(f"{where}: оценка — целое 1–5 (строки {', '.join(str(i + 2) for i in bad.index)})")

    if id_col == "goodreads_work_id" and p[id_col].isna().any() and "title_en" in p.columns:
        # книги, которых нет в Goodreads (после 2017), — в единой базе с Amazon по названию и автору
        from booksengine.data.merged import match_new_works
        found = match_new_works(clean_dir, p.title_en, p.get("author", pd.Series(None, index=p.index)))
        p[id_col] = p[id_col].fillna(pd.Series(found.to_numpy(), index=p.index, dtype="Int64"))
    merges, catalog = _catalog(clean_dir)
    p["work_id"] = p[id_col].map(lambda w: merges.get(w, w) if pd.notna(w) else w).astype("Int64")
    name = p.title.fillna(p[id_col].astype(str))
    why = np.select([p.work_id.isna(), ~p.work_id.isin(catalog), ~p.work_id.isin(work_ids)],
                    [NO_ID, NOT_IN_CATALOG, NOT_IN_CORE], "")
    skipped = [(n, w) for n, w in zip(name, why) if w]

    ok = (why == "") & ~unrated
    used = p[ok].assign(name=name[ok])
    g = used.assign(dnf=dnf[ok]).groupby("work_id", sort=True).agg(
        rating=("rating", "mean"), name=("name", "first"), dnf=("dnf", "all"))
    cols = np.searchsorted(work_ids, g.index.to_numpy(dtype=np.int64))
    x, d = (sp.csr_matrix((v, (np.zeros(len(cols), dtype=int), cols)), shape=(1, len(work_ids)))
            for v in (g.rating.to_numpy(np.float32), g.dnf.to_numpy(np.float32)))
    d.eliminate_zeros()
    outside = np.unique(p.work_id[why == NOT_IN_CORE].to_numpy(dtype=np.int64))

    def core_cols(mask) -> np.ndarray:
        return np.searchsorted(work_ids, np.unique(p.work_id[(why == "") & mask].to_numpy(dtype=np.int64)))
    read = np.setdiff1d(core_cols(unrated & ~wanted), cols)         # оценённое в другой строке — оценено
    want = np.setdiff1d(np.setdiff1d(core_cols(wanted), cols), read)
    names = dict(zip(cols.tolist(), g.name))
    for c, n in zip(np.searchsorted(work_ids, p.work_id[(why == "") & unrated].to_numpy(dtype=np.int64)).tolist(),
                    name[(why == "") & unrated]):
        names.setdefault(c, n)
    return Profile(x, d, names, skipped, outside, read, want)


@lru_cache(maxsize=4)
def _catalog(clean_dir: Path) -> tuple[dict, set]:
    """Слияния теней (тень → главное) и все work_id каталога — читаются один раз на папку (приложение спрашивает
    выдачу много раз за запуск)."""
    m = pd.read_parquet(clean_dir / "work_merges.parquet")
    catalog = set(duckdb.execute("SELECT work_id FROM read_parquet(?)",
                                 [str(clean_dir / "works.parquet")]).fetchnumpy()["work_id"].tolist())
    return dict(zip(m.shadow_work_id.tolist(), m.main_work_id.tolist())), catalog


def load_model(models_dir: Path, model: str | None = None):
    """Модель выдачи и её шанс: лучшая — слои «толпа + вкус» (models/layers), если выбрана; иначе смесь (models/mix).
    model="mix" — смесь явно."""
    from booksengine.model.layers import Layers
    layers_dir = models_dir / "layers"
    if model != "mix" and (layers_dir / "params.json").exists():
        model, d = Layers.load(layers_dir), layers_dir
    else:
        model, d = Mix.load(models_dir / "mix"), models_dir / "mix"
    if not (d / "chance.json").exists():
        raise FileNotFoundError(f"нет {d / 'chance.json'}: запустите `booksengine calibrate {d.name}`")
    return model, Chance.load(d / "chance.json", model_fp=fingerprint(d)), fingerprint(d)


def list_parts(clean_dir: Path, work_ids: np.ndarray, top: int,
               sections: bool = True) -> dict[str, tuple[np.ndarray | None, int]]:
    """Списки выдачи: название → (маска столбцов ядра, None — все книги; сколько книг). sections — художественная
    литература и нон-фикшн по top книг и, если в базе есть новые книги, «Новинки» (NEW_TOP); иначе один общий."""
    if not sections:
        return {"": (None, top)}
    from booksengine.data.merged import NEW_WORK_OFFSET
    nf, new = nonfiction(clean_dir, work_ids), work_ids >= NEW_WORK_OFFSET
    parts = {FICTION: (~nf & ~new, top), NONFICTION: (nf & ~new, top)}
    if new.any():
        parts[NEW] = (new, NEW_TOP)
    return parts


def pick_sections(order: np.ndarray, picker: ListPicker, rated: RatedFilter, rules: bool,
                  sections: dict[str, tuple[np.ndarray | None, int]]) -> tuple[list[tuple[str, int]], list[tuple[int, str]]]:
    """В каждый список sections (`list_parts`) — его число книг; правила списка — в каждом отдельно. Возвращает
    (список, столбец) по порядку и убранные правилами."""
    picked, removed = [], []
    for name, (mask, k) in sections.items():
        o = order if mask is None else order[mask[order]]
        p, r = picker.pick(o, k, rated, rules=rules)
        picked += [(name, c) for c in p]
        removed += r
    return picked, removed


class Engine:
    """Всё для выдачи, что не зависит от человека: модель и шанс, книги ядра, серии, правила списка. Грузится один раз
    (единая база — ~10 с); `recommend` — выдача одного профиля за доли секунды. Приложение держит его в памяти
    (`serve`), консольный `recommend` строит на один профиль."""

    def __init__(self, clean_dir: Path, models_dir: Path, model: str | None = None):
        self.clean_dir = clean_dir
        self.work_ids = catalog_works(clean_dir / "ratings.parquet")
        self.model, self.chance, self.model_fp = load_model(models_dir, model)
        self.info = work_info(clean_dir, self.work_ids)
        self.series = SeriesIndex(self.info.title.tolist())
        self.picker = ListPicker(self.info)
        self._parts: dict[tuple[int, bool], dict] = {}

    def parts(self, top: int, sections: bool) -> dict[str, tuple[np.ndarray | None, int]]:
        if (top, sections) not in self._parts:
            self._parts[top, sections] = list_parts(self.clean_dir, self.work_ids, top, sections)
        return self._parts[top, sections]

    def profile(self, frame: pd.DataFrame) -> Profile:
        return profile_from_frame(frame, self.work_ids, self.clean_dir)

    def scores(self, prof: Profile) -> tuple[np.ndarray, sp.csr_matrix]:
        """Баллы по всем книгам ядра (−∞ — не кандидат) и исключённые: как при калибровке шанса — вход, начатые серии
        и полка не кандидаты (и мест в отсечении вкуса не занимают)."""
        ex = (exclusion(prof.seen(), self.series) + prof._row(prof.want, np.ones(len(prof.want)))).tocsr()
        sc = self.model.score(prof.x, prof.dnf, prof.shelf(),
                              **({} if isinstance(self.model, Mix) else {"exclude": ex}))[0].astype(np.float64)
        sc[ex.indices] = -np.inf
        return sc, ex

    def rated(self, prof: Profile) -> RatedFilter:
        """«Уже оценено по сути»: оценённое и полка ядра, а книги профиля вне ядра — строками после ядра (в модель не
        входят, но их издания в ядре — то же произведение)."""
        n, books = len(self.work_ids), self.picker.books
        if len(prof.outside):
            books = Books.concat(books, Books.of(work_info(self.clean_dir, prof.outside)))
        return RatedFilter(books, np.concatenate([prof.not_advised(), np.arange(n, n + len(prof.outside))]))

    def chance_pct(self, prof: Profile, sc: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """Шанс (0–1) для столбцов cols по баллам sc (`scores`); не кандидат — NaN."""
        r = metrics.rounded(prof.x.data)
        return self.chance.predict(personal_pct(sc, cols), int((r >= self.chance.stars).sum()), prof.x.nnz)

    def recommend(self, prof: Profile, top: int = 20, rules: bool = True, sections: bool = True,
                  debug: bool = False, rank_tops: tuple[int, ...] = ()) -> Result:
        """debug — отладка слоёв для журнала (`Rec.debug`, «смелая» выдача в `Result.bold`); дольше на ~2 с.
        rank_tops — состав списков и других размеров (`Result.ranks`): правила списка зависят от размера (книг автора —
        одна на 10 мест), поэтому список из 20 — не первые 20 списка из 50."""
        if prof.x.nnz == 0:
            return Result([], skipped=prof.skipped)
        model, info, work_ids = self.model, self.info, self.work_ids
        sc, ex = self.scores(prof)
        order = np.argsort(-sc, kind="stable")
        order = order[np.isfinite(sc[order])]
        rated, parts = self.rated(prof), self.parts(top, sections)
        got, removed = pick_sections(order, self.picker, rated, rules, parts)
        picked = np.array([c for _, c in got], dtype=np.int64)

        pct = self.chance_pct(prof, sc, picked)
        shelf = prof.shelf()
        if isinstance(model, Mix):
            in_cols, contrib = explain.contributions(model, prof.x, picked, prof.dnf, shelf)
            taste, const = np.zeros_like(contrib), np.zeros(len(picked))
        else:
            in_cols, contrib, taste, const = explain.layers_parts(model, prof.x, picked, prof.dnf, shelf)
        names = [prof.names[c] for c in in_cols.tolist()]
        stars = [_label(prof, c) for c in in_cols.tolist()]
        rating_of = dict(zip(prof.x.indices.tolist(), prof.x.data.tolist()))
        x_in = np.array([rating_of.get(c, np.nan) for c in in_cols.tolist()])
        recs = []
        for k, c in enumerate(picked):
            why = explain.reason(contrib[:, k])
            shown = why.because + ([] if why.despite is None else [why.despite])
            note = explain.taste_note(taste[:, k], const[k])
            recs.append(Rec(int(work_ids[c]), display_title(info, c), info.author[c] or "", int(round(pct[k] * 100)),
                            [names[i] for i in why.because], None if why.despite is None else names[why.despite],
                            {names[i]: stars[i] for i in shown},
                            None if note is None else _taste_text(note, model.taste, c, in_cols, names, x_in, stars),
                            section=got[k][0]))
        bold = []
        if debug and not isinstance(model, Mix):
            bold = [(int(work_ids[c]), info.title[c], info.author[c] or "", sec) for sec, c in
                    _debug(model, prof, recs, picked, ex, self.picker, rated, rules, self.clean_dir, work_ids, parts)]
        ranks = {k: [(sec, int(work_ids[c])) for sec, c in pick_sections(order, self.picker, rated, rules,
                                                                          self.parts(k, sections))[0]]
                 for k in rank_tops}
        return Result(recs, [(f"{info.title[c]} — {info.author[c] or '?'}", WHY_REMOVED[w]) for c, w in removed],
                      prof.skipped, prof.x.nnz, len(prof.read), len(prof.want), self.chance.label(), bold, ranks)


def recommend(ratings_csv: Path, *, clean_dir: Path, models_dir: Path, top: int = 20,
              history_dir: Path | None = None, model: str | None = None, rules: bool = True,
              sections: bool = True) -> Result:
    """rules=False — без правил списка (сборники, поздние тома, книги автора), только «уже оценено». sections — два
    списка по top книг (художественная литература и нон-фикшн) и «Новинки», если они есть в базе (`list_parts`),
    иначе один общий."""
    prof = read_profile(ratings_csv, catalog_works(clean_dir / "ratings.parquet"), clean_dir)
    if prof.x.nnz == 0:
        return Result([], skipped=prof.skipped)
    eng = Engine(clean_dir, models_dir, model)
    res = eng.recommend(prof, top, rules, sections, debug=True)
    if history_dir is not None:
        journal.save(res.recs, ratings_csv.stem, eng.model_fp, history_dir, chance_of=res.chance,
                     code=code_fingerprint(), bold=res.bold)
    return res


def _label(prof: Profile, col: int) -> str:
    """Как книга входа показана в подписи: оценка («4★»), «не дочитал», «хочу прочитать», «прочитано»."""
    if col in set(prof.dnf.indices.tolist()):
        return "не дочитал"
    if col in set(prof.want.tolist()):
        return "хочу прочитать"
    if col in set(prof.read.tolist()):
        return "прочитано"
    return f"{prof.x[0, col]:g}★"


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
           rated: RatedFilter, rules: bool, clean_dir: Path, work_ids: np.ndarray,
           parts: dict[str, tuple[np.ndarray | None, int]]) -> list[tuple[str, int]]:
    """Отладка слоёв для журнала (пишет в `Rec.debug`): известность книги, её место у толпы без вкуса, поднял ли её
    вкус в список (списка толпы без вкуса она бы не попала), баллы толпы и вкуса. Возвращает «смелую» выдачу
    (вкус BOLD_TASTE) — (список, столбец ядра), по тем же спискам."""
    from dataclasses import replace

    from booksengine.model.layers import popularity
    top_cols = layers.mix.ease.top_cols
    v = layers.variant
    crowd = layers.crowd(v.crowd, prof.x, prof.dnf, v.als_weight, prof.shelf())[0]
    taste = layers.taste_z(prof.x, prof.dnf)[0]

    def full(s: np.ndarray) -> np.ndarray:
        out = np.full(len(work_ids), -np.inf)
        out[top_cols] = s
        out[ex.indices] = -np.inf
        return out

    def pick(s: np.ndarray) -> list[tuple[str, int]]:
        order = np.argsort(-s, kind="stable")
        return pick_sections(order[np.isfinite(s[order])], picker, rated, rules, parts)[0]

    c_full = full(crowd)
    by_crowd = {c for _, c in pick(c_full)}
    bold = pick(full(layers.combine(crowd[None, :], taste[None, :], np.isin(top_cols, ex.indices)[None, :],
                                    replace(v, taste_weight=BOLD_TASTE))[0]))
    place = np.argsort(np.argsort(-c_full, kind="stable"), kind="stable") + 1
    pop = popularity(clean_dir / "ratings.parquet", work_ids)
    pos = np.searchsorted(top_cols, picked)
    for rec, c, p in zip(recs, picked.tolist(), pos.tolist()):
        rec.debug = {"n_ratings": int(prof.x.nnz), "known": int(pop[c]), "crowd_place": int(place[c]),
                     "by_taste": int(c not in by_crowd), "crowd_z": round(float(crowd[p]), 3),
                     "taste_z": round(float(v.taste_weight * taste[p]), 3)}
    return bold


LEGEND = ("Как читать: «читатели» — книги из твоего профиля (рядом твоя оценка или «хочу прочитать» / «прочитано»), "
          "чьи читатели ценят и эту; "
          "«ценят те, кому не понравились» — книги, которые ты оценил низко, и эту ценят люди, которым они тоже не "
          "понравились; «несмотря на» — чьи читатели её не ценят. «Вкус» сравнивает твою оценку со средней у всех: книга похожа "
          "на ту, что ты оценил выше других, или не похожа на ту, что ниже; «обычно ставят» — книгу ценят все.")


def _book(name: str, rec: Rec) -> str:
    if name not in rec.rated:
        return name
    label = rec.rated[name]
    return f"{name} ({label})" if label in ("хочу прочитать", "прочитано") else f"{name} {label}"


def _low(stars: str | None) -> bool:
    """Книга профиля не понравилась: 1–2★ или не дочитана."""
    return stars is not None and (stars == "не дочитал" or (stars.endswith("★") and float(stars.rstrip("★")) <= 2))


def why_text(r: Rec) -> list[str]:
    """Подпись к книге: строка толпы и, если вкус сдвигает заметно, строка вкуса. Низко оценённая книга «за» —
    отдельно: эту книгу ценят те, кому та тоже не понравилась (а не «её читатели ценят эту»)."""
    liked = [b for b in r.because if not _low(r.rated.get(b))]
    low = [b for b in r.because if _low(r.rated.get(b))]
    parts = (["читатели: " + ", ".join(_book(b, r) for b in liked)] if liked else []) + (
        ["ценят те, кому не понравились: " + ", ".join(_book(b, r) for b in low)] if low else [])
    line = "; ".join(parts) or "по профилю в целом"
    if r.despite:
        line += f"; несмотря на: {_book(r.despite, r)}"
    return [line] + ([f"вкус {r.taste}"] if r.taste else [])


def format_result(res: Result) -> str:
    extra = [f"прочитано без оценки {res.n_read}"] * bool(res.n_read) + [f"хочу прочитать {res.n_want}"] * bool(res.n_want)
    out = [f"Учтено оценок: {res.n_used}" + (f"; без оценки (не советуются): {', '.join(extra)}" if extra else ""),
           "", f"В скобках — шанс, что поставишь книге {res.chance}, если прочтёшь (из десяти книг с шансом 40% "
               f"{res.chance} получат четыре). " + LEGEND]
    section, i = None, 0
    for r in res.recs:
        if r.section != section:
            section, i = r.section, 0
            out += ["", f"## {section}", ""] if section else [""]
            out += [NEW_NOTE, ""] if section == NEW else []
        i += 1
        out.append(f"{i:3}. {r.title} — {r.author}  [{r.chance}%]")
        out += [f"      {w}" for w in why_text(r)]
    if res.removed:
        out += ["", "Убраны из списка:"] + [f"  - {t}: {w}" for t, w in res.removed]
    if res.skipped:
        out += ["", "Не учтены:"] + [f"  - {n}: {w}" for n, w in res.skipped]
    return "\n".join(out)
