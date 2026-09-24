"""`booksengine recommend` (TODO п. 11): оценки человека из CSV → топ книг смеси с шансом и объяснением.

CSV: `goodreads_work_id`, `rating` 1–5, необязательно `status` (`dnf` без оценки = 1; во входе
EASE недочитанная книга весит 0 — `mix.DNF_INPUT`) и `title` (так книга
называется в объяснении). Книгу с произведением сопоставляет нейросеть заранее, своего поиска нет
(docs/resheniya.md, «поиск по каталогу»). Тень из `work_merges.parquet` заменяется главным произведением,
несколько строк одного произведения — средней оценкой (как издания при очистке).
"""
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import scipy.sparse as sp

from booksengine.model import explain, metrics
from booksengine.model.base import fingerprint
from booksengine.model.chance import Chance, personal_pct
from booksengine.model.filters import RatedFilter, work_info
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


@dataclass
class Rec:
    work_id: int
    title: str
    author: str
    chance: int
    because: list[str]
    despite: str | None


@dataclass
class Result:
    recs: list[Rec]
    filtered: list[str] = field(default_factory=list)  # убраны фильтром «уже оценено по сути»
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
    return Profile(x, d, dict(zip(cols.tolist(), g.name)), skipped)


def recommend(ratings_csv: Path, *, clean_dir: Path, models_dir: Path, top: int = 20) -> Result:
    work_ids = catalog_works(clean_dir / "ratings.parquet")
    prof = read_profile(ratings_csv, work_ids, clean_dir)
    if prof.x.nnz == 0:
        return Result([], skipped=prof.skipped)
    mix_dir = models_dir / "mix"
    mix = Mix.load(mix_dir)
    chance = Chance.load(mix_dir / "chance.json", model_fp=fingerprint(mix_dir))
    info = work_info(clean_dir, work_ids)

    # как при калибровке шанса: вход и начатые серии — не кандидаты
    sc = mix.score(prof.x, prof.dnf)[0].astype(np.float64)
    ex = exclusion(prof.x, SeriesIndex(info.title.tolist()))
    sc[ex.indices] = -np.inf
    order = np.argsort(-sc, kind="stable")
    order = order[np.isfinite(sc[order])]

    rated = RatedFilter(info, prof.x.indices)
    picked, filtered = [], []
    for c in order:
        if len(picked) == top:
            break
        (filtered if rated.is_rated_already(c) else picked).append(c)
    picked = np.array(picked, dtype=np.int64)

    r = metrics.rounded(prof.x.data)
    pct = chance.predict(personal_pct(sc, picked), int((r >= 4).sum()), prof.x.nnz)
    in_cols, contrib = explain.contributions(mix, prof.x, picked, prof.dnf)
    names = [prof.names[c] for c in in_cols.tolist()]
    recs = []
    for k, c in enumerate(picked):
        why = explain.reason(contrib[:, k])
        recs.append(Rec(int(work_ids[c]), info.title[c], info.author[c] or "", int(round(pct[k] * 100)),
                        [names[i] for i in why.because], None if why.despite is None else names[why.despite]))
    return Result(recs, [f"{info.title[c]} — {info.author[c]}" for c in filtered], prof.skipped, prof.x.nnz)


def format_result(res: Result) -> str:
    out = [f"Учтено оценок: {res.n_used}", ""]
    for i, r in enumerate(res.recs, 1):
        why = "потому что: " + ", ".join(r.because) if r.because else "по общему вкусу"
        if r.despite:
            why += f"; несмотря на: {r.despite}"
        out.append(f"{i:3}. {r.title} — {r.author}  [{r.chance}%]")
        out.append(f"      {why}")
    if res.filtered:
        out += ["", "Убраны как уже оценённые (дубль, сборник или часть оценённого сборника):"]
        out += [f"  - {t}" for t in res.filtered]
    if res.skipped:
        out += ["", "Не учтены:"] + [f"  - {n}: {w}" for n, w in res.skipped]
    return "\n".join(out)
