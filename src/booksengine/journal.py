"""Журнал выдач (profiles/history/): что и когда советовалось — и как потом оценено (`booksengine journal`).

Единственная проверка на настоящих будущих оценках, а не на отложенной выборке. Книга из выдачи в момент выдачи не
была оценена (оценённое не советуется), поэтому её оценка в профиле теперь — оценка, поставленная после совета.
Считается так же, как судья `layers`: доля пятёрок и доля 1–2★ среди прочитанного из советов (недочитанная = 1★);
рядом — те же доли среди остальных оценок человека, то есть книг, выбранных без модели.
"""
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


def save(recs, profile: str, model_fp: str, history_dir: Path) -> Path:
    """Выдача в журнал: место, книга, шанс, отпечаток модели. Одна выдача в день на профиль — повторный запуск
    перезаписывает. recs — `recommend.Rec`."""
    history_dir.mkdir(parents=True, exist_ok=True)
    path = history_dir / f"{profile}-{date.today().isoformat()}.csv"
    pd.DataFrame([{"rank": i, "goodreads_work_id": r.work_id, "title": r.title, "author": r.author,
                   "chance": r.chance, "model": model_fp} for i, r in enumerate(recs, 1)]).to_csv(path, index=False)
    return path


def files(history_dir: Path, profile: str) -> list[Path]:
    """Выдачи профиля по порядку дат: «<профиль>-ГГГГ-ММ-ДД.csv»."""
    return [p for p in sorted(history_dir.glob(f"{profile}-*.csv")) if p.stem[:-11] == profile]


def advised(history_dir: Path, profile: str) -> pd.DataFrame:
    """Все книги из выдач профиля — по первой выдаче, где книга появилась: дата, место, шанс, название."""
    parts = [pd.read_csv(p).assign(date=p.stem[-10:]) for p in files(history_dir, profile)]
    if not parts:
        return pd.DataFrame(columns=["goodreads_work_id", "date", "rank", "chance", "title", "author"])
    d = pd.concat(parts, ignore_index=True).sort_values(["date", "rank"], kind="stable")
    return d.drop_duplicates("goodreads_work_id")[["goodreads_work_id", "date", "rank", "chance", "title", "author"]]


def ratings(csv: Path, merges: dict[int, int] | None = None) -> pd.DataFrame:
    """Оценки профиля: произведение (тень → главное, как в `recommend`), оценка 1–5 (недочитанная — 1), название."""
    p = pd.read_csv(csv, dtype={"goodreads_work_id": "Int64"})
    dnf = p.get("status", pd.Series("", index=p.index)).eq("dnf")
    p["rating"] = p.rating.where(p.rating.notna() | ~dnf, 1)
    work = p.goodreads_work_id.map(lambda w: (merges or {}).get(int(w), int(w)) if pd.notna(w) else w)
    return pd.DataFrame({"work_id": work.astype("Int64"), "rating": p.rating.astype(float),
                         "title": p.get("title", pd.Series("", index=p.index)).fillna("")})


def _shares(r: np.ndarray) -> str:
    r = np.floor(np.asarray(r, dtype=np.float64) + 0.5)
    return (f"пятёрок {np.mean(r >= 5):.0%}, 1–2★ — {np.mean(r <= 2):.0%}, средняя {r.mean():.1f}"
            if len(r) else "—")


def report(profiles_dir: Path, history_dir: Path, clean_dir: Path | None = None) -> str:
    """Отчёт по всем профилям с выдачами в журнале. clean_dir — для слияния теней (work_merges.parquet), если есть."""
    merges = None
    if clean_dir is not None and (clean_dir / "work_merges.parquet").exists():
        m = pd.read_parquet(clean_dir / "work_merges.parquet")
        merges = dict(zip(m.shadow_work_id.astype(int), m.main_work_id.astype(int)))
    out = ["# Журнал выдач: что прочитано из советов и как оценено", "",
           "Прочитанное из советов — книги из выдач, которые теперь оценены в профиле (недочитанная = 1★). Если "
           "добавить в профиль книгу из советов, прочитанную давно, она тоже попадёт сюда."]
    for csv in sorted(profiles_dir.glob("*.csv")):
        adv = advised(history_dir, csv.stem)
        if adv.empty:
            continue
        r = ratings(csv, merges)
        rated = r.dropna(subset=["work_id"]).groupby("work_id").rating.mean()
        read = adv[adv.goodreads_work_id.isin(rated.index)].assign(
            rating=lambda d: d.goodreads_work_id.map(rated))
        others = r[~r.work_id.isin(read.goodreads_work_id).fillna(False).astype(bool)]
        days = [p.stem[-10:] for p in files(history_dir, csv.stem)]
        out += ["", f"## {csv.stem}", "",
                f"Выдач: {len(days)} ({days[0]}{'' if days[0] == days[-1] else ' … ' + days[-1]}), "
                f"разных книг в них: {len(adv)}. Прочитано из советов: {len(read)}"
                + (f" — {_shares(read.rating)}." if len(read) else " — пока сравнивать нечего."),
                f"Остальные оценки (книги, выбранные без модели, {len(others)}): {_shares(others.rating)}."]
        if len(read):
            out += ["", "| книга | посоветована | место | шанс | оценка |", "|---|---|---|---|---|"]
            out += [f"| {x.title} — {x.author} | {x.date} | {x.rank} | {x.chance}% | {x.rating:g} |"
                    for x in read.sort_values(["date", "rank"]).itertuples()]
    return "\n".join(out) + "\n"
