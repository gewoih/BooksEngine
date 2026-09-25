"""Журнал выдач (profiles/history/): что и когда советовалось — и как потом оценено (`booksengine journal`).

Единственная проверка на настоящих будущих оценках, а не на отложенной выборке. Книга из выдачи в момент выдачи не
была оценена (оценённое не советуется), поэтому её оценка в профиле теперь — оценка, поставленная после совета.
Считается так же, как судья `layers`: доля пятёрок и доля 1–2★ среди прочитанного из советов (недочитанная = 1★);
рядом — те же доли среди остальных оценок человека, то есть книг, выбранных без модели.

Отладка выдачи слоёв (колонки с `recommend._debug`) — чтобы проверить на живых оценках то, чего не проверить на
датасете: порог «угадано ≥ 80%» (`layers.GUARD`) выбран без замера и задаёт вес вкуса. Поэтому отдельно — книги,
которые поднял в список вкус (у толпы без вкуса их в списке не было), и «смелая» выдача (`recommend.BOLD_TASTE`):
она не показывается, но если книги из неё прочитаны сами и оценены высоко — порог стоит ослабить.
"""
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


def save(recs, profile: str, model_fp: str, history_dir: Path,
         bold: list[tuple[int, str, str]] | None = None) -> Path:
    """Выдача в журнал: место, книга, шанс, отпечаток модели, подпись и отладка слоёв (`Rec.debug`); строки
    `list = bold` — «смелая» выдача (id, название, автор), не показанная человеку. Одна выдача в день на профиль —
    повторный запуск перезаписывает. recs — `recommend.Rec`."""
    from booksengine.recommend import why_text
    history_dir.mkdir(parents=True, exist_ok=True)
    path = history_dir / f"{profile}-{date.today().isoformat()}.csv"
    rows = [{"list": "main", "rank": i, "goodreads_work_id": r.work_id, "title": r.title, "author": r.author,
             "chance": r.chance, "model": model_fp, "why": " | ".join(why_text(r)), **r.debug}
            for i, r in enumerate(recs, 1)]
    rows += [{"list": "bold", "rank": i, "goodreads_work_id": w, "title": t, "author": a, "model": model_fp}
             for i, (w, t, a) in enumerate(bold or [], 1)]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def files(history_dir: Path, profile: str) -> list[Path]:
    """Выдачи профиля по порядку дат: «<профиль>-ГГГГ-ММ-ДД.csv»."""
    return [p for p in sorted(history_dir.glob(f"{profile}-*.csv")) if p.stem[:-11] == profile]


ADVISED = ["goodreads_work_id", "date", "rank", "chance", "title", "author", "by_taste", "known"]


def _history(history_dir: Path, profile: str) -> pd.DataFrame:
    """Все строки выдач профиля; в старых выдачах нет отладки и «смелой» выдачи — там только показанное."""
    parts = [pd.read_csv(p).assign(date=p.stem[-10:]) for p in files(history_dir, profile)]
    if not parts:
        return pd.DataFrame(columns=ADVISED + ["list"])
    d = pd.concat(parts, ignore_index=True)
    for c in ("by_taste", "known"):
        if c not in d.columns:
            d[c] = np.nan
    d["list"] = d["list"].fillna("main") if "list" in d.columns else "main"
    return d.sort_values(["date", "rank"], kind="stable")


def advised(history_dir: Path, profile: str) -> pd.DataFrame:
    """Все книги из выдач профиля — по первой выдаче, где книга появилась: дата, место, шанс, название, поднял ли
    её вкус, известность (у выдач без отладки — пусто)."""
    d = _history(history_dir, profile)
    return d[d["list"] == "main"].drop_duplicates("goodreads_work_id")[ADVISED]


def bold_only(history_dir: Path, profile: str) -> pd.DataFrame:
    """Книги «смелых» выдач, которых не было ни в одной показанной, — по первой «смелой» выдаче."""
    d = _history(history_dir, profile)
    shown = set(d.goodreads_work_id[d["list"] == "main"])
    b = d[(d["list"] == "bold") & ~d.goodreads_work_id.isin(shown)]
    return b.drop_duplicates("goodreads_work_id")[["goodreads_work_id", "date", "rank", "title", "author"]]


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
        if adv.by_taste.notna().any():                  # выдачи с отладкой слоёв
            out += _debug_lines(read, bold_only(history_dir, csv.stem), rated)
        if len(read):
            out += ["", "| книга | посоветована | место | шанс | поднял вкус | известность | оценка |",
                    "|---|---|---|---|---|---|---|"]
            out += [f"| {x.title} — {x.author} | {x.date} | {x.rank} | {x.chance}% | {_yes(x.by_taste)} "
                    f"| {'' if pd.isna(x.known) else f'{x.known:,.0f}'} | {x.rating:g} |"
                    for x in read.sort_values(["date", "rank"]).itertuples()]
    return "\n".join(out) + "\n"


def _yes(v) -> str:
    return "" if pd.isna(v) else ("да" if v else "нет")


def _debug_lines(read: pd.DataFrame, bold: pd.DataFrame, rated: pd.Series) -> list[str]:
    """Проверка порога «угадано ≥ 80%»: прочитанное из советов — поднятое вкусом против поставленного толпой, и
    прочитанное из «смелой» выдачи, которую человек не видел."""
    from booksengine.recommend import BOLD_TASTE
    known = read.dropna(subset=["by_taste"])
    got = bold[bold.goodreads_work_id.isin(rated.index)]
    out = ["", "Порог «угадано ≥ 80%» (вес вкуса): если поднятое вкусом и прочитанное из «смелой» выдачи оценено не "
           "хуже поставленного толпой — порог стоит ослабить."]
    for name, m in (("поднял вкус", known.by_taste == 1), ("поставила толпа", known.by_taste == 0)):
        out.append(f"- {name}: прочитано {int(m.sum())}" + (f" — {_shares(known.rating[m])}" if m.any() else ""))
    out.append(f"- «смелая» выдача (вкус {BOLD_TASTE:g}, не показана): книг вне показанных {len(bold)}, прочитано "
               f"{len(got)}" + (f" — {_shares(got.goodreads_work_id.map(rated))}" if len(got) else ""))
    return out
