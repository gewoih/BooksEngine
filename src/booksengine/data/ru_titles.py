"""Русские названия книг и имена авторов для интерфейса (`booksengine ru-titles`) → data/ru/.

Источники по надёжности:
1. Fantlab (api.fantlab.ru): официальное русское название произведения и имя автора. Поиск — по английскому названию
   (до двоеточия и скобки), книга берётся, только если у найденного совпадает название (основное или одно из
   альтернативных) и фамилия основного автора; из нескольких — самое оцениваемое. Неверное русское название хуже
   английского: путает, какую книгу искать.
2. Русское издание Goodreads (язык rus или ISBN 978-5): самое оцениваемое, без «(серия, #N)».
3. У новых книг Amazon — `ru_title` единой базы (Wikidata и разметка `config/amazon_ru_titles.csv`).
Ответы Fantlab кэшируются (`fantlab_cache.jsonl`): прогон можно прерывать и продолжать.
"""
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

import duckdb
import pandas as pd

FANTLAB = "https://api.fantlab.ru/search-works?q={q}&page=1"
PAUSE = 0.3                        # между запросами одного потока — не нагружать чужой сервис
WORKERS = 4                        # параллельных запросов: ~3 в секунду (один запрос из Python — ~0.9 с)
SKIP_TYPES = {"cycle", "essay", "article", "review", "epigraph", "excerpt"}


def norm(s: str | None) -> str:
    """Название для сравнения: до двоеточия и скобки, без регистра, диакритики и знаков, без ведущего артикля."""
    s = re.sub(r"\s*[:(\[].*$", "", s if isinstance(s, str) else "")
    s = unicodedata.normalize("NFKD", s.lower())
    s = " ".join(re.findall(r"[^\W_]+", "".join(c for c in s if not unicodedata.combining(c))))
    return re.sub(r"^(the|a|an) ", "", s)


def surname(author: str | None) -> str:
    words = norm(author).split()
    return words[-1] if words else ""


def pick(matches: list[dict], title: str, author: str) -> dict | None:
    """Найденное Fantlab, если это та же книга (название и фамилия автора совпадают), самое оцениваемое."""
    t, last = norm(title), surname(author)
    if not t or not last:
        return None
    ok = []
    for m in matches:
        names = [m.get("name") or ""] + (m.get("altname") or "").split(";")
        if (m.get("name_eng") in SKIP_TYPES or not m.get("rusname")
                or t not in {norm(n) for n in names} or last not in norm(m.get("all_autor_name")).split()):
            continue
        ok.append(m)
    return max(ok, key=lambda m: m.get("markcount") or 0, default=None)


class Fantlab:
    def __init__(self, cache: Path, pause: float = PAUSE):
        self.cache, self.pause = cache, pause
        self.lock = threading.Lock()
        self.known: dict[str, list] = {}
        if cache.exists():
            for line in cache.read_text().splitlines():
                r = json.loads(line)
                self.known[r["q"]] = r["matches"]

    def search(self, q: str) -> list[dict]:
        if q not in self.known:
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(FANTLAB.format(q=urllib.parse.quote(q)), timeout=30) as r:
                        matches = json.load(r).get("matches", [])
                    break
                except (OSError, ValueError):
                    if attempt == 2:
                        raise
                    time.sleep(5 * (attempt + 1))
            keep = ("rusname", "name", "altname", "all_autor_name", "autor1_rusname", "autor2_id", "name_eng",
                    "markcount", "year")
            kept = [{k: m.get(k) for k in keep} for m in matches]
            with self.lock:
                self.known[q] = kept
                with self.cache.open("a") as f:
                    f.write(json.dumps({"q": q, "matches": kept}, ensure_ascii=False) + "\n")
            time.sleep(self.pause)
        return self.known[q]

    def prefetch(self, queries: list[str], workers: int = WORKERS, log=print) -> None:
        """Скачать незакэшированные запросы в несколько потоков."""
        todo = list(dict.fromkeys(q for q in queries if q not in self.known))
        t0 = time.monotonic()
        with ThreadPoolExecutor(workers) as ex:
            for i, _ in enumerate(ex.map(self.search, todo)):
                if (i + 1) % 500 == 0:
                    log(f"  Fantlab: {i + 1} из {len(todo)} ({time.monotonic() - t0:.0f} с)", flush=True)


def query(title: str) -> str:
    """Запрос к Fantlab — название до двоеточия и скобки («Dune (Dune Chronicles #1)» → «Dune»)."""
    return re.sub(r"\s*[:(\[].*$", "", title)


def books(clean_dir: Path, top: int | None) -> pd.DataFrame:
    """Книги ядра по убыванию числа оценок: work_id, название, основной автор и его id."""
    return duckdb.execute(f"""
        WITH prim AS (SELECT work_id, arg_min(author_id, position) AS author_id FROM read_parquet(?)
                      WHERE coalesce(role, '') = '' GROUP BY 1)
        SELECT w.work_id, w.title, a.name AS author, p.author_id, w.cf_ratings
        FROM read_parquet(?) w LEFT JOIN prim p USING (work_id) LEFT JOIN read_parquet(?) a USING (author_id)
        WHERE w.in_cf ORDER BY w.cf_ratings DESC, w.work_id {'' if top is None else f'LIMIT {int(top)}'}
    """, [str(clean_dir / "work_authors.parquet"), str(clean_dir / "works.parquet"),
          str(clean_dir / "authors.parquet")]).df()


def goodreads_ru(goodreads_dir: Path) -> pd.Series:
    """work_id → название самого оцениваемого русского издания (без «(серия, #N)»)."""
    d = duckdb.execute("""
        SELECT work_id, arg_max(coalesce(title_without_series, title), coalesce(ratings_count, 0)) AS t
        FROM read_parquet(?) WHERE language_code IN ('rus', 'ru') OR isbn13 LIKE '9785%' GROUP BY 1
    """, [str(goodreads_dir / "editions.parquet")]).df()
    t = d.t.str.replace(r"\s*\([^)]*#[^)]*\)\s*$", "", regex=True).str.strip()
    ok = t.str.contains(r"[А-Яа-яЁё]", regex=True)
    return pd.Series(t[ok].to_numpy(), index=d.work_id[ok].to_numpy())


def build(*, clean_dir: Path, goodreads_dir: Path, out_dir: Path, top: int | None = 30_000, log=print) -> dict:
    """Русские названия книг ядра (первые top по числу оценок — их видно в интерфейсе) и имена авторов →
    out_dir/titles.parquet (work_id, ru_title, source), out_dir/authors.parquet (author_id, ru_name)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    b = books(clean_dir, top)
    fl = Fantlab(out_dir / "fantlab_cache.jsonl")
    rows, names = [], []
    asked = [r for r in b.itertuples()
             if r.work_id < 100_000_000 and norm(r.title) and surname(r.author)]    # новые книги Amazon — ниже
    fl.prefetch([query(r.title) for r in asked], log=log)
    for r in asked:
        m = pick(fl.search(query(r.title)), r.title, r.author)
        if m:
            rows.append((r.work_id, m["rusname"], "fantlab"))
            if m.get("autor1_rusname") and not m.get("autor2_id"):
                names.append((r.author_id, m["autor1_rusname"]))
    fantlab = pd.DataFrame(rows, columns=["work_id", "ru_title", "source"])
    gr = goodreads_ru(goodreads_dir)
    gr = gr[gr.index.isin(b.work_id) & ~gr.index.isin(fantlab.work_id)]
    parts = [fantlab, pd.DataFrame({"work_id": gr.index, "ru_title": gr.to_numpy(), "source": "goodreads"})]
    cols = set(duckdb.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(clean_dir / "works.parquet")]).df()
               .column_name)
    if "ru_title" in cols:
        az = duckdb.execute("SELECT work_id, ru_title FROM read_parquet(?) WHERE source = 'amazon' AND ru_title <> ''",
                            [str(clean_dir / "works.parquet")]).df()
        parts.append(az.assign(source="amazon"))
    titles = pd.concat(parts, ignore_index=True).drop_duplicates("work_id")
    titles.to_parquet(out_dir / "titles.parquet", index=False)
    authors = (pd.DataFrame(names, columns=["author_id", "ru_name"]).dropna()
               .groupby("author_id").ru_name.agg(lambda s: s.value_counts().index[0]).reset_index())
    authors.to_parquet(out_dir / "authors.parquet", index=False)
    out = {"книг": len(b), "русских названий": len(titles), **titles.source.value_counts().to_dict(),
           "авторов": len(authors)}
    log(f"Готово: {out}", flush=True)
    return out
