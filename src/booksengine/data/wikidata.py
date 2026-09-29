"""Wikidata как сигнал «есть ли русский перевод» для книг без моста на Goodreads (см. amazon.py). По
ISBN-13 находим издание -> его произведение (P629) -> смотрим, есть ли среди изданий этого произведения
русское (P407 = Q7737). Публичный SPARQL-эндпоинт не для тысяч одиночных запросов — пачками, с диск-кэшем
(повторный прогон не бьёт эндпоинт заново по уже известным ISBN).

P212 в Wikidata хранится с дефисами по диапазонам регистрации (`978-1-944757-03-8`); запись Amazon
(`978-1944757038`) и голые цифры с ним не совпадают — поэтому ISBN сначала приводится к 13 цифрам, а в запрос
идёт каноническая расстановка дефисов (isbnlib.mask).
"""
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import isbnlib
import pandas as pd

ENDPOINT = "https://query.wikidata.org/sparql"
BATCH_SIZE = 200
RUSSIAN_LANG_QID = "Q7737"
CACHE_COLUMNS = ["isbn13", "status", "checked_at"]


def normalize_isbn13(value) -> str | None:
    """Любая запись ISBN-10/13 (дефисы, пробелы) -> 13 цифр; неверная контрольная цифра или мусор -> None."""
    if not isinstance(value, str) or not value.strip():
        return None
    isbn13 = isbnlib.to_isbn13(isbnlib.canonical(value))
    return isbn13 if isbn13 and isbnlib.is_isbn13(isbn13) else None


def _query(isbns: list[str], timeout: float = 60.0) -> dict[str, bool]:
    """Один SPARQL-запрос на пачку ISBN-13 (13 цифр) -> {isbn13: есть ли русское издание} только для найденных
    в Wikidata. Один ISBN бывает у нескольких элементов — перевод есть, если он есть хоть у одного."""
    variants = {}
    for isbn in isbns:
        variants[isbn] = isbn
        masked = isbnlib.mask(isbn)
        if masked:
            variants[masked] = isbn
    values = " ".join(f'"{v}"' for v in variants)
    query = f"""
        SELECT ?isbnraw ?hasRu WHERE {{
          VALUES ?isbnraw {{ {values} }}
          ?edition wdt:P212 ?isbnraw; wdt:P629 ?work .
          OPTIONAL {{ ?ruEdition wdt:P629 ?work; wdt:P407 wd:{RUSSIAN_LANG_QID} . }}
          BIND(BOUND(?ruEdition) AS ?hasRu)
        }}
    """
    req = urllib.request.Request(
        ENDPOINT, data=urllib.parse.urlencode({"query": query}).encode(),
        headers={"Accept": "application/sparql-results+json",
                 "User-Agent": "BooksEngine/0.1 (personal, non-commercial)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    result: dict[str, bool] = {}
    for b in data["results"]["bindings"]:
        isbn = variants.get(b["isbnraw"]["value"])
        if isbn is not None:
            result[isbn] = result.get(isbn, False) or b["hasRu"]["value"] == "true"
    return result


def _read_cache(cache_path: Path) -> pd.DataFrame:
    """Кэш старого формата (ключи не нормализованы, без status) — недействителен, начинаем заново."""
    if cache_path.exists():
        cache = pd.read_parquet(cache_path)
        if list(cache.columns) == CACHE_COLUMNS:
            return cache
    return pd.DataFrame(columns=CACHE_COLUMNS)


def _write_cache(cache: pd.DataFrame, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".tmp")
    cache.to_parquet(tmp, index=False)
    tmp.replace(cache_path)


def check_translations(isbns: list, cache_path: Path, query=_query, batch_size: int = BATCH_SIZE,
                       rate_limit_s: float = 1.0, retries: int = 4, backoff_s: float = 5.0) -> dict[str, str]:
    """ISBN (любая запись) -> статус по нормализованному ISBN-13: 'ru' — есть русское издание, 'found' —
    в Wikidata есть, русского нет, 'absent' — в Wikidata нет. Невалидные ISBN и пачки, упавшие после всех
    повторов, в ответ не попадают (и в кэш тоже — их нельзя считать «перевода нет»). Кэш (Parquet: isbn13,
    status, checked_at) пишется после каждой пачки — прерванный прогон продолжается с места.
    `query` подменяется в тестах (без похода в сеть)."""
    wanted = sorted({n for n in map(normalize_isbn13, isbns) if n})
    cache = _read_cache(cache_path)
    known = dict(zip(cache.isbn13, cache.status))
    todo = [i for i in wanted if i not in known]

    for start in range(0, len(todo), batch_size):
        batch = todo[start:start + batch_size]
        found = None
        for attempt in range(retries):
            try:
                found = query(batch)
                break
            except Exception:
                if attempt + 1 < retries:
                    time.sleep(backoff_s * 2 ** attempt)
        if found is not None:
            statuses = {i: ("ru" if found[i] else "found") if i in found else "absent" for i in batch}
            known.update(statuses)
            rows = pd.DataFrame({"isbn13": list(statuses), "status": list(statuses.values()),
                                 "checked_at": pd.Timestamp.now("UTC")})
            cache = rows if cache.empty else pd.concat([cache, rows], ignore_index=True)
            _write_cache(cache, cache_path)
        if start + batch_size < len(todo):
            time.sleep(rate_limit_s)

    return {i: known[i] for i in wanted if i in known}
