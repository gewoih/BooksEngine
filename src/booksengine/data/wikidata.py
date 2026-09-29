"""Wikidata как сигнал «есть ли русский перевод» для книг без моста на Goodreads (см. amazon.py). По
ISBN-13 находим издание -> его произведение (P629) -> смотрим, есть ли среди изданий этого произведения
русское (P407 = Q7737). Публичный SPARQL-эндпоинт не для тысяч одиночных запросов — пачками, с диск-кэшем
(повторный прогон не бьёт эндпоинт заново по уже известным ISBN).
"""
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ENDPOINT = "https://query.wikidata.org/sparql"
BATCH_SIZE = 200
RUSSIAN_LANG_QID = "Q7737"


def _query(isbns: list[str], timeout: float = 60.0) -> dict[str, bool]:
    """Один SPARQL-запрос на пачку ISBN-13 -> {isbn13: has_ru}. ISBN, не найденный в ответе, — до вызывающей
    стороны: он просто отсутствует в возвращённом словаре (check_translations трактует это как False)."""
    values = " ".join(f'"{i}"' for i in isbns)
    query = f"""
        SELECT ?isbn13 ?hasRu WHERE {{
          VALUES ?isbn13 {{ {values} }}
          ?edition wdt:P212 ?isbn13; wdt:P629 ?work .
          OPTIONAL {{ ?ruEdition wdt:P629 ?work; wdt:P407 wd:{RUSSIAN_LANG_QID} . }}
          BIND(BOUND(?ruEdition) AS ?hasRu)
        }}
    """
    url = ENDPOINT + "?" + urllib.parse.urlencode({"query": query, "format": "json"})
    req = urllib.request.Request(url, headers={"User-Agent": "BooksEngine/0.1 (personal, non-commercial)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    return {b["isbn13"]["value"]: b["hasRu"]["value"] == "true" for b in data["results"]["bindings"]}


def check_translations(isbns: list[str], cache_path: Path, query=_query, batch_size: int = BATCH_SIZE,
                       rate_limit_s: float = 1.0) -> dict[str, bool]:
    """isbn13 -> есть ли русский перевод. Кэш в cache_path (Parquet: isbn13, has_ru, checked_at) — уже
    проверенные ISBN не запрашиваются заново. `query` подменяется в тестах (без похода в сеть)."""
    isbns = sorted({i for i in isbns if i})
    cache = (pd.read_parquet(cache_path) if cache_path.exists()
             else pd.DataFrame(columns=["isbn13", "has_ru", "checked_at"]))
    known = dict(zip(cache.isbn13, cache.has_ru))
    todo = [i for i in isbns if i not in known]

    new_rows = []
    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        found = query(batch)
        now = pd.Timestamp.now("UTC")
        for isbn in batch:
            has_ru = bool(found.get(isbn, False))
            known[isbn] = has_ru
            new_rows.append({"isbn13": isbn, "has_ru": has_ru, "checked_at": now})
        if i + batch_size < len(todo):
            time.sleep(rate_limit_s)

    if new_rows:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True).to_parquet(cache_path, index=False)

    return {i: known[i] for i in isbns}
