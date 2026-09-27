"""TMDB (themoviedb.org): сведения о фильмах по tmdbId — русское название, режиссёр, франшиза, жанры,
документальное ли. Дополняет MovieLens: без этого — только английское название и грубые жанровые теги,
режиссёра и франшизы нет вообще, правила выдачи (аналог книжных «не больше одной книги автора», серия)
не на чем проверять. Кэш ответов — по одному JSON на tmdbId, не перекачивается повторно.
"""
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


class NotFound(Exception):
    """TMDB не знает такой tmdbId (404) — устаревшая или ошибочная ссылка в links.csv."""


def fetch_one(tmdb_id: int, api_key: str, timeout: float = 10.0) -> dict:
    """Один фильм: основные поля + credits.crew (режиссёр — среди Director) одним запросом, на русском
    (падает на английский у TMDB, если перевода нет)."""
    url = (f"https://api.themoviedb.org/3/movie/{tmdb_id}"
           f"?api_key={api_key}&language=ru-RU&append_to_response=credits")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise NotFound(str(tmdb_id)) from e
        if e.code == 429:
            time.sleep(float(e.headers.get("Retry-After", 1)))
            return fetch_one(tmdb_id, api_key, timeout)
        raise


def fetch_all(tmdb_ids: list[int], api_key: str, cache_dir: Path, fetch=fetch_one,
              rate_limit_s: float = 0.15) -> dict:
    """Кэширует ответ по каждому tmdbId в cache_dir/<id>.json; уже скачанные (файл существует, в т.ч. null для
    404) не перекачиваются — можно останавливать и продолжать. `fetch` — подменяется в тестах."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    stats = {"cached": 0, "fetched": 0, "missing": 0, "errors": []}
    for tmdb_id in tmdb_ids:
        path = cache_dir / f"{tmdb_id}.json"
        if path.exists():
            stats["cached"] += 1
            continue
        try:
            data = fetch(tmdb_id, api_key)
            path.write_text(json.dumps(data, ensure_ascii=False))
            stats["fetched"] += 1
        except NotFound:
            path.write_text("null")
            stats["missing"] += 1
        except Exception as e:
            stats["errors"].append((tmdb_id, str(e)))
        time.sleep(rate_limit_s)
    return stats


def extract(raw: dict | None) -> dict:
    """Поля из ответа TMDB, которые нужны модели: русское название, год, режиссёр (id + имя), франшиза
    (id + название), документальное ли (жанр «Документальный», id 99 у TMDB), жанры (список названий).
    raw = None (кэш 404) -> все поля пустые."""
    if raw is None:
        return {"title_ru": None, "year": None, "director_id": None, "director_name": None,
                "collection_id": None, "collection_name": None, "is_documentary": False, "genres": []}
    release = raw.get("release_date") or ""
    director = next((c for c in raw.get("credits", {}).get("crew", []) if c.get("job") == "Director"), None)
    collection = raw.get("belongs_to_collection")
    genres = [g["name"] for g in raw.get("genres", [])]
    return {
        "title_ru": raw.get("title"),
        "year": int(release[:4]) if release[:4].isdigit() else None,
        "director_id": director["id"] if director else None,
        "director_name": director["name"] if director else None,
        "collection_id": collection["id"] if collection else None,
        "collection_name": collection["name"] if collection else None,
        "is_documentary": any(g.get("id") == 99 for g in raw.get("genres", [])),
        "genres": genres,
    }
