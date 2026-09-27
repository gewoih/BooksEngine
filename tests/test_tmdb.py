import json

from booksengine.data import tmdb

RAW = {
    "id": 550,
    "title": "Бойцовский клуб",
    "release_date": "1999-10-15",
    "genres": [{"id": 18, "name": "Драма"}],
    "belongs_to_collection": None,
    "credits": {"crew": [
        {"id": 7467, "name": "Дэвид Финчер", "job": "Director"},
        {"id": 999, "name": "Кто-то ещё", "job": "Producer"},
    ]},
}

RAW_DOCUMENTARY_WITH_COLLECTION = {
    "id": 12345,
    "title": "Марш пингвинов",
    "release_date": "2005-01-26",
    "genres": [{"id": 99, "name": "Документальный"}],
    "belongs_to_collection": {"id": 77, "name": "Коллекция пингвинов"},
    "credits": {"crew": [{"id": 1, "name": "Люк Жаке", "job": "Director"}]},
}


def test_extract_pulls_director_year_genres():
    d = tmdb.extract(RAW)
    assert d["title_ru"] == "Бойцовский клуб"
    assert d["year"] == 1999
    assert d["director_id"] == 7467
    assert d["director_name"] == "Дэвид Финчер"
    assert d["collection_id"] is None
    assert d["is_documentary"] is False


def test_extract_pulls_collection_and_documentary_flag():
    d = tmdb.extract(RAW_DOCUMENTARY_WITH_COLLECTION)
    assert d["collection_id"] == 77
    assert d["collection_name"] == "Коллекция пингвинов"
    assert d["is_documentary"] is True


def test_extract_handles_missing_raw_and_missing_director():
    assert tmdb.extract(None)["title_ru"] is None
    no_director = {"id": 1, "title": "X", "release_date": "", "genres": [],
                   "belongs_to_collection": None, "credits": {"crew": []}}
    d = tmdb.extract(no_director)
    assert d["director_id"] is None and d["director_name"] is None
    assert d["year"] is None


def test_fetch_all_skips_already_cached_and_calls_fetch_for_new(tmp_path):
    calls = []

    def fake_fetch(tmdb_id, api_key, timeout=10.0):
        calls.append(tmdb_id)
        return {"id": tmdb_id}

    (tmp_path / "1.json").write_text(json.dumps({"id": 1}))   # уже скачан
    stats = tmdb.fetch_all([1, 2, 3], "key", tmp_path, fetch=fake_fetch, rate_limit_s=0.0)

    assert calls == [2, 3]
    assert stats == {"cached": 1, "fetched": 2, "missing": 0, "errors": []}
    assert json.loads((tmp_path / "2.json").read_text()) == {"id": 2}


def test_fetch_all_caches_not_found_as_null_and_counts_missing(tmp_path):
    def fake_fetch(tmdb_id, api_key, timeout=10.0):
        raise tmdb.NotFound(str(tmdb_id))

    stats = tmdb.fetch_all([5], "key", tmp_path, fetch=fake_fetch, rate_limit_s=0.0)
    assert stats == {"cached": 0, "fetched": 0, "missing": 1, "errors": []}
    assert json.loads((tmp_path / "5.json").read_text()) is None


def test_fetch_all_records_errors_without_stopping(tmp_path):
    def fake_fetch(tmdb_id, api_key, timeout=10.0):
        if tmdb_id == 2:
            raise RuntimeError("boom")
        return {"id": tmdb_id}

    stats = tmdb.fetch_all([1, 2, 3], "key", tmp_path, fetch=fake_fetch, rate_limit_s=0.0)
    assert stats["fetched"] == 2
    assert stats["errors"] == [(2, "boom")]
    assert not (tmp_path / "2.json").exists()
