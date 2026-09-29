import pandas as pd

from booksengine.data import wikidata


def test_check_translations_uses_cache_and_queries_only_missing_isbns(tmp_path):
    cache_path = tmp_path / "wikidata_cache.parquet"
    pd.DataFrame({"isbn13": ["9780000000001"], "has_ru": [True],
                 "checked_at": [pd.Timestamp.now("UTC")]}).to_parquet(cache_path)

    calls = []

    def fake_query(batch):
        calls.append(list(batch))
        return {"9780000000002": True, "9780000000003": False}

    result = wikidata.check_translations(
        ["9780000000001", "9780000000002", "9780000000003"], cache_path, query=fake_query, rate_limit_s=0)

    assert calls == [["9780000000002", "9780000000003"]]
    assert result == {"9780000000001": True, "9780000000002": True, "9780000000003": False}


def test_check_translations_defaults_missing_from_response_to_false(tmp_path):
    cache_path = tmp_path / "wikidata_cache.parquet"

    def fake_query(batch):
        return {}   # ISBN не нашёлся в Wikidata вообще

    result = wikidata.check_translations(["9780000000004"], cache_path, query=fake_query, rate_limit_s=0)
    assert result == {"9780000000004": False}


def test_check_translations_batches_requests_by_batch_size(tmp_path):
    cache_path = tmp_path / "wikidata_cache.parquet"
    isbns = [f"978000000{i:04d}" for i in range(5)]
    calls = []

    def fake_query(batch):
        calls.append(len(batch))
        return {}

    wikidata.check_translations(isbns, cache_path, query=fake_query, batch_size=2, rate_limit_s=0)
    assert calls == [2, 2, 1]


def test_check_translations_persists_new_results_to_cache(tmp_path):
    cache_path = tmp_path / "wikidata_cache.parquet"

    def fake_query(batch):
        return {"9780000000005": True}

    wikidata.check_translations(["9780000000005"], cache_path, query=fake_query, rate_limit_s=0)

    cached = pd.read_parquet(cache_path)
    assert cached.set_index("isbn13").loc["9780000000005", "has_ru"] == True

    # повторный вызов с тем же ISBN не должен снова спрашивать query
    calls = []
    wikidata.check_translations(["9780000000005"], cache_path,
                                query=lambda b: calls.append(b) or {}, rate_limit_s=0)
    assert calls == []
