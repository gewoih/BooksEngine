import io
import json
import urllib.parse

import pandas as pd
import pytest

from booksengine.data import wikidata

# валидные ISBN-13 (контрольная цифра сходится) — неверные нормализация отбрасывает
A, B, C, D, E = "9780000000002", "9780000000019", "9780000000026", "9780000000033", "9780000000040"


def test_normalize_isbn13_accepts_any_spelling_and_isbn10():
    assert wikidata.normalize_isbn13("978-1944757038") == "9781944757038"
    assert wikidata.normalize_isbn13("978-1-944757-03-8") == "9781944757038"
    assert wikidata.normalize_isbn13("0701169850") == "9780701169855"


def test_normalize_isbn13_rejects_bad_checksum_and_empty():
    assert wikidata.normalize_isbn13("9785171082532") is None   # контрольная цифра не сходится
    assert wikidata.normalize_isbn13("") is None
    assert wikidata.normalize_isbn13(None) is None
    assert wikidata.normalize_isbn13(float("nan")) is None


def test_check_translations_normalizes_input_before_query(tmp_path):
    calls = []

    def fake_query(batch):
        calls.append(list(batch))
        return {"9781944757038": True}

    result = wikidata.check_translations(["978-1944757038", "not an isbn"], tmp_path / "c.parquet",
                                         query=fake_query, rate_limit_s=0)
    assert calls == [["9781944757038"]]
    assert result == {"9781944757038": "ru"}


def test_check_translations_uses_cache_and_queries_only_missing_isbns(tmp_path):
    cache_path = tmp_path / "wikidata_cache.parquet"
    pd.DataFrame({"isbn13": [A], "status": ["ru"],
                 "checked_at": [pd.Timestamp.now("UTC")]}).to_parquet(cache_path)
    calls = []

    def fake_query(batch):
        calls.append(list(batch))
        return {B: True, C: False}

    result = wikidata.check_translations([A, B, C], cache_path, query=fake_query, rate_limit_s=0)

    assert calls == [[B, C]]
    assert result == {A: "ru", B: "ru", C: "found"}


def test_check_translations_marks_missing_from_response_as_absent(tmp_path):
    result = wikidata.check_translations([D], tmp_path / "c.parquet", query=lambda b: {}, rate_limit_s=0)
    assert result == {D: "absent"}


def test_check_translations_batches_requests_by_batch_size(tmp_path):
    calls = []

    def fake_query(batch):
        calls.append(len(batch))
        return {}

    wikidata.check_translations([A, B, C, D, E], tmp_path / "c.parquet", query=fake_query, batch_size=2,
                                rate_limit_s=0)
    assert calls == [2, 2, 1]


def test_check_translations_persists_results_and_does_not_requery(tmp_path):
    cache_path = tmp_path / "wikidata_cache.parquet"
    wikidata.check_translations([E], cache_path, query=lambda b: {E: True}, rate_limit_s=0)

    assert pd.read_parquet(cache_path).set_index("isbn13").loc[E, "status"] == "ru"
    calls = []
    wikidata.check_translations([E], cache_path, query=lambda b: calls.append(b) or {}, rate_limit_s=0)
    assert calls == []


def test_check_translations_retries_transient_errors(tmp_path):
    attempts = []

    def flaky(batch):
        attempts.append(1)
        if len(attempts) < 3:
            raise OSError("504 Gateway Timeout")
        return {A: True}

    result = wikidata.check_translations([A], tmp_path / "c.parquet", query=flaky, rate_limit_s=0,
                                         backoff_s=0)
    assert len(attempts) == 3
    assert result == {A: "ru"}


def test_check_translations_keeps_earlier_batches_when_a_later_one_keeps_failing(tmp_path):
    cache_path = tmp_path / "wikidata_cache.parquet"

    def query(batch):
        if batch == [A]:
            return {A: True}
        raise OSError("504 Gateway Timeout")

    result = wikidata.check_translations([A, B], cache_path, query=query, batch_size=1, rate_limit_s=0,
                                         retries=2, backoff_s=0)
    # сбойная пачка не записана как «нет перевода» — её просто нет в ответе, и нет в кэше
    assert result == {A: "ru"}
    cached = pd.read_parquet(cache_path)
    assert cached.isbn13.tolist() == [A]


def test_check_translations_ignores_cache_of_old_format(tmp_path):
    cache_path = tmp_path / "wikidata_cache.parquet"
    pd.DataFrame({"isbn13": ["978-1944757038"], "has_ru": [False],
                 "checked_at": [pd.Timestamp.now("UTC")]}).to_parquet(cache_path)
    calls = []

    def fake_query(batch):
        calls.append(list(batch))
        return {"9781944757038": True}

    result = wikidata.check_translations(["978-1944757038"], cache_path, query=fake_query, rate_limit_s=0)
    assert calls == [["9781944757038"]]
    assert result == {"9781944757038": "ru"}


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_query_sends_canonical_hyphenated_isbn_and_merges_duplicate_rows(monkeypatch):
    sent = {}

    def fake_urlopen(req, timeout):
        sent["body"] = urllib.parse.parse_qs(req.data.decode())["query"][0]
        bindings = [
            # один ISBN у нескольких элементов Wikidata: хоть один с русским изданием — значит есть
            {"isbnraw": {"value": "978-1-944757-03-8"}, "hasRu": {"value": "true"}},
            {"isbnraw": {"value": "978-1-944757-03-8"}, "hasRu": {"value": "false"}},
            {"isbnraw": {"value": "9780306406157"}, "hasRu": {"value": "false"}},
        ]
        return _FakeResponse(json.dumps({"results": {"bindings": bindings}}).encode())

    monkeypatch.setattr(wikidata.urllib.request, "urlopen", fake_urlopen)

    result = wikidata._query(["9781944757038", "9780306406157"])

    assert '"978-1-944757-03-8"' in sent["body"]      # так P212 хранится в Wikidata
    assert '"9781944757038"' in sent["body"]          # и без дефисов — на редкие записи в таком виде
    assert result == {"9781944757038": True, "9780306406157": False}
