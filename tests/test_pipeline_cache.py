"""Кэш profile.json: пересчёт при смене сырых файлов или кода профилирования."""
from booksengine.data import pipeline


def test_profile_cache_follows_raw_and_code(tmp_path):
    path = tmp_path / "profile.json"
    calls = []

    def compute():
        calls.append(1)
        return {"n": len(calls)}

    key = {"raw": "a", "code": "c1"}
    assert pipeline.cached_profile(path, key, compute) == {"n": 1}
    assert pipeline.cached_profile(path, key, compute) == {"n": 1}              # из кэша
    assert pipeline.cached_profile(path, {"raw": "a", "code": "c2"}, compute) == {"n": 2}   # код поменялся
    assert pipeline.cached_profile(path, {"raw": "b", "code": "c2"}, compute) == {"n": 3}   # сырые поменялись
    assert pipeline.cached_profile(path, {"raw": "b", "code": "c2"}, compute, force=True) == {"n": 4}


def test_old_cache_without_code_is_recomputed(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text('{"raw": "a", "metrics": {"n": 0}}')
    assert pipeline.cached_profile(path, {"raw": "a", "code": "c"}, lambda: {"n": 1}) == {"n": 1}


def test_profile_key_hashes_profile_code():
    key = pipeline.profile_key({"x": "1"})
    assert set(key) == {"raw", "code"} and len(key["code"]) == 16
