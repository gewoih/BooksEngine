import json

import numpy as np
import pandas as pd
import pytest

from booksengine.model import experiment as ex


def _ratings(rows):
    r = pd.DataFrame(rows, columns=["user_id", "work_id", "rating", "n_editions"])
    return r.astype({"rating": "float32", "n_editions": "int16"})


@pytest.fixture
def data(tmp_path):
    """Старое ядро: 6 книг, пользователи 1–5. Новое: без не-книги 4, тень 2 слита в 1, однообразный 5 удалён.
    Тестовые — 1 и 5, валидационный — 2."""
    works = pd.DataFrame({"work_id": [1, 2, 3, 4, 5, 6],
                          "title": ["Dune", "Dune.", "Emma", "Dune: Sheet Music", "Ulysses", "Beloved"]})
    old_rows = []
    for u in (1, 2, 3, 4):
        old_rows += [(u, 1, 5.0, 1), (u, 3, 3.0, 1), (u, 5, 4.0, 1), (u, 6, 2.0, 1)]
    old_rows += [(1, 2, 4.0, 1), (1, 4, 5.0, 1), (3, 4, 1.0, 1)] + [(5, w, 5.0, 1) for w in (1, 3, 5, 6)]
    new_rows = [r for r in old_rows if r[0] != 5 and r[1] not in (2, 4) and not (r[0] == 1 and r[1] == 1)]
    new_rows += [(1, 1, 4.5, 2)]
    cores = {}
    for name, rows, merges in (("old", old_rows, []), ("new", new_rows, [(2, 1)])):
        d = tmp_path / name
        d.mkdir()
        _ratings(rows).to_parquet(d / "ratings.parquet")
        works.to_parquet(d / "works.parquet")
        pd.DataFrame(merges, columns=["shadow_work_id", "main_work_id"]).astype("int64").to_parquet(
            d / "work_merges.parquet")
        cores[name] = d
    split = tmp_path / "split"
    split.mkdir()
    pd.DataFrame({"user_id": [1, 2, 5], "group": ["test", "val", "test"], "n_ratings": [6, 4, 4],
                  "bucket": ["10-19"] * 3}).to_parquet(split / "holdout_users.parquet")
    # у тестового 1 во входе тень 2 и не-книга 4, скрыто главное 1 и Emma 3
    pd.DataFrame({"user_id": [1, 1, 1, 1, 5, 5, 5], "work_id": [2, 4, 5, 6, 1, 3, 5],
                  "rating": [4.0, 5.0, 4.0, 2.0, 5.0, 5.0, 5.0]}).to_parquet(split / "test_input.parquet")
    pd.DataFrame({"user_id": [1, 1, 5], "work_id": [1, 3, 6], "rating": [5.0, 3.0, 5.0]}).to_parquet(
        split / "test_hidden.parquet")
    return cores, split, tmp_path


def test_common_split_same_hidden_no_leak_and_dropped_users(data):
    cores, split, tmp = data
    meta = ex.build_common_split(cores, split, tmp / "common")
    h_old = pd.read_parquet(tmp / "common" / "old" / "test_hidden.parquet")
    h_new = pd.read_parquet(tmp / "common" / "new" / "test_hidden.parquet")
    pd.testing.assert_frame_equal(h_old, h_new)
    assert set(h_old.user_id) == {1}                 # 5 выпал из нового ядра → исключён из обоих
    assert set(h_old.work_id) == {3}                 # главное 1 во входе нового (через тень 2) → не скрытое
    i_new = pd.read_parquet(tmp / "common" / "new" / "test_input.parquet")
    assert set(i_new.work_id) == {1, 5, 6}           # тень 2 → 1, не-книга 4 выкинута
    i_old = pd.read_parquet(tmp / "common" / "old" / "test_input.parquet")
    assert set(i_old.work_id) == {2, 4, 5, 6}        # старое ядро — вход как был
    hu = pd.read_parquet(tmp / "common" / "new" / "holdout_users.parquet")
    assert hu.to_dict("list") == {"user_id": [1], "group": ["test"], "n_ratings": [6], "bucket": ["10-19"]}
    assert meta["users"] == 1 and meta["users_dropped"] == 1


def test_profile_input_uses_own_five_point_rating(tmp_path):
    prof = tmp_path / "profile.csv"
    pd.DataFrame({"title": ["Дюна", "Эмма", "Вне", "Улисс", "Возлюбленная"], "rating": [5, None, 8, 1, 7],
                  "rating5": [2, 1, 4, 1, None], "status": ["read", "dnf", "read", "read", "read"],
                  "goodreads_work_id": [1, 3, 999, 5, 6]}).to_csv(prof, index=False)
    x = ex._profile_input(prof, np.array([1, 3, 5, 6]))
    # 5/10 → своя 2/5, не 2.5; 1/10 → 1, не 0.5; dnf — 1/5; без rating5 — пропуск
    assert x.toarray().tolist() == [[2.0, 1.0, 1.0, 0.0]]


def _run(cores, split, tmp, name, model_dir):
    prof = tmp / "profile.csv"
    pd.DataFrame({"title": ["Дюна", "Эмма", "Вне"], "rating": [10, None, 8], "rating5": [5, None, 4],
                  "status": ["read", "dnf", "read"], "goodreads_work_id": [1, 3, 999]}).to_csv(prof, index=False)
    return ex.run_core(name, ratings_path=cores[name] / "ratings.parquet", works_path=cores[name] / "works.parquet",
                       common_dir=tmp / "common" / name, holdout_path=split / "holdout_users.parquet",
                       profile_path=prof, model_dir=model_dir, eval_dir=tmp / "e", models=("popularity",))


def test_run_core_writes_metrics_and_profile(data):
    cores, split, tmp = data
    ex.build_common_split(cores, split, tmp / "common")
    out = _run(cores, split, tmp, "new", tmp / "m")
    res = out["models"]["popularity"]
    assert out["n_users"] == 1 and "ndcg20" in res["summary"]["all"]
    assert "Dune" not in res["profile"]                            # оценённое не советуется
    assert json.loads((tmp / "e" / "new.json").read_text())["core"] == "new"
    assert (tmp / "m" / "popularity" / "params.json").exists()     # обученная модель сохранена


def test_run_core_refuses_saved_model_with_other_params(data):
    cores, split, tmp = data
    ex.build_common_split(cores, split, tmp / "common")
    bad = tmp / "m" / "popularity"
    bad.mkdir(parents=True)
    (bad / "params.json").write_text(json.dumps({"formula": "bayes_log", "m": 1000.0}))
    with pytest.raises(ValueError, match="настройки"):
        _run(cores, split, tmp, "old", tmp / "m")


def test_report_renders_sections():
    from booksengine import report_exp
    s = {m: {"mean": 0.25, "lo": 0.24, "hi": 0.26, "n": 10} for m in ("ndcg20", "ndcg10", "recall20", "map20", "low20")}
    summ = {k: s for k in ("all", "10-19", "20-49", "50-199", "200+")}
    ev = {"n_users": 10, "models": {"als": {"fit_seconds": 1.0, "summary": summ, "coverage": 0.01, "profile": ["1984"]}}}
    man = {"outputs": {"ratings": {"rows": 100}, "users": {"rows": 10}},
           "cleaning_log": [{"rule": "nonbooks", "rows_removed": 5, "reason": "не книги", "detail": {}}]}
    text = report_exp.render({"users": 10, "users_dropped": 0, "hidden": 20, "hidden_dropped": 1},
                             {"old": ev, "new": ev}, {"old": man, "new": man})
    for x in ("## Что удалено", "## Метрики", "## Профиль", "0.250", "1984", "nonbooks"):
        assert x in text


def test_save_core_copies_merges_or_writes_empty(tmp_path):
    clean = tmp_path / "clean"
    clean.mkdir()
    for f in ("ratings.parquet", "works.parquet"):
        pd.DataFrame({"x": [1]}).to_parquet(clean / f)
    (clean / "manifest.json").write_text("{}")
    ex.save_core(clean, tmp_path / "a")
    assert len(pd.read_parquet(tmp_path / "a" / "work_merges.parquet")) == 0
    pd.DataFrame({"shadow_work_id": [2], "main_work_id": [1]}).to_parquet(clean / "work_merges.parquet")
    ex.save_core(clean, tmp_path / "b")
    assert pd.read_parquet(tmp_path / "b" / "work_merges.parquet").shadow_work_id.tolist() == [2]


def test_report_three_cores_compared_to_old():
    from booksengine import report_exp
    def summ(v):
        s = {m: {"mean": v, "lo": v - 0.01, "hi": v + 0.01, "n": 10} for m in ("ndcg20", "ndcg10", "recall20", "map20", "low20")}
        return {k: s for k in ("all", "10-19", "20-49", "50-199", "200+")}
    ev = lambda v: {"n_users": 10, "models": {"als": {"fit_seconds": 1.0, "summary": summ(v), "coverage": 0.01,
                                                      "profile": ["1984"]}}}
    man = lambda k: {"outputs": {"ratings": {"rows": 100}, "users": {"rows": 10}}, "config": {"kcore": {"min_work_ratings": k}},
                     "cleaning_log": [{"rule": "kcore", "rows_removed": k, "reason": "k-core", "detail": {}}]}
    text = report_exp.render({"users": 10, "users_dropped": 0, "hidden": 20, "hidden_dropped": 1},
                             {"old": ev(0.25), "k50": ev(0.25), "new": ev(0.30)},
                             {"old": man(20), "k50": man(50), "new": man(100)})
    assert "≥ 100" in text and "≥ 50" in text and "≥ 20" in text
    assert "≥ 50 против ≥ 20: NDCG@20 — нет" in text and "≥ 100 против ≥ 20: NDCG@20 — **да**" in text
