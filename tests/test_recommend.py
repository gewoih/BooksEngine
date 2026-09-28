import numpy as np
import pandas as pd
import pytest

from booksengine import recommend as rec


@pytest.fixture
def clean(tmp_path):
    pd.DataFrame({"work_id": [1, 2, 3, 4, 50]}).to_parquet(tmp_path / "works.parquet")    # 50 — вне ядра
    pd.DataFrame({"shadow_work_id": [4], "main_work_id": [2]}).to_parquet(tmp_path / "work_merges.parquet")
    return tmp_path


def _csv(tmp_path, rows):
    p = tmp_path / "r.csv"
    pd.DataFrame(rows, columns=["title", "goodreads_work_id", "rating", "status"]).to_csv(p, index=False)
    return p


def test_profile_maps_shadows_averages_and_reports_skipped(clean):
    p = _csv(clean, [("Дюна", 1, 5, "read"), ("Эмма", 2, 2, "read"), ("Эмма-тень", 4, 4, "read"),
                     ("Бросил", 3, None, "dnf"), ("Новая", None, 4, "read"), ("Редкая", 50, 3, "read"),
                     ("Чужая", 777, 3, "read")])
    prof = rec.read_profile(p, np.array([1, 2, 3]), clean)
    assert prof.x.toarray().tolist() == [[5.0, 3.0, 1.0]]            # тень → главное, среднее; dnf = 1
    assert prof.dnf.toarray().tolist() == [[0.0, 0.0, 1.0]]
    assert prof.names == {0: "Дюна", 1: "Эмма", 2: "Бросил"}
    assert prof.skipped == [("Новая", rec.NO_ID), ("Редкая", rec.NOT_IN_CORE), ("Чужая", rec.NOT_IN_CATALOG)]
    assert prof.outside.tolist() == [50]                              # вне ядра — для «уже оценено»


def test_work_is_dnf_only_if_every_row_is_dnf(clean):
    p = _csv(clean, [("Дюна", 1, None, "dnf"), ("Дюна-тень", 1, 4, "read"), ("Эмма", 2, 1, "dnf")])
    prof = rec.read_profile(p, np.array([1, 2, 3]), clean)
    assert prof.x.toarray().tolist() == [[2.5, 1.0, 0.0]] and prof.dnf.toarray().tolist() == [[0.0, 1.0, 0.0]]


@pytest.mark.parametrize("bad", [7, 3.5])
def test_rating_outside_one_to_five_is_rejected(clean, bad):
    p = _csv(clean, [("Дюна", 1, 5, "read"), ("Эмма", 2, bad, "read")])
    with pytest.raises(ValueError, match="строки 3"):
        rec.read_profile(p, np.array([1, 2, 3]), clean)


def test_empty_rating_means_read_without_rating(clean):
    p = _csv(clean, [("Дюна", 1, 5, "read"), ("Эмма", 2, None, "read"), ("Эмма-тень", 4, None, ""),
                     ("Бросил", 3, None, "dnf"), ("Редкая", 50, None, "read")])
    prof = rec.read_profile(p, np.array([1, 2, 3]), clean)
    assert prof.x.toarray().tolist() == [[5.0, 0.0, 1.0]]             # без оценки — не вход модели
    assert prof.read.tolist() == [1] and prof.outside.tolist() == [50]   # тень → главное; вне ядра — «уже прочитано»
    assert prof.seen().indices.tolist() == [0, 1, 2]
    both = _csv(clean, [("Дюна", 1, 5, "read"), ("Дюна ещё раз", 1, None, "read")])
    assert rec.read_profile(both, np.array([1, 2, 3]), clean).read.tolist() == []   # есть оценка — оценено


def test_want_status_is_a_shelf_not_a_rating(clean):
    from booksengine.model.mix import READ_INPUT, WANT_INPUT
    p = _csv(clean, [("Дюна", 1, 5, "read"), ("Эмма", 2, None, "want"), ("Бросил", 3, None, "dnf"),
                     ("Прочёл", 4, None, "read")])                  # 4 — тень 2: прочитанное важнее «хочу»
    prof = rec.read_profile(p, np.array([1, 2, 3]), clean)
    assert prof.x.toarray().tolist() == [[5.0, 0.0, 1.0]] and prof.read.tolist() == [1] and prof.want.tolist() == []
    p = _csv(clean, [("Дюна", 1, 5, "read"), ("Эмма", 2, None, "want"), ("Бросил", 3, None, "read")])
    prof = rec.read_profile(p, np.array([1, 2, 3]), clean)
    assert prof.want.tolist() == [1] and prof.read.tolist() == [2] and prof.names[1] == "Эмма"
    assert prof.shelf().toarray().tolist() == [[0.0, WANT_INPUT, READ_INPUT]]
    assert prof.seen().indices.tolist() == [0, 2]                   # «хочу» — не начатая серия
    assert prof.not_advised().tolist() == [0, 1, 2]


def test_why_text_labels_shelf_books():
    r = rec.Rec(1, "T", "A", 50, ["Дюна", "Эмма"], None, {"Дюна": "хочу прочитать", "Эмма": "прочитано"})
    assert rec.why_text(r) == ["читатели: Дюна (хочу прочитать), Эмма (прочитано)"]
