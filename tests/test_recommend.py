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


@pytest.mark.parametrize("bad", [7, 3.5, None])
def test_rating_outside_one_to_five_is_rejected(clean, bad):
    p = _csv(clean, [("Дюна", 1, 5, "read"), ("Эмма", 2, bad, "read")])
    with pytest.raises(ValueError, match="строки 3"):
        rec.read_profile(p, np.array([1, 2, 3]), clean)
