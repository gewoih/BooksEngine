import numpy as np
import pandas as pd

from booksengine.model.filters import RatedFilter, work_info


def _write(tmp_path, works, authors):
    pd.DataFrame(works, columns=["work_id", "title", "best_edition_title", "is_collection", "author_id"]).drop(
        columns="author_id").to_parquet(tmp_path / "works.parquet")
    wa = [(w[0], w[4], None, 1) for w in works] + [(works[0][0], 99, "Illustrator", 0)]
    pd.DataFrame(wa, columns=["work_id", "author_id", "role", "position"]).to_parquet(tmp_path / "work_authors.parquet")
    pd.DataFrame(authors, columns=["author_id", "name"]).to_parquet(tmp_path / "authors.parquet")


def test_filter_drops_duplicates_and_collections_of_rated(tmp_path):
    works = [(1, "1984", "1984", False, 10),
             (2, "Animal Farm / 1984", "Animal Farm / 1984", False, 10),
             (3, "1984 (Oxford Bookworms)", "1984", False, 10),          # пересказ — тот же ключ
             (4, "1984", "1984", False, 11),                              # другой автор
             (5, "Orwell Box Set 1984 Animal Farm", "x", True, 10),       # сборник с «1984»
             (6, "Burmese Days", "Burmese Days", False, 10),
             (7, "Dune (Dune, #1)", "Dune (Dune, #1)", False, 12),
             (8, "Dune (Dune, #2)", "Dune (Dune, #2)", False, 12),        # другой номер — не дубль
             (9, "Complete Dune Novels", "x", True, 12)]                  # сборник, «Dune» целым словом
    _write(tmp_path, works, [(10, "George Orwell"), (11, "Other"), (12, "Frank Herbert"), (99, "Artist")])
    info = work_info(tmp_path, np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]))
    assert info.author.tolist()[0] == "George Orwell"                     # иллюстратор — не основной
    f = RatedFilter(info, np.array([0, 6]))                               # оценены «1984» и «Dune #1»
    assert [f.is_rated_already(c) for c in range(9)] == [True, True, True, False, True, False, True, False, True]


def test_rated_collection_hides_its_parts(tmp_path):
    works = [(1, "Animal Farm / 1984", "x", False, 10), (2, "Animal Farm", "Animal Farm", False, 10),
             (3, "Burmese Days", "Burmese Days", False, 10)]
    _write(tmp_path, works, [(10, "George Orwell"), (99, "Artist")])
    f = RatedFilter(work_info(tmp_path, np.array([1, 2, 3])), np.array([0]))
    assert [f.is_rated_already(c) for c in range(3)] == [True, True, False]
