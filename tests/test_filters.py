import numpy as np
import pandas as pd

from booksengine.model.filters import RatedFilter, work_info


def _write(tmp_path, works, authors, original=None, extra_authors=()):
    """works: (work_id, title, best_edition_title, is_collection, author_id); original: work_id → оригинальное название;
    extra_authors: (work_id, author_id, role, position) сверх основного."""
    d = pd.DataFrame(works, columns=["work_id", "title", "best_edition_title", "is_collection", "author_id"])
    d.insert(2, "original_title", d.work_id.map(original or {}))
    d.drop(columns="author_id").to_parquet(tmp_path / "works.parquet")
    wa = [(w[0], w[4], None, 1) for w in works] + [(works[0][0], 99, "Illustrator", 0)] + list(extra_authors)
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


def test_same_work_under_other_title(tmp_path):
    works = [(1, "Alice in Wonderland", "Alice in Wonderland", False, 10),                                   # 0 оценена
             (2, "Alice's Adventures in Wonderland & Through the Looking-Glass", "x", False, 10),           # 1
             (3, "Don Quixote", "Don Quixote", False, 11),                                                  # 2 оценена
             (4, "Don Quijote de la Mancha I (Don Quijote de la Mancha, #1)", "x", False, 11),              # 3
             (5, "White Fang", "White Fang", False, 12),                                                    # 4 оценена
             (6, "The Call of the Wild/White Fang", "x", False, 12),                                        # 5
             (7, "The Lottery", "The Lottery", False, 13),                                                  # 6 оценена
             (8, "The Lottery and Other Stories", "x", False, 13),                                          # 7
             (9, "The White Queen (Cousins' War, #1)", "x", False, 14),                                     # 8 оценена
             (10, "The Other Queen", "The Other Queen", False, 14),                                         # 9 другая книга
             (11, "Faust: First Part", "Faust: First Part", False, 15),                                     # 10 оценена
             (12, "Faust, Part Two", "Faust, Part Two", False, 15),                                         # 11 другая книга
             (13, "Faust", "Faust", False, 15),                                                             # 12
             (14, "The Lottery", "The Lottery", False, 16),                                                 # 13 другой автор
             (15, "Dracula", "Dracula", False, 17),                                                         # 14 оценена
             (16, "Dracula", "Dracula", False, 18)]                                        # 15 основной — переводчик
    _write(tmp_path, works, [(10, "Carroll"), (11, "Cervantes"), (12, "London"), (13, "Jackson"), (14, "Gregory"),
                             (15, "Goethe"), (16, "Other"), (17, "Stoker"), (18, "Translator"), (99, "Artist")],
           original={3: "Don Quijote de La Mancha", 4: "El Ingenioso Hidalgo Don Quijote de La Mancha"},
           extra_authors=[(16, 17, "", 2)])
    f = RatedFilter(work_info(tmp_path, np.arange(1, 17)), np.array([0, 2, 4, 6, 8, 10, 14]))
    assert [c for c in range(16) if f.is_rated_already(c)] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 15]


def test_list_keeps_one_edition_of_the_same_work(tmp_path):
    from booksengine.model import filters as f
    from booksengine.model.series import SeriesIndex
    works = [(1, "Alice's Adventures in Wonderland", "x", False, 10), (2, "Alice in Wonderland", "x", False, 10),
             (3, "Solaris", "Solaris", False, 11)]
    _write(tmp_path, works, [(10, "Carroll"), (11, "Lem"), (99, "Artist")])
    info = work_info(tmp_path, np.array([1, 2, 3]))
    picked, removed = f.ListPicker(info, SeriesIndex(info.title.tolist())).pick([0, 1, 2], lambda c: 1.0, 20)
    assert picked == [0, 2] and removed == [(1, f.DUPLICATE)]


def _picker_world(tmp_path):
    """Серия Dune (#1–#3), сборник, три книги одного автора, поздний том без первого, оценённая книга и её дубль."""
    from booksengine.model.filters import ListPicker
    from booksengine.model.series import SeriesIndex
    works = [(1, "Dune (Dune, #1)", "Dune (Dune, #1)", False, 12),                              # 0
             (2, "Dune Messiah (Dune, #2)", "Dune Messiah (Dune, #2)", False, 12),              # 1
             (3, "Children of Dune (Dune, #3)", "Children of Dune (Dune, #3)", False, 12),      # 2
             (4, "Big Box Set", "Big Box Set", True, 13),                                       # 3
             (5, "Solo Alpha", "Solo Alpha", False, 14), (6, "Solo Beta", "Solo Beta", False, 14),            # 4, 5
             (7, "Solo Gamma", "Solo Gamma", False, 14),                                                # 6
             (8, "Lost (Saga, #2)", "Lost (Saga, #2)", False, 15),                              # 7
             (9, "Single", "Single", False, 16), (10, "Single", "Single", False, 16)]           # 8 — оценена, 9 — дубль
    _write(tmp_path, works, [(12, "Herbert"), (13, "Box"), (14, "Solo"), (15, "Saga"), (16, "One"), (99, "Artist")])
    info = work_info(tmp_path, np.arange(1, 11))
    picker = ListPicker(info, SeriesIndex(info.title.tolist()))
    sc = np.array([3, 9, -np.inf, 8, 7, 6, 5, 4, -np.inf, 2], dtype=np.float64)   # 8 — вход, 2 — не кандидат
    order = np.argsort(-sc, kind="stable")
    return info, picker, sc, order[np.isfinite(sc[order])]


def test_author_cap_is_one_book_per_ten_places():
    from booksengine.model.filters import author_cap
    assert [author_cap(t) for t in (5, 10, 19, 20, 25, 50)] == [1, 1, 1, 2, 2, 5]


def test_list_rules_replace_later_volume_drop_collections_and_cap_author(tmp_path):
    from booksengine.model import filters as f
    info, picker, sc, order = _picker_world(tmp_path)
    rated = RatedFilter(info, np.array([8]))
    picked, removed = picker.pick(order, lambda c: sc[c], 20, rated)
    assert picked == [0, 4, 5]                          # «Dune #2» → «Dune #1» на его месте, «Dune #1» не повторяется
    assert removed == [(1, f.LATER), (3, f.COLLECTION), (6, f.AUTHOR), (7, f.LATER_DROPPED), (9, f.RATED)]
    picked, removed = picker.pick(order, lambda c: sc[c], 5, rated)
    assert picked == [0, 4] and (5, f.AUTHOR) in removed  # список из 5 — одна книга автора
    no_first = sc.copy()
    no_first[0] = -np.inf                              # первая книга серии — не кандидат: поздний том просто убран
    picked, removed = picker.pick(order[order != 0], lambda c: no_first[c], 20, rated)
    assert 0 not in picked and 1 not in picked and removed[0] == (1, f.LATER_DROPPED)
    picked, removed = picker.pick(order, lambda c: sc[c], 20, rated, rules=False)
    assert picked == [1, 3, 4, 5, 6, 7, 0] and removed == [(9, f.RATED)]   # без правил — только «уже оценено»


def test_pick_top_matches_full_order_even_with_small_pool(tmp_path):
    from booksengine.model.filters import pick_top
    info, picker, sc, order = _picker_world(tmp_path)
    rated = RatedFilter(info, np.array([8]))
    want, _ = picker.pick(order, lambda c: sc[c], 20, rated)
    cols = np.arange(len(sc))
    for pool in (3, 100):                               # 3 — правила убрали больше, чем кандидатов: весь порядок
        got = pick_top(sc[None, :], cols, cols, picker, [rated], 20, pool=pool)
        assert got[0].tolist() == want
