import numpy as np
import pandas as pd
import pytest

from booksengine.model.filters import Books, RatedFilter, work_info


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
    works = [(1, "Alice's Adventures in Wonderland", "x", False, 10), (2, "Alice in Wonderland", "x", False, 10),
             (3, "Solaris", "Solaris", False, 11)]
    _write(tmp_path, works, [(10, "Carroll"), (11, "Lem"), (99, "Artist")])
    info = work_info(tmp_path, np.array([1, 2, 3]))
    picked, removed = f.ListPicker(info).pick([0, 1, 2], 20)
    assert picked == [0, 2] and removed == [(1, f.DUPLICATE)]


def _picker_world(tmp_path):
    """Серия Dune (#1–#3), сборник, три книги одного автора, поздний том без первого, оценённая книга и её дубль."""
    from booksengine.model.filters import ListPicker
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
    picker = ListPicker(info)
    sc = np.array([3, 9, -np.inf, 8, 7, 6, 5, 4, -np.inf, 2], dtype=np.float64)   # 8 — вход, 2 — не кандидат
    order = np.argsort(-sc, kind="stable")
    return info, picker, sc, order[np.isfinite(sc[order])]


def test_books_of_handles_domain_without_any_authors(tmp_path):
    """Домен без авторов совсем (фильмы, шаг 1 плана «Фильмы»): work_authors.parquet и authors.parquet
    пустые — LEFT JOIN в work_info даёт NULL (pandas: pd.NA, не None) в колонке authors. Books.of не должен
    падать: np.isscalar(pd.NA) — False, старая проверка это NULL не ловила."""
    works = pd.DataFrame([(1, "Toy Story (1995)", "Toy Story (1995)", False)],
                         columns=["work_id", "title", "best_edition_title", "is_collection"])
    works.insert(2, "original_title", None)
    works.to_parquet(tmp_path / "works.parquet")
    pd.DataFrame({"work_id": pd.Series(dtype="int64"), "author_id": pd.Series(dtype="int64"),
                 "role": pd.Series(dtype="object"), "position": pd.Series(dtype="int64")}
                ).to_parquet(tmp_path / "work_authors.parquet")
    pd.DataFrame({"author_id": pd.Series(dtype="int64"), "name": pd.Series(dtype="object")}
                ).to_parquet(tmp_path / "authors.parquet")

    info = work_info(tmp_path, np.array([1]))
    books = Books.of(info)
    assert books.author.tolist() == [-1]
    assert books.authors[0] == frozenset()
    f = RatedFilter(info, np.array([]))
    assert f.is_rated_already(0) is False


def test_author_with_role_when_nobody_is_without_role(tmp_path):
    """У «Маленького принца» автор записан с ролью «Author/Illustrator», у адаптации «Хоббита» — «Creator» после
    адаптатора: это основной автор, и адаптация прочитанной книги — «уже оценено». Одни иллюстраторы — автора нет,
    вместо имени пустая строка, а не «nan»."""
    pd.DataFrame([(1, "The Little Prince", None, "The Little Prince", False),
                  (2, "The Hobbit", None, "The Hobbit", False),
                  (3, "The Hobbit: Graphic Novel", None, "The Hobbit: Graphic Novel", False),
                  (4, "Pictures", None, "Pictures", False)],
                 columns=["work_id", "title", "original_title", "best_edition_title", "is_collection"]
                 ).to_parquet(tmp_path / "works.parquet")
    pd.DataFrame([(1, 10, "Author/Illustrator", 1), (1, 11, "Translator", 2), (2, 12, None, 1),
                  (3, 13, "Adapter", 1), (3, 12, "Creator", 2), (4, 14, "Illustrator", 1)],
                 columns=["work_id", "author_id", "role", "position"]).to_parquet(tmp_path / "work_authors.parquet")
    pd.DataFrame([(10, "Saint-Exupéry"), (11, "Howard"), (12, "Tolkien"), (13, "Dixon"), (14, "Artist")],
                 columns=["author_id", "name"]).to_parquet(tmp_path / "authors.parquet")
    info = work_info(tmp_path, np.array([1, 2, 3, 4]))
    assert info.author.tolist() == ["Saint-Exupéry", "Tolkien", "Tolkien", ""]
    assert RatedFilter(info, np.array([1])).is_rated_already(2)       # прочитан «Хоббит» — не советовать комикс


def test_author_cap_is_one_book_per_ten_places():
    from booksengine.model.filters import author_cap
    assert [author_cap(t) for t in (5, 10, 19, 20, 25, 50)] == [1, 1, 1, 2, 2, 5]


def test_list_rules_drop_later_volume_and_collections_and_cap_author(tmp_path):
    from booksengine.model import filters as f
    info, picker, sc, order = _picker_world(tmp_path)
    rated = RatedFilter(info, np.array([8]))
    picked, removed = picker.pick(order, 20, rated)
    # «Dune #2» убран, «Dune #1» — на своём месте по своему баллу, а не на месте позднего тома
    assert picked == [4, 5, 0]
    assert removed == [(1, f.LATER), (3, f.COLLECTION), (6, f.AUTHOR), (7, f.LATER), (9, f.RATED)]
    picked, removed = picker.pick(order, 5, rated)
    assert picked == [4, 0] and (5, f.AUTHOR) in removed  # список из 5 — одна книга автора
    picked, removed = picker.pick(order, 20, rated, rules=False)
    assert picked == [1, 3, 4, 5, 6, 7, 0] and removed == [(9, f.RATED)]   # без правил — только «уже оценено»


def test_pick_top_matches_full_order_even_with_small_pool(tmp_path):
    from booksengine.model.filters import pick_top
    info, picker, sc, order = _picker_world(tmp_path)
    rated = RatedFilter(info, np.array([8]))
    want, _ = picker.pick(order, 20, rated)
    cols = np.arange(len(sc))
    for pool in (3, 100):                               # 3 — правила убрали больше, чем кандидатов: весь порядок
        got = pick_top(sc[None, :], cols, picker, [rated], 20, pool=pool)
        assert got[0].tolist() == want


@pytest.mark.parametrize("title, later", [
    ("A Storm of Swords: Blood and Gold (A Song of Ice and Fire, #3: Part 2 of 2)", True),
    ("Locke & Key, Vol. 6: Alpha & Omega", True),
    ("The Way of Kings, Part 2 (The Stormlight Archive #1.2)", True),
    ("1Q84 BOOK 3 (1Q84, #3)", True),
    ("Preacher, Volume Two", True),
    ("Mort (Discworld, #4; Death, #1)", False),            # начало подсерии «Смерть»
    ("Death Note, Vol. 1: Boredom (Death Note, #1)", False),
    ("Harry Potter Boxset (Harry Potter, #1-7)", False),
    ("The Book Thief", False),
    ("Book 13", False),
    ("Faust, Part One", False),
])
def test_later_volume_by_title(title, later):
    from booksengine.model.filters import later_by_title
    assert later_by_title(title) is later


def test_list_skips_new_amazon_book_without_known_russian_translation(tmp_path):
    """Новая книга Amazon (после 2017, её нет в Goodreads) без известного русского перевода не советуется: читают
    по-русски, книга без перевода — мусор. У книг Goodreads (ru_known пусто) правило не действует."""
    from booksengine.model.filters import UNTRANSLATED, ListPicker
    works = [(1, "Old Goodreads Book", "Old Goodreads Book", False, 10),
             (2, "New Without Translation", "New Without Translation", False, 11),
             (3, "New With Translation", "New With Translation", False, 12)]
    _write(tmp_path, works, [(10, "A"), (11, "B"), (12, "C"), (99, "Artist")])
    w = pd.read_parquet(tmp_path / "works.parquet")
    w["source"] = ["goodreads", "amazon", "amazon"]
    w["ru_known"] = [None, False, True]
    w.to_parquet(tmp_path / "works.parquet")
    info = work_info(tmp_path, np.array([1, 2, 3]))
    picked, removed = ListPicker(info).pick([0, 1, 2], top=3)
    assert picked == [0, 2] and removed == [(1, UNTRANSLATED)]
    assert ListPicker(info).pick([0, 1, 2], top=3, rules=False)[0] == [0, 1, 2]


def test_work_info_without_source_columns_blocks_nothing(tmp_path):
    from booksengine.model.filters import ListPicker
    _write(tmp_path, [(1, "A Book", "A Book", False, 10)], [(10, "A"), (99, "Artist")])
    info = work_info(tmp_path, np.array([1]))
    assert not info.no_translation.any()
    assert ListPicker(info).pick([0], top=1)[0] == [0]


def test_work_info_exposes_russian_title_of_new_book(tmp_path):
    from booksengine.recommend import display_title
    _write(tmp_path, [(1, "Educated", "Educated", False, 10), (2, "Dune", "Dune", False, 11)],
           [(10, "Tara Westover"), (11, "Frank Herbert"), (99, "Artist")])
    w = pd.read_parquet(tmp_path / "works.parquet")
    w["source"], w["ru_known"], w["ru_title"] = ["amazon", "goodreads"], [True, None], ["Ученица", None]
    w.to_parquet(tmp_path / "works.parquet")
    info = work_info(tmp_path, np.array([1, 2]))
    assert display_title(info, 0) == "Ученица / Educated" and display_title(info, 1) == "Dune"
    _write(tmp_path, [(1, "Dune", "Dune", False, 11)], [(11, "Frank Herbert"), (99, "Artist")])
    assert display_title(work_info(tmp_path, np.array([1])), 0) == "Dune"   # база без русских названий
