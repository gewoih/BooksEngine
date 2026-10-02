"""ru-titles: русское название с Fantlab берётся, только если совпали название и фамилия автора."""
import json

from booksengine.data import ru_titles as rt


def _m(**kw):
    base = {"rusname": "Над пропастью во ржи", "name": "The Catcher in the Rye", "altname": "",
            "all_autor_name": "Jerome D. Salinger", "name_eng": "novel", "markcount": 2470}
    return base | kw


def test_norm_cuts_series_article_and_diacritics():
    assert rt.norm("The Catcher in the Rye") == "catcher in the rye"
    assert rt.norm("Dune (Dune Chronicles #1)") == rt.norm("Dune: Deluxe") == "dune"
    assert rt.norm("Amélie") == "amelie" and rt.norm(None) == "" and rt.norm(float("nan")) == ""
    assert rt.surname("J.D. Salinger") == "salinger"


def test_pick_needs_same_title_and_author_surname():
    assert rt.pick([_m()], "The Catcher in the Rye", "J.D. Salinger")["rusname"] == "Над пропастью во ржи"
    assert rt.pick([_m(all_autor_name="Alice Hoffman")], "The Catcher in the Rye", "J.D. Salinger") is None
    assert rt.pick([_m(name="On The Catcher in the Rye")], "The Catcher in the Rye", "J.D. Salinger") is None
    assert rt.pick([_m(name_eng="cycle")], "The Catcher in the Rye", "J.D. Salinger") is None
    assert rt.pick([_m(rusname="")], "The Catcher in the Rye", "J.D. Salinger") is None
    # английское — в альтернативных названиях (оригинал на иврите); из нескольких — самое оцениваемое
    sapiens = _m(rusname="Sapiens. Краткая история человечества", name="קיצור תולדות האנושות",
                 altname="Sapiens: A Brief History of Humankind", all_autor_name="יובל נח הררי / Yuval Noah Harari")
    assert rt.pick([sapiens], "Sapiens: A Brief History of Humankind", "Yuval Noah Harari") is sapiens
    rare = _m(rusname="Над пропастью", markcount=3)
    assert rt.pick([rare, _m()], "The Catcher in the Rye", "J.D. Salinger")["markcount"] == 2470


def test_fantlab_answers_are_cached(tmp_path):
    cache = tmp_path / "c.jsonl"
    cache.write_text(json.dumps({"q": "Dune", "matches": [_m(rusname="Дюна", name="Dune")]}, ensure_ascii=False) + "\n")
    fl = rt.Fantlab(cache)
    assert fl.search("Dune")[0]["rusname"] == "Дюна"                 # из кэша, без сети
