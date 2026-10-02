"""serve: выдача приложению — тот же код, что консольный recommend; книги — по внешнему id."""
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pandas as pd
import pytest

from booksengine import recommend as rec
from booksengine import serve
from booksengine.model import chance
from booksengine.model import layers as ly
from tests.test_layers import world  # noqa: F401 — фикстура: данные, разбиение и компоненты слоёв

RATED = {100: 5, 101: 5, 102: 4, 125: 1}


@pytest.fixture
def service(world):  # noqa: F811
    tp, sd, md = world
    ly.run("val", clean_dir=tp, split_dir=sd, models_dir=md, eval_dir=tp / "eval")
    chance.calibrate("layers", ratings_path=tp / "ratings.parquet", split_dir=sd, models_dir=md, eval_dir=tp / "eval")
    return serve.Service(rec.Engine(tp, md), "books"), tp, md


def _ratings(dnf=()):
    return [{"source": "goodreads", "id": str(w), "rating": r, "dnf": w in dnf, "title": f"T{w}"}
            for w, r in RATED.items()]


def test_recommend_matches_console_and_gives_ranks_of_other_sizes(service, tmp_path):
    s, tp, md = service
    csv = tmp_path / "p.csv"
    pd.DataFrame({"goodreads_work_id": list(RATED), "rating": list(RATED.values()),
                  "title": [f"T{w}" for w in RATED]}).to_csv(csv, index=False)
    want = rec.recommend(csv, clean_dir=tp, models_dir=md, top=5)
    got = s.recommend({"ratings": _ratings(), "top": 5, "rank_tops": [3, 5]})
    items = [(sec["name"], i) for sec in got["sections"] for i in sec["items"]]
    assert [(n, int(i["id"]), i["chance"]) for n, i in items] == [(r.section, r.work_id, r.chance) for r in want.recs]
    assert all(i["source"] == "goodreads" and i["why"] == rec.why_text(r) for (_, i), r in zip(items, want.recs))
    assert got["used"] == 4 and got["chance_label"] == "5★"
    assert [r["id"] for r in got["ranks"]["5"]] == [i["id"] for _, i in items]          # тот же список
    assert len([r for r in got["ranks"]["3"] if r["section"] == rec.FICTION]) == 3
    with pytest.raises(ValueError, match="от 1 до 100"):
        s.recommend({"ratings": _ratings(), "top": 0})


def test_chance_similar_and_unknown_books(service):
    s, *_ = service
    first = s.recommend({"ratings": _ratings(), "top": 5})["sections"][0]["items"][0]
    c = s.chance({"ratings": _ratings(), "work": {"source": "goodreads", "id": first["id"]}})
    assert c["chance"] == first["chance"] and c["in_model"]
    rated = s.chance({"ratings": _ratings(), "work": {"source": "goodreads", "id": "100"}})
    assert rated["chance"] is None and rated["in_model"]                               # оценена — не кандидат
    assert s.chance({"ratings": _ratings(), "work": {"source": "goodreads", "id": "999999"}}) == \
        {"chance": None, "chance_label": "5★", "in_model": False}
    sim = s.similar("goodreads", "100")["items"]
    assert 0 < len(sim) <= serve.SIMILAR and all(r["source"] == "goodreads" for r in sim)
    assert s.similar("goodreads", "999999") == {"items": []}
    assert s.match({"rows": [{"title_en": "X", "author": "Y"}]}) == {"items": [None]}     # база без новых книг
    with pytest.raises(ValueError, match="источник"):
        s.work_id("litres", "1")


def test_http_answers_json_and_refuses_body_without_length(service):
    s, *_ = service
    srv = ThreadingHTTPServer(("127.0.0.1", 0), serve.handler(s))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        assert json.load(urllib.request.urlopen(url + "/health"))["domain"] == "books"
        body = json.dumps({"ratings": _ratings(), "top": 3}).encode()
        got = json.load(urllib.request.urlopen(urllib.request.Request(url + "/recommend", body)))
        assert got["sections"][0]["items"]
        # тело частями (chunked) http.server не читает — ошибка, а не молча пустой профиль
        chunked = urllib.request.Request(url + "/recommend", iter([body]), {"Transfer-Encoding": "chunked"})
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(chunked)
        assert e.value.code == 411
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(urllib.request.Request(url + "/recommend", b'{"top": "x"}'))
        assert e.value.code == 400
    finally:
        srv.shutdown()
