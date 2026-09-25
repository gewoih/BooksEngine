import json

from booksengine import report_3a
from booksengine.model import split


def _summary(v):
    cell = {"mean": v, "lo": v - 0.01, "hi": v + 0.01, "n": 100}
    return {g: {m: cell for m in ("ndcg20", "ndcg10", "recall20", "map20", "low20")}
            for g in ["all", *split.BUCKET_ORDER]}


def test_report_contains_numbers_from_json(tmp_path):
    ed, sd = tmp_path / "eval", tmp_path / "split"
    ed.mkdir(); sd.mkdir()
    (sd / "split.json").write_text(json.dumps({"seed": 1, "share": 0.2, "groups": {
        "val": {"users": 5000, "buckets": {}}, "test": {"users": 20000, "buckets": {"20-39": 1800}}}}))
    (ed / "popularity_val.json").write_text(json.dumps([
        {"fit_params": {"formula": "count", "m": 0.0}, "score_params": {}, "fit_seconds": 1.0,
         "coverage": 0.001, "summary": _summary(0.05)}]))
    (ed / "popularity_test.json").write_text(json.dumps({
        "model": "popularity", "fit_params": {"formula": "count", "m": 0.0}, "fit_seconds": 1.0, "n_users": 20000,
        "variants": [{"label": "—", "score_params": {}, "deployable": True, "coverage": 0.001,
                      "summary": _summary(0.0512)}]}))
    out = report_3a.write(ed, sd, tmp_path / "r.md")
    text = out.read_text()
    assert "0.0512" in text and "Популярность" in text and "20 000" in text and "1 800" in text
    assert "вне начатых серий" in text and "вдвое больше" in text
