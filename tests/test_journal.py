import pandas as pd

from booksengine import journal
from booksengine.recommend import Rec


def test_report_compares_advised_reads_with_own_choices(tmp_path):
    hist, prof = tmp_path / "history", tmp_path / "profiles"
    prof.mkdir()
    recs = [Rec(10, "Ten", "A", 80, [], None), Rec(11, "Eleven", "B", 70, [], None), Rec(12, "Twelve", "C", 60, [], None)]
    journal.save(recs, "me", "fp", hist)
    first = next(hist.glob("me-*.csv"))
    first.rename(hist / "me-2026-01-01.csv")                          # более ранняя выдача
    journal.save([Rec(12, "Twelve", "C", 65, [], None), Rec(13, "Thirteen", "D", 55, [], None)], "me", "fp", hist)
    pd.DataFrame({"title": ["Своя", "Своя-2", "Ten", "Twelve-тень", "Thirteen"],
                  "goodreads_work_id": [1, 2, 10, 99, 13], "rating": [3, 5, 5, 2, None],
                  "status": ["read", "read", "read", "read", "dnf"]}).to_csv(prof / "me.csv", index=False)
    pd.DataFrame({"shadow_work_id": [99], "main_work_id": [12]}).to_parquet(tmp_path / "work_merges.parquet")
    adv = journal.advised(hist, "me")
    assert adv.set_index("goodreads_work_id").date.to_dict()[12] == "2026-01-01"   # книга — по первой выдаче
    text = journal.report(prof, hist, tmp_path)
    assert "Выдач: 2" in text and "разных книг в них: 4" in text
    assert "Прочитано из советов: 3 — пятёрок 33%, 1–2★ — 67%" in text          # 5, тень 12 → 2, dnf → 1
    assert "Остальные оценки (книги, выбранные без модели, 2): пятёрок 50%, 1–2★ — 0%" in text
    assert "| Twelve — C | 2026-01-01 | 3 | 60% | 2 |" in text


def test_report_without_reads_says_so(tmp_path):
    hist, prof = tmp_path / "history", tmp_path / "profiles"
    prof.mkdir()
    journal.save([Rec(10, "Ten", "A", 80, [], None)], "me", "fp", hist)
    pd.DataFrame({"goodreads_work_id": [1], "rating": [4]}).to_csv(prof / "me.csv", index=False)
    pd.DataFrame({"goodreads_work_id": [1], "rating": [4]}).to_csv(prof / "other.csv", index=False)   # без выдач
    text = journal.report(prof, hist)
    assert "## me" in text and "пока сравнивать нечего" in text and "## other" not in text
