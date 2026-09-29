# Amazon Reviews'23: приём данных, мост, сигнал перевода — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Разобрать Amazon Reviews'23 (Books + Kindle Store) в `data/amazon/clean/`, построить мост на
Goodreads по ISBN/ASIN и получить отчёт со сколькими книгами после 2017 года реально можно будет советовать
(есть достаточно оценок и известный русский перевод по Wikidata).

**Architecture:** Новый модуль `src/booksengine/data/amazon.py` — тот же DuckDB-паттерн, что `stage.py`/
`clean.py` (разбор → лёгкая очистка k-core → мост → экспорт в Parquet), плюс `src/booksengine/data/
wikidata.py` для пакетных SPARQL-запросов с диск-кэшем. Новая CLI-команда `amazon-bridge` запускает весь
пайплайн и пишет `reports/amazon_bridge.md`. Сырые файлы Amazon скачиваются отдельным устойчивым к обрывам
скриптом (`scripts/fetch_amazon_raw.sh`) в `RAW_DIR/amazon_reviews_2023/` — не автоматически внутри
пайплайна (сервер источника медленный и рвёт соединение).

**Tech Stack:** Python 3.12, DuckDB (staging/SQL), pandas + pyarrow (Parquet), typer (CLI), stdlib
`urllib.request` (Wikidata HTTP — без новых зависимостей, как в `tmdb.py`), pytest + `tmp_path`.

**Spec:** `docs/superpowers/specs/2026-09-29-amazon-reviews-bridge-design.md`

## Global Constraints

- Тесты пишут только в `tmp_path`, реальные пути (`RAW_DIR`, `~/Downloads`, боевой `data/`) в тестах не
  используются.
- `data/`, `reports/`, `models/` не коммитятся (некоммерческая лицензия датасетов).
- Коммиты — в рабочую ветку `amazon-reviews-probe` (эта сессия уже на ней), не в `main`.
- Сырые файлы Amazon — только в `RAW_DIR/amazon_reviews_2023/`, докачка — только через
  `scripts/fetch_amazon_raw.sh`; пайплайн (`amazon.py`, CLI) сеть не трогает вообще, кроме Wikidata-запросов.
- Без сигнала перевода книга не считается «с переводом» — `ru_translation_known = False` по умолчанию
  (нет ISBN-13 или книга уже с мостом на Goodreads — не проверяется).
- k-core для Amazon — `min_user=5, min_work=5` (не 20/100, как у книг/фильмов — оценки на человека в Amazon
  на порядок реже).
- Год издания из текста `Publisher` — берём **минимальное** правдоподобное 4-значное число (1900..текущий
  год), не максимальное: в проверке `max` иногда цеплял шумовое число из текста (найдены значения 2085–2099).
- Wikidata SPARQL — пачками по ISBN (`VALUES`, до 200 за запрос), не по одному запросу на книгу; результат
  кэшируется в Parquet, чтобы повторный прогон не бил эндпоинт заново.

## Review Focus

- Amazon-метаданные без ISBN-10/13 (`details` не содержит ключей, или книга без деталей вовсе) — мост и
  сигнал перевода должны трактовать это как «нет пары», не падать. → Task 4.
- Оценённая (прошла k-core) книга без соответствующей записи в meta-файле (битая/пропущенная строка JSON) —
  `items.parquet` всё равно должен получить строку для неё (title/author = NULL), не потерять и не уронить
  экспорт. → Task 5.
- Несколько изданий Goodreads с одинаковым ISBN (данные не идеальны) — мост не должен размножать строки на
  повторные совпадения. → Task 4.
- Wikidata возвращает ответ не по всем ISBN из пачки (нашлось не всё) — отсутствующие в ответе трактуются
  как «перевода нет», без ошибки и без потери остальных ISBN пачки. → Task 6.
- Повторный прогон с уже заполненным кэшем Wikidata — не отправляет запросы по уже известным ISBN. → Task 6.

---

## Task 1: `scripts/fetch_amazon_raw.sh` — устойчивая докачка сырья

**Files:**
- Create: `scripts/fetch_amazon_raw.sh`

**Interfaces:**
- Produces: 4 файла в `$RAW_DIR/amazon_reviews_2023/` (`Books.csv.gz`, `Kindle_Store.csv.gz`,
  `meta_Books.jsonl.gz`, `meta_Kindle_Store.jsonl.gz`) — вход для `amazon.prepare()` (Task 8).

Это инфраструктурный bash-скрипт для разового скачивания сырья на диск пользователя — не Python-код
проекта, не покрывается pytest (как и `scripts/night.sh`). Проверка — ручной прогон.

- [ ] **Step 1: Написать скрипт**

```bash
#!/bin/bash
# Устойчивая докачка сырых файлов Amazon Reviews'23 в RAW_DIR/amazon_reviews_2023/.
# Сервер McAuley Lab медленный (~0.8 МБ/с) и рвёт соединение на многогигабайтных файлах — докачка (-C -)
# с обрывом при зависании (--speed-limit/--speed-time) и повтором до полного Content-Length.
set -uo pipefail

RAW_DIR="${RAW_DIR:-$HOME/Downloads}"
OUT="$RAW_DIR/amazon_reviews_2023"
mkdir -p "$OUT"

BASE="https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023"

fetch() {
  local url="$1" out="$2" expected="$3" attempt=0
  while true; do
    attempt=$((attempt + 1))
    local cur
    cur=$(stat -f%z "$out" 2>/dev/null || stat -c%s "$out" 2>/dev/null || echo 0)
    if [ "$cur" -ge "$expected" ]; then
      echo "$out: готово ($cur байт)"
      return 0
    fi
    echo "$out: попытка $attempt, $cur/$expected байт"
    curl -sL -C - --speed-limit 2048 --speed-time 30 --retry 5 --retry-delay 3 -o "$out" "$url"
    if [ "$attempt" -gt 200 ]; then
      echo "$out: не удалось докачать за $attempt попыток" >&2
      return 1
    fi
    sleep 2
  done
}

fetch "$BASE/benchmark/0core/rating_only/Books.csv.gz" "$OUT/Books.csv.gz" 602182123
fetch "$BASE/benchmark/0core/rating_only/Kindle_Store.csv.gz" "$OUT/Kindle_Store.csv.gz" 464452653
fetch "$BASE/raw/meta_categories/meta_Books.jsonl.gz" "$OUT/meta_Books.jsonl.gz" 4942125770
fetch "$BASE/raw/meta_categories/meta_Kindle_Store.jsonl.gz" "$OUT/meta_Kindle_Store.jsonl.gz" 2269538269

status=0
for f in "$OUT"/*.gz; do
  if gzip -t "$f" 2>/dev/null; then
    echo "$f: gzip OK"
  else
    echo "$f: gzip BROKEN — удалите и перезапустите скрипт" >&2
    status=1
  fi
done
exit $status
```

- [ ] **Step 2: Сделать исполняемым**

Run: `chmod +x scripts/fetch_amazon_raw.sh`

- [ ] **Step 3: Ручная проверка (не в CI) — запустить и убедиться, что все 4 файла целы**

Run: `RAW_DIR=~/Downloads scripts/fetch_amazon_raw.sh` (полный прогон — часы из-за скорости источника;
можно прерывать Ctrl+C и перезапускать — докачает с места остановки). Ожидается: 4 строки `gzip OK` в конце,
код возврата 0.

- [ ] **Step 4: Commit**

```bash
git add scripts/fetch_amazon_raw.sh
git commit -m "Скрипт устойчивой докачки сырья Amazon Reviews'23"
```

---

## Task 2: `amazon.py` — `stage_ratings`

**Files:**
- Create: `src/booksengine/data/amazon.py`
- Test: `tests/test_amazon.py`

**Interfaces:**
- Produces: `stage_ratings(con: duckdb.DuckDBPyConnection, books_csv: Path, kindle_csv: Path) -> None` —
  создаёт temp-таблицу `_az_ratings(user_id VARCHAR, parent_asin VARCHAR, rating DOUBLE, timestamp BIGINT,
  source VARCHAR)`. Используется Task 4 (`apply_kcore`), Task 8 (`prepare`).

- [ ] **Step 1: Написать падающий тест**

```python
# tests/test_amazon.py
import duckdb
import pytest

from booksengine.data import amazon


@pytest.fixture
def con():
    return duckdb.connect()


def test_stage_ratings_unions_books_and_kindle_with_source_column(tmp_path, con):
    books = tmp_path / "books.csv"
    books.write_text("user_id,parent_asin,rating,timestamp\nU1,B001,5.0,1000\nU2,B002,3.0,1001\n")
    kindle = tmp_path / "kindle.csv"
    kindle.write_text("user_id,parent_asin,rating,timestamp\nU1,K001,4.0,1002\n")

    amazon.stage_ratings(con, books, kindle)

    rows = con.execute(
        "SELECT user_id, parent_asin, rating, timestamp, source FROM _az_ratings ORDER BY parent_asin"
    ).fetchall()
    assert rows == [
        ("U1", "B001", 5.0, 1000, "books"),
        ("U2", "B002", 3.0, 1001, "books"),
        ("U1", "K001", 4.0, 1002, "kindle"),
    ]
```

- [ ] **Step 2: Убедиться, что тест падает**

Run: `uv run pytest tests/test_amazon.py -v`
Expected: FAIL — `ModuleNotFoundError` или `AttributeError: module 'booksengine.data.amazon' has no
attribute 'stage_ratings'` (модуля ещё нет).

- [ ] **Step 3: Реализовать**

```python
# src/booksengine/data/amazon.py
"""Amazon Reviews'23 (McAuley Lab) — второй источник для книжной модели: мост на Goodreads по ISBN/ASIN и
сигнал перевода (Wikidata) для книг без моста. Дизайн: docs/superpowers/specs/2026-09-29-amazon-reviews-bridge-design.md.

Сырьё — RAW_DIR/amazon_reviews_2023/ (Books.csv.gz, Kindle_Store.csv.gz, meta_Books.jsonl.gz,
meta_Kindle_Store.jsonl.gz), см. scripts/fetch_amazon_raw.sh. Этот модуль сеть не трогает вообще, кроме
`apply_translation_signal` (через wikidata.check_translations).
"""
from pathlib import Path

import duckdb
import pandas as pd

MIN_USER = 5
MIN_WORK = 5


def stage_ratings(con: duckdb.DuckDBPyConnection, books_csv: Path, kindle_csv: Path) -> None:
    """_az_ratings: user_id, parent_asin, rating, timestamp, source ('books'|'kindle') — оба rating_only-
    файла как есть, без фильтрации."""
    con.execute("""
        CREATE OR REPLACE TEMP TABLE _az_ratings AS
        SELECT user_id::VARCHAR AS user_id, parent_asin::VARCHAR AS parent_asin, rating::DOUBLE AS rating,
               timestamp::BIGINT AS timestamp, 'books'::VARCHAR AS source
        FROM read_csv_auto(?)
        UNION ALL
        SELECT user_id::VARCHAR AS user_id, parent_asin::VARCHAR AS parent_asin, rating::DOUBLE AS rating,
               timestamp::BIGINT AS timestamp, 'kindle'::VARCHAR AS source
        FROM read_csv_auto(?)
    """, [str(books_csv), str(kindle_csv)])
```

- [ ] **Step 4: Убедиться, что тест проходит**

Run: `uv run pytest tests/test_amazon.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/booksengine/data/amazon.py tests/test_amazon.py
git commit -m "amazon.py: разбор rating_only Books/Kindle в единую таблицу"
```

---

## Task 3: `amazon.py` — `stage_meta` и `extract_year`

**Files:**
- Modify: `src/booksengine/data/amazon.py`
- Test: `tests/test_amazon.py`

**Interfaces:**
- Consumes: ничего нового (независимо от Task 2).
- Produces: `stage_meta(con, books_jsonl: Path, kindle_jsonl: Path) -> None` — temp-таблица
  `_az_meta(parent_asin, title, author, isbn10, isbn13, publisher_raw, language, categories, source)`.
  `extract_year(publisher_raw: str | None) -> int | None`. Используются Task 4 (`build_bridge`), Task 5
  (`export`).

- [ ] **Step 1: Написать падающие тесты**

```python
import json


def test_stage_meta_extracts_isbn_author_language_from_details(tmp_path, con):
    books = tmp_path / "meta_books.jsonl"
    books.write_text(json.dumps({
        "parent_asin": "0701169850", "title": "Chaucer",
        "author": {"name": "Peter Ackroyd"},
        "details": {"Publisher": "Chatto & Windus; First Edition (January 1, 2004)",
                    "Language": "English", "ISBN 10": "0701169850", "ISBN 13": "978-0701169855"},
        "categories": ["Books", "Literature & Fiction"],
    }) + "\n")
    kindle = tmp_path / "meta_kindle.jsonl"
    kindle.write_text(json.dumps({
        "parent_asin": "B0192CTMWI", "title": "Some Ebook",
        "author": {"name": "Jane Doe"},
        "details": {"Language": "English"},
        "categories": ["Kindle Store"],
    }) + "\n")

    amazon.stage_meta(con, books, kindle)

    rows = con.execute(
        "SELECT parent_asin, title, author, isbn10, isbn13, publisher_raw, language, categories, source "
        "FROM _az_meta ORDER BY source"
    ).fetchall()
    assert rows[0] == ("0701169850", "Chaucer", "Peter Ackroyd", "0701169850", "978-0701169855",
                       "Chatto & Windus; First Edition (January 1, 2004)", "English",
                       "Books|Literature & Fiction", "books")
    assert rows[1][0] == "B0192CTMWI" and rows[1][3] is None and rows[1][8] == "kindle"


def test_stage_meta_handles_book_without_details_or_categories(tmp_path, con):
    books = tmp_path / "meta_books.jsonl"
    books.write_text(json.dumps({"parent_asin": "X1", "title": "No Details Book"}) + "\n")
    kindle = tmp_path / "meta_kindle.jsonl"
    kindle.write_text("")

    amazon.stage_meta(con, books, kindle)

    row = con.execute("SELECT parent_asin, isbn10, categories FROM _az_meta").fetchone()
    assert row == ("X1", None, None)


def test_extract_year_picks_earliest_plausible_four_digit_year():
    assert amazon.extract_year("Chatto & Windus; First Edition (January 1, 2004)") == 2004
    assert amazon.extract_year("Heinemann; First Edition (May 20, 1996)") == 1996


def test_extract_year_ignores_out_of_range_noise_and_prefers_min_over_max():
    # в проверке на реальных данных max иногда цеплял шумовое число (2085-2099) из текста издателя
    assert amazon.extract_year("Some Press (2011); this edition 2018, item code 2094812") == 2011


def test_extract_year_returns_none_without_plausible_year():
    assert amazon.extract_year(None) is None
    assert amazon.extract_year("") is None
    assert amazon.extract_year("No year mentioned here") is None
```

- [ ] **Step 2: Убедиться, что тесты падают**

Run: `uv run pytest tests/test_amazon.py -v -k "stage_meta or extract_year"`
Expected: FAIL — атрибутов ещё нет.

- [ ] **Step 3: Реализовать**

```python
import re

CURRENT_YEAR = 2026


def stage_meta(con: duckdb.DuckDBPyConnection, books_jsonl: Path, kindle_jsonl: Path) -> None:
    """_az_meta: parent_asin, title, author, isbn10, isbn13, publisher_raw, language, categories, source —
    поля из `details` (карта строка->строка на диске, как popular_shelves у книжных editions) — автор,
    ISBN-10/13, издатель+дата текстом, язык."""
    columns = {
        "parent_asin": "VARCHAR", "title": "VARCHAR", "author": "STRUCT(name VARCHAR)",
        "details": "MAP(VARCHAR, VARCHAR)", "categories": "VARCHAR[]",
    }
    cols_sql = "{" + ", ".join(f"'{k}': '{v}'" for k, v in columns.items()) + "}"

    def _src(path: Path) -> str:
        return (f"read_json('{path}', format='newline_delimited', columns={cols_sql}, "
                f"ignore_errors=true, maximum_object_size=67108864)")

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _az_meta AS
        SELECT parent_asin, title, author.name AS author, details['ISBN 10'] AS isbn10,
               details['ISBN 13'] AS isbn13, details['Publisher'] AS publisher_raw,
               details['Language'] AS language, array_to_string(categories, '|') AS categories,
               'books' AS source
        FROM {_src(books_jsonl)}
        UNION ALL
        SELECT parent_asin, title, author.name AS author, details['ISBN 10'] AS isbn10,
               details['ISBN 13'] AS isbn13, details['Publisher'] AS publisher_raw,
               details['Language'] AS language, array_to_string(categories, '|') AS categories,
               'kindle' AS source
        FROM {_src(kindle_jsonl)}
    """)


def extract_year(publisher_raw: str | None) -> int | None:
    """Год издания — минимальное правдоподобное 4-значное число (1900..CURRENT_YEAR) в тексте `Publisher`
    («Chatto & Windus; First Edition (January 1, 2004)» -> 2004). Минимальное, не максимальное: в тексте
    попадаются шумовые числа (артикулы, вес) — они почти всегда больше настоящей даты издания."""
    if not publisher_raw:
        return None
    years = [int(y) for y in re.findall(r"(?:19|20)\d{2}", publisher_raw)]
    plausible = [y for y in years if 1900 <= y <= CURRENT_YEAR]
    return min(plausible) if plausible else None
```

- [ ] **Step 4: Убедиться, что тесты проходят**

Run: `uv run pytest tests/test_amazon.py -v`
Expected: PASS (все тесты, включая Task 2)

- [ ] **Step 5: Commit**

```bash
git add src/booksengine/data/amazon.py tests/test_amazon.py
git commit -m "amazon.py: разбор meta_Books/meta_Kindle_Store, извлечение года издания"
```

---

## Task 4: `amazon.py` — `build_bridge`

**Files:**
- Modify: `src/booksengine/data/amazon.py`
- Test: `tests/test_amazon.py`

**Interfaces:**
- Consumes: `_az_ratings` (Task 2), `_az_meta` (Task 3) — обе temp-таблицы уже в сессии `con`.
- Produces: `build_bridge(con, editions_path: Path) -> None` — temp-таблица `bridge(parent_asin, work_id)`,
  по одной строке на каждый `parent_asin`, встреченный в `_az_ratings` (оба источника), `work_id IS NULL` —
  без пары в Goodreads. Используется Task 5 (`export`), Task 7 (`apply_translation_signal` через
  `bridge.parquet`).

- [ ] **Step 1: Написать падающие тесты**

```python
import pandas as pd


def test_build_bridge_matches_books_by_isbn_and_kindle_by_asin(tmp_path, con):
    editions = pd.DataFrame([
        {"work_id": 1, "isbn": "0701169850", "isbn13": "9780701169855", "kindle_asin": None},
        {"work_id": 2, "isbn": None, "isbn13": None, "kindle_asin": "B0192CTMWI"},
    ])
    editions_path = tmp_path / "editions.parquet"
    editions.to_parquet(editions_path)

    con.execute("CREATE OR REPLACE TEMP TABLE _az_meta AS SELECT * FROM (VALUES "
                "('0701169850', 'books', '0701169850', '978-0701169855')) "
                "t(parent_asin, source, isbn10, isbn13)")
    con.execute("CREATE OR REPLACE TEMP TABLE _az_ratings AS SELECT * FROM (VALUES "
                "('U1', '0701169850', 5.0, 1000, 'books'), "
                "('U1', 'B0192CTMWI', 5.0, 1001, 'kindle'), "
                "('U1', 'ZZZUNKNOWN', 3.0, 1002, 'kindle')) "
                "t(user_id, parent_asin, rating, timestamp, source)")

    amazon.build_bridge(con, editions_path)

    rows = dict(con.execute("SELECT parent_asin, work_id FROM bridge").fetchall())
    assert rows == {"0701169850": 1, "B0192CTMWI": 2, "ZZZUNKNOWN": None}


def test_build_bridge_treats_missing_isbn_as_no_match(tmp_path, con):
    editions = pd.DataFrame([{"work_id": 1, "isbn": "1111111111", "isbn13": "9781111111111",
                              "kindle_asin": None}])
    editions_path = tmp_path / "editions.parquet"
    editions.to_parquet(editions_path)

    con.execute("CREATE OR REPLACE TEMP TABLE _az_meta AS SELECT * FROM (VALUES "
                "('NOISBN', 'books', NULL, NULL)) t(parent_asin, source, isbn10, isbn13)")
    con.execute("CREATE OR REPLACE TEMP TABLE _az_ratings AS SELECT * FROM (VALUES "
                "('U1', 'NOISBN', 5.0, 1000, 'books')) t(user_id, parent_asin, rating, timestamp, source)")

    amazon.build_bridge(con, editions_path)

    row = con.execute("SELECT work_id FROM bridge WHERE parent_asin = 'NOISBN'").fetchone()
    assert row == (None,)


def test_build_bridge_does_not_fan_out_on_duplicate_isbn_in_editions(tmp_path, con):
    # данные Goodreads не идеальны: два разных work_id с одним и тем же ISBN
    editions = pd.DataFrame([
        {"work_id": 1, "isbn": "2222222222", "isbn13": None, "kindle_asin": None},
        {"work_id": 2, "isbn": "2222222222", "isbn13": None, "kindle_asin": None},
    ])
    editions_path = tmp_path / "editions.parquet"
    editions.to_parquet(editions_path)

    con.execute("CREATE OR REPLACE TEMP TABLE _az_meta AS SELECT * FROM (VALUES "
                "('DUP', 'books', '2222222222', NULL)) t(parent_asin, source, isbn10, isbn13)")
    con.execute("CREATE OR REPLACE TEMP TABLE _az_ratings AS SELECT * FROM (VALUES "
                "('U1', 'DUP', 5.0, 1000, 'books')) t(user_id, parent_asin, rating, timestamp, source)")

    amazon.build_bridge(con, editions_path)

    rows = con.execute("SELECT parent_asin, work_id FROM bridge WHERE parent_asin = 'DUP'").fetchall()
    assert len(rows) == 1   # ровно одна строка мостa, не две — дубль ISBN не размножает bridge
    assert rows[0][1] in (1, 2)
```

- [ ] **Step 2: Убедиться, что тесты падают**

Run: `uv run pytest tests/test_amazon.py -v -k build_bridge`
Expected: FAIL — `build_bridge` не определена.

- [ ] **Step 3: Реализовать**

```python
def build_bridge(con: duckdb.DuckDBPyConnection, editions_path: Path) -> None:
    """bridge: parent_asin, work_id (NULL — книги без пары в Goodreads). Books — по isbn10/isbn13 из
    _az_meta (только цифры и X, без дефисов) на editions.isbn/isbn13; Kindle — parent_asin напрямую на
    editions.kindle_asin, метаданные не нужны. Покрывает все parent_asin из _az_ratings, даже без записи в
    _az_meta (не распарсилась/нет ISBN) — тогда просто нет сигнала для сопоставления, work_id = NULL.
    GROUP BY + max() в конце — защита от дублей ISBN в editions (не размножает строки моста)."""
    con.execute("CREATE OR REPLACE TEMP VIEW _editions AS SELECT * FROM read_parquet(?)", [str(editions_path)])
    con.execute("""
        CREATE OR REPLACE TEMP TABLE _editions_norm AS
        SELECT work_id, regexp_replace(upper(coalesce(isbn, '')), '[^0-9X]', '', 'g') AS isbn10_n,
               regexp_replace(coalesce(isbn13, ''), '[^0-9]', '', 'g') AS isbn13_n, kindle_asin
        FROM _editions
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE bridge AS
        SELECT parent_asin, max(work_id) AS work_id FROM (
            SELECT i.parent_asin,
                   CASE WHEN i.source = 'kindle' THEN ek.work_id
                        ELSE coalesce(e10.work_id, e13.work_id) END AS work_id
            FROM (SELECT DISTINCT parent_asin, source FROM _az_ratings) i
            LEFT JOIN (SELECT parent_asin,
                              regexp_replace(upper(coalesce(isbn10, '')), '[^0-9X]', '', 'g') AS isbn10_n,
                              regexp_replace(coalesce(isbn13, ''), '[^0-9]', '', 'g') AS isbn13_n
                       FROM _az_meta WHERE source = 'books') m ON m.parent_asin = i.parent_asin
            LEFT JOIN _editions_norm e10 ON e10.isbn10_n = m.isbn10_n AND m.isbn10_n != ''
            LEFT JOIN _editions_norm e13 ON e13.isbn13_n = m.isbn13_n AND m.isbn13_n != ''
            LEFT JOIN _editions_norm ek ON ek.kindle_asin = i.parent_asin AND i.source = 'kindle'
        )
        GROUP BY parent_asin
    """)
```

- [ ] **Step 4: Убедиться, что тесты проходят**

Run: `uv run pytest tests/test_amazon.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/booksengine/data/amazon.py tests/test_amazon.py
git commit -m "amazon.py: мост на Goodreads по ISBN (Books) и kindle_asin (Kindle)"
```

---

## Task 5: `amazon.py` — `apply_kcore` и `export`

**Files:**
- Modify: `src/booksengine/data/amazon.py`
- Test: `tests/test_amazon.py`

**Interfaces:**
- Consumes: `_az_ratings` (Task 2), `_az_meta` (Task 3), `bridge` (Task 4) — все temp-таблицы в `con`.
- Produces: `apply_kcore(con, min_user=MIN_USER, min_work=MIN_WORK) -> dict` (один элемент из
  `clean.CleaningLog.records()`, фильтрует `_az_ratings` на месте). `export(con, out_dir: Path) -> dict` —
  пишет `ratings.parquet`, `items.parquet`, `bridge.parquet` в `out_dir`; возвращает `{"ratings": int,
  "items": int, "bridged": int, "new": int}`. Используются Task 8 (`prepare`).

- [ ] **Step 1: Написать падающие тесты**

```python
def test_apply_kcore_filters_low_volume_users_and_items(con):
    rows = ", ".join(f"('U{u}', 'B{w}', 5.0, 1000, 'books')" for u in range(5) for w in range(5))
    con.execute(f"CREATE OR REPLACE TEMP TABLE _az_ratings AS SELECT * FROM (VALUES {rows}, "
                f"('U9', 'B9', 5.0, 1000, 'books')) "   # 1 оценка у пользователя и книги — не пройдёт k-core
                f"t(user_id, parent_asin, rating, timestamp, source)")

    step = amazon.apply_kcore(con, min_user=5, min_work=5)

    remaining = con.execute("SELECT count(*) FROM _az_ratings").fetchone()[0]
    assert remaining == 25   # плотная матрица 5x5 проходит k-core целиком
    assert step["rule"] == "kcore"
    assert con.execute("SELECT count(*) FROM _az_ratings WHERE parent_asin = 'B9'").fetchone()[0] == 0


def test_export_writes_ratings_items_and_bridge_parquet(tmp_path, con):
    con.execute("CREATE OR REPLACE TEMP TABLE _az_ratings AS SELECT * FROM (VALUES "
                "('U1', 'B001', 5.0, 1000, 'books'), ('U2', 'B001', 4.0, 1001, 'books'), "
                "('U1', 'B002', 3.0, 1002, 'books')) "   # B002 оценена, но без записи в _az_meta
                "t(user_id, parent_asin, rating, timestamp, source)")
    con.execute("CREATE OR REPLACE TEMP TABLE _az_meta AS SELECT * FROM (VALUES "
                "('B001', 'Some Title', 'Some Author', '0701169850', '978-0701169855', "
                "'Publisher (2004)', 'English', 'Books|Fiction', 'books')) "
                "t(parent_asin, title, author, isbn10, isbn13, publisher_raw, language, categories, source)")
    con.execute("CREATE OR REPLACE TEMP TABLE bridge AS SELECT * FROM (VALUES "
                "('B001', 1), ('B002', NULL)) t(parent_asin, work_id)")

    counts = amazon.export(con, tmp_path)
    assert counts == {"ratings": 3, "items": 2, "bridged": 1, "new": 1}

    ratings = pd.read_parquet(tmp_path / "ratings.parquet")
    assert list(ratings.columns) == ["user_id", "parent_asin", "rating", "timestamp", "source"]
    assert len(ratings) == 3

    items = pd.read_parquet(tmp_path / "items.parquet").set_index("parent_asin")
    assert list(items.columns) == ["title", "author", "isbn10", "isbn13", "year", "categories",
                                   "n_ratings", "source"]
    assert items.loc["B001", "year"] == 2004
    assert items.loc["B001", "n_ratings"] == 2
    # B002 оценена, но нет meta-строки для неё — не должна пропасть из items, просто с пустыми полями
    assert items.loc["B002", "n_ratings"] == 1
    assert pd.isna(items.loc["B002", "title"])

    bridge = pd.read_parquet(tmp_path / "bridge.parquet")
    assert list(bridge.columns) == ["parent_asin", "work_id"]
    assert len(bridge) == 2
```

- [ ] **Step 2: Убедиться, что тесты падают**

Run: `uv run pytest tests/test_amazon.py -v -k "apply_kcore or export"`
Expected: FAIL — функций ещё нет.

- [ ] **Step 3: Реализовать**

```python
def apply_kcore(con: duckdb.DuckDBPyConnection, min_user: int = MIN_USER, min_work: int = MIN_WORK) -> dict:
    """Итеративный k-core на _az_ratings по (user_id, parent_asin). Переиспользует `clean.apply_kcore` —
    внутри него имя колонки произведения жёстко `work_id`, поэтому parent_asin временно переименовывается."""
    from booksengine.data import clean

    con.execute("CREATE OR REPLACE TEMP TABLE ratings_w AS SELECT user_id, parent_asin AS work_id, rating, "
                "timestamp, source FROM _az_ratings")
    log = clean.CleaningLog()
    clean.apply_kcore(con, log, min_user, min_work)
    con.execute("CREATE OR REPLACE TEMP TABLE _az_ratings AS SELECT user_id, work_id AS parent_asin, rating, "
                "timestamp, source FROM ratings_w")
    con.execute("DROP TABLE ratings_w")
    return log.records()[-1]


def export(con: duckdb.DuckDBPyConnection, out_dir: Path) -> dict:
    """ratings/items/bridge.parquet в out_dir. items — от _az_ratings (после k-core), не от _az_meta:
    оценённая книга без meta-строки (JSON не распарсился) не теряется, просто с пустыми полями."""
    out_dir.mkdir(parents=True, exist_ok=True)
    opts = "(FORMAT parquet, COMPRESSION zstd)"
    con.execute(f"COPY (SELECT user_id, parent_asin, rating, timestamp, source FROM _az_ratings "
                f"ORDER BY parent_asin, user_id) TO '{out_dir / 'ratings.parquet'}' {opts}")

    counts = con.execute("SELECT parent_asin, count(*) AS n_ratings FROM _az_ratings GROUP BY 1").df()
    meta = con.execute("SELECT parent_asin, title, author, isbn10, isbn13, publisher_raw, language, "
                       "categories, source FROM _az_meta").df()
    items = counts.merge(meta, on="parent_asin", how="left")
    items["year"] = items["publisher_raw"].map(extract_year)
    items = items[["parent_asin", "title", "author", "isbn10", "isbn13", "year", "categories",
                   "n_ratings", "source"]].set_index("parent_asin").reset_index()
    items.to_parquet(out_dir / "items.parquet", index=False)

    con.execute(f"COPY (SELECT b.parent_asin, b.work_id FROM bridge b "
                f"JOIN (SELECT DISTINCT parent_asin FROM _az_ratings) r USING (parent_asin) "
                f"ORDER BY b.parent_asin) TO '{out_dir / 'bridge.parquet'}' {opts}")

    bridged = con.execute(
        "SELECT count(*) FROM bridge b JOIN (SELECT DISTINCT parent_asin FROM _az_ratings) r "
        "USING (parent_asin) WHERE b.work_id IS NOT NULL").fetchone()[0]
    return {"ratings": int(con.execute("SELECT count(*) FROM _az_ratings").fetchone()[0]),
            "items": len(items), "bridged": bridged, "new": len(items) - bridged}
```

- [ ] **Step 4: Убедиться, что тесты проходят**

Run: `uv run pytest tests/test_amazon.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/booksengine/data/amazon.py tests/test_amazon.py
git commit -m "amazon.py: k-core (5/5) и экспорт ratings/items/bridge.parquet"
```

---

## Task 6: `wikidata.py` — `check_translations`

**Files:**
- Create: `src/booksengine/data/wikidata.py`
- Test: `tests/test_wikidata.py`

**Interfaces:**
- Produces: `check_translations(isbns: list[str], cache_path: Path, query=_query, batch_size: int = 200,
  rate_limit_s: float = 1.0) -> dict[str, bool]`. `query` — подменяется в тестах (как `fetch` в
  `tmdb.fetch_all`), сигнатура `query(batch: list[str]) -> dict[str, bool]`. Используется Task 7
  (`apply_translation_signal`).

- [ ] **Step 1: Написать падающие тесты**

```python
# tests/test_wikidata.py
import pandas as pd

from booksengine.data import wikidata


def test_check_translations_uses_cache_and_queries_only_missing_isbns(tmp_path):
    cache_path = tmp_path / "wikidata_cache.parquet"
    pd.DataFrame({"isbn13": ["9780000000001"], "has_ru": [True],
                 "checked_at": [pd.Timestamp.utcnow()]}).to_parquet(cache_path)

    calls = []

    def fake_query(batch):
        calls.append(list(batch))
        return {"9780000000002": True, "9780000000003": False}

    result = wikidata.check_translations(
        ["9780000000001", "9780000000002", "9780000000003"], cache_path, query=fake_query, rate_limit_s=0)

    assert calls == [["9780000000002", "9780000000003"]]
    assert result == {"9780000000001": True, "9780000000002": True, "9780000000003": False}


def test_check_translations_defaults_missing_from_response_to_false(tmp_path):
    cache_path = tmp_path / "wikidata_cache.parquet"

    def fake_query(batch):
        return {}   # ISBN не нашёлся в Wikidata вообще

    result = wikidata.check_translations(["9780000000004"], cache_path, query=fake_query, rate_limit_s=0)
    assert result == {"9780000000004": False}


def test_check_translations_batches_requests_by_batch_size(tmp_path):
    cache_path = tmp_path / "wikidata_cache.parquet"
    isbns = [f"978000000{i:04d}" for i in range(5)]
    calls = []

    def fake_query(batch):
        calls.append(len(batch))
        return {}

    wikidata.check_translations(isbns, cache_path, query=fake_query, batch_size=2, rate_limit_s=0)
    assert calls == [2, 2, 1]


def test_check_translations_persists_new_results_to_cache(tmp_path):
    cache_path = tmp_path / "wikidata_cache.parquet"

    def fake_query(batch):
        return {"9780000000005": True}

    wikidata.check_translations(["9780000000005"], cache_path, query=fake_query, rate_limit_s=0)

    cached = pd.read_parquet(cache_path)
    assert cached.set_index("isbn13").loc["9780000000005", "has_ru"] == True

    # повторный вызов с тем же ISBN не должен снова спрашивать query
    calls = []
    wikidata.check_translations(["9780000000005"], cache_path,
                                query=lambda b: calls.append(b) or {}, rate_limit_s=0)
    assert calls == []
```

- [ ] **Step 2: Убедиться, что тесты падают**

Run: `uv run pytest tests/test_wikidata.py -v`
Expected: FAIL — модуля `wikidata.py` ещё нет.

- [ ] **Step 3: Реализовать**

```python
# src/booksengine/data/wikidata.py
"""Wikidata как сигнал «есть ли русский перевод» для книг без моста на Goodreads (см. amazon.py). По
ISBN-13 находим издание -> его произведение (P629) -> смотрим, есть ли среди изданий этого произведения
русское (P407 = Q7737). Публичный SPARQL-эндпоинт не для тысяч одиночных запросов — пачками, с диск-кэшем
(повторный прогон не бьёт эндпоинт заново по уже известным ISBN).
"""
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ENDPOINT = "https://query.wikidata.org/sparql"
BATCH_SIZE = 200
RUSSIAN_LANG_QID = "Q7737"


def _query(isbns: list[str], timeout: float = 60.0) -> dict[str, bool]:
    """Один SPARQL-запрос на пачку ISBN-13 -> {isbn13: has_ru}. ISBN, не найденный в ответе, — до вызывающей
    стороны: он просто отсутствует в возвращённом словаре (check_translations трактует это как False)."""
    values = " ".join(f'"{i}"' for i in isbns)
    query = f"""
        SELECT ?isbn13 ?hasRu WHERE {{
          VALUES ?isbn13 {{ {values} }}
          ?edition wdt:P212 ?isbn13; wdt:P629 ?work .
          OPTIONAL {{ ?ruEdition wdt:P629 ?work; wdt:P407 wd:{RUSSIAN_LANG_QID} . }}
          BIND(BOUND(?ruEdition) AS ?hasRu)
        }}
    """
    url = ENDPOINT + "?" + urllib.parse.urlencode({"query": query, "format": "json"})
    req = urllib.request.Request(url, headers={"User-Agent": "BooksEngine/0.1 (personal, non-commercial)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    return {b["isbn13"]["value"]: b["hasRu"]["value"] == "true" for b in data["results"]["bindings"]}


def check_translations(isbns: list[str], cache_path: Path, query=_query, batch_size: int = BATCH_SIZE,
                       rate_limit_s: float = 1.0) -> dict[str, bool]:
    """isbn13 -> есть ли русский перевод. Кэш в cache_path (Parquet: isbn13, has_ru, checked_at) — уже
    проверенные ISBN не запрашиваются заново. `query` подменяется в тестах (без похода в сеть)."""
    isbns = sorted({i for i in isbns if i})
    cache = (pd.read_parquet(cache_path) if cache_path.exists()
             else pd.DataFrame(columns=["isbn13", "has_ru", "checked_at"]))
    known = dict(zip(cache.isbn13, cache.has_ru))
    todo = [i for i in isbns if i not in known]

    new_rows = []
    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        found = query(batch)
        now = pd.Timestamp.utcnow()
        for isbn in batch:
            has_ru = bool(found.get(isbn, False))
            known[isbn] = has_ru
            new_rows.append({"isbn13": isbn, "has_ru": has_ru, "checked_at": now})
        if i + batch_size < len(todo):
            time.sleep(rate_limit_s)

    if new_rows:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True).to_parquet(cache_path, index=False)

    return {i: known[i] for i in isbns}
```

- [ ] **Step 4: Убедиться, что тесты проходят**

Run: `uv run pytest tests/test_wikidata.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/booksengine/data/wikidata.py tests/test_wikidata.py
git commit -m "wikidata.py: пакетная проверка русского перевода по ISBN с диск-кэшем"
```

---

## Task 7: `amazon.py` — `apply_translation_signal`

**Files:**
- Modify: `src/booksengine/data/amazon.py`
- Test: `tests/test_amazon.py`

**Interfaces:**
- Consumes: `wikidata.check_translations` (Task 6), `items.parquet`/`bridge.parquet` на диске (Task 5).
- Produces: `apply_translation_signal(clean_dir: Path, cache_path: Path, query=None) -> dict` — дописывает
  колонку `ru_translation_known: bool` в `items.parquet`, возвращает `{"candidates": int,
  "with_translation": int}`. Используется Task 8 (`prepare`).

- [ ] **Step 1: Написать падающий тест**

```python
def test_apply_translation_signal_flags_only_unbridged_books_with_isbn_and_ru_edition(tmp_path):
    pd.DataFrame([
        {"parent_asin": "A1", "isbn13": "9780000000001", "n_ratings": 100},   # без моста, есть перевод
        {"parent_asin": "A2", "isbn13": "9780000000002", "n_ratings": 100},   # без моста, без перевода
        {"parent_asin": "A3", "isbn13": None, "n_ratings": 100},              # без моста, без ISBN
        {"parent_asin": "A4", "isbn13": "9780000000001", "n_ratings": 100},   # с мостом — не проверяем
    ]).to_parquet(tmp_path / "items.parquet")
    pd.DataFrame([
        {"parent_asin": "A1", "work_id": None}, {"parent_asin": "A2", "work_id": None},
        {"parent_asin": "A3", "work_id": None}, {"parent_asin": "A4", "work_id": 1},
    ]).to_parquet(tmp_path / "bridge.parquet")

    def fake_query(batch):
        return {"9780000000001": True, "9780000000002": False}

    stats = amazon.apply_translation_signal(tmp_path, tmp_path / "cache.parquet", query=fake_query)
    assert stats == {"candidates": 2, "with_translation": 1}

    items = pd.read_parquet(tmp_path / "items.parquet").set_index("parent_asin")
    assert items.loc["A1", "ru_translation_known"] == True
    assert items.loc["A2", "ru_translation_known"] == False
    assert items.loc["A3", "ru_translation_known"] == False
    assert items.loc["A4", "ru_translation_known"] == False


def test_apply_translation_signal_is_noop_without_candidates(tmp_path):
    pd.DataFrame([{"parent_asin": "A1", "isbn13": None, "n_ratings": 100}]).to_parquet(
        tmp_path / "items.parquet")
    pd.DataFrame([{"parent_asin": "A1", "work_id": None}]).to_parquet(tmp_path / "bridge.parquet")

    calls = []
    stats = amazon.apply_translation_signal(tmp_path, tmp_path / "cache.parquet",
                                             query=lambda b: calls.append(b) or {})
    assert stats == {"candidates": 0, "with_translation": 0}
    assert calls == []
```

- [ ] **Step 2: Убедиться, что тест падает**

Run: `uv run pytest tests/test_amazon.py -v -k translation_signal`
Expected: FAIL — функции ещё нет.

- [ ] **Step 3: Реализовать**

```python
def apply_translation_signal(clean_dir: Path, cache_path: Path, query=None) -> dict:
    """items.parquet получает колонку ru_translation_known: True только для книг без моста на Goodreads, у
    которых нашёлся русский перевод в Wikidata по ISBN-13. С мостом или без ISBN-13 — False (нет сигнала —
    не советуем, решение принято на этапе дизайна)."""
    from booksengine.data import wikidata

    items = pd.read_parquet(clean_dir / "items.parquet")
    bridge = pd.read_parquet(clean_dir / "bridge.parquet")
    unbridged = set(bridge.loc[bridge.work_id.isna(), "parent_asin"])

    items["ru_translation_known"] = False
    candidates = items[items.parent_asin.isin(unbridged) & items.isbn13.notna()]
    if len(candidates):
        result = wikidata.check_translations(candidates.isbn13.tolist(), cache_path,
                                             query=query or wikidata._query)
        has_ru = candidates.isbn13.map(result).fillna(False)
        items.loc[candidates.index, "ru_translation_known"] = has_ru.values

    items.to_parquet(clean_dir / "items.parquet", index=False)
    return {"candidates": len(candidates), "with_translation": int(items.ru_translation_known.sum())}
```

- [ ] **Step 4: Убедиться, что тесты проходят**

Run: `uv run pytest tests/test_amazon.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/booksengine/data/amazon.py tests/test_amazon.py
git commit -m "amazon.py: сигнал перевода из Wikidata в items.parquet"
```

---

## Task 8: `amazon.py` — `prepare()` (оркестрация, end-to-end)

**Files:**
- Modify: `src/booksengine/data/amazon.py`
- Test: `tests/test_amazon.py`

**Interfaces:**
- Consumes: все функции Task 2–7.
- Produces: `prepare(raw_dir: Path, out_dir: Path, editions_path: Path, cache_path: Path,
  min_user: int = MIN_USER, min_work: int = MIN_WORK, translation_query=None) -> dict` — читает
  `raw_dir/amazon_reviews_2023/{Books,Kindle_Store}.csv.gz` и `meta_*.jsonl.gz`, пишет `out_dir/{ratings,
  items,bridge}.parquet`. Возвращает `{"kcore": dict, "export": dict, "translation": dict}`. Используется
  Task 9 (CLI).

- [ ] **Step 1: Написать падающий end-to-end тест**

```python
import gzip


def _write_gz(path, text: str) -> None:
    with gzip.open(path, "wt") as f:
        f.write(text)


def test_prepare_end_to_end_with_synthetic_raw_files(tmp_path):
    raw = tmp_path / "raw"
    base = raw / "amazon_reviews_2023"
    base.mkdir(parents=True)

    ratings_rows = "user_id,parent_asin,rating,timestamp\n" + "".join(
        f"U{u},B1,5.0,{1000 + u}\n" for u in range(5))
    _write_gz(base / "Books.csv.gz", ratings_rows)
    _write_gz(base / "Kindle_Store.csv.gz", "user_id,parent_asin,rating,timestamp\n")

    meta_row = json.dumps({
        "parent_asin": "B1", "title": "Test Book", "author": {"name": "A. Uthor"},
        "details": {"ISBN 10": "1111111111", "ISBN 13": "9781111111111", "Publisher": "Pub (2019)",
                    "Language": "English"},
        "categories": ["Books", "Fiction"],
    })
    _write_gz(base / "meta_Books.jsonl.gz", meta_row + "\n")
    _write_gz(base / "meta_Kindle_Store.jsonl.gz", "")

    # пустой editions — мост не найдётся, B1 останется книгой без пары в Goodreads
    editions_path = tmp_path / "editions.parquet"
    pd.DataFrame(columns=["work_id", "isbn", "isbn13", "kindle_asin"]).astype(
        {"work_id": "int64"}).to_parquet(editions_path)

    out_dir = tmp_path / "clean"
    cache_path = tmp_path / "wikidata_cache.parquet"

    manifest = amazon.prepare(raw, out_dir, editions_path, cache_path, min_user=1, min_work=1,
                              translation_query=lambda batch: {"9781111111111": True})

    assert manifest["kcore"]["rule"] == "kcore"
    assert manifest["export"] == {"ratings": 5, "items": 1, "bridged": 0, "new": 1}
    assert manifest["translation"] == {"candidates": 1, "with_translation": 1}

    items = pd.read_parquet(out_dir / "items.parquet")
    assert items.iloc[0]["ru_translation_known"] == True
    assert items.iloc[0]["year"] == 2019
```

- [ ] **Step 2: Убедиться, что тест падает**

Run: `uv run pytest tests/test_amazon.py -v -k prepare`
Expected: FAIL — `prepare` не определена.

- [ ] **Step 3: Реализовать**

```python
def prepare(raw_dir: Path, out_dir: Path, editions_path: Path, cache_path: Path, min_user: int = MIN_USER,
           min_work: int = MIN_WORK, translation_query=None) -> dict:
    """Полный прогон: raw_dir/amazon_reviews_2023/{Books,Kindle_Store}.csv.gz + meta_*.jsonl.gz -> staging ->
    k-core -> мост на Goodreads (editions_path) -> экспорт -> сигнал перевода (Wikidata, кэш в cache_path) ->
    out_dir. `translation_query` — подменяется в тестах, без него — настоящий Wikidata."""
    base = raw_dir / "amazon_reviews_2023"
    con = duckdb.connect()
    stage_ratings(con, base / "Books.csv.gz", base / "Kindle_Store.csv.gz")
    stage_meta(con, base / "meta_Books.jsonl.gz", base / "meta_Kindle_Store.jsonl.gz")
    kcore_step = apply_kcore(con, min_user, min_work)
    build_bridge(con, editions_path)
    counts = export(con, out_dir)
    con.close()
    translation = apply_translation_signal(out_dir, cache_path, query=translation_query)
    return {"kcore": kcore_step, "export": counts, "translation": translation}
```

- [ ] **Step 4: Убедиться, что тесты проходят**

Run: `uv run pytest tests/test_amazon.py tests/test_wikidata.py -v`
Expected: PASS — все тесты модуля.

- [ ] **Step 5: Commit**

```bash
git add src/booksengine/data/amazon.py tests/test_amazon.py
git commit -m "amazon.py: prepare() — полный прогон staging -> мост -> сигнал перевода"
```

---

## Task 9: CLI-команда `amazon-bridge`, отчёт, README

**Files:**
- Modify: `src/booksengine/paths.py`
- Modify: `src/booksengine/cli.py`
- Modify: `src/booksengine/data/amazon.py`
- Modify: `README.md`
- Test: `tests/test_amazon.py`

**Interfaces:**
- Consumes: `amazon.prepare` (Task 8).
- Produces: `paths.AMAZON_DIR`, `paths.AMAZON_CLEAN_DIR`, `paths.AMAZON_CACHE_PATH`; `amazon.report(manifest:
  dict, clean_dir: Path) -> str` (путь к записанному отчёту); команда `uv run booksengine amazon-bridge`.

- [ ] **Step 1: Добавить пути в `paths.py`**

Amazon — не домен (`--domain books|movies`), а второй источник только для книг: пути фиксированы, не через
`DATA_DIR`/`DOMAIN`.

```python
# добавить в src/booksengine/paths.py, после REPORTS_DIR
AMAZON_DIR = PROJECT_ROOT / "data" / "amazon"
AMAZON_CLEAN_DIR = AMAZON_DIR / "clean"
AMAZON_CACHE_PATH = AMAZON_DIR / "wikidata_cache.parquet"
```

- [ ] **Step 2: Написать падающий тест на отчёт**

```python
def test_report_writes_markdown_with_bridge_and_translation_numbers(tmp_path, monkeypatch):
    from booksengine import paths
    monkeypatch.setattr(paths, "REPORTS_DIR", tmp_path)

    pd.DataFrame([
        {"user_id": "U1", "parent_asin": "B1", "rating": 5.0, "timestamp": 1, "source": "books"},
        {"user_id": "U1", "parent_asin": "K1", "rating": 4.0, "timestamp": 2, "source": "kindle"},
    ]).to_parquet(tmp_path / "ratings.parquet")
    pd.DataFrame([
        {"parent_asin": "B1", "title": "T1", "author": "A1", "isbn10": None, "isbn13": "9781111111111",
         "year": 2020, "categories": "Books", "n_ratings": 1, "source": "books",
         "ru_translation_known": True},
        {"parent_asin": "K1", "title": "T2", "author": "A2", "isbn10": None, "isbn13": None, "year": 2021,
         "categories": "Kindle Store", "n_ratings": 1, "source": "kindle",
         "ru_translation_known": False},
    ]).to_parquet(tmp_path / "items.parquet")
    pd.DataFrame([{"parent_asin": "B1", "work_id": None}, {"parent_asin": "K1", "work_id": None}]
                ).to_parquet(tmp_path / "bridge.parquet")

    manifest = {"kcore": {}, "export": {"ratings": 2, "items": 2, "bridged": 0, "new": 2},
                "translation": {"candidates": 1, "with_translation": 1}}

    path = amazon.report(manifest, tmp_path)
    text = Path(path).read_text()
    assert "books" in text and "kindle" in text
    assert "Мост на Goodreads" in text
    assert "1" in text   # с известным переводом
```

- [ ] **Step 3: Убедиться, что тест падает**

Run: `uv run pytest tests/test_amazon.py -v -k report`
Expected: FAIL — `report` не определена.

- [ ] **Step 4: Реализовать `report()` в `amazon.py`**

```python
def report(manifest: dict, clean_dir: Path) -> str:
    """reports/amazon_bridge.md — люди/книги/оценки по источникам, сила моста, книги после 2017 без моста
    с известным переводом (Wikidata) — главное число для решения о следующем шаге."""
    from booksengine.paths import REPORTS_DIR

    items = pd.read_parquet(clean_dir / "items.parquet")
    bridge = pd.read_parquet(clean_dir / "bridge.parquet")
    ratings = pd.read_parquet(clean_dir / "ratings.parquet")

    lines = ["# Amazon Reviews'23: мост на Goodreads и сигнал перевода", ""]
    kcore = manifest.get("kcore") or {}
    if kcore:
        lines.append(f"- **k-core** ({MIN_USER}/{MIN_WORK}): {kcore.get('rows_before', 0):,} → "
                     f"{kcore.get('rows_after', 0):,} оценок ({kcore.get('users_after', 0):,} человек, "
                     f"{kcore.get('works_after', 0):,} книг)")
    for source in ("books", "kindle"):
        r = ratings[ratings.source == source]
        lines.append(f"- **{source}**: {r.user_id.nunique():,} человек, {r.parent_asin.nunique():,} книг, "
                     f"{len(r):,} оценок (после k-core {MIN_USER}/{MIN_WORK})")

    bridged = int(bridge.work_id.notna().sum())
    lines.append(f"- **Мост на Goodreads**: {bridged:,} из {len(bridge):,} книг "
                 f"({bridged / len(bridge) * 100:.1f}%)" if len(bridge) else "- **Мост на Goodreads**: нет данных")

    unbridged_asins = set(bridge.loc[bridge.work_id.isna(), "parent_asin"])
    post2017 = items[(items.year > 2017) & items.parent_asin.isin(unbridged_asins)]
    with_ru = int(post2017.get("ru_translation_known", pd.Series(dtype=bool)).sum())
    lines.append(f"- **Книг после 2017 без моста**: {len(post2017):,}, из них с известным переводом "
                 f"(Wikidata): {with_ru:,}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "amazon_bridge.md"
    out.write_text("\n".join(lines) + "\n")
    return str(out)
```

- [ ] **Step 5: Убедиться, что тест проходит**

Run: `uv run pytest tests/test_amazon.py -v`
Expected: PASS

- [ ] **Step 6: Добавить CLI-команду**

```python
# добавить в src/booksengine/cli.py, после команды report()
@app.command("amazon-bridge")
def amazon_bridge() -> None:
    """Amazon Reviews'23: приём, мост на Goodreads по ISBN/ASIN, сигнал перевода (Wikidata) ->
    reports/amazon_bridge.md. Сырьё — RAW_DIR/amazon_reviews_2023/ (scripts/fetch_amazon_raw.sh)."""
    from booksengine.data import amazon
    from booksengine.paths import AMAZON_CACHE_PATH, AMAZON_CLEAN_DIR, CLEAN_DIR, RAW_DIR
    manifest = amazon.prepare(RAW_DIR, AMAZON_CLEAN_DIR, CLEAN_DIR / "editions.parquet", AMAZON_CACHE_PATH)
    path = amazon.report(manifest, AMAZON_CLEAN_DIR)
    print(f"Отчёт: {path}")
```

- [ ] **Step 7: Проверить, что CLI регистрирует команду**

Run: `uv run booksengine --help`
Expected: в списке команд присутствует `amazon-bridge`.

- [ ] **Step 8: Обновить README**

В README.md добавить после раздела «## Датасет» (перед «## Запуск») короткую врезку:

```markdown
### Второй источник: Amazon Reviews'23 (опционально)

Для книг после 2017 года (датасет Goodreads ими не пополняется) — `amazon-bridge` строит мост на Goodreads
по ISBN/ASIN и проверяет сигнал перевода (Wikidata). Сырьё — 4 файла в одну папку `RAW_DIR/
amazon_reviews_2023/` со страницы [amazon-reviews-2023.github.io](https://amazon-reviews-2023.github.io/)
(0-Core → rating_only: `Books.csv.gz`, `Kindle_Store.csv.gz`; raw/meta_categories: `meta_Books.jsonl.gz`,
`meta_Kindle_Store.jsonl.gz`) — сервер источника медленный и рвёт соединение на больших файлах, докачивать
через `scripts/fetch_amazon_raw.sh` (устойчив к обрывам, можно прерывать и перезапускать).

```bash
RAW_DIR=~/Downloads scripts/fetch_amazon_raw.sh   # разово, часы из-за скорости источника
uv run booksengine amazon-bridge                  # -> data/amazon/clean/, reports/amazon_bridge.md
```
```

И добавить строку в таблицу «## Команды» (раздел «данные»):

```markdown
| `amazon-bridge` | Amazon Reviews'23: мост на Goodreads по ISBN/ASIN, сигнал перевода (Wikidata) → `data/amazon/clean/`, `reports/amazon_bridge.md` |
```

- [ ] **Step 9: Прогнать полный набор тестов модуля**

Run: `uv run pytest tests/test_amazon.py tests/test_wikidata.py -v`
Expected: PASS — все тесты обеих задач.

- [ ] **Step 10: Commit**

```bash
git add src/booksengine/paths.py src/booksengine/cli.py src/booksengine/data/amazon.py \
        tests/test_amazon.py README.md
git commit -m "amazon-bridge: CLI-команда, отчёт reports/amazon_bridge.md, README"
```

---

## После выполнения плана

`uv run pytest` — полный прогон юнит-тестов (без реальных данных). Ручной прогон на боевых данных —
отдельно, пользователем или агентом на Mac с доступом к `data/`:
`RAW_DIR=~/Downloads scripts/fetch_amazon_raw.sh && uv run booksengine amazon-bridge` — займёт часы
(скорость источника) + время на Wikidata (пачками, с паузами). Результат — `reports/amazon_bridge.md` с
главным числом: сколько книг после 2017 без моста реально имеют русский перевод. Это число определяет
объём следующего цикла (новые произведения в каталоге, модель со вторым источником, `source_id`,
отложенные Amazon-люди для замера) — задача записана в `TODO.md`.
