# Фильмы, шаг 1: проверка толпы и вкуса на MovieLens — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Собрать `data/movies/clean/*.parquet` из MovieLens 32M в схеме книг, прогнать через них
`split`/старую смесь-опору/`taste`/`ease-like-tune`/`layers val|test` без изменений кода модели и получить
отчёты, по которым видно, даёт ли модель вкуса выигрыш на фильмах.

**Architecture:** Новый модуль `src/booksengine/data/movielens.py` — DuckDB-загрузчик MovieLens (по образцу
`data/clean.py`), переиспользующий `clean.filter_users`/`clean.apply_kcore` без изменений; экспортирует
`data/movies/clean/{ratings,users,works,work_authors,authors}.parquet` в той же схеме, что читает модель книг.
Разовый скрипт `scripts/movies_check.py` связывает пути `data/movies/…`, `models/movies/…` и вызывает готовые
функции `split.build`, `evaluate.tune` (старая смесь-опора), `taste.tune`, `layers.tune_like`, `layers.run` —
без CLI-обвязки и без домена в `paths.py` (это шаг 2 плана «Фильмы», не этого шага).

**Tech Stack:** Python, DuckDB, pandas, pytest, существующие `booksengine.model.*` и `booksengine.data.clean`.

**Spec:** `docs/superpowers/specs/2026-09-27-movies-crowd-taste-check-design.md`

## Global Constraints

- Округление оценок: `rating = ceil(raw_rating)` — полузвёзды 0.5–5.0 → целые 1–5 вверх (без сдвига целых).
- k-core фиксирован решением из TODO, не перебирается: `min_user_ratings = 20`, `min_work_ratings = 100`.
- Схема экспорта — та же, что читает модель книг: `ratings.parquet(user_id, work_id, rating)`,
  `users.parquet(user_id, external_id)`, `works.parquet(work_id, title, original_title, best_edition_title,
  is_collection)`, `work_authors.parquet(work_id, author_id, role, position)` и `authors.parquet(author_id, name)`
  — оба пустые (0 строк), но с этими колонками и типами.
- Код `model/*.py` (`ease.py`, `taste.py`, `layers.py`, `split.py`, `evaluate.py`) не меняется — только пути и
  явные параметры (`grid`, `clean_dir`, `models_dir` и т.д.), которые эти функции и так принимают.
- Тесты — только в `tmp_path` на синтетических данных; `RAW_DIR`, `CONFIG_PATH` и другие пути по умолчанию из
  `booksengine.paths` в тестах не читать.
- `scripts/movies_check.py` — разовый скрипт, не часть `booksengine` CLI и не покрывается pytest как команда
  (реальный прогон — вручную, в фоне, лог в `reports/`).

## Review Focus

- Оценка ровно на границе звезды (3.0, 5.0) — `ceil` не должен её сдвигать вверх (округляется только дробная
  часть).
- Повторная строка `(userId, movieId)` в `ratings.csv` — без дедупа `scipy.sparse` тихо суммирует дубли при
  построении матрицы (`matrix.to_csr`), незаметно испортив оценку.
- Фильм есть в `ratings.csv`, но отсутствует в `movies.csv` — без явной проверки он молча пропадёт из
  `works.parquet`, а через несколько часов прогона упадёт `layers.run` (`work_info` требует названия для всех
  книг ядра).
- Пустые `work_authors.parquet`/`authors.parquet` — важна не только пустота, но и типы колонок: `filters.py`
  джойнит их с `work_id`/`author_id` типа BIGINT, несовпадение типов — ошибка DuckDB при первом реальном вызове
  `layers.run`, не раньше.
- `evaluate.tune("mix", …)` без явного `grid` тихо подхватывает книжные ALS/EASE через модульный `MODELS_DIR`
  (`evaluate.MODELS["mix"]` — уточнение, найденное при планировании, не в исходной спеке) — риск получить
  осмысленно выглядящую, но чужую опору сравнения.

---

## Файлы

- Modify: `config/cleaning.yaml` — новый раздел `movies:`.
- Create: `src/booksengine/data/movielens.py` — загрузчик MovieLens.
- Create: `tests/test_movielens.py`.
- Create: `scripts/movies_check.py` — разовый скрипт прогона.

### Task 1: Загрузчик MovieLens — механика (`load`, `to_ratings_w`, `export`)

**Files:**
- Create: `src/booksengine/data/movielens.py`
- Create: `tests/test_movielens.py`

**Interfaces:**
- Produces: `load(con: duckdb.DuckDBPyConnection, ratings_csv: Path, movies_csv: Path) -> None` — временные
  таблицы `_ml_ratings` (userId, movieId, rating, timestamp), `_ml_movies` (movieId, title, genres).
- Produces: `to_ratings_w(con: duckdb.DuckDBPyConnection) -> None` — таблица `ratings_w(user_id, work_id,
  rating)`, ожидается `clean.filter_users`/`clean.apply_kcore` (Task 2) как есть.
- Produces: `export(con: duckdb.DuckDBPyConnection, out_dir: Path) -> dict` — пишет 5 parquet-файлов в
  `out_dir`, возвращает `{"ratings": int, "users": int, "works": int}`.

- [ ] **Step 1: Написать падающий тест на округление и дедуп**

```python
# tests/test_movielens.py
import duckdb
import pandas as pd
import pytest

from booksengine.data import movielens


@pytest.fixture
def con():
    return duckdb.connect()


def test_to_ratings_w_rounds_half_stars_up_and_averages_duplicates(con):
    con.execute("""
        CREATE TEMP TABLE _ml_ratings AS SELECT * FROM (VALUES
            (1, 10, 3.0, 100), (2, 10, 0.5, 101), (3, 10, 5.0, 102),
            (4, 10, 3.0, 103), (4, 10, 4.0, 104))
        t(userId, movieId, rating, timestamp)
    """)
    movielens.to_ratings_w(con)
    assert con.execute("SELECT count(*) FROM ratings_w").fetchone()[0] == 4
    rows = dict(((u, w), r) for u, w, r in
                con.execute("SELECT user_id, work_id, rating FROM ratings_w").fetchall())
    assert rows[(1, 10)] == 3   # целая звезда не сдвигается
    assert rows[(2, 10)] == 1   # 0.5 -> 1
    assert rows[(3, 10)] == 5   # 5.0 -> 5
    assert rows[(4, 10)] == 4   # дубль (user, movie): среднее 3.5 -> ceil 4
```

- [ ] **Step 2: Убедиться, что тест падает**

Run: `uv run pytest tests/test_movielens.py -v`
Expected: FAIL — `AttributeError: module 'booksengine.data.movielens' has no attribute 'to_ratings_w'` (модуля
ещё нет).

- [ ] **Step 3: Написать `load` и `to_ratings_w`**

```python
# src/booksengine/data/movielens.py
"""MovieLens 32M → matrices в той же схеме, что у книг (проверка толпы и вкуса на фильмах, шаг 1 плана
«Фильмы» — TODO.md, docs/superpowers/specs/2026-09-27-movies-crowd-taste-check-design.md). Переиспользует
очистку пользователей и k-core из `booksengine.data.clean` без изменений — книжные правила (сборники, дубли
произведений, не-книги) фильмам не нужны.
"""
from pathlib import Path

import duckdb


def load(con: duckdb.DuckDBPyConnection, ratings_csv: Path, movies_csv: Path) -> None:
    """_ml_ratings (userId, movieId, rating, timestamp), _ml_movies (movieId, title, genres) — сырые CSV как есть."""
    con.execute("CREATE OR REPLACE TEMP TABLE _ml_ratings AS SELECT * FROM read_csv_auto(?)", [str(ratings_csv)])
    con.execute("CREATE OR REPLACE TEMP TABLE _ml_movies AS SELECT * FROM read_csv_auto(?)", [str(movies_csv)])


def to_ratings_w(con: duckdb.DuckDBPyConnection) -> None:
    """ratings_w: user_id, work_id, rating — полузвёзды 0.5–5.0 округлены вверх до целых 1–5; повтор пары
    (userId, movieId) — среднее до округления, как повтор издания у книг (`clean.to_work_level`)."""
    con.execute("""
        CREATE OR REPLACE TEMP TABLE ratings_w AS
        SELECT userId AS user_id, movieId AS work_id, ceil(avg(rating))::TINYINT AS rating
        FROM _ml_ratings GROUP BY userId, movieId
    """)
```

- [ ] **Step 4: Убедиться, что тест проходит**

Run: `uv run pytest tests/test_movielens.py -v`
Expected: PASS

- [ ] **Step 5: Написать падающий тест на проверку ссылочной целостности при экспорте**

```python
def test_export_raises_when_ratings_reference_missing_movie(con, tmp_path):
    con.execute("CREATE TEMP TABLE ratings_w AS SELECT * FROM (VALUES (1, 10, 3)) t(user_id, work_id, rating)")
    con.execute("CREATE TEMP TABLE _ml_movies AS SELECT * FROM (VALUES (99, 'Other', 'Drama')) "
                "t(movieId, title, genres)")
    with pytest.raises(ValueError, match="не хватает"):
        movielens.export(con, tmp_path)
```

- [ ] **Step 6: Убедиться, что тест падает**

Run: `uv run pytest tests/test_movielens.py -v`
Expected: FAIL — `AttributeError: module 'booksengine.data.movielens' has no attribute 'export'`.

- [ ] **Step 7: Написать падающий тест на схему и содержимое экспорта**

```python
def test_export_writes_ratings_users_works_and_trivial_author_tables(con, tmp_path):
    con.execute("CREATE TEMP TABLE ratings_w AS SELECT * FROM (VALUES "
                "(1, 10, 5), (1, 20, 3), (2, 10, 4)) t(user_id, work_id, rating)")
    con.execute("CREATE TEMP TABLE _ml_movies AS SELECT * FROM (VALUES "
                "(10, 'Toy Story (1995)', 'Adventure'), (20, 'Heat (1995)', 'Action')) "
                "t(movieId, title, genres)")
    counts = movielens.export(con, tmp_path)
    assert counts == {"ratings": 3, "users": 2, "works": 2}

    ratings = pd.read_parquet(tmp_path / "ratings.parquet")
    assert list(ratings.columns) == ["user_id", "work_id", "rating"]
    assert len(ratings) == 3

    users = pd.read_parquet(tmp_path / "users.parquet")
    assert sorted(users.user_id) == [1, 2]
    assert (users.user_id == users.external_id).all()

    works = pd.read_parquet(tmp_path / "works.parquet")
    assert sorted(works.title) == ["Heat (1995)", "Toy Story (1995)"]
    assert not works.is_collection.any()
    assert works.original_title.isna().all() and works.best_edition_title.isna().all()

    work_authors = pd.read_parquet(tmp_path / "work_authors.parquet")
    assert list(work_authors.columns) == ["work_id", "author_id", "role", "position"]
    assert len(work_authors) == 0

    authors = pd.read_parquet(tmp_path / "authors.parquet")
    assert list(authors.columns) == ["author_id", "name"]
    assert len(authors) == 0
```

- [ ] **Step 8: Написать `export`**

```python
def export(con: duckdb.DuckDBPyConnection, out_dir: Path) -> dict:
    """ratings/users/works/work_authors/authors.parquet — схема, которую читает модель книг
    (`filters.work_info`); авторы и коллекции — пустые/тривиальные, режиссёр появится только с TMDB."""
    out_dir.mkdir(parents=True, exist_ok=True)
    opts = "(FORMAT parquet, COMPRESSION zstd)"
    missing = con.execute("""
        SELECT count(*) FROM (SELECT DISTINCT work_id FROM ratings_w) r
        LEFT JOIN _ml_movies m ON m.movieId = r.work_id WHERE m.movieId IS NULL
    """).fetchone()[0]
    if missing:
        raise ValueError(f"movies.csv: не хватает {missing} строк для movieId из ratings.csv")
    con.execute(f"COPY (SELECT user_id, work_id, rating FROM ratings_w ORDER BY user_id, work_id) "
                f"TO '{out_dir / 'ratings.parquet'}' {opts}")
    con.execute(f"COPY (SELECT DISTINCT user_id, user_id AS external_id FROM ratings_w ORDER BY 1) "
                f"TO '{out_dir / 'users.parquet'}' {opts}")
    con.execute(f"""
        COPY (SELECT r.work_id, m.title, NULL::VARCHAR AS original_title, NULL::VARCHAR AS best_edition_title,
                     false AS is_collection
              FROM (SELECT DISTINCT work_id FROM ratings_w) r JOIN _ml_movies m ON m.movieId = r.work_id
              ORDER BY r.work_id)
        TO '{out_dir / 'works.parquet'}' {opts}
    """)
    con.execute(f"COPY (SELECT NULL::BIGINT AS work_id, NULL::BIGINT AS author_id, NULL::VARCHAR AS role, "
                f"NULL::SMALLINT AS position WHERE false) TO '{out_dir / 'work_authors.parquet'}' {opts}")
    con.execute(f"COPY (SELECT NULL::BIGINT AS author_id, NULL::VARCHAR AS name WHERE false) "
                f"TO '{out_dir / 'authors.parquet'}' {opts}")
    counts = con.execute(
        "SELECT count(*), count(DISTINCT user_id), count(DISTINCT work_id) FROM ratings_w").fetchone()
    return {"ratings": counts[0], "users": counts[1], "works": counts[2]}
```

- [ ] **Step 9: Убедиться, что все тесты проходят**

Run: `uv run pytest tests/test_movielens.py -v`
Expected: PASS (3 теста)

- [ ] **Step 10: Commit**

```bash
git add src/booksengine/data/movielens.py tests/test_movielens.py
git commit -m "Загрузчик MovieLens: округление, дедуп, экспорт в схему книг"
```

### Task 2: `prepare()` — очистка пользователей, k-core, конфиг

**Files:**
- Modify: `config/cleaning.yaml` — добавить раздел `movies:`
- Modify: `src/booksengine/data/movielens.py` — добавить `prepare`
- Modify: `tests/test_movielens.py`

**Interfaces:**
- Consumes: `load`, `to_ratings_w`, `export` из Task 1 (те же сигнатуры); `clean.CleaningLog`,
  `clean.filter_users(con, log, max_sd, min_ratings_for_sd, max_ratings, table="ratings_w",
  max_mode_share=None)`, `clean.apply_kcore(con, log, min_user, min_work, table="ratings_w")` из
  `booksengine.data.clean` (существующий код, без изменений).
- Produces: `prepare(raw_dir: Path, out_dir: Path, cfg: dict, min_user: int = 20, min_work: int = 100) -> dict`
  — `{"cleaning_log": list[dict], "outputs": {"ratings": int, "users": int, "works": int}}`. `cfg` — словарь с
  ключами `low_variance_max_sd`, `low_variance_min_ratings`, `monotone_max_mode_share`, `max_ratings` (раздел
  `movies:` из `config/cleaning.yaml`, читает вызывающий код — Task 3).

- [ ] **Step 1: Добавить раздел `movies:` в `config/cleaning.yaml`**

```yaml
movies:
  # MovieLens: пороги отдельные от книжных (users:), хоть стартовые значения и совпадают — у MovieLens
  # люди часто оценивают пачкой при регистрации (по памяти, не по ходу просмотра), и тюнинг одного домена
  # не должен задевать пороги другого. min_user/min_work (k-core) фиксированы TODO — 20/100, не в конфиге.
  low_variance_max_sd: 0.2
  low_variance_min_ratings: 10
  monotone_max_mode_share: 0.9
  max_ratings: 3000
```

Добавить в конец `config/cleaning.yaml`, после раздела `kcore:`.

- [ ] **Step 2: Написать падающий тест на `prepare` с синтетическим датасетом**

```python
def _synthetic_ml_csvs(tmp_path) -> Path:
    raw = tmp_path / "raw"
    (raw / "ml-32m").mkdir(parents=True)
    (raw / "ml-32m" / "ratings.csv").write_text(
        "userId,movieId,rating,timestamp\n"
        "1,10,5.0,100\n1,20,4.0,101\n1,30,3.0,102\n"
        "2,10,4.5,110\n2,20,3.5,111\n2,30,2.0,112\n"
        "3,10,5.0,120\n3,20,5.0,121\n3,30,5.0,122\n"
    )
    (raw / "ml-32m" / "movies.csv").write_text(
        "movieId,title,genres\n"
        "10,Toy Story (1995),Adventure\n20,Heat (1995),Action\n30,Se7en (1995),Thriller\n"
    )
    return raw


MOVIES_CFG = {"low_variance_max_sd": 0.2, "low_variance_min_ratings": 10, "monotone_max_mode_share": 0.9,
              "max_ratings": 3000}


def test_prepare_end_to_end_with_synthetic_dataset(tmp_path):
    raw = _synthetic_ml_csvs(tmp_path)
    manifest = movielens.prepare(raw, tmp_path / "clean", MOVIES_CFG, min_user=1, min_work=1)
    assert manifest["outputs"] == {"ratings": 9, "users": 3, "works": 3}
    ratings = pd.read_parquet(tmp_path / "clean" / "ratings.parquet")
    assert set(ratings.rating.unique()) <= {1, 2, 3, 4, 5}
    assert (tmp_path / "clean" / "works.parquet").exists()


def test_prepare_drops_hyperactive_users_per_movies_cfg(tmp_path):
    raw = _synthetic_ml_csvs(tmp_path)
    cfg = MOVIES_CFG | {"max_ratings": 2}   # у всех троих по 3 оценки — «гиперактивны» при пороге 2
    manifest = movielens.prepare(raw, tmp_path / "clean", cfg, min_user=1, min_work=1)
    assert manifest["outputs"]["users"] == 0
    dropped = next(s for s in manifest["cleaning_log"] if s["rule"] == "users_hyperactive")
    assert dropped["users_after"] == 0
```

- [ ] **Step 3: Убедиться, что тесты падают**

Run: `uv run pytest tests/test_movielens.py -v`
Expected: FAIL — `AttributeError: module 'booksengine.data.movielens' has no attribute 'prepare'`.

- [ ] **Step 4: Написать `prepare`**

```python
def prepare(raw_dir: Path, out_dir: Path, cfg: dict, min_user: int = 20, min_work: int = 100) -> dict:
    """Загрузка → округление и дедуп → очистка пользователей (`cfg` — раздел movies: cleaning.yaml) →
    k-core (min_user/min_work — фиксированные решением из TODO, не в конфиге) → экспорт."""
    from booksengine.data import clean

    con = duckdb.connect()
    load(con, raw_dir / "ml-32m" / "ratings.csv", raw_dir / "ml-32m" / "movies.csv")
    to_ratings_w(con)
    log = clean.CleaningLog()
    clean.filter_users(con, log, cfg["low_variance_max_sd"], cfg["low_variance_min_ratings"], cfg["max_ratings"],
                       max_mode_share=cfg["monotone_max_mode_share"])
    clean.apply_kcore(con, log, min_user, min_work)
    outputs = export(con, out_dir)
    con.close()
    return {"cleaning_log": log.records(), "outputs": outputs}
```

- [ ] **Step 5: Убедиться, что все тесты проходят**

Run: `uv run pytest tests/test_movielens.py -v`
Expected: PASS (5 тестов)

- [ ] **Step 6: Commit**

```bash
git add config/cleaning.yaml src/booksengine/data/movielens.py tests/test_movielens.py
git commit -m "movielens.prepare: очистка пользователей и k-core по конфигу movies:"
```

### Task 3: Скрипт прогона `scripts/movies_check.py`

**Files:**
- Create: `scripts/movies_check.py`

**Interfaces:**
- Consumes: `movielens.prepare(raw_dir, out_dir, cfg, min_user=20, min_work=100) -> dict` (Task 2);
  `split.build(ratings_path, users_path, out_dir, data_fp, force=False) -> dict`;
  `evaluate.tune(name, *, ratings_path, split_dir, eval_dir, models_dir, grid=None) -> list[dict]`;
  `taste.tune(*, ratings_path, split_dir, models_dir, eval_dir) -> dict`, `taste.report(dict) -> str`;
  `layers.tune_like(*, clean_dir, split_dir, models_dir, eval_dir) -> dict`, `layers.report_like(dict) -> str`;
  `layers.run(stage, *, clean_dir, split_dir, models_dir, eval_dir) -> dict`, `layers.report(dict) -> str`
  (все — существующий код `booksengine.model.*`, без изменений).
- Produces: файлы `data/movies/clean/*.parquet`, `data/movies/model/split/*`, `models/movies/**`,
  `reports/movies_{taste_val,ease_like_tune,layers_val,layers_test}.md`.

- [ ] **Step 1: Написать скрипт**

```python
#!/usr/bin/env python3
"""Фильмы, шаг 1 плана «Фильмы» (TODO.md): MovieLens 32M -> split -> опорная смесь (als_neg+ease+mix, код
evaluate.py) -> толпа «ценность» (ease-like-tune) -> вкус -> layers val|test. Код model/*.py не меняется —
только пути и явные grid-параметры. Разовый скрипт, не часть booksengine CLI: домен в CLI (paths.py,
data/<domain>/, models/<domain>/) — задача шага 2 плана «Фильмы», пока его нет.
Дизайн: docs/superpowers/specs/2026-09-27-movies-crowd-taste-check-design.md.

Запуск из корня проекта: uv run python scripts/movies_check.py
Долгий (часы на 32М строк) — в фоне:
  uv run python scripts/movies_check.py > reports/movies_check.log 2>&1 &
"""
import json
import time

import yaml

from booksengine.data import movielens
from booksengine.model import evaluate, layers, split, taste
from booksengine.paths import CONFIG_PATH, PROJECT_ROOT, RAW_DIR

DATA_DIR = PROJECT_ROOT / "data" / "movies"
CLEAN_DIR = DATA_DIR / "clean"
SPLIT_DIR = DATA_DIR / "model" / "split"
MODELS_DIR = PROJECT_ROOT / "models" / "movies"
EVAL_DIR = MODELS_DIR / "eval"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Копия сетки evaluate.MODELS["mix"] (evaluate.py) с movies-путями к als_neg/ease. У evaluate.py als_dir/ease_dir
# зашиты через модульный (книжный) MODELS_DIR на уровне импорта — своя копия обходит это без изменений
# evaluate.py. Если сетка als_weight в evaluate.py изменится, эту копию нужно поправить вручную.
MIX_GRID = [({"als_dir": str(MODELS_DIR / "als_neg"), "ease_dir": str(MODELS_DIR / "ease")},
             [{"als_weight": w} for w in (0.0, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 1.0)])]


def _write_report(name: str, text: str) -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / f"movies_{name}.md").write_text(text)
    print(text)


def main() -> None:
    t0 = time.time()
    cfg = yaml.safe_load(CONFIG_PATH.read_text())["movies"]

    print("[movielens] загрузка и очистка")
    manifest = movielens.prepare(RAW_DIR, CLEAN_DIR, cfg)
    print(json.dumps(manifest["outputs"], ensure_ascii=False, indent=1))

    print("[split]")
    meta = split.build(CLEAN_DIR / "ratings.parquet", CLEAN_DIR / "users.parquet", SPLIT_DIR,
                       data_fp=json.dumps(manifest["outputs"], sort_keys=True))
    print(json.dumps(meta["groups"], ensure_ascii=False, indent=1))

    print("[опорная смесь] als_neg")
    evaluate.tune("als_neg", ratings_path=CLEAN_DIR / "ratings.parquet", split_dir=SPLIT_DIR, eval_dir=EVAL_DIR,
                 models_dir=MODELS_DIR)
    print("[опорная смесь] ease")
    evaluate.tune("ease", ratings_path=CLEAN_DIR / "ratings.parquet", split_dir=SPLIT_DIR, eval_dir=EVAL_DIR,
                 models_dir=MODELS_DIR)
    print("[опорная смесь] mix — grid с movies-путями:", MIX_GRID)
    evaluate.tune("mix", ratings_path=CLEAN_DIR / "ratings.parquet", split_dir=SPLIT_DIR, eval_dir=EVAL_DIR,
                 models_dir=MODELS_DIR, grid=MIX_GRID)

    print("[taste]")
    _write_report("taste_val", taste.report(taste.tune(
        ratings_path=CLEAN_DIR / "ratings.parquet", split_dir=SPLIT_DIR, models_dir=MODELS_DIR, eval_dir=EVAL_DIR)))

    print("[ease-like-tune]")
    _write_report("ease_like_tune", layers.report_like(layers.tune_like(
        clean_dir=CLEAN_DIR, split_dir=SPLIT_DIR, models_dir=MODELS_DIR, eval_dir=EVAL_DIR)))

    for stage in ("val", "test"):
        print(f"[layers {stage}]")
        _write_report(f"layers_{stage}", layers.report(layers.run(
            stage, clean_dir=CLEAN_DIR, split_dir=SPLIT_DIR, models_dir=MODELS_DIR, eval_dir=EVAL_DIR)))

    print(f"--- готово за {(time.time() - t0) / 60:.0f} мин")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Проверить, что файл компилируется и импорты разрешаются**

Run: `uv run python -m py_compile scripts/movies_check.py && uv run python -c "import ast, pathlib; ast.parse(pathlib.Path('scripts/movies_check.py').read_text()); from booksengine.data import movielens; from booksengine.model import evaluate, layers, split, taste; print('ok')"`
Expected: `ok`, без ошибок импорта.

- [ ] **Step 3: Commit**

```bash
git add scripts/movies_check.py
git commit -m "Скрипт прогона фильмов: split → опорная смесь → taste → ease-like-tune → layers val|test"
```

- [ ] **Step 4: Запустить в фоне на реальном MovieLens и убедиться, что первые стадии стартуют без ошибок**

Run: `uv run python scripts/movies_check.py > reports/movies_check.log 2>&1 &` (из корня проекта; датасет уже в
`~/Downloads/ml-32m/`).
Проверить через 1–2 минуты: `tail -50 reports/movies_check.log` — должны появиться строки `[movielens]
загрузка и очистка`, затем JSON с `outputs` (ratings/users/works — счётчики без ошибок DuckDB), затем `[split]`
и JSON с `groups`. Если на этих двух стадиях исключений нет — код корректен; дальше (опорная смесь, taste,
ease-like-tune, layers) — многочасовое обучение без присмотра, полный лог — в `reports/movies_check.log`, итог —
последняя строка `--- готово за N мин`.

Expected: лог без traceback на первых двух стадиях; процесс продолжает работать в фоне.

---

## Self-Review

**Spec coverage:** округление/дедуп — Task 1; очистка пользователей (`movies:` в cleaning.yaml) и k-core —
Task 2; схема экспорта (5 parquet) — Task 1; прогон split/taste/ease-like-tune/layers — Task 3;
уточнение про опорную смесь (`evaluate.tune` + `MIX_GRID`) — Task 3. Профиль, TMDB, домен в CLI, правила
списка — вне этого шага (TODO.md, следующие шаги плана «Фильмы»), в план не включены намеренно.

**Placeholder scan:** не найдено — весь код в шагах законченный, без TBD/«добавить обработку».

**Type consistency:** `prepare(raw_dir, out_dir, cfg, min_user=20, min_work=100) -> dict` — сигнатура из Task 2
совпадает с вызовом в Task 3 (`movielens.prepare(RAW_DIR, CLEAN_DIR, cfg)`, min_user/min_work по умолчанию).
`export(con, out_dir) -> dict` (Task 1) используется внутри `prepare` (Task 2) без изменений сигнатуры.

**Review Focus:** все 5 пунктов покрыты тестами Task 1 (округление/границы звёзд, дедуп, ссылочная
целостность, схема пустых таблиц) и явным комментарием + печатью grid в Task 3 (опорная смесь).
