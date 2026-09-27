#!/usr/bin/env python3
"""Проверка толпы и вкуса на MovieLens: загрузка -> split -> опорная смесь (als_neg+ease+mix, код
evaluate.py) -> толпа «ценность» (ease-like-tune) -> вкус -> layers val|test. Код model/*.py и
data/clean.py не меняется — только пути и явные grid-параметры; общий код модели работает с матрицей
«человек × произведение × оценка», книжной специфики (сборники, дубли произведений) в нём нет.
Разовый скрипт, не часть booksengine CLI — пути даны напрямую, без домена в paths.py/CLI.
Дизайн: docs/superpowers/specs/2026-09-27-movies-crowd-taste-check-design.md.

Запуск из корня проекта: uv run python scripts/movies_check.py
Долгий (часы на 32М строк) — в фоне, без буферизации вывода (иначе лог пуст, пока процесс не выйдет):
  uv run python -u scripts/movies_check.py > reports/movies_check.log 2>&1 &
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

# Сетка mix из evaluate.MODELS["mix"] (score_grid — веса ALS в смеси) с movies-путями к als_neg/ease вместо
# книжных. У evaluate.py als_dir/ease_dir зашиты через модульный (книжный) MODELS_DIR на уровне импорта —
# tune() принимает свой grid параметром (как --weights у ease-like-tune), это и обходит без изменений
# evaluate.py; веса сохраняются те же, что у книг, — не копия вручную, а чтение из evaluate.MODELS.
_MIX_FIT = {"als_dir": str(MODELS_DIR / "als_neg"), "ease_dir": str(MODELS_DIR / "ease")}
MIX_GRID = [(_MIX_FIT, score_grid) for _, score_grid in evaluate.MODELS["mix"][1]]


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

    ratings_path = CLEAN_DIR / "ratings.parquet"
    pool = movielens.bucket_pool_size(ratings_path)
    print("[split] пул по этапам (свой, не книжный):", json.dumps(pool, ensure_ascii=False))
    meta = split.build(ratings_path, CLEAN_DIR / "users.parquet", SPLIT_DIR,
                       data_fp=json.dumps(manifest["outputs"], sort_keys=True), bucket_pool_size=pool)
    print(json.dumps(meta["groups"], ensure_ascii=False, indent=1))

    print("[опорная смесь] als_neg")
    evaluate.tune("als_neg", ratings_path=ratings_path, split_dir=SPLIT_DIR, eval_dir=EVAL_DIR,
                 models_dir=MODELS_DIR)
    print("[опорная смесь] ease")
    evaluate.tune("ease", ratings_path=ratings_path, split_dir=SPLIT_DIR, eval_dir=EVAL_DIR,
                 models_dir=MODELS_DIR)
    print("[опорная смесь] mix — grid с movies-путями:", MIX_GRID)
    evaluate.tune("mix", ratings_path=ratings_path, split_dir=SPLIT_DIR, eval_dir=EVAL_DIR,
                 models_dir=MODELS_DIR, grid=MIX_GRID)

    print("[taste]")
    _write_report("taste_val", taste.report(taste.tune(
        ratings_path=ratings_path, split_dir=SPLIT_DIR, models_dir=MODELS_DIR, eval_dir=EVAL_DIR)))

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
