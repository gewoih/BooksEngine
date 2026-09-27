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
