import typer

app = typer.Typer(help="BooksEngine: офлайн-часть рекомендательной системы книг", no_args_is_help=True)


@app.command()
def prepare(force: bool = typer.Option(False, "--force", help="пересобрать всё, игнорируя кэш")) -> None:
    """Staging → профилирование → очистка → валидация → отчёт (reports/stage1_report.md)."""
    from booksengine.data import pipeline
    pipeline.prepare(force=force)


@app.command()
def validate() -> None:
    """Проверить инварианты уже очищенных данных."""
    from booksengine.data import validate as v
    v.run()


@app.command("load-db")
def load_db(force: bool = typer.Option(False, "--force", help="загрузить, даже если этот manifest уже загружен")) -> None:
    """Загрузить каталог из data/clean/*.parquet в PostgreSQL (схема — EF-миграции dotnet/BooksEngine.Db)."""
    from booksengine import db_load
    db_load.load(force=force)


@app.command()
def report() -> None:
    """Пересобрать отчёт по готовым profile.json и manifest.json."""
    import json

    from booksengine import report as r
    from booksengine.paths import CLEAN_DIR
    prof = json.loads((CLEAN_DIR.parent / "profile.json").read_text())["metrics"]
    manifest = json.loads((CLEAN_DIR / "manifest.json").read_text())
    print(r.write(prof, manifest))


@app.command()
def split(force: bool = typer.Option(False, "--force", help="пересобрать сплит")) -> None:
    """Отложенная выборка: тест и валидация из групп 20-49/50-199/200+, вне обучения (data/model/split/)."""
    import json

    from booksengine.model import split as s
    from booksengine.paths import CLEAN_DIR, SPLIT_DIR
    manifest = json.loads((CLEAN_DIR / "manifest.json").read_text())
    meta = s.build(CLEAN_DIR / "ratings.parquet", CLEAN_DIR / "users.parquet", SPLIT_DIR,
                   manifest["outputs"]["ratings"]["checksum"], force=force)
    print(json.dumps(meta["groups"], ensure_ascii=False, indent=1))


@app.command()
def evaluate(model: str = typer.Argument(..., help="popularity | als | als_neg | knn | ease | mix"),
             stage: str = typer.Option("val", help="val — перебор настроек; test — замер лучшей и сохранение")) -> None:
    """Метрики модели на валидации или тесте (models/eval/)."""
    from booksengine.model import evaluate as ev
    if stage == "val":
        ev.tune(model)
    elif stage == "test":
        ev.test(model)
    else:
        raise typer.BadParameter("stage: val | test")


@app.command()
def calibrate(model: str = typer.Argument("mix", help="сохранённая модель в models/")) -> None:
    """Шанс «понравится» (п. 29): учится на валидации, проверяется на тесте → models/<model>/chance.json."""
    import json

    from booksengine.model import chance
    from booksengine.model.evaluate import RATINGS
    from booksengine.paths import EVAL_DIR, MODELS_DIR, SPLIT_DIR
    out = chance.calibrate(model, ratings_path=RATINGS, split_dir=SPLIT_DIR, models_dir=MODELS_DIR, eval_dir=EVAL_DIR)
    print(json.dumps({k: out[k] for k in ("chance", "test", "reliability")}, ensure_ascii=False, indent=1))


@app.command()
def recommend(ratings: str = typer.Option(..., "--ratings", help="CSV: goodreads_work_id, rating 1–5, status, title"),
              top: int = typer.Option(20, "--top", help="сколько книг показать")) -> None:
    """Оценки из CSV → рекомендации смеси с шансом «понравится» и объяснением (п. 11)."""
    from pathlib import Path

    from booksengine import recommend as rec
    from booksengine.paths import CLEAN_DIR, MODELS_DIR
    print(rec.format_result(rec.recommend(Path(ratings), clean_dir=CLEAN_DIR, models_dir=MODELS_DIR, top=top)))


@app.command("export-model")
def export_model() -> None:
    """Смесь models/mix → PostgreSQL для C# API: перезаписывает модель целиком (веб-интерфейс)."""
    from booksengine import export_model as ex
    from booksengine.db_load import pg_dsn
    from booksengine.paths import CLEAN_DIR, MODELS_DIR, PROJECT_ROOT, TMP_DIR
    ex.run(clean_dir=CLEAN_DIR, models_dir=MODELS_DIR, profiles_dir=PROJECT_ROOT / "profiles",
           tmp_dir=TMP_DIR / "export", dsn=pg_dsn())


@app.command("report-3a")
def report_3a() -> None:
    """Отчёт этапа 3a (reports/stage3a_report.md) из models/eval/*.json."""
    from booksengine import report_3a as r
    print(r.write())


exp_app = typer.Typer(help="Сравнение вариантов CF-ядра (правила очистки, пороги) на общем тесте")
app.add_typer(exp_app, name="exp")


def _exp_cores() -> dict:
    """Ядра сравнения: сохранённые папки data/exp/<имя>/ и «new» — ссылки на текущий data/clean."""
    from booksengine.paths import CLEAN_DIR, EXP_DIR
    new = EXP_DIR / "new"
    new.mkdir(parents=True, exist_ok=True)
    for f in ("ratings.parquet", "works.parquet", "work_merges.parquet", "manifest.json"):
        (new / f).unlink(missing_ok=True)
        (new / f).symlink_to(CLEAN_DIR / f)
    return {d.name: d for d in sorted(EXP_DIR.iterdir()) if d.is_dir() and d.name != "common"}


@exp_app.command("save")
def exp_save(name: str = typer.Argument(..., help="имя ядра, например old или k50")) -> None:
    """Скопировать текущее ядро (data/clean) в data/exp/<name>/ — до `prepare --force` с другими правилами."""
    from booksengine.model import experiment as ex
    from booksengine.paths import CLEAN_DIR, EXP_DIR
    if name in ("new", "common"):
        raise typer.BadParameter("имена new и common заняты")
    if (EXP_DIR / name / "ratings.parquet").exists():
        raise typer.BadParameter(f"data/exp/{name}/ уже есть — сохранённое ядро не перезаписывается")
    ex.save_core(CLEAN_DIR, EXP_DIR / name)


@exp_app.command("split")
def exp_split() -> None:
    """Общий тест всех ядер сравнения (data/exp/common/)."""
    from booksengine.model import experiment as ex
    from booksengine.paths import EXP_DIR, SPLIT_DIR
    print(ex.build_common_split(_exp_cores(), SPLIT_DIR, EXP_DIR / "common"))


@exp_app.command("run")
def exp_run(core: str = typer.Option(None, help="одно ядро; по умолчанию все")) -> None:
    """Популярность, ALS, kNN с настройками 3a на общем тесте (models/eval/exp/<core>.json).
    Модель ядра, уже лежащая в models/exp/<core>/, загружается, а не обучается заново."""
    from booksengine.model import experiment as ex
    from booksengine.paths import EXP_DIR, EXP_EVAL_DIR, EXP_MODELS_DIR, PROJECT_ROOT, SPLIT_DIR
    for name in [core] if core else list(_exp_cores()):
        ex.run_core(name, ratings_path=EXP_DIR / name / "ratings.parquet", works_path=EXP_DIR / name / "works.parquet",
                    common_dir=EXP_DIR / "common" / name, holdout_path=SPLIT_DIR / "holdout_users.parquet",
                    profile_path=PROJECT_ROOT / "profiles" / "my_ratings.csv",
                    model_dir=EXP_MODELS_DIR / name, eval_dir=EXP_EVAL_DIR)


@exp_app.command("report")
def exp_report() -> None:
    """reports/cleanup_experiment.md."""
    from booksengine import report_exp
    print(report_exp.write())


if __name__ == "__main__":
    app()
