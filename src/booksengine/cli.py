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
    """Отложенная выборка: валидация 5K и тест 20K пользователей вне обучения (data/model/split/)."""
    import json

    from booksengine.model import split as s
    from booksengine.paths import CLEAN_DIR, SPLIT_DIR
    manifest = json.loads((CLEAN_DIR / "manifest.json").read_text())
    meta = s.build(CLEAN_DIR / "ratings.parquet", SPLIT_DIR, manifest["outputs"]["ratings"]["checksum"], force=force)
    print(json.dumps(meta["groups"], ensure_ascii=False, indent=1))


@app.command()
def evaluate(model: str = typer.Argument(..., help="popularity | als | knn | ease"),
             stage: str = typer.Option("val", help="val — перебор настроек; test — замер лучшей и сохранение")) -> None:
    """Метрики модели на валидации или тесте (models/eval/)."""
    from booksengine.model import evaluate as ev
    if stage == "val":
        ev.tune(model)
    elif stage == "test":
        ev.test(model)
    else:
        raise typer.BadParameter("stage: val | test")


@app.command("report-3a")
def report_3a() -> None:
    """Отчёт этапа 3a (reports/stage3a_report.md) из models/eval/*.json."""
    from booksengine import report_3a as r
    print(r.write())


if __name__ == "__main__":
    app()
