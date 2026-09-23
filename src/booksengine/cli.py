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


if __name__ == "__main__":
    app()
