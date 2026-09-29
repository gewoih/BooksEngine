from pathlib import Path

from booksengine.paths import domain_dirs, domain_profiles


def test_domain_dirs_puts_data_and_models_under_domain_name():
    data_dir, models_dir = domain_dirs(Path("/proj"), "movies")
    assert data_dir == Path("/proj/data/movies")
    assert models_dir == Path("/proj/models/movies")


def test_domain_dirs_default_domain_is_books():
    from booksengine.paths import DOMAIN
    assert DOMAIN == "books"


def test_cli_rejects_unknown_domain():
    from typer.testing import CliRunner

    from booksengine.cli import app
    result = CliRunner().invoke(app, ["--domain", "nonsense", "validate"])
    assert result.exit_code != 0


def test_cli_domain_sets_env_before_subcommand(monkeypatch):
    """Колбэк `--domain` в cli.py.main ставит переменную окружения раньше, чем выполнится подкоманда — на
    отдельном приложении с тем же колбэком, чтобы не засорять общий `app` тестовыми командами."""
    import os

    import typer
    from typer.testing import CliRunner

    from booksengine.cli import main
    monkeypatch.delenv("BOOKSENGINE_DOMAIN", raising=False)
    seen = {}

    probe_app = typer.Typer()
    probe_app.callback()(main)

    @probe_app.command()
    def probe() -> None:
        seen["domain"] = os.environ.get("BOOKSENGINE_DOMAIN")

    result = CliRunner().invoke(probe_app, ["--domain", "movies", "probe"])
    assert result.exit_code == 0, result.output
    assert seen["domain"] == "movies"


def test_domain_profiles_split_books_and_movies(tmp_path):
    for name in ("my_ratings.csv", "lera_ratings.csv", "my_movies.csv", "notes.txt"):
        (tmp_path / name).write_text("")
    assert [p.name for p in domain_profiles(tmp_path, "books")] == ["lera_ratings.csv", "my_ratings.csv"]
    assert [p.name for p in domain_profiles(tmp_path, "movies")] == ["my_movies.csv"]


def test_history_dir_is_separate_for_non_book_domains(tmp_path):
    """Выдача другого домена (единая база с Amazon) не затирает книжную запись дня в журнале: своя подпапка."""
    from booksengine.paths import history_dir
    assert history_dir(tmp_path, "books") == tmp_path / "profiles" / "history"
    assert history_dir(tmp_path, "books-amazon") == tmp_path / "profiles" / "history" / "books-amazon"
