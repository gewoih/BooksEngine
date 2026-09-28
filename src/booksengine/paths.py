"""Пути к сырым и производным данным. Переопределяются через RAW_DIR / DATA_DIR (или .env).

Домен (`--domain` в CLI, книги по умолчанию) — данные и модели разных доменов не смешиваются:
`data/<domain>/`, `models/<domain>/`; сырые файлы — общий RAW_DIR (у них разные имена/подпапки).
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv() -> None:
    env = PROJECT_ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()

RAW_DIR = Path(os.environ.get("RAW_DIR", "~/Downloads")).expanduser()
DOMAIN = os.environ.get("BOOKSENGINE_DOMAIN", "books")


def domain_profiles(profiles_dir: Path, domain: str) -> list[Path]:
    """Профили домена: у фильмов — `<имя>_movies.csv`, у книг — остальные CSV папки. Без разбора книжные команды
    брали и профиль фильмов и падали на нём (в нём нет goodreads_work_id)."""
    movies = sorted(profiles_dir.glob("*_movies.csv"))
    return movies if domain == "movies" else [p for p in sorted(profiles_dir.glob("*.csv")) if p not in movies]


def domain_dirs(project_root: Path, domain: str) -> tuple[Path, Path]:
    """(data_dir, models_dir) для домена — чистая функция, без чтения окружения (тестируется без .env/env)."""
    return project_root / "data" / domain, project_root / "models" / domain


_DATA_DIR, MODELS_DIR = domain_dirs(PROJECT_ROOT, DOMAIN)
DATA_DIR = Path(os.environ.get("DATA_DIR", _DATA_DIR)).expanduser()
STAGING_DIR = DATA_DIR / "staging"
CLEAN_DIR = DATA_DIR / "clean"
TMP_DIR = DATA_DIR / "tmp"
REPORTS_DIR = PROJECT_ROOT / "reports"
CONFIG_PATH = PROJECT_ROOT / "config" / "cleaning.yaml"
SPLIT_DIR = DATA_DIR / "model" / "split"
EVAL_DIR = MODELS_DIR / "eval"
EXP_DIR = DATA_DIR / "exp"                 # копия старого ядра и общий тест сравнения очистки
EXP_MODELS_DIR = MODELS_DIR / "exp"
EXP_EVAL_DIR = EVAL_DIR / "exp"

RAW_FILES = {
    "interactions": "goodreads_interactions.csv",
    "book_id_map": "book_id_map.csv",
    "user_id_map": "user_id_map.csv",
    "books": "goodreads_books.json",
    "works": "goodreads_book_works.json",
    "authors": "goodreads_book_authors.json",
    "genres": "goodreads_book_genres_initial.json",
}


def raw_path(name: str) -> Path:
    """Путь к сырому файлу; допускается и сжатая версия (.gz)."""
    p = RAW_DIR / RAW_FILES[name]
    if p.exists():
        return p
    gz = p.with_name(p.name + ".gz")
    if gz.exists():
        return gz
    raise FileNotFoundError(f"Нет файла датасета: {p} (или {gz.name})")
