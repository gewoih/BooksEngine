"""Пути к сырым и производным данным. Переопределяются через RAW_DIR / DATA_DIR (или .env)."""
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
DATA_DIR = Path(os.environ.get("DATA_DIR", PROJECT_ROOT / "data")).expanduser()
STAGING_DIR = DATA_DIR / "staging"
CLEAN_DIR = DATA_DIR / "clean"
TMP_DIR = DATA_DIR / "tmp"
REPORTS_DIR = PROJECT_ROOT / "reports"
CONFIG_PATH = PROJECT_ROOT / "config" / "cleaning.yaml"

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
