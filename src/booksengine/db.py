import shutil

import duckdb

from booksengine.paths import TMP_DIR


def connect(memory_limit: str = "8GB", threads: int = 8) -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB с ограничением памяти и спиллом на диск."""
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET threads={threads}")
    con.execute(f"SET temp_directory='{TMP_DIR}'")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET enable_progress_bar=false")
    return con


def ensure_free_space(min_gb: float = 3.0) -> None:
    free = shutil.disk_usage(TMP_DIR.parent if TMP_DIR.parent.exists() else "/").free / 1e9
    if free < min_gb:
        raise RuntimeError(f"Свободно {free:.1f} ГБ на диске, нужно минимум {min_gb} ГБ")
