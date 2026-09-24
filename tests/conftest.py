"""Общие фикстуры: чистая тестовая БД booksengine_test со схемой из EF-миграций.

Схема — `dotnet ef migrations script`, БД — отдельная booksengine_test в контейнере из docker-compose.
Нет Postgres или dotnet — тесты с `dsn` пропускаются.
"""
import shutil
import subprocess

import psycopg
import pytest

from booksengine import db_load
from booksengine.paths import PROJECT_ROOT

TEST_DB = "booksengine_test"
DB_PROJECT = PROJECT_ROOT / "dotnet" / "BooksEngine.Db"


@pytest.fixture(scope="session")
def schema_sql(tmp_path_factory) -> str:
    if shutil.which("dotnet") is None:
        pytest.skip("нет dotnet")
    try:
        psycopg.connect(db_load.pg_dsn(), connect_timeout=3).close()
    except psycopg.OperationalError:
        pytest.skip("PostgreSQL недоступен (docker compose up -d)")
    out = tmp_path_factory.mktemp("schema") / "schema.sql"
    subprocess.run(["dotnet", "ef", "migrations", "script", "-o", str(out)],
                   cwd=DB_PROJECT, check=True, capture_output=True)
    return out.read_text(encoding="utf-8-sig")  # EF пишет скрипт с BOM


@pytest.fixture
def dsn(schema_sql) -> str:
    with psycopg.connect(db_load.pg_dsn(), autocommit=True) as admin:
        admin.execute(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")
        admin.execute(f"CREATE DATABASE {TEST_DB}")
    with psycopg.connect(db_load.pg_dsn(TEST_DB)) as conn:
        conn.execute(schema_sql)
    return db_load.pg_dsn(TEST_DB)
