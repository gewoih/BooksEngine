"""Инварианты очищенных данных. Любое нарушение — исключение: сломанный набор не должен уйти дальше."""
import yaml

from booksengine.db import connect
from booksengine.paths import CLEAN_DIR, CONFIG_PATH

TABLES = ["works", "editions", "authors", "work_authors", "work_genres", "users", "ratings", "work_merges"]


def checks(k_user: int, k_work: int) -> dict[str, str]:
    """Имя проверки → SQL, возвращающий число нарушений (0 = ок)."""
    return {
        "ratings: уникальность (user, work)":
            "SELECT count(*) - count(DISTINCT (user_id, work_id)) FROM ratings",
        "ratings: оценка в [1; 5]": "SELECT count(*) FROM ratings WHERE rating < 1 OR rating > 5 OR rating IS NULL",
        "ratings → works (in_cf)":
            "SELECT count(*) FROM ratings r LEFT JOIN works w USING (work_id) WHERE w.work_id IS NULL OR NOT w.in_cf",
        "ratings → users": "SELECT count(*) FROM ratings r ANTI JOIN users u USING (user_id)",
        f"k-core: у пользователя ≥ {k_user} оценок":
            f"SELECT count(*) FROM (SELECT user_id FROM ratings GROUP BY 1 HAVING count(*) < {k_user})",
        f"k-core: у произведения ≥ {k_work} оценок":
            f"SELECT count(*) FROM (SELECT work_id FROM ratings GROUP BY 1 HAVING count(*) < {k_work})",
        "works: уникальность work_id": "SELECT count(*) - count(DISTINCT work_id) FROM works",
        "works: есть название": "SELECT count(*) FROM works WHERE title IS NULL",
        "works: in_cf согласован с ratings":
            "SELECT count(*) FROM works w WHERE in_cf <> EXISTS (SELECT 1 FROM ratings r WHERE r.work_id = w.work_id)",
        "works: не-книги вне ядра": "SELECT count(*) FROM works WHERE is_nonbook AND in_cf",
        "work_merges: тень вне ядра":
            "SELECT count(*) FROM work_merges m JOIN works w ON w.work_id = m.shadow_work_id WHERE w.in_cf",
        "work_merges → works (главное)":
            "SELECT count(*) FROM work_merges m ANTI JOIN works w ON w.work_id = m.main_work_id",
        "editions: уникальность book_id": "SELECT count(*) - count(DISTINCT book_id) FROM editions",
        "editions → works": "SELECT count(*) FROM editions e ANTI JOIN works w USING (work_id)",
        "works: у каждого есть издание": "SELECT count(*) FROM works w ANTI JOIN editions e USING (work_id)",
        "work_authors → authors": "SELECT count(*) FROM work_authors wa ANTI JOIN authors a USING (author_id)",
        "work_authors → works": "SELECT count(*) FROM work_authors wa ANTI JOIN works w USING (work_id)",
        "work_genres → works": "SELECT count(*) FROM work_genres g ANTI JOIN works w USING (work_id)",
        "users: уникальность и внешний id":
            "SELECT count(*) - count(DISTINCT user_id) + count(*) FILTER (external_id IS NULL) FROM users",
    }


def run() -> dict[str, int]:
    cfg = yaml.safe_load(CONFIG_PATH.read_text())["kcore"]
    con = connect()
    for t in TABLES:
        con.execute(f"CREATE VIEW {t} AS SELECT * FROM '{CLEAN_DIR / t}.parquet'")
    results = {}
    print("[validate]")
    for name, sql in checks(cfg["min_user_ratings"], cfg["min_work_ratings"]).items():
        results[name] = con.execute(sql).fetchone()[0]
        print(f"  {'OK ' if results[name] == 0 else 'FAIL'} {name}" + ("" if results[name] == 0 else f": {results[name]}"))
    con.close()
    failed = {k: v for k, v in results.items() if v}
    if failed:
        raise AssertionError(f"Нарушены инварианты: {failed}")
    return results
