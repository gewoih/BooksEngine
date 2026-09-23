"""`booksengine prepare`: staging → профилирование → очистка → экспорт → валидация → отчёт."""
import hashlib
import json
import shutil
import time
from pathlib import Path

import yaml

from booksengine.data import clean, profile, stage
from booksengine.db import connect
from booksengine.paths import CLEAN_DIR, CONFIG_PATH, RAW_DIR, RAW_FILES, STAGING_DIR, TMP_DIR

STAGING_TABLES = ["interactions", "book_id_map", "user_id_map", "editions", "works", "authors", "genres"]
PARQUET = "(FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 1000000)"


def _file_fingerprint(path: Path) -> str:
    st = path.stat()
    return f"{path.name}:{st.st_size}:{int(st.st_mtime)}"


def _raw_fingerprints() -> dict:
    out = {}
    for name in RAW_FILES:
        try:
            out[name] = _file_fingerprint(stage.raw_path(name))
        except FileNotFoundError:
            out[name] = None
    return out


def _code_hash() -> str:
    h = hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob("*.py")):
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _export(con, sql: str, name: str) -> dict:
    out = CLEAN_DIR / f"{name}.parquet"
    con.execute(f"COPY ({sql}) TO '{out}' {PARQUET}")
    rows, checksum = con.execute(f"SELECT count(*), coalesce(sum(hash(t)::HUGEINT), 0)::VARCHAR "
                                 f"FROM '{out}' t").fetchone()
    print(f"  {name}: {rows:,} строк, {out.stat().st_size / 1e6:,.0f} МБ")
    return {"rows": rows, "checksum": checksum, "bytes": out.stat().st_size}


def export(con) -> dict:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    clean.author_source(con)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE _work_authors AS
        SELECT DISTINCT ON (work_id, a.author_id) work_id, a.author_id AS author_id, a.role AS role,
               pos::SMALLINT AS position
        FROM (SELECT work_id, unnest(authors) AS a, generate_subscripts(authors, 1) AS pos FROM _author_src)
        WHERE a.author_id IS NOT NULL AND a.author_id IN (SELECT author_id FROM authors)
        ORDER BY work_id, a.author_id, pos
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE _cf_works AS
        SELECT work_id, count(*) AS cf_ratings, round(avg(rating), 6) AS cf_mean_rating FROM ratings_w GROUP BY 1
    """)
    res = {
        "works": _export(con, """
            SELECT w.*, c.work_id IS NOT NULL AS in_cf, coalesce(c.cf_ratings, 0) AS cf_ratings, c.cf_mean_rating
            FROM works_valid w LEFT JOIN _cf_works c USING (work_id) ORDER BY work_id""", "works"),
        "editions": _export(con, """
            SELECT book_id, work_id, title, title_without_series, list_transform(authors, a -> a.author_id) AS author_ids,
                   isbn, isbn13, asin, kindle_asin, language_code, country_code, format, is_ebook, num_pages,
                   publication_year, publisher, ratings_count, average_rating, image_url, url
            FROM editions WHERE work_id IN (SELECT work_id FROM works_valid) ORDER BY book_id""", "editions"),
        "work_authors": _export(con, "SELECT * FROM _work_authors ORDER BY work_id, position", "work_authors"),
        "authors": _export(con, """
            SELECT author_id, name, average_rating, ratings_count FROM authors
            WHERE author_id IN (SELECT author_id FROM _work_authors) ORDER BY author_id""", "authors"),
        "work_genres": _export(con, """
            SELECT work_id, genre, votes, round(votes / sum(votes) OVER (PARTITION BY work_id), 4) AS share
            FROM (SELECT e.work_id, g.genre, sum(g.votes)::INTEGER AS votes
                  FROM genres g JOIN editions e ON e.book_id = g.book_id
                  WHERE e.work_id IN (SELECT work_id FROM works_valid) AND g.votes > 0
                  GROUP BY e.work_id, g.genre)
            ORDER BY work_id, votes DESC, genre""", "work_genres"),
        "users": _export(con, f"""
            SELECT r.user_id, m.user_id AS external_id, count(*) AS n_ratings, round(avg(r.rating), 6) AS mean_rating,
                   round(coalesce(stddev_pop(r.rating), 0), 6) AS sd_rating
            FROM ratings_w r JOIN user_id_map m ON m.user_id_csv = r.user_id GROUP BY ALL ORDER BY r.user_id""",
            "users"),
        "ratings": _export(con, "SELECT user_id, work_id, rating, n_editions FROM ratings_w ORDER BY user_id, work_id",
                           "ratings"),
        # тень → главное: тени остаются в каталоге вне ядра, оценки по ним — у главного (TODO п. 17)
        "work_merges": _export(con, "SELECT shadow_work_id, main_work_id FROM dup_map ORDER BY 1", "work_merges"),
    }
    for t in ("_author_src", "_work_authors", "_cf_works"):
        con.execute(f"DROP TABLE {t}")
    return res


def collection_examples(con) -> list[dict]:
    cur = con.execute("""
        SELECT w.best_edition_title AS title, count(r.user_id) AS explicit_ratings
        FROM works_valid w JOIN ratings_w r USING (work_id) WHERE w.is_collection
        GROUP BY 1 ORDER BY 2 DESC LIMIT 12
    """)
    return [dict(zip([d[0] for d in cur.description], row)) for row in cur.fetchall()]


def collection_totals(con) -> dict:
    works, ratings = con.execute("""
        SELECT count(DISTINCT work_id), count(*) FROM ratings_w
        WHERE work_id IN (SELECT work_id FROM works_valid WHERE is_collection)
    """).fetchone()
    return {"works_rated": works, "ratings": ratings,
            "works_flagged": con.execute("SELECT count(*) FROM works_valid WHERE is_collection").fetchone()[0]}


def prepare(force: bool = False, skip_profile: bool = False) -> dict:
    t0 = time.time()
    cfg_text = CONFIG_PATH.read_text()
    cfg = yaml.safe_load(cfg_text)
    fingerprint = {"raw": _raw_fingerprints(), "config": hashlib.sha256(cfg_text.encode()).hexdigest()[:16],
                   "code": _code_hash()}
    manifest_path = CLEAN_DIR / "manifest.json"
    if manifest_path.exists() and not force:
        old = json.loads(manifest_path.read_text())
        if old.get("fingerprint") == fingerprint:
            print("Входы, конфиг и код не менялись — результат актуален (используйте --force для пересборки).")
            return old

    stage_stats = stage.run(force=force)
    con = connect()
    for t in STAGING_TABLES:
        con.execute(f"CREATE OR REPLACE VIEW {t} AS SELECT * FROM '{STAGING_DIR / t}.parquet'")

    profile_path = CLEAN_DIR.parent / "profile.json"
    raw_fp = json.dumps(fingerprint["raw"], sort_keys=True)
    if profile_path.exists() and not force and json.loads(profile_path.read_text()).get("raw") == raw_fp:
        prof = json.loads(profile_path.read_text())["metrics"]
        print("[profile] из кэша")
    elif skip_profile:
        prof = {}
    else:
        prof = profile.run(con)
        profile_path.write_text(json.dumps({"raw": raw_fp, "metrics": prof}, ensure_ascii=False, indent=1,
                                           default=str))

    print("[clean]")
    clean.SPILL_DIR = TMP_DIR / "clean"
    log = clean.CleaningLog()
    clean.structural(con, log)
    clean.build_catalog(con, log, cfg["collections"]["title_patterns"])
    clean.link_to_works(con, log)
    s = cfg["signal"]
    clean.to_work_level(con, log, s["explicit_min"], s["explicit_max"])
    extra = {"collections": collection_totals(con), "collection_examples": collection_examples(con)}
    if cfg["collections"]["drop_from_ratings"]:
        clean.drop_collections(con, log)
    nb = cfg["nonbooks"]
    clean.flag_nonbooks(con, nb["title_patterns"], nb["exceptions"])
    clean.drop_nonbooks(con, log)
    clean.primary_authors(con)
    d = cfg["duplicates"]
    clean.find_duplicates(con, d["adaptation_patterns"], d["max_shadow_share"])
    clean.merge_duplicates(con, log)
    u = cfg["users"]
    clean.filter_users(con, log, u["low_variance_max_sd"], u["low_variance_min_ratings"], u["max_ratings"],
                       max_mode_share=u["monotone_max_mode_share"])
    k = cfg["kcore"]
    print("[kcore] варианты порогов")
    extra["kcore_options"] = clean.kcore_options(con, k["report_options"])
    clean.apply_kcore(con, log, k["min_user_ratings"], k["min_work_ratings"])

    print("[export]")
    outputs = export(con)
    con.close()

    from booksengine.data import validate
    checks = validate.run()
    manifest = {
        "fingerprint": fingerprint, "config": cfg, "outputs": outputs, "cleaning_log": log.records(),
        "extra": extra, "validation": checks, "seconds": round(time.time() - t0, 1),
        "raw_dir": str(RAW_DIR), "staging_rejected": stage_stats["interactions"].get("rejected_lines", 0),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1, default=str))

    from booksengine import report
    path = report.write(prof, manifest)
    print(f"Отчёт: {path}")
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    return manifest
