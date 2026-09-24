"""`booksengine export-model`: готовая смесь (models/mix) → PostgreSQL для онлайн-выдачи в C# API.

Всё «умное» считается здесь и попадает в БД данными, а не переписанным кодом: векторы ALS, соседи EASE,
параметры смеси и шанса, пары «оценил X → не советовать Y» (серии — `series.py`, дубли и сборники —
`filters.py`), обложки, слияния теней и эталонная выдача `recommend`, по которой C# проверяет себя.
Экспорт перезаписывает модель целиком одной транзакцией.
"""
import io
import json
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import psycopg
import pyarrow as pa
import pyarrow.csv as pacsv
import scipy.sparse as sp

from booksengine import recommend as rec
from booksengine.db import ensure_free_space
from booksengine.model import explain
from booksengine.model.base import fingerprint
from booksengine.model.chance import Chance
from booksengine.model.filters import RatedFilter, work_info
from booksengine.model.matrix import catalog_works
from booksengine.model.mix import DNF_INPUT, Mix
from booksengine.model.series import SeriesIndex

SERIES, RATED = "series", "rated"
# эталонные пользователи датасета: по одному из целевых групп — 50–199 и 200+ оценок
DATASET_GROUPS = {"goodreads_50_199": (50, 199), "goodreads_200plus": (200, 3000)}


@dataclass
class ModelExport:
    meta: dict
    items: pd.DataFrame          # gr_work_id, col, ease_pos (Int32, NA — вне EASE), embedding
    ease: pd.DataFrame           # from_pos, to_pos, weight
    exclusions: pd.DataFrame     # rated_gr, excluded_gr, reason
    merges: pd.DataFrame         # shadow_gr, main_gr
    covers: pd.DataFrame         # gr_work_id, image_url
    golden_inputs: pd.DataFrame  # profile, gr_work_id, rating, dnf
    golden_recs: pd.DataFrame    # profile, rank, gr_work_id, score, chance, because, despite


def exclusion_pairs(info: pd.DataFrame) -> pd.DataFrame:
    """Пары столбцов (оценённый, исключаемый) — правила `recommend` для каждой книги по отдельности.
    Оба фильтра «или» по оценённым книгам, поэтому объединение пар равно фильтру набора.
    Серия важнее «уже оценено»: серия убирает книгу до расчёта шанса, фильтр — только из выдачи."""
    rows = []
    idx = SeriesIndex(info.title.tolist())
    for i in range(len(info)):
        rows += [(i, int(j), SERIES) for j in idx.continuations(np.array([i])).tolist() if j != i]
    for _, group in info.dropna(subset=["author_id"]).groupby("author_id"):
        cols = group.index.to_numpy()
        if len(cols) < 2:
            continue
        for i in cols:
            f = RatedFilter(info, np.array([i]))
            rows += [(int(i), int(j), RATED) for j in cols if j != i and f.is_rated_already(int(j))]
    d = pd.DataFrame(rows, columns=["rated", "excluded", "reason"])
    d = d.sort_values("reason", key=lambda s: s.ne(SERIES), kind="stable")
    return d.drop_duplicates(["rated", "excluded"]).sort_values(["rated", "excluded"]).reset_index(drop=True)


def covers(clean_dir: Path, work_ids: np.ndarray) -> pd.DataFrame:
    """Обложка — самое популярное издание с картинкой: у лучшего издания она есть у 65% ядра, у любого — у 88%."""
    d = duckdb.execute("""
        SELECT DISTINCT ON (work_id) work_id AS gr_work_id, image_url FROM read_parquet(?)
        WHERE image_url IS NOT NULL AND image_url NOT LIKE '%nophoto%'
        ORDER BY work_id, ratings_count DESC NULLS LAST, book_id""", [str(clean_dir / "editions.parquet")]).df()
    return d[d.gr_work_id.isin(work_ids)].sort_values("gr_work_id").reset_index(drop=True)


def dataset_profiles(clean_dir: Path, out_dir: Path) -> dict[str, Path]:
    """CSV эталонных пользователей датасета: наименьший user_id группы с целыми оценками."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ratings = str(clean_dir / "ratings.parquet")
    out = {}
    for name, (lo, hi) in DATASET_GROUPS.items():
        d = duckdb.execute("""
            WITH u AS (SELECT user_id FROM read_parquet(?) GROUP BY 1
                       HAVING count(*) BETWEEN ? AND ? AND bool_and(rating = round(rating))
                       ORDER BY user_id LIMIT 1)
            SELECT work_id AS goodreads_work_id, CAST(rating AS INTEGER) AS rating, 'read' AS status
            FROM read_parquet(?) JOIN u USING (user_id) ORDER BY work_id""", [ratings, lo, hi, ratings]).df()
        out[name] = out_dir / f"{name}.csv"
        d.to_csv(out[name], index=False)
    return out


def golden(profiles: dict[str, Path], *, clean_dir: Path, models_dir: Path, tmp_dir: Path,
           top: int = 50) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Выдача `recommend` на эталонных профилях: вход, топ, балл смеси, шанс, книги объяснения (id)."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    work_ids = catalog_works(clean_dir / "ratings.parquet")
    mix = Mix.load(models_dir / "mix")
    inputs, recs = [], []
    for name, csv in profiles.items():
        src = pd.read_csv(csv, dtype={"goodreads_work_id": "Int64"})
        src["title"] = [f"#{i}" for i in range(len(src))]      # имя в объяснении однозначно → книга
        path = tmp_dir / f"golden_{name}.csv"
        src.to_csv(path, index=False)
        prof = rec.read_profile(path, work_ids, clean_dir)
        col_of_name = {n: c for c, n in prof.names.items()}
        inputs.append(pd.DataFrame({"profile": name, "gr_work_id": work_ids[prof.x.indices],
                                    "rating": prof.x.data.astype(np.float64),
                                    "dnf": np.isin(prof.x.indices, prof.dnf.indices)}))
        score = mix.score(prof.x, prof.dnf)[0]
        res = rec.recommend(path, clean_dir=clean_dir, models_dir=models_dir, top=top)
        for k, r in enumerate(res.recs, 1):
            recs.append({"profile": name, "rank": k, "gr_work_id": r.work_id,
                         "score": float(score[np.searchsorted(work_ids, r.work_id)]), "chance": r.chance,
                         "because": [int(work_ids[col_of_name[b]]) for b in r.because],
                         "despite": None if r.despite is None else int(work_ids[col_of_name[r.despite]])})
    g = pd.DataFrame(recs, columns=["profile", "rank", "gr_work_id", "score", "chance", "because", "despite"])
    g["despite"] = g.despite.astype("Int64")
    return pd.concat(inputs, ignore_index=True), g


def build(*, clean_dir: Path, models_dir: Path, profiles: dict[str, Path], tmp_dir: Path,
          top: int = 50) -> ModelExport:
    work_ids = catalog_works(clean_dir / "ratings.parquet")
    mix_dir = models_dir / "mix"
    mix = Mix.load(mix_dir)
    if mix.ease.n_items != len(work_ids):
        raise ValueError(f"модель обучена на {mix.ease.n_items} книгах, в ядре {len(work_ids)}: переобучите")
    fp = fingerprint(mix_dir)
    chance = Chance.load(mix_dir / "chance.json", model_fp=fp)

    ease_pos = np.full(len(work_ids), -1)
    ease_pos[mix.ease.top_cols] = np.arange(len(mix.ease.top_cols))
    items = pd.DataFrame({"gr_work_id": work_ids, "col": np.arange(len(work_ids)),
                          "ease_pos": pd.Series(ease_pos).astype("Int32").mask(ease_pos < 0),
                          "embedding": list(mix.als.item_factors)})
    B = sp.coo_matrix(mix.ease._B)
    ease = pd.DataFrame({"from_pos": B.row, "to_pos": B.col, "weight": B.data.astype(np.float32)})

    pairs = exclusion_pairs(work_info(clean_dir, work_ids))
    exclusions = pd.DataFrame({"rated_gr": work_ids[pairs.rated], "excluded_gr": work_ids[pairs.excluded],
                               "reason": pairs.reason})
    merges = (pd.read_parquet(clean_dir / "work_merges.parquet")
              .rename(columns={"shadow_work_id": "shadow_gr", "main_work_id": "main_gr"})[["shadow_gr", "main_gr"]])
    golden_inputs, golden_recs = golden(profiles, clean_dir=clean_dir, models_dir=models_dir, tmp_dir=tmp_dir, top=top)

    meta = {"fingerprint": fp, "als_weight": mix.als_weight, "ease_input": list(mix.ease_input),
            "dnf_input": DNF_INPUT, "als_alpha": float(mix.als.alpha),
            "als_regularization": float(mix.als.regularization),
            "als_neg_rule": mix.als.neg_rule, "als_neg_weight": float(mix.als.neg_weight),
            "chance_coef": list(chance.coef), "chance_prior": chance.prior, "chance_p0": chance.p0,
            "max_because": explain.MAX_BECAUSE, "min_of_leader": explain.MIN_OF_LEADER,
            "despite_of_leader": explain.DESPITE_OF_LEADER}
    return ModelExport(meta, items, ease, exclusions, merges, covers(clean_dir, work_ids), golden_inputs, golden_recs)


MODEL_TABLES = ["model_meta", "work_embeddings", "ease_weights", "work_exclusions", "work_merges",
                "work_covers", "golden_inputs", "golden_recommendations"]


def _vec(v: np.ndarray) -> str:
    return "[" + ",".join(repr(float(x)) for x in v) + "]"   # float32 → double → текст без потерь


def _copy_csv(conn: psycopg.Connection, table: str, columns: list[str], t: pa.Table) -> None:
    buf = io.BytesIO()
    pacsv.write_csv(t, buf, pacsv.WriteOptions(include_header=False))
    with conn.cursor().copy(f"COPY {table} ({', '.join(columns)}) FROM STDIN (FORMAT csv)") as cp:
        cp.write(buf.getvalue())


def write(e: ModelExport, dsn: str) -> dict[str, int]:
    """Перезаписать модель в БД одной транзакцией: упало — остаётся прежняя целиком."""
    need = pd.concat([e.items.gr_work_id, e.exclusions.rated_gr, e.exclusions.excluded_gr, e.merges.main_gr,
                      e.covers.gr_work_id, e.golden_inputs.gr_work_id, e.golden_recs.gr_work_id,
                      e.golden_recs.despite.dropna(), e.golden_recs.because.explode().dropna()]).astype("int64").unique()
    with psycopg.connect(dsn) as conn:
        if not conn.execute("SELECT pg_try_advisory_xact_lock(hashtext('booksengine.export-model'))").fetchone()[0]:
            raise RuntimeError("Другой экспорт модели уже идёт в эту БД")
        found = dict(conn.execute("""
            SELECT x.external_id::bigint, x.internal_id FROM external_ids x JOIN sources s ON s.id = x.source_id
            WHERE s.code = 'goodreads' AND x.entity_type = 'work' AND x.external_id = ANY(%s)""",
                                  ([str(g) for g in need],)).fetchall())
        missing = [int(g) for g in need if int(g) not in found]
        if missing:
            raise ValueError(f"книг модели нет в каталоге БД: {len(missing)} (например {missing[:5]}); "
                             "сначала `booksengine load-db`")

        def m(s: pd.Series) -> pd.Series:   # Goodreads → внутренний id
            return s.astype("int64").map(found)

        conn.execute(f"TRUNCATE {', '.join(MODEL_TABLES)}")
        conn.execute("INSERT INTO model_meta (id, fingerprint, params) VALUES (1, %s, %s)",
                     (e.meta["fingerprint"], json.dumps(e.meta)))
        with conn.cursor().copy("COPY work_embeddings (work_id, col, ease_pos, embedding) FROM STDIN") as cp:
            for w, c, p, v in zip(m(e.items.gr_work_id), e.items.col, e.items.ease_pos, e.items.embedding):
                cp.write_row((int(w), int(c), None if pd.isna(p) else int(p), _vec(v)))
        _copy_csv(conn, "ease_weights", ["from_pos", "to_pos", "weight"], pa.table({
            "f": pa.array(e.ease.from_pos, pa.int32()), "t": pa.array(e.ease.to_pos, pa.int32()),
            "w": pa.array(e.ease.weight, pa.float32())}))
        _copy_csv(conn, "work_exclusions", ["rated_work_id", "excluded_work_id", "reason"], pa.table({
            "r": pa.array(m(e.exclusions.rated_gr), pa.int64()),
            "x": pa.array(m(e.exclusions.excluded_gr), pa.int64()),
            "why": pa.array(e.exclusions.reason, pa.string())}))
        with conn.cursor().copy("COPY work_merges (shadow_external_id, main_work_id) FROM STDIN") as cp:
            for s, w in zip(e.merges.shadow_gr, m(e.merges.main_gr)):
                cp.write_row((str(int(s)), int(w)))
        with conn.cursor().copy("COPY work_covers (work_id, image_url) FROM STDIN") as cp:
            for w, u in zip(m(e.covers.gr_work_id), e.covers.image_url):
                cp.write_row((int(w), u))
        with conn.cursor().copy("COPY golden_inputs (profile, work_id, rating, dnf) FROM STDIN") as cp:
            for p, w, r, d in zip(e.golden_inputs.profile, m(e.golden_inputs.gr_work_id), e.golden_inputs.rating,
                                  e.golden_inputs.dnf):
                cp.write_row((p, int(w), float(r), bool(d)))
        with conn.cursor().copy("COPY golden_recommendations (profile, rank, work_id, score, chance, because, "
                                "despite) FROM STDIN") as cp:
            for r in e.golden_recs.itertuples():
                cp.write_row((r.profile, int(r.rank), found[int(r.gr_work_id)], float(r.score), int(r.chance),
                              [found[int(b)] for b in r.because],
                              None if pd.isna(r.despite) else found[int(r.despite)]))
        counts = {t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in MODEL_TABLES}
        # commit — на выходе из with; исключение выше откатывает всё, включая TRUNCATE
    return counts


def run(*, clean_dir: Path, models_dir: Path, profiles_dir: Path, tmp_dir: Path, dsn: str) -> dict[str, int]:
    ensure_free_space(3.0)
    profiles = {"my_ratings": profiles_dir / "my_ratings.csv"} | dataset_profiles(clean_dir, tmp_dir)
    e = build(clean_dir=clean_dir, models_dir=models_dir, profiles=profiles, tmp_dir=tmp_dir)
    print(f"  пар исключений: {len(e.exclusions)}, соседей EASE: {len(e.ease)}, обложек: {len(e.covers)}", flush=True)
    counts = write(e, dsn)
    print(f"Выгружено: {counts}")
    return counts
