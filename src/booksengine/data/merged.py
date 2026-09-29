"""Единая база Goodreads + Amazon Reviews'23 — домен `books-amazon` (`data/books-amazon/clean/`, та же схема, что у
книг): модель, правила списка и выдача работают с ней без изменений.

- Люди Amazon — дополнительные читатели (user_id от AMAZON_USER_OFFSET). Их оценки ложатся на книги ядра Goodreads
  по мосту ISBN/ASIN (`amazon.build_bridge`), а без моста — по основному автору и названию до двоеточия/скобки:
  переиздание старой книги с новым ISBN — то же произведение, а не новая книга. Звёзды — по шкале (`AMAZON_SCALES`:
  у Amazon 64% пятёрок против 33% у Goodreads), в строку — среднее по изданиям. Порог людей — книг у человека.
- Новые книги — книги после 2017 года без пары в Goodreads (датасет Goodreads кончается 2017-м): издания одного
  произведения (Books и Kindle) склеены по автору и названию; нужно не меньше min_new_ratings читателей. work_id — от
  NEW_WORK_OFFSET, автор — автор Goodreads с тем же именем, иначе новый (от NEW_AUTHOR_OFFSET): правила «один автор
  на 10 мест» и «одно произведение» работают через оба источника. Нон-фикшн — по категории Amazon.
- Перевод: у новой книги works.ru_known — есть ли известный русский перевод (Wikidata, `amazon.apply_translation_signal`);
  без него правила списка книгу не советуют (`filters.ListPicker`). У книг Goodreads ru_known пусто — не трогаем.
- Отложенные люди — те же, что у книг (копия `data/books/model/split`): судья меряет обе базы на одних людях, а люди
  Amazon в проверку не попадают.
"""
import shutil
from pathlib import Path

import duckdb
import pandas as pd

from booksengine.data.amazon import AMAZON_SCALES, NEW_AFTER_YEAR
from booksengine.data.clean import title_key_sql

AMAZON_USER_OFFSET = 1_000_000
NEW_WORK_OFFSET = 100_000_000
NEW_AUTHOR_OFFSET = 100_000_000
MIN_NEW_RATINGS = 100          # как порог книги в ядре Goodreads
# верхняя категория Amazon (Books|<она>|…, Kindle Store|Kindle eBooks|<она>|…) — нон-фикшн
NONFICTION_CATEGORIES = frozenset({
    "Biographies & Memoirs", "History", "Politics & Social Sciences", "Cookbooks, Food & Wine", "Arts & Photography",
    "Crafts, Hobbies & Home", "Health, Fitness & Dieting", "Science & Math", "Business & Money", "Self-Help",
    "Computers & Technology", "Reference", "Medical Books", "Engineering & Transportation", "Parenting & Relationships",
    "Travel", "Professional & Technical", "Education & Teaching", "Law", "Sports & Outdoors",
    "Religion & Spirituality"})
_COLLECTION = r"box(ed)? set|collection set|\d+ books? (collection|set)|books? \d+-\d+"


def author_key_sql(col: str) -> str:
    """Ключ имени автора: только буквы, без регистра («J.K. Rowling» = «J. K. Rowling»)."""
    return f"regexp_replace(lower(coalesce({col}, '')), '[^\\p{{L}}]+', '', 'g')"


def short_key_sql(col: str) -> str:
    """Ключ названия до двоеточия или скобки: «Educated: A Memoir» = «Educated», «Dune (Dune, #1)» = «Dune»."""
    return title_key_sql(f"regexp_replace(coalesce({col}, ''), '\\s*[:(\\[].*$', '')")


def _category_sql(col: str) -> str:
    return f"CASE WHEN {col} LIKE 'Kindle%' THEN split_part({col}, '|', 3) ELSE split_part({col}, '|', 2) END"


def build(goodreads_dir: Path, amazon_dir: Path, out_dir: Path, *, scale: str = "q", min_user: int = 5,
          min_new_ratings: int = MIN_NEW_RATINGS, memory_limit: str = "8GB") -> dict:
    """Собирает out_dir (clean) и копию отложенных людей рядом (`out_dir.parent / model / split`)."""
    from booksengine.model.filters import work_info
    from booksengine.model.matrix import catalog_works

    if scale not in AMAZON_SCALES:
        raise ValueError(f"шкала {scale}: нужно {' | '.join(AMAZON_SCALES)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    spill = out_dir.parent / "tmp"
    spill.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET temp_directory='{spill}'")
    g = {n: str(goodreads_dir / f"{n}.parquet")
         for n in ("ratings", "users", "works", "authors", "work_authors", "work_genres", "work_merges")}
    a = {n: str(amazon_dir / f"{n}.parquet") for n in ("ratings", "items", "bridge")}

    core = catalog_works(goodreads_dir / "ratings.parquet")
    info = work_info(goodreads_dir, core)[["work_id", "title", "author"]]
    info["cf_ratings"] = duckdb.execute("SELECT work_id, cf_ratings FROM read_parquet(?)", [g["works"]]).df(
        ).set_index("work_id").cf_ratings.reindex(info.work_id).to_numpy()
    con.register("core_info", info)
    con.execute(f"""
        CREATE TEMP TABLE gr AS
        SELECT arg_max(work_id, cf_ratings) AS work_id, akey, tkey FROM (
            SELECT work_id, cf_ratings, {author_key_sql('author')} AS akey, {short_key_sql('title')} AS tkey
            FROM core_info)
        WHERE akey <> '' AND tkey <> '' GROUP BY akey, tkey""")
    con.execute(f"""
        CREATE TEMP TABLE az0 AS
        SELECT i.parent_asin, i.title, i.author, try_cast(i.year AS INTEGER) AS year, i.categories, i.wikidata,
               b.work_id AS by_isbn, {author_key_sql('i.author')} AS akey, {short_key_sql('i.title')} AS tkey
        FROM read_parquet(?) i JOIN read_parquet(?) b USING (parent_asin)
    """, [a["items"], a["bridge"]])
    # только равенство в ON: условие на левую сторону превращает LEFT JOIN в DuckDB во вложенный цикл
    con.execute("""
        CREATE TEMP TABLE az AS
        SELECT az0.*, CASE WHEN az0.by_isbn IS NULL THEN gr.work_id END AS by_title
        FROM az0 LEFT JOIN gr ON gr.akey = az0.akey AND gr.tkey = az0.tkey""")
    con.register("core_ids", pd.DataFrame({"work_id": core}))
    # произведение каждого издания Amazon: книга ядра (по ISBN или по автору и названию) или новая книга
    con.execute(f"""
        CREATE TEMP TABLE asin_work AS
        SELECT parent_asin, coalesce(by_isbn, by_title) AS work_id, NULL AS new_key FROM az
        WHERE coalesce(by_isbn, by_title) IN (SELECT work_id FROM core_ids)
        UNION ALL
        SELECT parent_asin, NULL, akey || '|' || tkey FROM az
        WHERE by_isbn IS NULL AND by_title IS NULL AND year > {NEW_AFTER_YEAR} AND akey <> '' AND tkey <> ''""")
    con.execute("""
        CREATE TEMP TABLE r AS
        SELECT r.user_id AS amazon_user, w.work_id, w.new_key, avg(r.rating) AS rating, count(*) AS n_editions
        FROM read_parquet(?) r JOIN asin_work w USING (parent_asin)
        WHERE r.rating >= 1 GROUP BY 1, 2, 3""", [a["ratings"]])
    con.execute(f"""
        CREATE TEMP TABLE new_works AS
        SELECT new_key, {NEW_WORK_OFFSET} + row_number() OVER (ORDER BY new_key) AS work_id, n_users
        FROM (SELECT new_key, count(DISTINCT amazon_user) AS n_users FROM r WHERE new_key IS NOT NULL GROUP BY 1)
        WHERE n_users >= {int(min_new_ratings)}""")
    con.execute(f"""
        CREATE TEMP TABLE ar AS
        SELECT amazon_user, coalesce(r.work_id, n.work_id) AS work_id, rating, n_editions
        FROM r LEFT JOIN new_works n USING (new_key)
        WHERE r.work_id IS NOT NULL OR n.work_id IS NOT NULL""")
    con.execute(f"""
        CREATE TEMP TABLE au AS
        SELECT amazon_user, {AMAZON_USER_OFFSET} + row_number() OVER (ORDER BY amazon_user) AS user_id
        FROM ar GROUP BY 1 HAVING count(*) >= {int(min_user)}""")
    stars = AMAZON_SCALES[scale]
    star_case = "CASE " + " ".join(f"WHEN floor(ar.rating + 0.5) = {k} THEN {v}" for k, v in
                                   zip(range(1, 6), stars)) + " END"
    opts = "(FORMAT parquet, COMPRESSION zstd)"
    con.execute(f"""
        COPY (SELECT user_id::INTEGER AS user_id, work_id::BIGINT AS work_id, rating::FLOAT AS rating,
                     n_editions::SMALLINT AS n_editions FROM read_parquet('{g['ratings']}')
              UNION ALL
              SELECT au.user_id::INTEGER, ar.work_id::BIGINT, ({star_case})::FLOAT, ar.n_editions::SMALLINT
              FROM ar JOIN au USING (amazon_user))
        TO '{out_dir / 'ratings.parquet'}' {opts}""")
    con.execute(f"""
        COPY (SELECT * FROM read_parquet('{g['users']}')
              UNION ALL
              SELECT au.user_id::INTEGER, 'amazon:' || au.amazon_user, count(*), round(avg(ar.rating), 6),
                     round(coalesce(stddev_pop(ar.rating), 0), 6)
              FROM ar JOIN au USING (amazon_user) GROUP BY au.user_id, au.amazon_user)
        TO '{out_dir / 'users.parquet'}' {opts}""")

    # новые книги: название, автор, категория — у самого оцениваемого издания (при равенстве — с кратким названием:
    # по нему лучше ловятся дубли), год — самый ранний, перевод — у любого издания
    con.execute(f"""
        CREATE TEMP TABLE nw AS
        SELECT n.work_id, first(z.title ORDER BY c.n DESC NULLS LAST, length(z.title), z.title) AS title,
               first(z.author ORDER BY c.n DESC NULLS LAST, length(z.title), z.title) AS author, min(z.year) AS year,
               first(z.categories ORDER BY c.n DESC NULLS LAST, length(z.title), z.title) AS categories,
               bool_or(z.wikidata = 'ru') AS ru_known,
               count(DISTINCT z.parent_asin) AS n_editions, any_value(z.akey) AS akey
        FROM new_works n
        JOIN az z ON z.akey || '|' || z.tkey = n.new_key AND z.by_isbn IS NULL AND z.by_title IS NULL
            AND z.year > {NEW_AFTER_YEAR}
        LEFT JOIN (SELECT parent_asin, count(*) AS n FROM read_parquet('{a['ratings']}') GROUP BY 1) c USING (parent_asin)
        GROUP BY n.work_id""")
    stats_new = con.execute("""
        SELECT n.work_id, count(*) AS n, avg(rating) AS mean FROM ar JOIN au USING (amazon_user)
        JOIN nw n USING (work_id) GROUP BY 1""").df()
    con.register("nstats", stats_new)
    con.execute(f"""
        COPY (SELECT *, 'goodreads' AS source, NULL::BOOLEAN AS ru_known FROM read_parquet('{g['works']}')
              UNION ALL BY NAME
              SELECT nw.work_id::BIGINT AS work_id, NULL::BIGINT AS best_book_id, title, NULL AS original_title,
                     title AS best_edition_title, year AS publication_year, 'eng' AS language_code,
                     'book' AS media_type, n_editions::INTEGER AS books_count,
                     regexp_matches(lower(title), '{_COLLECTION}') AS is_collection, false AS is_nonbook,
                     true AS in_cf, s.n::BIGINT AS cf_ratings, round(s.mean, 6) AS cf_mean_rating,
                     'amazon' AS source, coalesce(ru_known, false) AS ru_known
              FROM nw LEFT JOIN nstats s USING (work_id))
        TO '{out_dir / 'works.parquet'}' {opts}""")
    con.execute(f"""
        CREATE TEMP TABLE known_author AS
        SELECT {author_key_sql('name')} AS akey, arg_max(author_id, ratings_count) AS author_id
        FROM read_parquet('{g['authors']}') GROUP BY 1""")
    con.execute(f"""
        CREATE TEMP TABLE nw_author AS
        SELECT nw.work_id, nw.author, coalesce(k.author_id,
               {NEW_AUTHOR_OFFSET} + dense_rank() OVER (ORDER BY CASE WHEN k.author_id IS NULL THEN nw.akey END))
               AS author_id, k.author_id IS NULL AS is_new
        FROM nw LEFT JOIN known_author k USING (akey) WHERE nw.akey <> ''""")
    con.execute(f"""
        COPY (SELECT * FROM read_parquet('{g['authors']}')
              UNION ALL
              SELECT author_id, any_value(author), NULL::DOUBLE, NULL::BIGINT FROM nw_author WHERE is_new GROUP BY 1)
        TO '{out_dir / 'authors.parquet'}' {opts}""")
    con.execute(f"""
        COPY (SELECT * FROM read_parquet('{g['work_authors']}')
              UNION ALL
              SELECT work_id::BIGINT, author_id::BIGINT, NULL::VARCHAR, 1::SMALLINT FROM nw_author)
        TO '{out_dir / 'work_authors.parquet'}' {opts}""")
    con.register("nonfiction", pd.DataFrame({"cat": sorted(NONFICTION_CATEGORIES)}))
    con.execute(f"""
        COPY (SELECT * FROM read_parquet('{g['work_genres']}')
              UNION ALL
              SELECT work_id::BIGINT, 'non-fiction', 1, 1.0 FROM nw
              WHERE {_category_sql('categories')} IN (SELECT cat FROM nonfiction))
        TO '{out_dir / 'work_genres.parquet'}' {opts}""")
    shutil.copy(g["work_merges"], out_dir / "work_merges.parquet")

    split_src, split_dst = goodreads_dir.parent / "model" / "split", out_dir.parent / "model" / "split"
    if split_dst.exists():
        shutil.rmtree(split_dst)
    shutil.copytree(split_src, split_dst)

    stats = con.execute(f"""
        SELECT (SELECT count(*) FROM au) AS amazon_users,
               (SELECT count(*) FROM ar JOIN au USING (amazon_user)) AS amazon_ratings,
               (SELECT count(*) FROM ar JOIN au USING (amazon_user) WHERE work_id < {NEW_WORK_OFFSET}) AS on_core,
               (SELECT count(*) FROM nw) AS new_works,
               (SELECT count(*) FROM nw WHERE ru_known) AS new_with_ru,
               (SELECT count(*) FROM az WHERE by_title IS NOT NULL) AS by_title""").df().iloc[0].to_dict()
    con.close()
    shutil.rmtree(spill, ignore_errors=True)
    return {k: int(v) for k, v in stats.items()}


def match_new_works(clean_dir: Path, titles, authors) -> pd.Series:
    """work_id новой книги Amazon (`source = 'amazon'`) для пар «английское название, автор» — по тем же ключам, что
    склейка изданий (`short_key_sql`, `author_key_sql`); не нашлось или база без новых книг — NA."""
    titles = pd.Series(titles, dtype=object).reset_index(drop=True)
    authors = pd.Series(authors, dtype=object).reset_index(drop=True)
    out = pd.Series(pd.NA, index=titles.index, dtype="Int64")
    works = clean_dir / "works.parquet"
    cols = set(duckdb.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(works)]).df().column_name)
    if "source" not in cols or titles.isna().all():
        return out
    con = duckdb.connect()
    con.register("q", pd.DataFrame({"i": titles.index, "title": titles, "author": authors}))
    d = con.execute(f"""
        WITH new AS (
            SELECT w.work_id, {short_key_sql('w.title')} AS tkey, {author_key_sql('a.name')} AS akey
            FROM read_parquet(?) w JOIN read_parquet(?) wa USING (work_id) JOIN read_parquet(?) a USING (author_id)
            WHERE w.source = 'amazon')
        SELECT q.i, min(new.work_id) AS work_id FROM q
        JOIN new ON new.tkey = {short_key_sql('q.title')} AND new.akey = {author_key_sql('q.author')}
        WHERE {short_key_sql('q.title')} <> '' GROUP BY 1
    """, [str(works), str(clean_dir / "work_authors.parquet"), str(clean_dir / "authors.parquet")]).df()
    con.close()
    out.loc[d.i.to_numpy()] = d.work_id.to_numpy()
    return out


def refit(clean_dir: Path, split_dir: Path, models_dir: Path, source_models: Path, min_user: int, log=print) -> None:
    """Все компоненты выдачи на единой базе — с теми же настройками, что у моделей source_models (только Goodreads):
    ALS, EASE, смесь, вкус, толпа «ценность» (порог людей — min_user: у Goodreads в ядре все от 20, так что это порог
    людей Amazon) и её смесь; выбор варианта слоёв — копия, `layers val` выберет заново. Подбора настроек здесь нет:
    сравнивается база, а не настройки."""
    import time

    from booksengine.model.als import ALS
    from booksengine.model.base import read_params
    from booksengine.model.ease import EASE, EASELike
    from booksengine.model.matrix import load_train
    from booksengine.model.mix import Mix
    from booksengine.model.taste import Taste

    def timed(what, fn):
        t0 = time.perf_counter()
        fn()
        log(f"{what}: {time.perf_counter() - t0:.0f} с")

    train = load_train(clean_dir / "ratings.parquet", split_dir / "holdout_users.parquet")
    log(f"обучение: {train.X.shape[0]:,} человек × {train.X.shape[1]:,} книг, {train.X.nnz:,} оценок")
    p = read_params(source_models / "als_neg")
    als = ALS(**{k: p[k] for k in ("factors", "regularization", "alpha", "iterations", "seed")})
    timed("ALS", lambda: als.fit(train))
    als.configure(p["neg_rule"], p["neg_weight"])
    als.save(models_dir / "als_neg")
    del als
    p = read_params(source_models / "ease")
    ease = EASE(lam=p["lam"], n_top=p["n_top"], block=p["block"])
    timed("EASE", lambda: ease.fit(train))
    ease.configure(topk=p["topk"])
    ease.save(models_dir / "ease")
    del ease
    p = read_params(source_models / "mix")
    mix = Mix(models_dir / "als_neg", models_dir / "ease")
    mix.fit(train)
    mix.configure(als_weight=p["als_weight"], ease_input=p["ease_input"])
    mix.save(models_dir / "mix")
    p = read_params(source_models / "taste")
    taste = Taste(factors=p["factors"], reg=p["reg"], iterations=p["iterations"], seed=p["seed"])
    timed("вкус", lambda: taste.fit(train, log=lambda *a: None))
    taste.save(models_dir / "taste")
    del taste
    p = read_params(source_models / "ease_like")
    like = EASELike(lam=p["lam"], n_top=p["n_top"], block=p["block"], topk=p["topk"], weights=p["weights"],
                    min_user=min_user, min_support=p.get("min_support", 0), amazon="в базе")
    timed("толпа «ценность»", lambda: like.fit(train))
    like.save(models_dir / "ease_like")
    mix_like = Mix(models_dir / "als_neg", models_dir / "ease_like")
    mix_like.fit(train)
    mix_like.configure(als_weight=0.5, ease_input=like.weights)
    mix_like.save(models_dir / "mix_like")
    (models_dir / "layers").mkdir(parents=True, exist_ok=True)
    shutil.copy(source_models / "layers" / "params.json", models_dir / "layers" / "params.json")
