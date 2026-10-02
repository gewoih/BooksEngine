import typer

app = typer.Typer(help="BooksEngine: офлайн-часть рекомендательной системы книг", no_args_is_help=True)


@app.callback()
def main(domain: str = typer.Option("books", "--domain", help="books | movies — данные и модели домена "
                                     "(data/<domain>/, models/<domain>/)")) -> None:
    import os
    if domain not in ("books", "movies", "books-amazon"):
        raise typer.BadParameter("domain: books | movies | books-amazon")
    os.environ["BOOKSENGINE_DOMAIN"] = domain


@app.command()
def prepare(force: bool = typer.Option(False, "--force", help="пересобрать всё, игнорируя кэш (только books)"),
            amazon_scale: str = typer.Option("q", "--amazon-scale", help="books-amazon: шкала звёзд Amazon, raw | q"),
            amazon_min_user: int = typer.Option(5, "--amazon-min-user",
                                                help="books-amazon: человек Amazon — от стольких книг")) -> None:
    """books: staging → профилирование → очистка → валидация → отчёт (reports/stage1_report.md).
    movies: MovieLens → очистка (округление звёзд, k-core) → data/movies/clean/.
    books-amazon: единая база data/books/clean + data/amazon/clean (`amazon-bridge`) → data/books-amazon/clean/,
    отложенные люди — копия книжных."""
    from booksengine.paths import DOMAIN
    if DOMAIN == "books-amazon":
        import json

        from booksengine.data import merged
        from booksengine.paths import AMAZON_CLEAN_DIR, CLEAN_DIR, PROJECT_ROOT, domain_dirs
        books_clean = domain_dirs(PROJECT_ROOT, "books")[0] / "clean"
        stats = merged.build(books_clean, AMAZON_CLEAN_DIR, CLEAN_DIR, scale=amazon_scale, min_user=amazon_min_user,
                             ru_titles=merged.RU_TITLES)
        print(json.dumps(stats, ensure_ascii=False, indent=1))
        return
    if DOMAIN == "movies":
        import json

        import yaml

        from booksengine.data import movielens
        from booksengine.paths import CLEAN_DIR, CONFIG_PATH, RAW_DIR
        cfg = yaml.safe_load(CONFIG_PATH.read_text())["movies"]
        manifest = movielens.prepare(RAW_DIR, CLEAN_DIR, cfg)
        print(json.dumps(manifest["outputs"], ensure_ascii=False, indent=1))
        return
    from booksengine.data import pipeline
    pipeline.prepare(force=force)


@app.command()
def refit(min_user: int = typer.Option(5, "--min-user", help="толпа «ценность»: человек — от стольких оценок "
                                        "(у Goodreads в ядре все от 20 — это порог людей Amazon)"),
          taste_goodreads: bool = typer.Option(False, "--taste-goodreads", help="вкус — только на людях Goodreads"),
          taste_only: bool = typer.Option(False, "--taste-only", help="переобучить только вкус"),
          taste_amazon_min: int = typer.Option(None, "--taste-amazon-min",
                                               help="вкус — на людях Goodreads и людях Amazon от стольких книг")) -> None:
    """books-amazon: все компоненты выдачи на единой базе с настройками моделей models/books (ALS, EASE, смесь, вкус,
    толпа «ценность»); после — `layers val`, `layers test`, `calibrate layers`, затем `compare-bases books books-amazon`."""
    from booksengine.data import merged
    from booksengine.paths import CLEAN_DIR, DOMAIN, MODELS_DIR, PROJECT_ROOT, SPLIT_DIR, domain_dirs
    if DOMAIN != "books-amazon":
        raise typer.BadParameter("refit — только для --domain books-amazon")
    merged.refit(CLEAN_DIR, SPLIT_DIR, MODELS_DIR, domain_dirs(PROJECT_ROOT, "books")[1], min_user=min_user,
                 taste_goodreads=taste_goodreads, taste_only=taste_only, taste_amazon_min=taste_amazon_min)


@app.command("compare-bases")
def compare_bases(a: str = typer.Argument("books", help="домен A"), b: str = typer.Argument("books-amazon", help="домен B"),
                  stage: str = typer.Option("test", help="val | test")) -> None:
    """Две базы на одних отложенных людях, каждая своей выдачей: парная разность качества и угаданного →
    reports/compare_<a>_<b>.md."""
    from booksengine.model import layers as ly
    from booksengine.paths import PROJECT_ROOT, REPORTS_DIR, domain_dirs

    def dirs(domain: str):
        data, models = domain_dirs(PROJECT_ROOT, domain)
        return data / "clean", data / "model" / "split", models
    text = ly.report_compare(ly.compare_bases(dirs(a), dirs(b), stage=stage), a, b)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / f"compare_{a}_{b}_{stage}.md").write_text(text)
    print(text)


@app.command("new-books")
def new_books() -> None:
    """Новые книги (единая база, `--domain books-amazon`) на отложенных людях Amazon: список новинок выдачи против
    самых популярных и лучших по оценкам → reports/<domain>/new_books.md."""
    from booksengine.model import new_books as nb
    from booksengine.paths import CLEAN_DIR, EVAL_DIR, MODELS_DIR, REPORTS_DIR, SPLIT_DIR
    text = nb.report(nb.run(clean_dir=CLEAN_DIR, split_dir=SPLIT_DIR, models_dir=MODELS_DIR, eval_dir=EVAL_DIR))
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "new_books.md").write_text(text)
    print(text)


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


@app.command("amazon-bridge")
def amazon_bridge() -> None:
    """Amazon Reviews'23: приём, мост на Goodreads по ISBN/ASIN, сигнал перевода (Wikidata) ->
    reports/amazon_bridge.md. Сырьё — RAW_DIR/amazon_reviews_2023/ (scripts/fetch_amazon_raw.sh)."""
    from booksengine.data import amazon
    from booksengine.paths import AMAZON_CACHE_PATH, AMAZON_CLEAN_DIR, CLEAN_DIR, RAW_DIR
    manifest = amazon.prepare(RAW_DIR, AMAZON_CLEAN_DIR, CLEAN_DIR / "editions.parquet", AMAZON_CACHE_PATH)
    path = amazon.report(manifest, AMAZON_CLEAN_DIR)
    print(f"Отчёт: {path}")


@app.command()
def split(force: bool = typer.Option(False, "--force", help="пересобрать сплит")) -> None:
    """Отложенная выборка: тест и валидация поровну из этапов 20-39/40-79/80-159/160-319/320-999 оценок, вне обучения (data/<domain>/model/split/)."""
    import json

    from booksengine.model import split as s
    from booksengine.paths import CLEAN_DIR, DOMAIN, SPLIT_DIR
    if DOMAIN == "books-amazon":
        raise typer.BadParameter("books-amazon: отложенные люди — копия книжных (`prepare`), чтобы обе базы судились на "
                                 "одних людях; своего сплита нет")
    kw = {}
    if DOMAIN == "movies":
        from booksengine.data import movielens
        kw["bucket_pool_size"] = movielens.bucket_pool_size(CLEAN_DIR / "ratings.parquet")
        data_fp = str((CLEAN_DIR / "ratings.parquet").stat().st_mtime_ns)
    else:
        manifest = json.loads((CLEAN_DIR / "manifest.json").read_text())
        data_fp = manifest["outputs"]["ratings"]["checksum"]
    meta = s.build(CLEAN_DIR / "ratings.parquet", CLEAN_DIR / "users.parquet", SPLIT_DIR,
                   data_fp, force=force, **kw)
    print(json.dumps(meta["groups"], ensure_ascii=False, indent=1))


@app.command()
def evaluate(model: str = typer.Argument(..., help="popularity | als | als_neg | knn | ease | mix"),
             stage: str = typer.Option("val", help="val — перебор настроек; test — замер лучшей и сохранение")) -> None:
    """Метрики модели на валидации или тесте (models/eval/)."""
    from booksengine.model import evaluate as ev
    if stage == "val":
        ev.tune(model)
    elif stage == "test":
        ev.test(model)
    else:
        raise typer.BadParameter("stage: val | test")


@app.command()
def calibrate(model: str = typer.Argument("mix", help="сохранённая модель в models/")) -> None:
    """Шанс «понравится»: учится на валидации, проверяется на тесте → models/<model>/chance.json,
    отчёт — reports/chance_<model>.md."""
    from booksengine.model import chance
    from booksengine.model.evaluate import RATINGS
    from booksengine.paths import EVAL_DIR, MODELS_DIR, REPORTS_DIR, SPLIT_DIR
    out = chance.calibrate(model, ratings_path=RATINGS, split_dir=SPLIT_DIR, models_dir=MODELS_DIR, eval_dir=EVAL_DIR)
    text = chance.report(out)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / f"chance_{model}.md").write_text(text)
    print(text)


@app.command()
def recommend(ratings: str = typer.Option(..., "--ratings", help="CSV: goodreads_work_id, rating 1–5, status, title"),
              top: int = typer.Option(20, "--top", help="сколько книг в каждом списке"),
              one_list: bool = typer.Option(False, "--one-list", help="один общий список вместо двух")) -> None:
    """Оценки из CSV → два списка (художественная литература и нон-фикшн) с шансом «понравится» и объяснением;
    выдача пишется в profiles/history/."""
    from pathlib import Path

    from booksengine import recommend as rec
    from booksengine.paths import CLEAN_DIR, DOMAIN, MODELS_DIR, PROJECT_ROOT, history_dir
    print(rec.format_result(rec.recommend(Path(ratings), clean_dir=CLEAN_DIR, models_dir=MODELS_DIR, top=top,
                                          history_dir=history_dir(PROJECT_ROOT, DOMAIN), sections=not one_list)))


@app.command()
def serve(port: int = typer.Option(5090, "--port", help="порт на localhost; его же ждёт API (Recommender:Url)")) -> None:
    """Выдача для приложения: модели домена в памяти, API спрашивает по HTTP (тот же код, что у `recommend`)."""
    from booksengine import serve as sv
    from booksengine.paths import CLEAN_DIR, DOMAIN, MODELS_DIR
    sv.run(clean_dir=CLEAN_DIR, models_dir=MODELS_DIR, domain=DOMAIN, port=port)


@app.command()
def journal() -> None:
    """Журнал выдач (profiles/history/) против оценок, поставленных позже: что прочитано из советов и как оценено
    → reports/journal.md."""
    from booksengine import journal as jr
    from booksengine.paths import CLEAN_DIR, DOMAIN, PROJECT_ROOT, REPORTS_DIR, history_dir
    text = jr.report(PROJECT_ROOT / "profiles", history_dir(PROJECT_ROOT, DOMAIN), CLEAN_DIR)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "journal.md").write_text(text)
    print(text)


@app.command("ease-size")
def ease_size() -> None:
    """Что теряется из-за границы EASE (30 000 книг): профили, тест, выдача ALS → reports/ease_size.md."""
    from booksengine.model import ease_size as es
    from booksengine.paths import CLEAN_DIR, MODELS_DIR, PROJECT_ROOT, REPORTS_DIR, SPLIT_DIR
    text = es.run(clean_dir=CLEAN_DIR, split_dir=SPLIT_DIR, models_dir=MODELS_DIR, profiles_dir=PROJECT_ROOT / "profiles")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "ease_size.md").write_text(text)
    print(text)


@app.command()
def taste(factors: str = typer.Option(None, help="размеры через запятую, например 64,128"),
          reg: str = typer.Option(None, help="регуляризации через запятую, например 0.02,0.05")) -> None:
    """Модель вкуса: перебор на валидации по личной точности → models/taste, reports/taste_val.md.
    Без параметров — стандартная сетка; с ними — все сочетания, дописываются к прежнему перебору."""
    from booksengine.model import taste as tm
    from booksengine.model.evaluate import RATINGS
    from booksengine.paths import EVAL_DIR, MODELS_DIR, REPORTS_DIR, SPLIT_DIR
    grid = tm.GRID
    if factors or reg:
        fs = [int(v) for v in (factors or "64").split(",")]
        rs = [float(v) for v in (reg or "0.05").split(",")]
        grid = [(f, r) for f in fs for r in rs]
    text = tm.report(tm.tune(ratings_path=RATINGS, split_dir=SPLIT_DIR, models_dir=MODELS_DIR, eval_dir=EVAL_DIR,
                             grid=grid))
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "taste_val.md").write_text(text)
    print(text)


@app.command()
def layers(stage: str = typer.Argument(..., help="val | test | profiles"),
           top: int = typer.Option(20, "--top", help="сколько книг показать в profiles")) -> None:
    """Толпа + вкус → models/layers: val — перебор и выбор, test — один замер, profiles — топ-N рядом с нынешней
    выдачей (смесь)."""
    from booksengine.model import layers as ly
    from booksengine.paths import CLEAN_DIR, EVAL_DIR, MODELS_DIR, PROJECT_ROOT, REPORTS_DIR, SPLIT_DIR
    if stage in ("val", "test"):
        text = ly.report(ly.run(stage, clean_dir=CLEAN_DIR, split_dir=SPLIT_DIR, models_dir=MODELS_DIR,
                                eval_dir=EVAL_DIR))
    elif stage == "profiles":
        text = ly.profiles(clean_dir=CLEAN_DIR, models_dir=MODELS_DIR, profiles_dir=PROJECT_ROOT / "profiles", top=top)
    else:
        raise typer.BadParameter("stage: val | test | profiles")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / f"layers_{stage}.md").write_text(text)
    print(text)


@app.command("ease-like-tune")
def ease_like_tune(lam: float = typer.Option(None, help="одна настройка вместо сетки: λ"),
                   weights: str = typer.Option(None, help="одна настройка: веса 1★…5★ через запятую, например 2,4,8,16,32"),
                   force: bool = typer.Option(False, "--force", help="записать эту настройку, даже если она не лучшая"),
                   min_user: int = typer.Option(20, "--min-user", help="обучать только на людях с ≥ N оценок"),
                   min_support: int = typer.Option(None, "--min-support",
                                                   help="связь книг — только при ≥ N общих читателях (по умолчанию "
                                                        "layers.MIN_SUPPORT)"),
                   amazon: str = typer.Option(None, "--amazon",
                                              help="варианты с людьми Amazon при λ и весах сохранённой толпы: "
                                                   "«шкала-порог» через запятую, шкала raw | q, например q-5,raw-20 "
                                                   "(нужен amazon-bridge)")) -> None:
    """Подбор толпы «ценность» (λ, веса звёзд) на валидации → лучшая в models/ease_like и models/mix_like;
    сетка ~50–70 мин. Сохранённая толпа всегда в сравнении; --lam / --weights — проверить одну настройку против неё
    (без --weights берутся −2/−1/0/1/2). После — `layers val` и `layers test`."""
    from booksengine.model import layers as ly
    from booksengine.paths import CLEAN_DIR, EVAL_DIR, MODELS_DIR, REPORTS_DIR, SPLIT_DIR
    from booksengine.paths import AMAZON_CLEAN_DIR
    grid = ly.LIKE_GRID
    if lam is not None or weights:
        grid = [(lam if lam is not None else 500.0,
                 tuple(float(w) for w in weights.split(",")) if weights else ly.W0)]
    if amazon:
        from booksengine.model.base import read_params
        saved = read_params(MODELS_DIR / "ease_like")
        grid = [(lam if lam is not None else saved["lam"],
                 tuple(float(w) for w in weights.split(",")) if weights else tuple(saved["weights"]), tag.strip())
                for tag in amazon.split(",")]
    text = ly.report_like(ly.tune_like(clean_dir=CLEAN_DIR, split_dir=SPLIT_DIR, models_dir=MODELS_DIR,
                                       eval_dir=EVAL_DIR, grid=grid, force=force and (lam is not None or bool(weights)),
                                       min_user=min_user,
                                       min_support=ly.MIN_SUPPORT if min_support is None else min_support,
                                       amazon_dir=AMAZON_CLEAN_DIR))
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "ease_like_tune.md").write_text(text)
    print(text)


@app.command("profile-check")
def profile_check() -> None:
    """Проверка на своих оценках: каждая книга профиля прячется по очереди — на каком месте её поставила бы выдача."""
    from booksengine.model import layers as ly
    from booksengine.paths import CLEAN_DIR, MODELS_DIR, PROJECT_ROOT, REPORTS_DIR
    text = ly.profile_check(clean_dir=CLEAN_DIR, models_dir=MODELS_DIR, profiles_dir=PROJECT_ROOT / "profiles")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "profile_check.md").write_text(text)
    print(text)


@app.command()
def why(profile: str = typer.Option("my_ratings", help="профиль из profiles/ без .csv"),
        book: str = typer.Argument(..., help="goodreads_work_id или часть названия")) -> None:
    """Почему книга стоит там, где стоит (models/layers): место по частям модели и вклады книг профиля."""
    from booksengine.model import layers as ly
    from booksengine.paths import CLEAN_DIR, MODELS_DIR, PROJECT_ROOT
    print(ly.why(clean_dir=CLEAN_DIR, models_dir=MODELS_DIR, profile_csv=PROJECT_ROOT / "profiles" / f"{profile}.csv",
                 query=book))


@app.command("taste-gap")
def taste_gap() -> None:
    """Личная точность: ставит ли модель понравившиеся книги выше непонравившихся → reports/taste_gap.md."""
    from booksengine.model import taste_gap as tg
    from booksengine.model.evaluate import RATINGS
    from booksengine.paths import EVAL_DIR, MODELS_DIR, REPORTS_DIR, SPLIT_DIR
    text = tg.report(tg.run(ratings_path=RATINGS, split_dir=SPLIT_DIR, models_dir=MODELS_DIR, eval_dir=EVAL_DIR))
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "taste_gap.md").write_text(text)
    print(text)


@app.command("ru-titles")
def ru_titles(top: int = typer.Option(30_000, "--top", help="сколько самых оцениваемых книг ядра переводить")) -> None:
    """Русские названия книг и имена авторов (Fantlab, русские издания Goodreads, новинки Amazon) → data/ru/;
    в БД — `export-app`. Прерывать можно: ответы Fantlab кэшируются."""
    from booksengine.data import ru_titles as rt
    from booksengine.paths import CLEAN_DIR, PROJECT_ROOT, domain_dirs
    rt.build(clean_dir=CLEAN_DIR, goodreads_dir=domain_dirs(PROJECT_ROOT, "books")[0] / "clean",
             out_dir=PROJECT_ROOT / "data" / "ru", top=top)


@app.command("export-app")
def export_app() -> None:
    """Данные приложения → PostgreSQL: новые книги единой базы (`--domain books-amazon`), русские названия
    (`ru-titles`), обложки, слияния теней.
    Выдачу приложению считает `serve`."""
    from booksengine import app_export
    from booksengine.db_load import pg_dsn
    from booksengine.paths import CLEAN_DIR, PROJECT_ROOT, domain_dirs
    app_export.run(clean_dir=CLEAN_DIR, goodreads_dir=domain_dirs(PROJECT_ROOT, "books")[0] / "clean",
                   ru_dir=PROJECT_ROOT / "data" / "ru", dsn=pg_dsn())

@app.command("report-3a")
def report_3a() -> None:
    """Сравнение моделей стенда `evaluate` на тесте → reports/stage3a_report.md (из models/eval/*.json)."""
    from booksengine import report_3a as r
    print(r.write())


exp_app = typer.Typer(help="Сравнение вариантов CF-ядра (правила очистки, пороги) на общем тесте")
app.add_typer(exp_app, name="exp")


def _exp_cores() -> dict:
    """Ядра сравнения: сохранённые папки data/exp/<имя>/ и «new» — ссылки на текущий data/clean."""
    from booksengine.paths import CLEAN_DIR, EXP_DIR
    new = EXP_DIR / "new"
    new.mkdir(parents=True, exist_ok=True)
    for f in ("ratings.parquet", "works.parquet", "work_merges.parquet", "manifest.json"):
        (new / f).unlink(missing_ok=True)
        (new / f).symlink_to(CLEAN_DIR / f)
    return {d.name: d for d in sorted(EXP_DIR.iterdir()) if d.is_dir() and d.name != "common"}


@exp_app.command("save")
def exp_save(name: str = typer.Argument(..., help="имя ядра, например old или k50")) -> None:
    """Скопировать текущее ядро (data/clean) в data/exp/<name>/ — до `prepare --force` с другими правилами."""
    from booksengine.model import experiment as ex
    from booksengine.paths import CLEAN_DIR, EXP_DIR
    if name in ("new", "common"):
        raise typer.BadParameter("имена new и common заняты")
    if (EXP_DIR / name / "ratings.parquet").exists():
        raise typer.BadParameter(f"data/exp/{name}/ уже есть — сохранённое ядро не перезаписывается")
    ex.save_core(CLEAN_DIR, EXP_DIR / name)


@exp_app.command("split")
def exp_split() -> None:
    """Общий тест всех ядер сравнения (data/exp/common/)."""
    from booksengine.model import experiment as ex
    from booksengine.paths import EXP_DIR, SPLIT_DIR
    print(ex.build_common_split(_exp_cores(), SPLIT_DIR, EXP_DIR / "common"))


@exp_app.command("run")
def exp_run(core: str = typer.Option(None, help="одно ядро; по умолчанию все")) -> None:
    """Популярность, ALS, kNN с настройками `experiment.FIXED` на общем тесте (models/eval/exp/<core>.json).
    Модель ядра, уже лежащая в models/exp/<core>/, загружается, а не обучается заново."""
    from booksengine.model import experiment as ex
    from booksengine.paths import EXP_DIR, EXP_EVAL_DIR, EXP_MODELS_DIR, PROJECT_ROOT, SPLIT_DIR
    for name in [core] if core else list(_exp_cores()):
        ex.run_core(name, ratings_path=EXP_DIR / name / "ratings.parquet", works_path=EXP_DIR / name / "works.parquet",
                    common_dir=EXP_DIR / "common" / name, holdout_path=SPLIT_DIR / "holdout_users.parquet",
                    profile_path=PROJECT_ROOT / "profiles" / "my_ratings.csv",
                    model_dir=EXP_MODELS_DIR / name, eval_dir=EXP_EVAL_DIR)


@exp_app.command("report")
def exp_report() -> None:
    """reports/cleanup_experiment.md."""
    from booksengine import report_exp
    print(report_exp.write())


if __name__ == "__main__":
    app()
