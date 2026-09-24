import typer

app = typer.Typer(help="BooksEngine: офлайн-часть рекомендательной системы книг", no_args_is_help=True)


@app.command()
def prepare(force: bool = typer.Option(False, "--force", help="пересобрать всё, игнорируя кэш")) -> None:
    """Staging → профилирование → очистка → валидация → отчёт (reports/stage1_report.md)."""
    from booksengine.data import pipeline
    pipeline.prepare(force=force)


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


@app.command()
def split(force: bool = typer.Option(False, "--force", help="пересобрать сплит")) -> None:
    """Отложенная выборка: тест и валидация из групп 20-49/50-199/200+, вне обучения (data/model/split/)."""
    import json

    from booksengine.model import split as s
    from booksengine.paths import CLEAN_DIR, SPLIT_DIR
    manifest = json.loads((CLEAN_DIR / "manifest.json").read_text())
    meta = s.build(CLEAN_DIR / "ratings.parquet", CLEAN_DIR / "users.parquet", SPLIT_DIR,
                   manifest["outputs"]["ratings"]["checksum"], force=force)
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
    """Шанс «понравится» (п. 29): учится на валидации, проверяется на тесте → models/<model>/chance.json."""
    import json

    from booksengine.model import chance
    from booksengine.model.evaluate import RATINGS
    from booksengine.paths import EVAL_DIR, MODELS_DIR, SPLIT_DIR
    out = chance.calibrate(model, ratings_path=RATINGS, split_dir=SPLIT_DIR, models_dir=MODELS_DIR, eval_dir=EVAL_DIR)
    print(json.dumps({k: out[k] for k in ("chance", "test", "reliability")}, ensure_ascii=False, indent=1))


@app.command()
def recommend(ratings: str = typer.Option(..., "--ratings", help="CSV: goodreads_work_id, rating 1–5, status, title"),
              top: int = typer.Option(20, "--top", help="сколько книг показать")) -> None:
    """Оценки из CSV → рекомендации смеси с шансом «понравится» и объяснением (п. 11)."""
    from pathlib import Path

    from booksengine import recommend as rec
    from booksengine.paths import CLEAN_DIR, MODELS_DIR
    print(rec.format_result(rec.recommend(Path(ratings), clean_dir=CLEAN_DIR, models_dir=MODELS_DIR, top=top)))


@app.command("ease-size")
def ease_size() -> None:
    """Что теряется из-за границы EASE (30 000 книг): профили, тест, выдача ALS → reports/ease_size.md."""
    from booksengine.model import ease_size as es
    from booksengine.paths import CLEAN_DIR, MODELS_DIR, PROJECT_ROOT, REPORTS_DIR, SPLIT_DIR
    text = es.run(clean_dir=CLEAN_DIR, split_dir=SPLIT_DIR, models_dir=MODELS_DIR, profiles_dir=PROJECT_ROOT / "profiles")
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "ease_size.md").write_text(text)
    print(text)


@app.command()
def taste(factors: str = typer.Option(None, help="размеры через запятую, например 64,128"),
          reg: str = typer.Option(None, help="регуляризации через запятую, например 0.02,0.05")) -> None:
    """Модель вкуса (п. 37, шаг 1): перебор на валидации по личной точности → models/taste, reports/taste_val.md.
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
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "taste_val.md").write_text(text)
    print(text)


@app.command()
def layers(stage: str = typer.Argument(..., help="val | test | profiles"),
           top: int = typer.Option(20, "--top", help="сколько книг показать в profiles")) -> None:
    """Толпа + вкус (п. 37, шаг 2): val — перебор и выбор, test — один замер, profiles — топ-20 рядом с нынешним."""
    from booksengine.model import layers as ly
    from booksengine.paths import CLEAN_DIR, EVAL_DIR, MODELS_DIR, PROJECT_ROOT, REPORTS_DIR, SPLIT_DIR
    if stage in ("val", "test"):
        text = ly.report(ly.run(stage, clean_dir=CLEAN_DIR, split_dir=SPLIT_DIR, models_dir=MODELS_DIR,
                                eval_dir=EVAL_DIR))
    elif stage == "profiles":
        text = ly.profiles(clean_dir=CLEAN_DIR, models_dir=MODELS_DIR, profiles_dir=PROJECT_ROOT / "profiles", top=top)
    else:
        raise typer.BadParameter("stage: val | test | profiles")
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / f"layers_{stage}.md").write_text(text)
    print(text)


@app.command("ease-like")
def ease_like(lam: float = typer.Option(500.0, help="регуляризация λ, как у EASE смеси")) -> None:
    """П. 38: EASE с целью «оценка − 3» → models/ease_like и смесь с ALS models/mix_like; дальше `layers val`."""
    import time

    from booksengine.model.ease import EASELike
    from booksengine.model.evaluate import RATINGS
    from booksengine.model.matrix import load_train
    from booksengine.model.mix import Mix
    from booksengine.paths import MODELS_DIR, SPLIT_DIR
    train = load_train(RATINGS, SPLIT_DIR / "holdout_users.parquet")
    t0 = time.perf_counter()
    m = EASELike(lam=lam)
    m.fit(train)
    m.save(MODELS_DIR / "ease_like")
    mix = Mix(MODELS_DIR / "als_neg", MODELS_DIR / "ease_like")
    mix.fit(train)
    mix.save(MODELS_DIR / "mix_like")
    print(f"EASE «ценность» λ = {lam}: обучение {time.perf_counter() - t0:.0f} с → models/ease_like, models/mix_like")


@app.command("ease-like-tune")
def ease_like_tune() -> None:
    """П. 38: подбор толпы «ценность» (λ, веса звёзд) на валидации → лучшая в models/ease_like; ~40–60 мин."""
    from booksengine.model import layers as ly
    from booksengine.paths import CLEAN_DIR, EVAL_DIR, MODELS_DIR, REPORTS_DIR, SPLIT_DIR
    text = ly.report_like(ly.tune_like(clean_dir=CLEAN_DIR, split_dir=SPLIT_DIR, models_dir=MODELS_DIR,
                                       eval_dir=EVAL_DIR))
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "ease_like_tune.md").write_text(text)
    print(text)


@app.command("taste-gap")
def taste_gap() -> None:
    """Личная точность: ставит ли модель понравившиеся книги выше непонравившихся (п. 37) → reports/taste_gap.md."""
    from booksengine.model import taste_gap as tg
    from booksengine.model.evaluate import RATINGS
    from booksengine.paths import EVAL_DIR, MODELS_DIR, REPORTS_DIR, SPLIT_DIR
    text = tg.report(tg.run(ratings_path=RATINGS, split_dir=SPLIT_DIR, models_dir=MODELS_DIR, eval_dir=EVAL_DIR))
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "taste_gap.md").write_text(text)
    print(text)


@app.command("export-model")
def export_model() -> None:
    """Смесь models/mix → PostgreSQL для C# API: перезаписывает модель целиком (веб-интерфейс)."""
    from booksengine import export_model as ex
    from booksengine.db_load import pg_dsn
    from booksengine.paths import CLEAN_DIR, MODELS_DIR, PROJECT_ROOT, TMP_DIR
    ex.run(clean_dir=CLEAN_DIR, models_dir=MODELS_DIR, profiles_dir=PROJECT_ROOT / "profiles",
           tmp_dir=TMP_DIR / "export", dsn=pg_dsn())


@app.command("report-3a")
def report_3a() -> None:
    """Отчёт этапа 3a (reports/stage3a_report.md) из models/eval/*.json."""
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
    """Популярность, ALS, kNN с настройками 3a на общем тесте (models/eval/exp/<core>.json).
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
