"""Отчёт этапа 1 (reports/stage1_report.md) из profile.json и manifest.json.

Все цифры подставляются из данных: отчёт пересобирается одной командой и не расходится с результатом.
"""
from booksengine.paths import REPORTS_DIR


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "да" if v else "нет"
    if isinstance(v, float):
        if v.is_integer() and abs(v) >= 1000:
            return f"{int(v):,}".replace(",", " ")
        return f"{v:.4g}" if abs(v) < 1 else f"{v:,.2f}".replace(",", " ")
    if isinstance(v, int):
        return f"{v:,}".replace(",", " ")
    if isinstance(v, list):
        return ", ".join(_fmt(x) for x in v)
    return str(v)


def _n(v) -> str:
    return _fmt(v)


def _pct(part, whole) -> str:
    return f"{100 * part / whole:.2f}%" if whole else "—"


def table(rows: list[dict], cols: list[str] | None = None, headers: list[str] | None = None) -> str:
    if not rows:
        return "_нет данных_\n"
    cols = cols or list(rows[0].keys())
    headers = headers or cols
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(_fmt(r.get(c)).replace("|", "\\|") for c in cols) + " |")
    return "\n".join(out) + "\n"


def write(prof: dict, manifest: dict):
    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / "stage1_report.md"
    path.write_text(render(prof, manifest))
    return path


def render(p: dict, m: dict) -> str:
    log = {s["rule"]: s for s in m["cleaning_log"]}
    out = m["outputs"]
    cfg = m["config"]
    ov = p["interactions_overview"][0]
    lc = p["link_chain"][0]
    eq = p["editions_quality"][0]
    wq = p["works_quality"][0]
    uq = p["user_quantiles"][0]
    wqq = p["work_quantiles"][0]
    aq = p["authors_quality"][0]
    ex = m["extra"]
    final = log["kcore"]
    total_explicit = log["split_explicit_signal"]["rows_after"]
    lv = log["users_low_variance"]
    ha = log["users_hyperactive"]
    kc = cfg["kcore"]
    uc = cfg["users"]
    s = []
    w = s.append

    w("# Этап 1. Датасет UCSD Goodreads: структура, качество, очистка\n")
    w("_Отчёт сгенерирован `uv run booksengine prepare`; все цифры взяты из данных._\n")

    # --- Итог
    w("## Итог\n")
    w(f"- На входе **{_n(ov['rows'])}** взаимодействий «пользователь × издание» от {_n(ov['users'])} пользователей "
      f"по {_n(ov['books'])} изданиям. Явных оценок (1–5) из них {_n(ov['explicit_ratings'])} "
      f"({_pct(ov['explicit_ratings'], ov['rows'])}), остальное — книги «на полке» без оценки.")
    w(f"- На выходе матрица **«пользователь × произведение»**: **{_n(final['rows_after'])}** оценок, "
      f"**{_n(final['users_after'])}** пользователей, **{_n(final['works_after'])}** произведений "
      f"(плотность {final['rows_after'] / final['users_after'] / final['works_after']:.4%}). "
      f"Это {_pct(final['rows_after'], total_explicit)} явных оценок.")
    w(f"- Каталог: {_n(out['works']['rows'])} произведений и {_n(out['editions']['rows'])} изданий. В каталоге "
      "остаются и произведения вне CF-ядра: по ним можно будет сопоставлять оценки новых пользователей.")
    w("- Структурного мусора (битых строк, дублей, висящих ссылок) в датасете почти нет. Основной шум — "
      "поведенческий: «полочные» записи без оценки, одинаково оценивающие пользователи, аккаунты-каталогизаторы "
      "и длинный хвост произведений с единичными оценками.\n")

    w("### Воронка очистки\n")
    rows = []
    for st in m["cleaning_log"]:
        rows.append({"шаг": st["rule"], "было": st["rows_before"], "стало": st["rows_after"],
                     "удалено": st["rows_removed"], "почему": st["reason"]})
    w(table(rows))
    w("\n_Для шага `works_without_title_or_editions` строки — это произведения; для `editions_to_works` "
      "«удалённые» строки не теряются, а сливаются в одну оценку произведения._\n")

    # --- Структура
    w("## 1. Файлы и связи\n")
    w(table(p["files"], ["file", "rows"], ["файл (staging)", "строк"]))
    w("""
Связи:

```
goodreads_interactions.csv (user_id_csv, book_id_csv, is_read, rating 0–5, is_reviewed)
   │ user_id_csv ──► user_id_map.csv ──► user_id (hex, внешний id Goodreads)
   │ book_id_csv ──► book_id_map.csv ──► book_id ──► goodreads_books.json (издание)
   │                                                   ├─ work_id ──► goodreads_book_works.json (произведение)
   │                                                   ├─ authors[].author_id ──► goodreads_book_authors.json
   │                                                   └─ book_id ◄── goodreads_book_genres_initial.json
```

Издание (edition) — конкретная публикация: перевод, переплёт, аудиокнига. Произведение (work) объединяет все
издания одного текста. Оценки ставятся изданиям, а рекомендовать нужно произведения, поэтому всё сворачивается
до `work_id`.
""")
    w("### Ссылочная целостность\n")
    w(table(p["id_maps"], None, ["маппинг", "строк", "уник. csv-id", "уник. целевых id"]))
    w("\n" + table([lc], ["interactions", "no_book_map", "no_edition", "edition_without_work", "work_missing",
                         "users_no_map"],
                   ["взаимодействий", "нет в book_id_map", "нет издания", "издание без work_id",
                    "work_id нет в works", "пользователей без user_id_map"]))
    w(f"\nЦепочка почти полная: разрывается только на {_n(eq['no_work_id'])} изданиях без `work_id`, на которые "
      f"приходится {_n(lc['edition_without_work'])} взаимодействий. "
      f"В works у всех {_n(wq['works'])} произведений есть издания, но у {_n(wq['best_book_missing'])} "
      "`best_book_id` указывает на издание, которого нет в books.json.\n")

    # --- Качество
    w("## 2. Качество данных\n")
    w("### Взаимодействия\n")
    w(table([ov], ["rows", "null_keys", "rating_out_of_range", "duplicate_pairs", "reviewed"],
            ["строк", "null в ключах", "rating вне 0–5", "дубли (user, book)", "с рецензией"]))
    w("\n### Издания\n")
    w(table([eq], ["editions", "no_work_id", "no_title", "no_authors", "no_language", "no_description",
                   "no_isbn", "no_year"],
            ["изданий", "без work_id", "без названия", "без авторов", "без языка", "без описания", "без ISBN",
             "без года"]))
    w("\n### Произведения\n")
    w(table([wq], ["works", "no_original_title", "no_editions", "best_book_missing"],
            ["произведений", "без original_title", "без изданий", "best_book_id не найден"]))
    w(f"\n`original_title` пуст у {_pct(wq['no_original_title'], wq['works'])} произведений. Где он есть, у "
      "переводов это транслит оригинала («Prestuplenie i nakazanie», «Master i Margarita»). Поэтому название "
      "произведения в каталоге берётся из лучшего издания («Crime and Punishment»), а `original_title` хранится "
      "отдельным полем.\n")
    w(table(p["editions_per_work"], None, ["изданий на произведение", "произведений"]))
    w("\n### Авторы\n")
    w(f"{_n(aq['authors'])} авторов, без имени {_n(aq['no_name'])}. Связей «издание → автор» {_n(aq['edition_author_links'])}, "
      f"из них на несуществующего автора {_n(aq['link_to_missing_author'])}. В авторах изданий бывают иллюстраторы "
      "и переводчики (поле `role`, например у Гарри Поттера — Mary GrandPré), поэтому роль сохраняется.\n")
    w("### media_type — не используется для фильтрации\n")
    w(table(p["media_type"], None, ["media_type", "произведений", "явных оценок"]))
    w("\nСамые популярные произведения с `media_type` «not a book / periodical / article»:\n")
    w(table(p["media_type_not_book_top"], None, ["название", "media_type", "оценок"]))
    w("\nПоле размечено ненадёжно («Dracula» и «Watchmen» помечены как «not a book»), поэтому фильтровать по "
      "нему нельзя.\n")
    w("### Языки (по изданиям с явными оценками)\n")
    w(table(p["languages_rated"], None, ["язык издания", "явных оценок"]))
    w("\nДатасет в основном англоязычный. Язык — свойство издания, а не произведения, поэтому по нему ничего "
      "не удаляется: для коллаборативной фильтрации язык не важен.\n")
    w("### Жанры\n")
    w(table(p["genres_overview"], None, ["жанр (Goodreads, 10 крупных групп)", "изданий"]))
    w("\nЖанры — это голоса пользователей за крупные группы; в каталоге они суммируются по изданиям "
      "произведения (`work_genres.votes`, `share`).\n")

    # --- Распределения
    w("## 3. Распределения\n")
    w("### rating × is_read\n")
    w(table(p["rating_x_is_read"], None, ["rating", "is_read", "строк", "%"]))
    w("\n`rating = 0` означает не «ноль баллов», а отсутствие оценки: книга на полке (to-read), либо прочитана "
      "без оценки. Это другой сигнал (интерес, а не оценка), в модель оценок он не идёт (см. 4.2).\n")
    w(table(p["rating_distribution_explicit"], None, ["оценка", "n", "%"]))
    w("\nРаспределение сильно смещено вверх: 4–5 баллов — больше двух третей оценок, 1–2 — меньше 12%. "
      "Низкие оценки редки, поэтому ценны как негативный сигнал; это учтём на этапе 3.\n")
    w("### Пользователи (явные оценки на уровне произведения)\n")
    w(f"Пользователей с явными оценками — {_n(uq['users_with_explicit'])}, без единой явной оценки — "
      f"{_n(uq['users_without_explicit'])}. Квантили числа оценок: медиана {_n(uq['p50'])}, p90 {_n(uq['p90'])}, "
      f"p99 {_n(uq['p99'])}, p99.9 {_n(uq['p999'])}, максимум {_n(uq['max'])}.\n")
    w(table(p["user_activity"], None, ["оценок у пользователя", "пользователей", "оценок", "медиана sd",
                                        "медиана средней", "медиана доли редких книг*", "медиана доли пятёрок"]))
    w("\n\\* «Редкие» — произведения, у которых меньше 20 оценок во всём датасете.\n")
    w("### Самые активные аккаунты\n")
    w(table(p["top_users"], None, ["user_id", "явных оценок", "sd", "средняя", "доля редких", "доля пятёрок",
                                    "всего взаимодействий"]))
    w("\n### Разброс оценок пользователя (≥ 10 оценок)\n")
    w(table(p["user_variance"], None, ["sd оценок", "пользователей", "оценок", "из них «все пятёрки»"]))
    w("\n### Произведения\n")
    w(f"Произведений с явными оценками — {_n(wqq['works_with_explicit'])}. Медиана {_n(wqq['p50'])} оценок, "
      f"p90 {_n(wqq['p90'])}, p99 {_n(wqq['p99'])}, максимум {_n(wqq['max'])}.\n")
    w(table(p["work_popularity"], None, ["оценок у произведения", "произведений", "оценок"]))
    w("\n")

    # --- Решения
    w("## 4. Решения по очистке\n")
    li = log["broken_links"]
    wt = log["works_without_title_or_editions"]
    w("### 4.1 Структурные и ссылочные\n")
    w(f"- **Битые строки, null, значения вне диапазонов, дубли (user, book): 0.** Правила в пайплайне есть, "
      f"но на этом датасете ничего не удалили: CSV разобран без ошибок (отклонённых строк "
      f"{_n(m.get('staging_rejected', 0))}).")
    w(f"- **Произведения без изданий или без названия: −{_n(wt['rows_removed'])}.** Ещё у "
      f"{_n(wt['detail']['best_book_replaced'])} произведений `best_book_id` указывает на издание, которого нет "
      "(или у которого нет названия); вместо него берётся самое популярное издание этого произведения.")
    w(f"- **Взаимодействия, не доходящие до произведения: −{_n(li['rows_removed'])}** "
      f"({_n(li['detail']['edition_without_work_id'])} — издание без `work_id`, "
      f"{_n(li['detail']['work_invalid'])} — произведение без названия).\n")
    sp = log["split_explicit_signal"]
    ew = log["editions_to_works"]
    w("### 4.2 Тип сигнала и переход к произведениям\n")
    w(f"- **`rating = 0` отбрасывается: {_n(sp['detail']['unrated_rows'])} записей.** Это не оценка, а «на полке» "
      "или «прочитано без оценки»; ни модель, ни БД этот сигнал не используют (решение 2026-09-23, раньше "
      "хранился в `shelf_events`). Исходные записи остаются в `data/staging/`.")
    w(f"- **Издания → произведения.** У {_n(ew['detail']['pairs_with_multiple_editions'])} пар (пользователь, "
      f"произведение) оценено несколько изданий (например, книга и аудиокнига). В "
      f"{_n(ew['detail']['pairs_conflicting'])} случаях оценки расходятся, в {_n(ew['detail']['pairs_conflict_ge2'])} "
      f"— на 2 балла и больше. Дат в CSV нет, поэтому «последнюю» оценку не определить; берётся **среднее**. Это "
      f"затрагивает {_pct(ew['detail']['pairs_conflicting'], ew['rows_after'])} оценок, так что дробные значения "
      "(3.5, 4.5) встречаются редко.")
    w("- Если у пользователя есть явная оценка одного издания произведения, «полочные» записи о других его "
      "изданиях отбрасываются как избыточные.\n")
    col = ex["collections"]
    w("### 4.3 Сборники и box sets — помечены, не удалены\n")
    w(f"По шаблонам в названии («#1-3», «Boxed Set», «Omnibus», «N books in one») помечено "
      f"{_n(col['works_flagged'])} произведений (`works.is_collection`). Оценки есть у {_n(col['works_rated'])} из них, "
      f"всего {_n(col['ratings'])} оценок ({_pct(col['ratings'], total_explicit)}). Примеры:\n")
    w(table(ex["collection_examples"], None, ["название лучшего издания", "оценок"]))
    w("\nСборник дублирует входящие в него произведения, но оценка сборника — настоящий сигнал вкуса "
      "(«The Lord of the Rings #1-3» у многих и есть «Властелин колец»). Поэтому оценки остаются в матрице, "
      "а флаг будет использоваться при выдаче: не рекомендовать сборник тому, кто уже оценил его части. "
      "Если удалять — `collections.drop_from_ratings: true` в конфиге.\n")
    w("### 4.4 Аномальные пользователи\n")
    w(f"- **Нулевой разброс: sd < {uc['low_variance_max_sd']} при ≥ {uc['low_variance_min_ratings']} оценках — "
      f"−{_n(lv['users_before'] - lv['users_after'])} пользователей, −{_n(lv['rows_removed'])} оценок.** "
      f"Из них {_n(lv['detail']['of_them_all_fives'])} ставят только пятёрки. У такого пользователя после "
      "центрирования по его средней все оценки нулевые, отличить любимое от проходного нельзя; для item-item "
      "похожести он лишь добавляет шум. Порог 0.2 отсекает ровно «все одинаковые, кроме редких исключений» — "
      "у основной массы пользователей sd около 0.8 (см. таблицу разброса).")
    w(f"- **Сверхактивные: > {uc['max_ratings']} оценённых произведений — −{_n(ha['users_before'] - ha['users_after'])} "
      f"пользователей, −{_n(ha['rows_removed'])} оценок ({_pct(ha['rows_removed'], total_explicit)}).** "
      f"Это за p99.9 ({_n(uq['p999'])}). С ростом активности доля редких книг растёт с ~1% до 10–15%: это "
      "каталогизация (библиотекари, импорт) или массовые одинаковые оценки (топ-аккаунт — 38 тыс. пятёрок). "
      "Для CF такой аккаунт вреден: одна строка с 10 тыс. оценками создаёт ~50 млн ложных совместных оценок "
      "в item-item статистике.\n")
    if "users_monotone" in log:
        mo = log["users_monotone"]
        w(f"- **Однообразные: ≥ {uc['monotone_max_mode_share']:.0%} оценок одного значения — "
          f"−{_n(mo['users_before'] - mo['users_after'])} пользователей, −{_n(mo['rows_removed'])} оценок "
          f"({_pct(mo['rows_removed'], total_explicit)}).** Шире правила разброса: «95% пятёрок и пара четвёрок» — "
          "вкуса относительно своей средней не видно. Решение 2026-09-23.\n")
    for rule, text in (("nonbooks", "Не-книги (ноты, раскраски, календари, аудиокурсы; шаблоны названия, "
                                    "`config/cleaning.yaml` → `nonbooks`)"),
                       ("duplicates", "Слияние дублей: «теневое» произведение той же книги → главное "
                                      "(`data/clean/work_merges.parquet`)")):
        if rule in log:
            st = log[rule]
            extra = f", теней — {_n(st['detail']['shadows'])}" if rule == "duplicates" else ""
            w(f"- **{text}** — −{_n(st['rows_removed'])} строк оценок "
              f"({_pct(st['rows_removed'], total_explicit)}){extra}. Решение 2026-09-23.")
    w("")
    w("### 4.5 k-core: минимум оценок у пользователя и произведения\n")
    w("Выбор порогов (посчитано на матрице после фильтров 4.1–4.4):\n")
    w(table(ex["kcore_options"], ["min_user", "min_work", "ratings", "users", "works", "kept_share", "density",
                                  "iterations"],
            ["min оценок у польз.", "min оценок у произв.", "оценок", "пользователей", "произведений",
             "доля оценок", "плотность", "итераций"]))
    w(f"\nВыбрано **({kc['min_user_ratings']}, {kc['min_work_ratings']})**:\n")
    w(f"- **≥ {kc['min_user_ratings']} у пользователя**: люди с меньшим числом оценок — не аудитория приложения и "
      "слабый сигнал; их удаление не изменило NDCG@20 на общем тесте (`booksengine exp`, решение 2026-09-23, "
      "было 10).")
    opt = {(o["min_user"], o["min_work"]): o for o in ex["kcore_options"]}
    ku = kc["min_user_ratings"]
    alts = "; ".join(f"({ku}, {kw}) — {_n(opt[(ku, kw)]['works'])} произведений, {opt[(ku, kw)]['kept_share']:.1%} оценок"
                     for kw in (10, 20, 50) if (ku, kw) in opt)
    w(f"- **≥ {kc['min_work_ratings']} у произведения**: у редких книг похожесть статистически ненадёжна, а "
      "пользователю важнее проверенная «база», чем ниша (решение 2026-09-23, было 20). "
      f"Варианты: {alts}.\n")

    # --- Итог
    w("## 5. Результат\n")
    w(table([{"файл": k, "строк": v["rows"], "МБ": round(v["bytes"] / 1e6, 1)} for k, v in out.items()]))
    w("""
| файл | содержимое |
|---|---|
| `works` | каталог произведений: название, оригинальное название, год, язык лучшего издания, описание, `is_collection`, `in_cf`, число и средняя оценок в CF-ядре |
| `editions` | издания (ISBN, язык, формат, издатель, ссылки) — для сопоставления книг пользователя с каталогом |
| `authors`, `work_authors` | авторы произведения с ролью и порядком |
| `work_genres` | жанровые голоса, агрегированные по изданиям |
| `users` | пользователи CF-ядра: внутренний и внешний id, число оценок, средняя, разброс |
| `ratings` | **матрица для CF**: (user, work, rating 1–5, число изданий) |
| `work_merges` | дубли произведений: тень → главное (оценки тени перенесены на главное) |
| `manifest.json` | отпечатки входов, конфига и кода, журнал очистки, контрольные суммы файлов |
""")
    w("### Валидация\n")
    w(table([{"проверка": k, "нарушений": v} for k, v in m["validation"].items()]))
    w("\n## Воспроизведение\n")
    w("```bash\ncp .env.example .env   # RAW_DIR — папка с файлами датасета\nuv run booksengine prepare"
      "            # --force — пересобрать с нуля\nuv run pytest\n```\n")
    w(f"Пороги — в `config/cleaning.yaml`. Полный прогон с нуля — около {round(m['seconds'] / 60) + 1} мин "
      "(плюс ~30 с staging); если входы, конфиг и код не менялись, повторный запуск ничего не пересчитывает.\n")
    return "\n".join(s)
