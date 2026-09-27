# Фильмы, шаг 1: работают ли толпа и вкус на MovieLens

Первый шаг плана «Фильмы» из `TODO.md` — проверка, даёт ли модель вкуса выигрыш на втором домене
(MovieLens 32M), прежде чем делать домен в CLI, TMDB и остальное. Датасет уже скачан и распакован —
`~/Downloads/ml-32m/{ratings.csv, movies.csv, links.csv, tags.csv}`.

## Что уже верно, что уточнено разведкой

`ease.py`, `taste.py`, `layers.py`, `split.py`, `metrics.py` работают только с матрицей «человек × произведение
× оценка» и не зависят от книжной схемы — подтверждено чтением кода (`matrix.load_train`, `split.build` читают
только `user_id, work_id, rating` и `user_id, external_id`).

Уточнение, которого не было в TODO: `layers.run` (и `val`, и `test`) безусловно вызывает
`filters.work_info(clean_dir, work_ids)`, а тот читает `works.parquet`, `work_authors.parquet`, `authors.parquet`
(колонки `title`, `original_title`, `best_edition_title`, `is_collection`, авторы). Без них замер падает даже на
шаге 1. Отдельно «выключать» правила списка (`ListPicker`) не нужно: если авторов и коллекций нет (данные пустые/
тривиальные), правила сами не делают ничего — «один автор не больше раза на 10 мест» не сработает, если автор
везде неизвестен; сборников и серий в названиях фильмов MovieLens тоже нет.

## Данные и очистка

Сырьё: `<RAW_DIR>/ml-32m/ratings.csv` (userId, movieId, rating 0.5–5.0, timestamp), `movies.csv` (movieId, title,
genres). `links.csv` пока не читаем — он для TMDB (шаг 3 плана).

- Оценки уже все явные (в отличие от Goodreads, нет «на полке» / `rating = 0`) — фильтровать нечего.
- Округление: `rating = ceil(raw_rating)` — полузвёзды 0.5–5.0 → целые 1–5 вверх (решено в TODO).
- Очистка пользователей — переиспользуем `clean.filter_users` без изменений кода. Пороги — новый раздел
  `movies:` в `config/cleaning.yaml`, копия стартовых значений `users:` (`low_variance_max_sd: 0.2`,
  `low_variance_min_ratings: 10`, `monotone_max_mode_share: 0.9`, `max_ratings: 3000`), но отдельно настраиваемый:
  у MovieLens пользователи часто ставят оценки пачкой при регистрации (по памяти), поведение может отличаться от
  Goodreads, и тюнинг одного домена не должен задевать пороги другого.
- k-core — переиспользуем `clean.apply_kcore(min_user=20, min_work=100)`. Оба порога фиксированы решением из
  TODO (не перебираются): человек ≥ 20 в MovieLens и так почти everyone, фильм ≥ 100 даёт ~10–15 тыс. фильмов.

## Экспорт → `data/movies/clean/`

Та же схема, что читает модель книг (позволяет не трогать код модели):

| файл | колонки | источник |
|---|---|---|
| `ratings.parquet` | `user_id` (=userId), `work_id` (=movieId), `rating` | ratings.csv после округления и очистки |
| `users.parquet` | `user_id`, `external_id` (=userId) | нужен `split.build` |
| `works.parquet` | `work_id`, `title` (=название MovieLens), `original_title` = NULL, `best_edition_title` = NULL, `is_collection` = false | movies.csv |
| `work_authors.parquet` | `work_id`, `author_id`, `role`, `position` — 0 строк | режиссёра нет до TMDB (шаг 3) |
| `authors.parquet` | `author_id`, `name` — 0 строк | — |

## Прогон

Разовый скрипт `scripts/movies_check.py` (не часть `booksengine` CLI — домен в CLI появится на шаге 2 плана и
заменит это ad hoc-связывание путей; постоянного места в CLI это не заслуживает раньше времени):

1. Загрузка `ratings.csv` + `movies.csv` → очистка (см. выше) → экспорт в `data/movies/clean/`.
2. `split.build(ratings_path, users_path, data/movies/model/split, ...)` — без изменений кода.
3. `taste.tune(ratings_path=..., split_dir=..., models_dir=data/../models/movies, eval_dir=...)`.
4. `layers.tune_like(clean_dir=..., split_dir=..., models_dir=models/movies, eval_dir=...)` (это и есть
   `ease-like-tune`).
5. `layers.run("val", ...)`, затем `layers.run("test", ...)`.

Модели — `models/movies/{taste, ease_like, mix_like, layers}`, отчёты — `reports/movies_{taste_val,
ease_like_tune, layers_val, layers_test}.md`, теми же функциями `report()` / `report_like()`, что у книг.

Долгие шаги (весь прогон — split на 32М строк, обучение ALS/EASE) — в фоне, лог в `reports/` (правило проекта
для долгих прогонов на Mac пользователя).

## Тесты

Юнит-тесты на загрузчик — округление половин звёзд, форма экспортированных таблиц (в т.ч. что
`work_authors`/`authors` пустые, но с правильными колонками) — на маленьком синтетическом CSV в `tmp_path`, по
конвенции репозитория (тесты не используют реальные пути). Полный прогон на 32М строк — вручную, не в pytest.

## Решение по итогу

Смотрим, как для книг: выигрыш вкуса по этапам (`120–319` и т.д. — сравнение `layers val`/`layers test`) и
колонку «известность». TODO уже предупреждает, что MovieLens смещена к известным фильмам (пачечные оценки при
регистрации) — «известность» может быть завышена относительно книжного случая, это не повод отклонять результат
сходу, но повод не удивляться, если она выше книжной. Если вкус не даёт выигрыша (интервал разницы не выше
нуля) — дальше (домен в CLI, TMDB, правила списка, профиль) не делаем, разбираемся, в чём отличие от книг.
