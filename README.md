# BooksEngine

Рекомендации книг на основе коллаборативной фильтрации. Источник данных —
[UCSD Goodreads Book Graph](https://mengtingwan.github.io/data/goodreads.html)
(228M взаимодействий, 876K пользователей, 2.36M изданий).

Сейчас готова офлайн-часть этапа 1: изучение и очистка датасета. Дальше — каталог в
PostgreSQL + pgvector и базовая модель рекомендаций (см. `TODO.md`).

## Требования

- [uv](https://docs.astral.sh/uv/) (Python 3.12 ставится автоматически)
- ~8 ГБ свободного места под `data/` (staging + очищенные данные) и 16 ГБ RAM
- позже: Docker (PostgreSQL + pgvector), .NET SDK 10

## Датасет

Скачать со страницы датасета в одну папку (по умолчанию `~/Downloads`), можно в `.gz`:

| файл | раздел на странице | размер |
|---|---|---|
| `goodreads_interactions.csv` | Book Shelves | 4.1 ГБ |
| `book_id_map.csv`, `user_id_map.csv` | Book Shelves | 38 + 35 МБ |
| `goodreads_books.json` | Meta-Data of Books | 9.2 ГБ (2 ГБ в .gz) |
| `goodreads_book_works.json` | Meta-Data of Books | 731 МБ |
| `goodreads_book_authors.json` | Meta-Data of Books | 106 МБ |
| `goodreads_book_genres_initial.json` | Meta-Data of Books | 200 МБ |

`goodreads_interactions_dedup.json.gz` (11 ГБ, с датами и рецензиями) не нужен.

**Лицензия датасета — только некоммерческое использование.** Сырые данные и всё, что из них
получено (`data/`, `reports/`, модели), в репозиторий не попадают.

## Запуск

```bash
cp .env.example .env               # RAW_DIR — папка с файлами датасета
uv run booksengine prepare         # ~4 мин с нуля; повторный запуск без изменений — мгновенно
uv run pytest
```

Результат:

- `data/clean/*.parquet` — каталог (`works`, `editions`, `authors`, `work_authors`, `work_genres`),
  матрица оценок `ratings` (user × work, 1–5), неявный сигнал `shelf_events`, `users`;
- `data/clean/manifest.json` — отпечатки входов/конфига/кода, журнал очистки, контрольные суммы;
- `reports/stage1_report.md` — структура данных, качество, распределения, все решения по очистке с цифрами.

Пороги очистки — в `config/cleaning.yaml`.

## Команды

| команда | что делает |
|---|---|
| `booksengine prepare [--force]` | staging → профиль → очистка → валидация → отчёт |
| `booksengine validate` | проверить инварианты готовых данных |
| `booksengine report` | пересобрать отчёт без пересчёта |
