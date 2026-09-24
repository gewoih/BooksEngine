# BooksEngine

Рекомендации книг на основе коллаборативной фильтрации. Источник данных —
[UCSD Goodreads Book Graph](https://mengtingwan.github.io/data/goodreads.html)
(228M взаимодействий, 876K пользователей, 2.36M изданий).

Готово: изучение и очистка датасета (этап 1), каталог книг в PostgreSQL + pgvector (этап 2),
модель «толпа + вкус» (`models/layers`: EASE с целью «оценка − 3» + ALS) с шансом «понравится», объяснением
и журналом выдач, CLI `recommend`; приложение пока на прежней смеси ALS + EASE,
веб-интерфейс: библиотека с поиском и оценками, рекомендации, карточка книги, импорт CSV.
Открытые задачи — `TODO.md`.

## Требования

- [uv](https://docs.astral.sh/uv/) (Python 3.12 ставится автоматически)
- ~8 ГБ свободного места под `data/` (staging + очищенные данные) и 16 ГБ RAM
- Docker (PostgreSQL + pgvector), .NET SDK 10 и `dotnet-ef` (`dotnet tool install -g dotnet-ef`)
- Node.js 22+ (фронт `web/`)

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
  матрица оценок `ratings` (user × work, 1–5), `users`, дубли произведений `work_merges`;
- `data/clean/manifest.json` — отпечатки входов/конфига/кода, журнал очистки, контрольные суммы;
- `reports/stage1_report.md` — структура данных, качество, распределения, все решения по очистке с цифрами.

Пороги очистки — в `config/cleaning.yaml`.

### Каталог в PostgreSQL

```bash
docker compose up -d                                        # параметры — POSTGRES_* в .env
(cd dotnet/BooksEngine.Db && dotnet ef database update)     # схема: EF Core-миграции
uv run booksengine load-db                                  # каталог из data/clean → БД
```

В БД попадает каталог: произведения, издания, авторы, жанры (с поиском по названию и автору,
`pg_trgm`). Оценки датасета остаются в Parquet — на них учится модель. Повторная загрузка
того же `manifest.json` пропускается; изменённый каталог обновляется на месте, внутренние id
произведений сохраняются.

## Команды

| команда | что делает |
|---|---|
| `booksengine prepare [--force]` | staging → профиль → очистка → валидация → отчёт |
| `booksengine validate` | проверить инварианты готовых данных |
| `booksengine report` | пересобрать отчёт без пересчёта |
| `booksengine load-db [--force]` | загрузить каталог в PostgreSQL, сверить с `manifest.json` |
| `booksengine split [--force]` | отложенная выборка: 5K пользователей для настройки, 20K для теста |
| `booksengine evaluate <model> --stage val\|test` | перебор настроек / замер модели на тесте |
| `booksengine report-3a` | `reports/stage3a_report.md` — сравнение моделей |
| `booksengine exp save\|split\|run\|report` | сравнить варианты очистки на общем тесте |
| `booksengine calibrate [model]` | шанс «понравится» → `models/<model>/chance.json` |
| `booksengine recommend --ratings <csv> [--top 20]` | рекомендации по CSV (`goodreads_work_id`, `rating` 1–5, `status`) |
| `booksengine export-model` | модель → БД для веб-интерфейса (перезаписывает целиком) |

### Веб-интерфейс

```bash
uv run booksengine export-model                     # models/mix → БД (~3 мин); после load-db — обязательно
dotnet run --project dotnet/BooksEngine.Api         # API на :5080, модель грузится при старте (~16 с)
(cd web && npm ci && npm run dev)                   # фронт на http://localhost:5173
(cd dotnet && dotnet test --project BooksEngine.Api.Tests)  # в т. ч. GoldenTests: выдача C# = Python
```

Вход без пароля («войти как», приложение локальное). Импорт — CSV в нашем формате или экспорт
Goodreads (My Books → Export); оценки книг из файла перезаписываются, остальные остаются.
После нового `export-model` — `POST /api/admin/reload-model` или перезапуск API.
