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

### Второй источник: Amazon Reviews'23 (опционально)

Для книг после 2017 года (датасет Goodreads ими не пополняется) — `amazon-bridge` строит мост на Goodreads
по ISBN/ASIN и проверяет сигнал перевода (Wikidata). Сырьё — 4 файла в одну папку `RAW_DIR/
amazon_reviews_2023/` со страницы [amazon-reviews-2023.github.io](https://amazon-reviews-2023.github.io/)
(0-Core → rating_only: `Books.csv.gz`, `Kindle_Store.csv.gz`; raw/meta_categories: `meta_Books.jsonl.gz`,
`meta_Kindle_Store.jsonl.gz`) — сервер источника медленный и рвёт соединение на больших файлах, докачивать
через `scripts/fetch_amazon_raw.sh` (устойчив к обрывам, можно прерывать и перезапускать).

```bash
RAW_DIR=~/Downloads scripts/fetch_amazon_raw.sh   # разово, часы из-за скорости источника
uv run booksengine amazon-bridge                  # -> data/amazon/clean/, reports/amazon_bridge.md
```

Единая база Goodreads + Amazon — домен `books-amazon` (`data/books-amazon/`, `models/books-amazon/`,
`reports/books-amazon/`, журнал — `profiles/history/books-amazon/`): люди Amazon — дополнительные читатели,
новые книги (после 2017, их нет в Goodreads) — свои произведения; новая книга советуется, только если известен
русский перевод (Wikidata или разметка `config/amazon_ru_titles.csv`). Отложенные люди — копия книжных, поэтому
обе базы судятся на одних людях:

```bash
uv run booksengine --domain books-amazon prepare                  # база (~1 мин), --amazon-scale q|raw, --amazon-min-user
uv run booksengine --domain books-amazon refit                    # компоненты с настройками models/books (~40 мин)
uv run booksengine --domain books-amazon layers val               # затем layers test, calibrate layers
uv run booksengine compare-bases books books-amazon --stage test  # парное сравнение → reports/compare_*.md
uv run booksengine --domain books-amazon recommend --ratings profiles/my_ratings.csv
```

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

Все — `uv run booksengine <команда>`, справка по каждой — `--help`.

| команда | что делает |
|---|---|
| **данные** | |
| `prepare [--force]` | staging → профиль → очистка → валидация → `reports/stage1_report.md` |
| `validate` | проверить инварианты готовых данных |
| `report` | пересобрать отчёт очистки без пересчёта |
| `load-db [--force]` | загрузить каталог в PostgreSQL, сверить с `manifest.json` |
| `amazon-bridge` | Amazon Reviews'23: мост на Goodreads по ISBN/ASIN, сигнал перевода (Wikidata) → `data/amazon/clean/`, `reports/amazon_bridge.md` |
| `--domain books-amazon prepare` | единая база Goodreads + Amazon → `data/books-amazon/clean/`, отложенные люди — копия книжных |
| `--domain books-amazon refit [--min-user N]` | все компоненты выдачи на единой базе с настройками `models/books` |
| `compare-bases A B [--stage test\|val]` | две базы на одних отложенных людях, каждая своей выдачей: парная разность качества и угаданного по этапам → `reports/compare_A_B_<stage>.md` |
| `split [--force]` | отложенная выборка: ~5 000 человек для настройки, ~6 000 для теста (поровну из этапов 20–39, 40–79, 80–159, 160–319, 320–999 оценок) |
| **модели и замеры** | |
| `evaluate <model> --stage val\|test` | стенд `popularity`, `als`, `als_neg`, `knn`, `ease`, `mix`: перебор по NDCG@20 → `models/<model>` / замер на тесте |
| `report-3a` | `reports/stage3a_report.md` — сравнение моделей стенда |
| `calibrate [model]` | шанс: у `layers` (выдача) — пятёрки (5★), у `mix` (приложение) — «понравится» (4–5★) → `models/<model>/chance.json`, отчёт `reports/chance_<model>.md` |
| `taste [--factors … --reg …]` | модель вкуса → `models/taste` |
| `ease-like-tune [--lam … --weights … --force --min-user N --min-support N --amazon q-5,raw-20]` | подбор толпы «ценность» → `models/ease_like`, `models/mix_like`; связь книг — только при ≥ N общих читателях (по умолчанию 25); `--amazon` — варианты с людьми Amazon (шкала raw/q, порог книг) против сохранённой толпы |
| `layers val\|test\|profiles [--top N]` | слои «толпа + вкус»: выбор по качеству списка (средняя ценность угаданных книг топ-20: 5★ = 2 … 1★ = −1) → `models/layers`, замер на тесте с контролем «шум вместо вкуса» («сверх шума» — честный вклад вкуса), топ профилей рядом с прежней выдачей |
| `profile-check` | каждая книга `profiles/*.csv` по очереди прячется — на каком месте её поставит выдача |
| `why [--profile имя] "<книга>"` | почему книга стоит на своём месте: части модели и вклады книг профиля |
| `taste-gap` | личная точность моделей: понравившиеся скрытые книги выше непонравившихся? |
| `ease-size` | книги профилей и теста за границей EASE (30 000) — стоит ли её расширять |
| `exp save\|split\|run\|report` | сравнить варианты очистки на общем тесте |
| **выдача** | |
| `journal` | журнал выдач (`profiles/history/`) против оценок, поставленных позже: доля 5★ и 1–2★ среди прочитанного из советов и среди остальных оценок; отдельно — поднятое вкусом и прочитанное из «смелой» выдачи (проверка порога 80%) → `reports/journal.md` |
| `recommend --ratings <csv> [--top 20] [--one-list]` | рекомендации по CSV (`goodreads_work_id`, `rating` 1–5, `status`, `title`): два списка — художественная литература и нон-фикшн (`--one-list` — один); слои `models/layers`, без них — смесь; без сборников и поздних томов неначатых серий, не больше одной книги автора на 10 мест; без оценки: `status=want` — «хочу прочитать», иначе — «прочитано, оценку не помню» (не советуются, во входе толпы — слабый плюс; полка до 20 книг даёт +2% угаданного, вся полка Goodreads — +21%); выдача пишется в `profiles/history/` |
| `export-model` | смесь → БД для веб-интерфейса (перезаписывает целиком) |

Порядок сборки моделей: `split` → `evaluate als_neg` и `evaluate ease` (val, затем test) → `evaluate mix` →
`calibrate mix` (приложение) → `taste` → `ease-like-tune` → `layers val` → `layers test` → `calibrate layers`
(`recommend`). Переобученный компонент ломает загрузку смеси и слоёв с подсказкой, что пересобрать.

### Веб-интерфейс

```bash
uv run booksengine export-model                     # models/mix → БД (~3 мин); после load-db — обязательно
dotnet run --project dotnet/BooksEngine.Api         # API на :5080, модель грузится при старте (~16 с)
(cd web && npm ci && npm run dev)                   # фронт на http://localhost:5173
(cd dotnet && dotnet test --project BooksEngine.Api.Tests)  # в т. ч. GoldenTests: выдача C# = Python
```

Python — только офлайн: `export-model` кладёт в БД всё для выдачи (векторы ALS, соседей EASE, параметры смеси
и шанса, пары «оценил X → не советовать Y», эталонную выдачу), а C# API считает по ним выдачу смеси — повтор
Python (`Recommendations/Recommender.cs`), расхождение ловит `GoldenTests`.

Вход без пароля («войти как», приложение локальное). Импорт — CSV в нашем формате или экспорт
Goodreads (My Books → Export); оценки книг из файла перезаписываются, остальные остаются.
После нового `export-model` — `POST /api/admin/reload-model` или перезапуск API.
