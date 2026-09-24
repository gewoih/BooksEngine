# Фронт BooksEngine

React + TypeScript + Vite + Mantine, данные — TanStack Query, адреса — React Router.
Выдачу считает C# API (`dotnet/BooksEngine.Api`) по модели, выгруженной `booksengine export-model`; запуск всего
приложения — корневой `README.md`, «Веб-интерфейс».

```bash
npm ci
npm run dev       # http://localhost:5173, /api проксируется на API :5080
npm run build     # tsc + vite build — проверка перед коммитом (автотестов фронта нет)
npm run lint      # oxlint
npm run gen:api   # src/api/schema.d.ts из OpenAPI запущенного API — после изменения контрактов
```

| путь | что там |
|---|---|
| `src/api/` | клиент (`openapi-fetch` по сгенерированной схеме) и хуки запросов |
| `src/pages/` | библиотека (с панелью «мои оценки»), рекомендации |
| `src/components/` | строка книги, звёзды и «бросил», карточка книги (`?work=<id>`), импорт, вход |
