#!/usr/bin/env bash
# Сайт одной командой: БД (Docker), сервис выдачи (`serve`, единая база), API, фронт — в фоне, логи в reports/app-*.log.
# Уже запущенное (порт занят) не трогается. Запуск из корня проекта: scripts/app.sh [stop]
set -uo pipefail
cd "$(dirname "$0")/.."
DOMAIN=${DOMAIN:-books-amazon}

up() { curl -s -o /dev/null -m 2 "$1"; }

if [[ "${1:-}" == "stop" ]]; then
  for port in 5173 5080 5090; do
    pids=$(lsof -tiTCP:$port -sTCP:LISTEN)
    [[ -n "$pids" ]] && kill $pids && echo "остановлено: порт $port"
  done
  exit 0
fi

if ! docker info >/dev/null 2>&1; then
  echo "запускаю Docker…"
  open -a Docker
  until docker info >/dev/null 2>&1; do sleep 3; done
fi
docker compose up -d >/dev/null 2>&1
until docker compose exec -T db pg_isready -U "${POSTGRES_USER:-booksengine}" >/dev/null 2>&1; do sleep 2; done
echo "БД: готова"

if ! up localhost:5090/health; then
  nohup uv run booksengine --domain "$DOMAIN" serve > reports/app-serve.log 2>&1 &
  echo "сервис выдачи: запускаю (модели грузятся ~10 с)"
fi
if ! up localhost:5080/openapi/v1.json; then
  nohup dotnet run --project dotnet/BooksEngine.Api > reports/app-api.log 2>&1 &
  echo "API: запускаю"
fi
if ! up localhost:5173; then
  (cd web && [[ -d node_modules ]] || npm ci >/dev/null 2>&1; nohup npm run dev > ../reports/app-web.log 2>&1 &)
  echo "фронт: запускаю"
fi

for i in $(seq 1 60); do
  up localhost:5090/health && up localhost:5080/openapi/v1.json && up localhost:5173 && break
  sleep 2
done
for u in "сервис выдачи:localhost:5090/health" "API:localhost:5080/openapi/v1.json" "фронт:localhost:5173"; do
  name=${u%%:*}; url=${u#*:}
  if up "$url"; then echo "$name: работает"; else echo "$name: не отвечает — лог в reports/app-*.log"; fi
done
echo "Сайт: http://localhost:5173"
