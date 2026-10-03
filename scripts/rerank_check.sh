#!/usr/bin/env bash
# Дешёвая проверка бустинга-переранжирования на проверочных людях книжной базы: вкусы на шкале судьи (если ещё не
# обучены), затем `rerank check` → reports/rerank_check.md. Если идёт другой тяжёлый прогон — ждёт его (WAIT_PID):
# двум прогонам не хватает 16 ГБ.
# Запуск из корня проекта: nohup caffeinate -i scripts/rerank_check.sh [WAIT_PID] > reports/rerank-<дата>.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
WAIT_PID=${1:-}
if [[ -n "$WAIT_PID" ]]; then
  echo "=== $(date +%T) жду окончания процесса $WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
fi

step() {
  echo; echo "=== $(date +%T) $*"
  local t0=$SECONDS
  if uv run booksengine "$@"; then
    echo "--- готово за $(( (SECONDS - t0) / 60 )) мин"
  else
    echo "!!! упало через $(( (SECONDS - t0) / 60 )) мин: $*"
    return 1
  fi
}

[[ -f models/books/taste_value/params.json && -f models/books/taste_five/params.json ]] || step rerank tastes
step rerank check
echo; echo "=== ИТОГ $(date +%T)"
[[ -f reports/rerank_check.md ]] && head -5 reports/rerank_check.md
