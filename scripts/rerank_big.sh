#!/usr/bin/env bash
# Большая проверка бустинга-переранжирования на книжной базе: ещё 50 000 отложенных людей для его обучения, части
# формулы — без них (models/books-rank), бустинг, выбор на проверке, замер на тесте против нынешней выдачи →
# reports/rerank_final.md. Если идёт другой тяжёлый прогон — ждёт его (WAIT_PID): обучению EASE нужна вся память.
# Запуск из корня проекта: nohup caffeinate -i scripts/rerank_big.sh [WAIT_PID] > reports/rerank-big-<дата>.log 2>&1 &
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

step rerank split &&
  step rerank refit &&
  step rerank final
echo; echo "=== ИТОГ $(date +%T)"
[[ -f reports/rerank_final.md ]] && head -5 reports/rerank_final.md
