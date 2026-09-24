#!/usr/bin/env bash
# Ночной подбор параметров: веса звёзд и λ толпы «ценность» → вес вкуса, ALS и отсечение → тест → шанс → списки.
# Запуск из корня проекта: scripts/night.sh   (лог — reports/night-<дата>.log, итог — в конце лога)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p reports
LOG="reports/night-$(date +%F-%H%M).log"
exec > >(tee "$LOG") 2>&1

step() {
  echo; echo "=== $(date +%T) $*"
  local t0=$SECONDS
  uv run booksengine "$@"
  echo "--- готово за $(( (SECONDS - t0) / 60 )) мин"
}

step ease-like-tune                 # λ и веса звёзд (сетка ~1 ч)
step layers val                     # вес вкуса × вес ALS × отсечение
step layers test                    # один замер выбранного на тесте
step calibrate layers               # шанс «понравится» под выбранный вариант
step layers profiles                # прежний и новый топ-20 рядом
step profile-check                  # места своих оценённых книг
for p in profiles/*.csv; do step recommend --ratings "$p"; done

echo; echo "=== ИТОГ"
grep -h "^Выбрано:" reports/ease_like_tune.md || true
grep -h "Выбран:" reports/layers_val.md | sed 's/.*\*\*Выбран:/Выбран:/' || true
echo "Отчёты: reports/ease_like_tune.md, layers_val.md, layers_test.md, layers_profiles.md, profile_check.md; лог: $LOG"
