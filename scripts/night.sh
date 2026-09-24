#!/usr/bin/env bash
# Ночной подбор параметров: веса звёзд и λ толпы «ценность» → вес вкуса, ALS и отсечение → тест → шанс → списки.
# Запуск из корня проекта: scripts/night.sh [--taste]   (лог — reports/night-<дата>.log, итог — в конце лога)
# --taste — сначала дообучить модель вкуса (64/128 координат × регуляризация 0.05/0.08; посчитанное не повторяется,
#           сохраняется только лучшая по личной точности). Долго: по часу и больше на вариант.
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

[[ "${1:-}" == "--taste" ]] && step taste --factors 64,128 --reg 0.05,0.08
step ease-like-tune                 # λ 250/500/1000 × 6 наборов весов звёзд (18 настроек, ~3 ч)
step layers val                     # вес вкуса (10) × вес ALS (5) × отсечение (нет/100/300/500/1000)
step layers test                    # один замер выбранного на тесте
step calibrate layers               # шанс «понравится» под выбранный вариант
step layers profiles                # прежний и новый топ-20 рядом
step profile-check                  # места своих оценённых книг
for p in profiles/*.csv; do step recommend --ratings "$p"; done

echo; echo "=== ИТОГ"
grep -h "^Выбрано:" reports/ease_like_tune.md || true
grep -h "Выбран:" reports/layers_val.md | sed 's/.*\*\*Выбран:/Выбран:/' || true
echo "Отчёты: reports/ease_like_tune.md, layers_val.md, layers_test.md, layers_profiles.md, profile_check.md; лог: $LOG"
