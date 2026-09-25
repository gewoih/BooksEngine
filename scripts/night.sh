#!/usr/bin/env bash
# Ночной подбор параметров: веса звёзд и λ толпы «ценность» → вес вкуса, ALS и отсечение → тест → шанс → списки.
# Запуск из корня проекта: scripts/night.sh [--taste]   (лог — reports/night-<дата>.log, итог — в конце лога)
# --taste — сначала дообучить модель вкуса (64/128 координат × регуляризация 0.05/0.08; посчитанное не повторяется,
#           сохраняется только лучшая по личной точности). Долго: по часу и больше на вариант.
# Упавший шаг не останавливает ночь: следующие считают на том, что уже сохранено; список упавших — в итоге.
# Подбор толпы продолжает прошлый прогон: посчитанные настройки не пересчитываются.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p reports
LOG="reports/night-$(date +%F-%H%M).log"
exec > >(tee "$LOG") 2>&1
FAILED=()

step() {
  echo; echo "=== $(date +%T) $*"
  local t0=$SECONDS
  if uv run booksengine "$@"; then
    echo "--- готово за $(( (SECONDS - t0) / 60 )) мин"
  else
    echo "!!! упало через $(( (SECONDS - t0) / 60 )) мин: $*"
    FAILED+=("$*")
    return 1
  fi
}

if [[ "${1:-}" == "--taste" ]]; then step taste --factors 64,128 --reg 0.05,0.08; fi
step ease-like-tune                 # λ 125/250/500/1000, веса звёзд −1/−0.5/0.5/1/2 (посчитанное не повторяется)
# шанс нужен под тот вариант, что выбран: без свежего layers val старые params и chance.json остаются рабочими
step layers val && step layers test && step calibrate layers
step layers profiles
step profile-check
for p in profiles/*.csv; do step recommend --ratings "$p"; done

echo; echo "=== ИТОГ $(date +%T)"
grep -h "^Выбрано:" reports/ease_like_tune.md 2>/dev/null || true
grep -h "Выбран:" reports/layers_val.md 2>/dev/null | sed 's/.*\*\*Выбран:/Выбран:/' || true
if ((${#FAILED[@]})); then printf 'Упало: %s\n' "${FAILED[@]}"; else echo "Все шаги прошли."; fi
echo "Отчёты: reports/ease_like_tune.md, layers_val.md, layers_test.md, layers_profiles.md, chance_layers.md, profile_check.md; лог: $LOG"
