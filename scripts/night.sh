#!/usr/bin/env bash
# Ночь: пересобрать модель и проверить её — толпа «ценность» → вес вкуса, ALS и отсечение → тест с контролем шума →
# шанс → списки и проверки на профилях → журнал выдач.
# Это не поиск «лучших настроек»: вес вкуса, отсечение и вес ALS — одна ручка «насколько смело уходить от известного»
# (все варианты лежат на одной кривой «угадано → качество»), и подбор доходит до порога угаданного (`layers.GUARD`).
# Повторный прогон на тех же данных и моделях даёт тот же ответ: запускать после изменения данных, компонентов модели
# или правил списка.
# Запуск из корня проекта: scripts/night.sh [--taste]   (лог — reports/night-<дата>.log, итог — в конце лога)
# --taste — сначала дообучить модель вкуса (64/128 координат × регуляризация 0.05/0.08; посчитанное не повторяется,
#           сохраняется только лучшая по личной точности). 64 координаты — ~10 мин на вариант, 128 — дольше.
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
step ease-like-tune                 # λ 250/500, веса звёзд −1/−0.5/0.5/1/2 (посчитанное не повторяется)
# шанс нужен под тот вариант, что выбран: без свежего layers val старые params и chance.json остаются рабочими
step layers val && step layers test && step calibrate layers
step layers profiles
step profile-check
for p in profiles/*.csv; do [[ $p == *_movies.csv ]] || step recommend --ratings "$p"; done   # профили фильмов — не здесь
step journal

echo; echo "=== ИТОГ $(date +%T)"
for r in ease_like_tune layers_val layers_test; do        # вердикт отчёта — первая строка со звёздочками и «сверх шума»
  [[ -f reports/$r.md ]] && { echo "$r:"; grep -m1 '^\*\*' "reports/$r.md"; grep -m1 '^Сверх шума' "reports/$r.md"; }
done
if ((${#FAILED[@]})); then printf 'Упало: %s\n' "${FAILED[@]}"; else echo "Все шаги прошли."; fi
echo "Отчёты: reports/ease_like_tune.md, layers_val.md, layers_test.md, layers_profiles.md, chance_layers.md, profile_check.md, journal.md; лог: $LOG"
