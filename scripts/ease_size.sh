#!/usr/bin/env bash
# Опыт с размером EASE на единой базе: толпа видит N книг вместо 30 000 — больше новинок в поле зрения толпы.
# Модели — в отдельной папке models/books-amazon-<N/1000>k (рабочие models/books-amazon не трогаются): ALS и вкус —
# копией из рабочих, EASE и толпа «ценность» — заново на N книгах; выбор варианта слоёв, тест, шанс. Затем сравнение
# с рабочими на одних людях: люди Goodreads (compare-bases, тест) и новинки на людях Amazon (new-books --vs).
# Запуск из корня проекта: nohup caffeinate -i scripts/ease_size.sh 40000 > reports/ease-40k-<дата>.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
N=${1:?размер EASE, например 40000}
EXP="books-amazon-$((N / 1000))k"
BASE=books-amazon
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

# блок 1000 вместо 2000 — меньше памяти на обучение толпы при том же результате
step --domain $BASE --models "$EXP" refit --ease-top "$N" --ease-block 1000 --reuse $BASE &&
  step --domain $BASE --models "$EXP" layers val &&
  step --domain $BASE --models "$EXP" layers test &&
  step --domain $BASE --models "$EXP" calibrate layers &&
  step --domain $BASE new-books &&
  step --domain $BASE --models "$EXP" new-books --vs $BASE &&
  step --domain $BASE compare-bases $BASE "$BASE:$EXP" --stage test
step --domain $BASE --models "$EXP" layers profiles

echo; echo "=== ИТОГ $(date +%T)"
for r in layers_val layers_test; do
  [[ -f reports/$EXP/$r.md ]] && { echo "$r:"; grep -m1 '^\*\*' "reports/$EXP/$r.md"; }
done
[[ -f reports/$BASE/compare_${BASE}_${EXP}_test.md ]] && head -12 "reports/$BASE/compare_${BASE}_${EXP}_test.md"
[[ -f reports/$EXP/new_books_vs_$BASE.md ]] && cat "reports/$EXP/new_books_vs_$BASE.md"
if ((${#FAILED[@]})); then printf 'Упало: %s\n' "${FAILED[@]}"; else echo "Все шаги прошли."; fi
