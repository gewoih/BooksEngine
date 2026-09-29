#!/bin/bash
# Устойчивая докачка сырых файлов Amazon Reviews'23 в RAW_DIR/amazon_reviews_2023/.
# Сервер McAuley Lab медленный (~0.8 МБ/с) и рвёт соединение на многогигабайтных файлах — докачка (-C -)
# с обрывом при зависании (--speed-limit/--speed-time) и повтором до полного Content-Length.
set -uo pipefail

RAW_DIR="${RAW_DIR:-$HOME/Downloads}"
OUT="$RAW_DIR/amazon_reviews_2023"
mkdir -p "$OUT"

BASE="https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023"

fetch() {
  local url="$1" out="$2" expected="$3" attempt=0
  while true; do
    attempt=$((attempt + 1))
    local cur
    cur=$(stat -f%z "$out" 2>/dev/null || stat -c%s "$out" 2>/dev/null || echo 0)
    if [ "$cur" -ge "$expected" ]; then
      echo "$out: готово ($cur байт)"
      return 0
    fi
    echo "$out: попытка $attempt, $cur/$expected байт"
    curl -sL -C - --speed-limit 2048 --speed-time 30 --retry 5 --retry-delay 3 -o "$out" "$url"
    if [ "$attempt" -gt 200 ]; then
      echo "$out: не удалось докачать за $attempt попыток" >&2
      return 1
    fi
    sleep 2
  done
}

fetch "$BASE/benchmark/0core/rating_only/Books.csv.gz" "$OUT/Books.csv.gz" 602182123
fetch "$BASE/benchmark/0core/rating_only/Kindle_Store.csv.gz" "$OUT/Kindle_Store.csv.gz" 464452653
fetch "$BASE/raw/meta_categories/meta_Books.jsonl.gz" "$OUT/meta_Books.jsonl.gz" 4942125770
fetch "$BASE/raw/meta_categories/meta_Kindle_Store.jsonl.gz" "$OUT/meta_Kindle_Store.jsonl.gz" 2269538269

status=0
for f in "$OUT"/*.gz; do
  if gzip -t "$f" 2>/dev/null; then
    echo "$f: gzip OK"
  else
    echo "$f: gzip BROKEN — удалите и перезапустите скрипт" >&2
    status=1
  fi
done
exit $status
