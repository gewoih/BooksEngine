"""Отчёт этапа 3a: сравнение моделей на тесте. Все цифры — из models/eval/*.json и split.json."""
import json
from pathlib import Path

from booksengine.model.split import BUCKET_ORDER
from booksengine.paths import EVAL_DIR, REPORTS_DIR, SPLIT_DIR

NAMES = {"popularity": "Популярность", "als": "ALS", "als_neg": "ALS", "knn": "item-kNN",
         "mix": "Смесь ALS + EASE", "ease": "EASE^R"}
ORDER = ["popularity", "als", "als_neg", "knn", "ease", "mix"]


def _n(x: int) -> str:
    return f"{x:,}".replace(",", " ")


def _cell(m: dict) -> str:
    return "—" if m["mean"] is None else f"{m['mean']:.4f} [{m['lo']:.4f}; {m['hi']:.4f}]"


def _label(name: str, v: dict) -> str:
    tail = "" if v["label"] == "—" else f" ({v['label']})"
    if v["deployable"]:
        why = ""
    elif v["score_params"].get("neg_rule") == "none":
        why = " — контроль без отрицательного сигнала"
    else:
        why = " — не переносится в C#"
    return f"{NAMES.get(name, name)}{tail}{why}"


def write(eval_dir: Path = EVAL_DIR, split_dir: Path = SPLIT_DIR,
          out_path: Path = REPORTS_DIR / "stage3a_report.md") -> Path:
    meta = json.loads((split_dir / "split.json").read_text())
    tests = {p.name.removesuffix("_test.json"): json.loads(p.read_text()) for p in eval_dir.glob("*_test.json")}
    names = [n for n in ORDER if n in tests] + sorted(set(tests) - set(ORDER))
    t = meta["groups"]["test"]
    L = ["# Этап 3a. Сравнение моделей рекомендаций", "",
         f"Тест: {_n(t['users'])} пользователей Goodreads, целиком исключённых из обучения; у каждого скрыто "
         f"{round(meta['share'] * 100)}% оценок, остальное подано модели как профиль нового человека. "
         f"Настройки подбирались на отдельных {_n(meta['groups']['val']['users'])} пользователях.", "",
         "Группы теста по активности: " + ", ".join(
             f"{b} оценок — {_n(t['buckets'].get(b, 0))}" for b in BUCKET_ORDER) + ".", "",
         "## Как читать", "",
         "- **NDCG@20** — главная: насколько высоко в топ-20 стоят спрятанные книги, которые человек оценил на 5★ "
         "(вдвое важнее) и 4★. 1.0 — идеальный порядок, 0 — ни одной в топе.",
         "- **Recall@20** — какая доля спрятанных 4–5★ вообще попала в топ-20.",
         "- **MAP@20** — то же, но с учётом, насколько рано в списке попадания.",
         "- **Coverage** — доля каталога, которая хоть кому-то попала в топ: низкая — всем советуют одно и то же.",
         "- **Low@20** — доля спрятанных 1–2★, которые модель всё-таки посоветовала. Чем ниже, тем лучше.",
         "- В квадратных скобках — 95% интервал. Если интервалы двух моделей перекрываются, разница может быть случайной.",
         "- Все метрики — вне начатых серий: продолжения и части серий из профиля человека не советуются и не "
         "считаются попаданием (иначе метрика награждает «угадай следующий том»).",
         "", "## Итог на тесте", "",
         "| модель | NDCG@20 | NDCG@10 | Recall@20 | MAP@20 | Coverage | Low@20 | обучение, с |",
         "|---|---|---|---|---|---|---|---|"]
    for n in names:
        r = tests[n]
        for v in r["variants"]:
            a = v["summary"]["all"]
            L.append(f"| {_label(n, v)} | {_cell(a['ndcg20'])} | {_cell(a['ndcg10'])} | {_cell(a['recall20'])} | "
                     f"{_cell(a['map20'])} | {v['coverage']:.4f} | {_cell(a['low20'])} | {r['fit_seconds']} |")
    L += ["", "## NDCG@20 по активности пользователя", "",
          "Целевые группы — читающие, 50–199 и 200+ оценок; 20–49 — вторична.", "",
          "| модель | " + " | ".join(BUCKET_ORDER) + " |", "|---|" + "---|" * len(BUCKET_ORDER)]
    for n in names:
        for v in tests[n]["variants"]:
            L.append(f"| {_label(n, v)} | " + " | ".join(_cell(v["summary"][b]["ndcg20"]) for b in BUCKET_ORDER) + " |")
    L += ["", "## Перебор настроек на валидации", ""]
    for n in names:
        vp = eval_dir / f"{n}_val.json"
        if not vp.exists():
            continue
        L += [f"### {NAMES.get(n, n)}", "", "| обучение | выдача | NDCG@20 | Coverage | обучение, с |", "|---|---|---|---|---|"]
        for r in json.loads(vp.read_text()):
            L.append(f"| {r['fit_params']} | {r['score_params'] or '—'} | {_cell(r['summary']['all']['ndcg20'])} | "
                     f"{r['coverage']:.4f} | {r['fit_seconds']} |")
        L.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(L) + "\n")
    return out_path
