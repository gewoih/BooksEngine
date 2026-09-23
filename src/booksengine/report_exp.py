"""Отчёт сравнения вариантов CF-ядра (`booksengine exp report`). Все цифры — из JSON и manifest."""
import json
from pathlib import Path

from booksengine.model.split import BUCKET_ORDER
from booksengine.paths import EXP_DIR, EXP_EVAL_DIR, REPORTS_DIR
from booksengine.report_3a import NAMES, ORDER, _cell, _n

# Порядок ядер в отчёте; «old» — эталон, с которым сравниваются остальные. «new» — текущий data/clean.
CORE_ORDER = ["old", "k50", "new"]
NEW_RULES = ("nonbooks", "duplicates", "users_monotone", "users_low_variance", "users_hyperactive", "kcore")


def _label(core: str, man: dict) -> str:
    k = man.get("config", {}).get("kcore", {}).get("min_work_ratings")
    return f"≥ {k}" if k is not None else core


def _overlap(a: dict, b: dict) -> str:
    """Отличие значимо, если 95% интервалы не перекрываются."""
    if a["mean"] is None or b["mean"] is None:
        return "—"
    return "нет" if a["lo"] <= b["hi"] and b["lo"] <= a["hi"] else "**да**"


def render(common: dict, evals: dict, manifests: dict) -> str:
    cores = [c for c in CORE_ORDER if c in manifests] + sorted(set(manifests) - set(CORE_ORDER))
    lab = {c: _label(c, manifests[c]) for c in cores}
    L = ["# Очистка от шума: сравнение CF-ядер", "",
         "Ядра подписаны порогом k-core по числу оценок у книги. Одни и те же тестовые пользователи и скрытые "
         "книги для всех ядер: из сравнения исключены люди и книги, которых нет хотя бы в одном ядре, иначе "
         "жёсткое ядро выиграло бы только оттого, что из теста ушли трудные случаи.", "",
         f"Общий тест: {_n(common['users'])} пользователей (исключено {_n(common['users_dropped'])}), "
         f"{_n(common['hidden'])} скрытых оценок (исключено {_n(common['hidden_dropped'])}).", "",
         "## Что удалено", "",
         "| ядро | оценок | пользователей |", "|---|---|---|"]
    for c in cores:
        man = manifests[c]
        L.append(f"| {lab[c]} | {_n(man['outputs']['ratings']['rows'])} | {_n(man['outputs']['users']['rows'])} |")
    logs = {c: {s["rule"]: s for s in manifests[c]["cleaning_log"]} for c in cores}
    rules = [r for r in NEW_RULES if any(r in logs[c] for c in cores)]
    L += ["", "Удалено строк оценок по шагам очистки:", "",
          "| шаг | " + " | ".join(lab[c] for c in cores) + " | почему |", "|---|" + "---|" * (len(cores) + 1)]
    for r in rules:
        cells = []
        for c in cores:
            st = logs[c].get(r)
            extra = f"; теней {_n(st['detail']['shadows'])}" if st and r == "duplicates" and st.get("detail") else ""
            cells.append(f"{_n(st['rows_removed'])}{extra}" if st else "—")
        reason = next(logs[c][r]["reason"] for c in reversed(cores) if r in logs[c])
        L.append(f"| `{r}` | " + " | ".join(cells) + f" | {reason} |")
    L += ["", "## Метрики", "",
          "NDCG@20 — насколько высоко в топ-20 спрятанные книги на 4–5★; Low@20 — доля спрятанных 1–2★, попавших "
          "в топ (ниже — лучше). В скобках — 95% интервал; «значимо» — интервалы не перекрываются.", ""]
    ecores = [c for c in cores if c in evals]
    models = [m for m in ORDER if any(m in evals[c]["models"] for c in ecores)]
    for m in models:
        L += [f"### {NAMES.get(m, m)}", "",
              "| ядро | NDCG@20 | " + " | ".join(f"NDCG@20 {b}" for b in BUCKET_ORDER[:2])
              + " | Low@20 | coverage | обучение, с |", "|---|---|---|---|---|---|---|"]
        rows = {}
        for c in ecores:
            if m not in evals[c]["models"]:
                continue
            r = evals[c]["models"][m]
            rows[c] = r["summary"]["all"]
            fit = "—" if r["fit_seconds"] is None else r["fit_seconds"]
            L.append(f"| {lab[c]} | {_cell(r['summary']['all']['ndcg20'])} | "
                     + " | ".join(_cell(r["summary"][b]["ndcg20"]) for b in BUCKET_ORDER[:2])
                     + f" | {_cell(r['summary']['all']['low20'])} | {r['coverage']:.4f} | {fit} |")
        if "old" in rows:
            L.append("")
            for c in rows:
                if c != "old":
                    L.append(f"- Значимо, {lab[c]} против {lab['old']}: NDCG@20 — "
                             f"{_overlap(rows['old']['ndcg20'], rows[c]['ndcg20'])}, "
                             f"Low@20 — {_overlap(rows['old']['low20'], rows[c]['low20'])}.")
        L.append("")
    L += ["coverage — доля каталога ядра, попавшая хоть кому-то в топ; у меньшего ядра она выше просто от "
          "меньшего знаменателя. Обучение «—» — модель загружена с диска, не обучалась.", "",
          "## Профиль", "", "Топ-20 для `profiles/my_ratings.csv` (dnf и книги вне ядра пропущены).", ""]
    for m in models:
        L += [f"### {NAMES.get(m, m)}", "", "| # | " + " | ".join(lab[c] for c in ecores) + " |",
              "|---|" + "---|" * len(ecores)]
        tops = [evals[c]["models"].get(m, {}).get("profile", []) for c in ecores]
        for i in range(max(map(len, tops), default=0)):
            L.append(f"| {i + 1} | " + " | ".join(t[i] if i < len(t) else "" for t in tops) + " |")
        L.append("")
    return "\n".join(L) + "\n"


def write(exp_dir: Path = EXP_DIR, eval_dir: Path = EXP_EVAL_DIR,
          out_path: Path = REPORTS_DIR / "cleanup_experiment.md") -> Path:
    """Ядра — папки exp_dir/<core>/ с manifest.json (кроме common/)."""
    common = json.loads((exp_dir / "common" / "common.json").read_text())
    dirs = [d for d in exp_dir.iterdir() if d.is_dir() and d.name != "common" and (d / "manifest.json").exists()]
    manifests = {d.name: json.loads((d / "manifest.json").read_text()) for d in dirs}
    evals = {c: json.loads((eval_dir / f"{c}.json").read_text()) for c in manifests if (eval_dir / f"{c}.json").exists()}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render(common, evals, manifests))
    return out_path
