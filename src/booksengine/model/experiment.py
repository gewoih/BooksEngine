"""Сравнение вариантов CF-ядра (правила очистки, пороги k-core) на общем тесте — `booksengine exp`.

Варианты — копии data/clean в data/exp/<имя>/ (`exp save`) плюс текущий data/clean («new»). Тестовые
пользователи и скрытые книги — общие для всех вариантов: иначе метрика жёсткого ядра вырастет сама, потому
что из теста уйдут трудные случаи (CLAUDE.md). Настройки моделей фиксированы (FIXED), не перебираются.
"""
import json
import shutil
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import scipy.sparse as sp

from booksengine.model import metrics
from booksengine.model.base import read_params
from booksengine.model.evaluate import MODELS, run_eval
from booksengine.model.matrix import columns, load_holdout, load_train

# Лучшие по валидации 3a (models/eval/*_test.json); здесь не перебираются.
FIXED: dict[str, tuple[dict, dict]] = {
    "popularity": ({"formula": "bayes_log", "m": 1000.0}, {}),
    "als": ({"factors": 64, "regularization": 0.1, "alpha": 1.0}, {}),
    "knn": ({"beta": 0.0, "k_max": 200}, {"k": 50, "normalize": False}),
}


def save_core(clean_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in ("ratings.parquet", "works.parquet", "manifest.json"):
        shutil.copy2(clean_dir / f, out_dir / f)
    if (clean_dir / "work_merges.parquet").exists():
        shutil.copy2(clean_dir / "work_merges.parquet", out_dir / "work_merges.parquet")
    else:  # ядро до слияния дублей
        pd.DataFrame({"shadow_work_id": pd.Series([], dtype="int64"),
                      "main_work_id": pd.Series([], dtype="int64")}).to_parquet(out_dir / "work_merges.parquet")


def _remapped_input(inp: pd.DataFrame, core_dir: Path, works: set) -> pd.DataFrame:
    """Вход тестовых пользователей в пространстве ядра: тень → главное (среднее), вне ядра — выкинуть."""
    wm = pd.read_parquet(core_dir / "work_merges.parquet")
    m = dict(zip(wm.shadow_work_id, wm.main_work_id))
    x = inp.assign(work_id=inp.work_id.map(lambda w: m.get(w, w)))
    x = x[x.work_id.isin(works)]
    return x.groupby(["user_id", "work_id"], as_index=False)["rating"].mean()


def build_common_split(cores: dict[str, Path], split_dir: Path, out_dir: Path) -> dict:
    holdout = pd.read_parquet(split_dir / "holdout_users.parquet")
    test_users = holdout[holdout.group == "test"]
    inp = pd.read_parquet(split_dir / "test_input.parquet")
    hid = pd.read_parquet(split_dir / "test_hidden.parquet")
    users, works_all, inputs = set(test_users.user_id), None, {}
    for name, d in cores.items():
        r = duckdb.execute("SELECT DISTINCT user_id, work_id FROM read_parquet(?)", [str(d / "ratings.parquet")]).df()
        users &= set(r.user_id)
        w = set(r.work_id)
        works_all = w if works_all is None else works_all & w
        inputs[name] = _remapped_input(inp, d, w)
    used = pd.concat([x[["user_id", "work_id"]] for x in inputs.values()]).drop_duplicates()
    h = hid[hid.user_id.isin(users) & hid.work_id.isin(works_all)]
    h = h.merge(used, how="left", indicator=True).query("_merge == 'left_only'").drop(columns="_merge")
    keep = set(h.user_id)
    for x in inputs.values():
        keep &= set(x.user_id)
    h = h[h.user_id.isin(keep)].sort_values(["user_id", "work_id"], ignore_index=True)
    hu = test_users[test_users.user_id.isin(keep)].sort_values("user_id", ignore_index=True)
    for name, x in inputs.items():
        o = out_dir / name
        o.mkdir(parents=True, exist_ok=True)
        x[x.user_id.isin(keep)].sort_values(["user_id", "work_id"], ignore_index=True).to_parquet(
            o / "test_input.parquet", index=False)
        h.to_parquet(o / "test_hidden.parquet", index=False)
        hu.to_parquet(o / "holdout_users.parquet", index=False)
    meta = {"users": len(keep), "users_dropped": len(test_users) - len(keep), "hidden": len(h),
            "hidden_dropped": int(hid.user_id.isin(set(test_users.user_id)).sum()) - len(h)}
    (out_dir / "common.json").write_text(json.dumps(meta, indent=1))
    return meta


def _profile_input(profile_path: Path, work_ids: np.ndarray) -> sp.csr_matrix:
    """Книги профиля по его собственной шкале 1–5 (`rating5`, у dnf — 1); книги вне ядра пропускаются."""
    p = pd.read_csv(profile_path)
    p = p[p.rating5.notna() & p.goodreads_work_id.isin(work_ids)].sort_values(
        "goodreads_work_id")
    cols = columns(work_ids, p.goodreads_work_id.to_numpy())
    vals = p.rating5.to_numpy(dtype=np.float32)
    return sp.csr_matrix((vals, (np.zeros(len(cols), dtype=int), cols)), shape=(1, len(work_ids)))


def _model(m: str, path: Path, train):
    """Сохранённая модель с настройками FIXED — или новая, обученная и сохранённая. Чужие настройки — ошибка:
    иначе сравнение молча меряет не ту модель."""
    cls = MODELS[m][0]
    fit_params, score_params = FIXED[m]
    if (path / "params.json").exists():
        saved = read_params(path)
        if any(saved.get(k) != v for k, v in fit_params.items()):
            raise ValueError(f"{path}: настройки {saved} не совпадают с {fit_params}")
        model, fit_s = cls.load(path), None
    else:
        model = cls(**fit_params)
        t0 = time.perf_counter()
        model.fit(train)
        fit_s = round(time.perf_counter() - t0, 1)
        model.configure(**score_params)
        model.save(path)
    model.configure(**score_params)
    return model, fit_s


def run_core(name: str, *, ratings_path: Path, works_path: Path, common_dir: Path, holdout_path: Path,
             profile_path: Path, model_dir: Path, eval_dir: Path, models=("popularity", "als", "knn")) -> dict:
    train = load_train(ratings_path, holdout_path)
    hold = load_holdout(common_dir, "test", train.work_ids)
    titles = duckdb.execute("SELECT work_id, title FROM read_parquet(?)", [str(works_path)]).df() \
        .set_index("work_id")["title"]
    prof = _profile_input(profile_path, train.work_ids)
    out = {"core": name, "n_users": int(len(hold.user_ids)), "models": {}}
    eval_dir.mkdir(parents=True, exist_ok=True)
    for m in models:
        model, fit_s = _model(m, model_dir / m, train)
        per_user, cov = run_eval(model, hold)
        top = metrics.top_k(model.score(prof), prof, min(metrics.K, prof.shape[1]))[0]
        out["models"][m] = {"fit_seconds": fit_s, "summary": metrics.summarize(per_user), "coverage": cov,
                            "profile": [str(titles.get(train.work_ids[c], c)) for c in top if c >= 0]}
        print(f"{name} {m}: NDCG@20 = {out['models'][m]['summary']['all']['ndcg20']['mean']}", flush=True)
        (eval_dir / f"{name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return out
