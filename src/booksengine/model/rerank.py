"""Переранжирование бустингом (опыт): первые POOL книг толпы упорядочивают две модели LightGBM, обученные прямо под
судью, вместо ручной формулы «толпа + 3 × вкус» (`layers.Layers.combine`).

- «Прочтёт ли» — шанс p, что книга среди спрятанных книг человека (судья её увидит — «угадана»).
- «Насколько ценна, если прочтёт» — v, ожидаемая ценность по шкале судьи (5★ = 2 … 1★ = −1).

Балл книги — p·(v − c). Судья — средняя ценность угаданных: книга ценнее c поднимает среднюю, дешевле — опускает, и
тем сильнее, чем вероятнее, что её прочтут. c — ручка смелости, как вес вкуса у формулы: больше c — угаданные
ценнее, но их меньше. c выбирается тем же правилом, что вариант формулы (`layers.choose`: лучшее качество среди
угадывающих не меньше порога), сравнение с формулой — парно на тех же людях и при том же числе угаданных.

Признаки (`FEATURES`): баллы толпы, её частей, старого EASE и формулы, вкус, места у толпы и формулы; человек — сколько
оценок, средняя, доля пятёрок и 1–2★, личная шкала вкуса; книга — известность, средняя и доля пятёрок у людей
обучения, год, жанры, серия; пара — сколько книг автора человек оценил, насколько выше своей средней и лучшая оценка,
совпадение жанров с понравившимся и с прочитанным. Если обучены (`fit_value_tastes`), — ещё вкус на шкале судьи и
вкус «пятёрка или нет».

Учиться можно только на людях, которых не видели толпа и вкус: у остальных спрятанные книги уже в связях толпы.
`check` — дешёвая проверка на проверочных людях: перекрёстно (учим на четырёх частях, судим пятую) и кривая обучения —
растёт ли выигрыш с числом людей обучения; растёт — стоит отложить ещё десятки тысяч людей.
"""
import json
import time
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import scipy.sparse as sp

from booksengine.model import layers as ly
from booksengine.model import metrics
from booksengine.model.mix import _z
from booksengine.model.split import BUCKET_ORDER, SEED

POOL = 1000          # кандидаты — первые столько книг толпы: те же, что переставляет вкус формулы
GENRES = ("fiction", "non-fiction", "history, historical fiction, biography", "romance", "fantasy, paranormal",
          "mystery, thriller, crime", "children", "young-adult", "comics, graphic", "poetry")
SCORES = ("crowd", "crowd_ease", "crowd_als", "ease_old", "taste", "taste_z", "formula", "crowd_rank", "formula_rank")
PERSON = ("n_input", "mean_input", "share5_input", "share_low_input", "user_bias")
BOOK = ("log_count", "book_mean", "book_share5", "year", "item_bias", "series_first", "in_series") + tuple(
    f"genre_{i}" for i in range(len(GENRES)))
PAIR = ("author_n", "author_dev", "author_max", "genre_like", "genre_read")
VALUE_TASTES = ("taste_value", "taste_five")   # вкус на шкале судьи и на «пятёрка или нет» (`fit_value_tastes`)
FEATURES = SCORES + PERSON + BOOK + PAIR + VALUE_TASTES
# наборы признаков для сравнения: что знает формула; всё, кроме вкусов на шкале судьи; всё
SETS = {"scores": SCORES, "full": SCORES + PERSON + BOOK + PAIR, "value": FEATURES}
# c в балле p·(v − c); −inf — только «прочтёт ли» (самый осторожный список)
C_GRID = (-np.inf, -1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75)
FOLDS = 5
SIZES = (1000, 2000, 4000)    # кривая обучения: людей обучения в каждой части
NEG_RATE = 0.1                # доля непрочитанных кандидатов в обучении «прочтёт ли» (с весом 1 / NEG_RATE)
# формула с разным весом вкуса — кривая «угадано → качество», с которой сравнивается бустинг при том же угаданном
FORMULA_WEIGHTS = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0)


def judge_values(r: np.ndarray, kind: str) -> np.ndarray:
    """Оценки 1–5 → цель вкуса: «value» — ценность по шкале судьи, «five» — 1 у пятёрки, иначе 0."""
    k = np.clip(metrics.rounded(r), 1, 5).astype(int)
    return (ly.JUDGE_STARS[k - 1] if kind == "value" else (k == 5).astype(np.float64)).astype(np.float32)


def _as_kind(X: sp.csr_matrix, kind: str) -> sp.csr_matrix:
    out = X.tocsr().astype(np.float32, copy=True)
    out.data = judge_values(out.data, kind)
    return out


def fit_value_tastes(*, ratings_path: Path, split_dir: Path, models_dir: Path, skip_ready: bool = False,
                     log=print) -> None:
    """Вкус с настройками models/taste, но цель — не звёзды: ценность судьи (models/taste_value) и «пятёрка или нет»
    (models/taste_five). Звёзды считают шаг 4 → 5 равным шагу 3 → 4, а судья — вдвое дороже. skip_ready — уже
    обученный не переобучать (продолжить прерванное; после нового разбиения — удалить обе папки)."""
    from booksengine.model.base import read_params
    from booksengine.model.matrix import RatingMatrix, load_train
    from booksengine.model.taste import Taste
    train = load_train(ratings_path, split_dir / "holdout_users.parquet")
    p = read_params(models_dir / "taste")
    for kind in ("value", "five"):
        if skip_ready and (models_dir / f"taste_{kind}" / "items.npz").exists():
            log(f"вкус «{kind}»: уже обучен")
            continue
        t0 = time.perf_counter()
        m = Taste(factors=p["factors"], reg=p["reg"], iterations=p["iterations"], seed=p["seed"])
        m.fit(RatingMatrix(_as_kind(train.X, kind), train.user_ids, train.work_ids), log=lambda *a: None)
        (models_dir / f"taste_{kind}").mkdir(parents=True, exist_ok=True)
        m.save(models_dir / f"taste_{kind}")
        log(f"вкус «{kind}»: {time.perf_counter() - t0:.0f} с")


def value_tastes(models_dir: Path) -> dict:
    """Обученные вкусы на шкале судьи: имя признака → (модель, цель)."""
    from booksengine.model.taste import Taste
    return {f"taste_{k}": (Taste.load(models_dir / f"taste_{k}"), k) for k in ("value", "five")
            if (models_dir / f"taste_{k}" / "params.json").exists()}


def book_table(clean_dir: Path, holdout_path: Path, work_ids: np.ndarray, info: pd.DataFrame,
               item_bias: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Признаки книг ядра (строки — столбцы матрицы × BOOK) и доли жанров (× GENRES). Известность, средняя и доля
    пятёрок — по людям обучения: у отложенного человека его спрятанная книга не должна прибавлять себе оценку."""
    con = duckdb.connect()
    st = con.execute("""SELECT work_id, count(*) AS n, avg(rating) AS m, avg((rating >= 4.5)::DOUBLE) AS f
                        FROM read_parquet(?) r ANTI JOIN read_parquet(?) h USING (user_id) GROUP BY 1""",
                     [str(clean_dir / "ratings.parquet"), str(holdout_path)]).df().set_index("work_id").reindex(work_ids)
    works = str(clean_dir / "works.parquet")
    has_year = "publication_year" in set(con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [works]).df().column_name)
    year = con.execute(f"SELECT work_id, {'publication_year' if has_year else 'NULL::INT'} AS y FROM read_parquet(?)",
                       [works]).df().set_index("work_id").y.reindex(work_ids)
    g = con.execute("SELECT work_id, genre, share FROM read_parquet(?)", [str(clean_dir / "work_genres.parquet")]).df()
    con.close()
    G = (g[g.genre.isin(GENRES)].pivot_table(index="work_id", columns="genre", values="share", aggfunc="max")
         .reindex(index=work_ids, columns=list(GENRES)).fillna(0.0).to_numpy(dtype=np.float32))
    sno = pd.to_numeric(info.series_no, errors="coerce").to_numpy(dtype=np.float64)
    B = np.column_stack([np.log1p(st.n.fillna(0).to_numpy()), st.m.to_numpy(), st.f.to_numpy(),
                         year.to_numpy(dtype=np.float64), item_bias, (sno == 1).astype(float),
                         np.isfinite(sno).astype(float), G]).astype(np.float32)
    return B, G


@dataclass
class Part:
    """Кандидаты отложенных людей, по убыванию балла толпы. Все кандидаты (pos — люди × POOL) — для списков и судьи;
    обучающая выборка (`build(neg_rate=...)`) — прочитанные и доля непрочитанных с весом, pos нет."""
    pos: np.ndarray | None     # люди × POOL — место книги среди книг EASE (столбец top_cols)
    X: np.ndarray              # строки × признаки FEATURES (без необученных вкусов — NaN)
    rating: np.ndarray         # оценка спрятанной книги, 0 — книга не среди спрятанных
    person: np.ndarray         # номер человека (в hold) каждой строки
    weight: np.ndarray | None = None   # вес строки в «прочтёт ли»; None — прорежает `fit`

    def label_value(self) -> np.ndarray:
        r = np.clip(self.rating, 1, 5).astype(int)
        return np.where(self.rating > 0, ly.JUDGE_STARS[r - 1], np.nan)


def _lookup(M: sp.csr_matrix, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    return np.asarray(M[rows, cols]).ravel()


def build(layers: "ly.Layers", hold, info: pd.DataFrame, books: np.ndarray, genres: np.ndarray,
          extra: dict | None = None, batch: int = 250, pool: int = POOL, neg_rate: float | None = None,
          seed: int = SEED, log=print) -> Part:
    """Кандидаты и признаки всех людей hold (входы, спрятанные, `exclude` — вход и начатые серии). neg_rate —
    обучающая выборка: непрочитанные кандидаты — эта доля с весом 1 / neg_rate (у 50 000 человек все кандидаты —
    7 ГБ)."""
    extra = extra or {}
    rng = np.random.default_rng(seed)
    persons, weights = [], []
    top = layers.mix.ease.top_cols
    taste, v = layers.taste, layers.variant
    F_top = taste._F()[top].astype(np.float64)
    author = info.author_id.fillna(-1).to_numpy(dtype=np.int64)
    K = int(author.max()) + 2
    exclude = (hold.inputs if hold.exclude is None else hold.exclude).tocsr()
    n_items = hold.inputs.shape[1]
    poss, feats, ratings = [], [], []
    t0 = time.perf_counter()
    for s in range(0, len(hold.user_ids), batch):
        X = hold.inputs[s:s + batch].tocsr()
        b = X.shape[0]
        excl = exclude[s:s + batch][:, top].toarray() != 0
        za, ze = layers.like_parts(X)
        crowd = layers.like_crowd(za, ze, v.als_weight)
        theta = taste.fold_in(layers.taste_input(X))
        tp = taste.mu + taste.item_bias[top][None, :] + theta @ F_top.T
        tz = _z(tp)
        formula = layers.combine(crowd, tz, excl, v)
        eo = layers.mix.ease_inputs(X) @ layers.mix.ease._B
        eo = _z(eo.toarray() if sp.issparse(eo) else np.asarray(eo))
        c = np.where(excl, -np.inf, crowd)
        p = np.argpartition(-c, pool - 1, axis=1)[:, :pool]
        p = np.take_along_axis(p, np.argsort(-np.take_along_axis(c, p, 1), axis=1, kind="stable"), 1)
        take = lambda M: np.take_along_axis(M, p, 1).ravel()
        fp = np.take_along_axis(formula, p, 1)
        frank = np.empty_like(p)
        np.put_along_axis(frank, np.argsort(-fp, axis=1, kind="stable"), np.arange(pool)[None, :].repeat(b, 0), 1)
        rows = np.repeat(np.arange(b), pool)
        cols = top[p.ravel()]
        # человек
        n = np.diff(X.indptr).astype(np.float64)
        ri = np.repeat(np.arange(b), np.diff(X.indptr))
        r = metrics.rounded(X.data)
        mean = np.bincount(ri, r, b) / n
        person = np.column_stack([np.log1p(n), mean, np.bincount(ri, (r >= 5).astype(float), b) / n,
                                  np.bincount(ri, (r <= 2).astype(float), b) / n, theta[:, 0]])
        # автор: сколько его книг оценено, насколько выше своей средней, лучшая оценка
        ai = author[X.indices]
        ok = ai >= 0
        agg = (pd.DataFrame({"k": ri[ok] * K + ai[ok], "dev": (r - mean[ri])[ok], "r": r[ok]})
               .groupby("k").agg(n=("r", "size"), dev=("dev", "mean"), mx=("r", "max")))
        keys = agg.index.to_numpy()
        ac = author[cols]
        q = rows * K + ac
        if len(keys):
            i = np.searchsorted(keys, q).clip(max=len(keys) - 1)
            hit = (keys[i] == q) & (ac >= 0)
        else:
            i, hit = np.zeros(len(q), dtype=int), np.zeros(len(q), dtype=bool)
        a_n = np.where(hit, agg.n.to_numpy()[i], 0.0) if len(keys) else np.zeros(len(q))
        a_dev = np.where(hit, agg.dev.to_numpy()[i], np.nan) if len(keys) else np.full(len(q), np.nan)
        a_max = np.where(hit, agg.mx.to_numpy()[i], np.nan) if len(keys) else np.full(len(q), np.nan)
        # жанры: понравившееся (оценка выше своей средней) и прочитанное
        Xdev = sp.csr_matrix(((r - mean[ri]).astype(np.float32), X.indices, X.indptr), shape=X.shape)
        Xone = sp.csr_matrix((np.ones(X.nnz, np.float32), X.indices, X.indptr), shape=X.shape)
        like_g, read_g = (Xdev @ genres) / n[:, None], (Xone @ genres) / n[:, None]
        gc = genres[cols]
        pair = np.column_stack([a_n, a_dev, a_max, (like_g[rows] * gc).sum(1), (read_g[rows] * gc).sum(1)])
        vt = []
        for name in VALUE_TASTES:
            if name not in extra:
                vt.append(np.full(len(rows), np.nan))
                continue
            m, kind = extra[name]
            th = m.fold_in(_as_kind(X, kind))
            vt.append(take(m.mu + m.item_bias[top][None, :] + th @ m._F()[top].astype(np.float64).T))
        scores = np.column_stack([take(crowd), take(ze), take(za), take(eo), take(tp), take(tz), fp.ravel(),
                                  np.tile(np.arange(pool), b), frank.ravel()])
        feats.append(np.column_stack([scores, person[rows], books[cols], pair, *vt]).astype(np.float32))
        # спрятанные книги
        u = range(s, s + b)
        hc = np.concatenate([hold.hidden_cols[k] for k in u])
        hr = np.concatenate([hold.hidden_ratings[k] for k in u])
        hrow = np.repeat(np.arange(b), [len(hold.hidden_cols[k]) for k in u])
        H = sp.csr_matrix((metrics.rounded(hr).astype(np.float32), (hrow, hc)), shape=(b, n_items))
        rt = _lookup(H, rows, cols).astype(np.float32)
        if neg_rate is None:
            poss.append(p)
            ratings.append(rt)
            persons.append(s + rows)
        else:
            keep = (rt > 0) | (rng.random(len(rt)) < neg_rate)
            feats[-1] = feats[-1][keep]
            ratings.append(rt[keep])
            persons.append(s + rows[keep])
            weights.append(np.where(rt[keep] > 0, 1.0, 1.0 / neg_rate).astype(np.float32))
        if (s // batch) % 10 == 0:
            log(f"  признаки: {s + b:,} из {len(hold.user_ids):,} человек, {time.perf_counter() - t0:.0f} с")
    return Part(np.concatenate(poss) if poss else None, np.concatenate(feats), np.concatenate(ratings),
                np.concatenate(persons), np.concatenate(weights) if weights else None)


ROUNDS, STOP = 2000, 50      # деревьев не больше стольких; остановка, если на отложенной десятой части нет улучшения


def _lgb_params(objective: str, seed: int) -> dict:
    return dict(objective=objective, learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, bagging_fraction=0.8,
                bagging_freq=1, feature_fraction=0.8, lambda_l2=1.0, seed=seed, verbose=-1)


@dataclass
class Rerank:
    """Две модели: «прочтёт ли» (p) и «ценность, если прочтёт» (v); балл — p·(v − c)."""
    read: object           # lightgbm.Booster
    value: object
    columns: list[int]     # номера признаков FEATURES, на которых учились

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Z = X[:, self.columns]
        return (self.read.predict(Z, num_iteration=self.read.best_iteration),
                self.value.predict(Z, num_iteration=self.value.best_iteration))

    @staticmethod
    def key(p: np.ndarray, v: np.ndarray, c: float) -> np.ndarray:
        return p if c == -np.inf else p * (v - c)


def fit(part: Part, people: np.ndarray, columns: list[int], seed: int = SEED, neg_rate: float = NEG_RATE) -> Rerank:
    """Учит обе модели на людях people (номера в hold); десятая часть людей — для ранней остановки.
    Непрочитанных кандидатов в «прочтёт ли» — доля neg_rate с весом 1 / neg_rate (если выборка не прорежена при
    сборке, `build(neg_rate=...)`): шанс остаётся честным."""
    import lightgbm as lgb
    rng = np.random.default_rng(seed)
    people = rng.permutation(people)
    n_stop = max(1, len(people) // 10)
    group = np.zeros(int(part.person.max()) + 1, dtype=np.int8)
    group[people[n_stop:]], group[people[:n_stop]] = 1, 2
    g = group[part.person]
    fit_r, stop_r = np.flatnonzero(g == 1), np.flatnonzero(g == 2)
    read = part.rating > 0

    def sample(r):
        if part.weight is not None:
            return r, part.weight[r]
        keep = read[r] | (rng.random(len(r)) < neg_rate)
        r = r[keep]
        return r, np.where(read[r], 1.0, 1.0 / neg_rate)

    def train(objective, r, y, w, sr, sy, sw):
        X = part.X[:, columns]
        d = lgb.Dataset(X[r], y, weight=w, free_raw_data=True)
        e = lgb.Dataset(X[sr], sy, weight=sw, reference=d)
        return lgb.train(_lgb_params(objective, seed), d, ROUNDS, valid_sets=[e],
                         callbacks=[lgb.early_stopping(STOP, verbose=False)])

    (fr, fw), (sr, sw) = sample(fit_r), sample(stop_r)
    clf = train("binary", fr, read[fr].astype(float), fw, sr, read[sr].astype(float), sw)
    y = part.label_value()
    fv, sv = fit_r[read[fit_r]], stop_r[read[stop_r]]
    reg = train("regression", fv, y[fv], None, sv, y[sv], None)
    return Rerank(clf, reg, list(columns))


def rated_filters(hold, picker) -> list:
    """«Уже оценено по сути» каждого человека hold — один раз на все списки."""
    from booksengine.model.filters import RatedFilter
    X = hold.inputs
    return [RatedFilter(picker.books, X.indices[X.indptr[u]:X.indptr[u + 1]]) for u in range(X.shape[0])]


def judge(keys: np.ndarray, part: Part, people: np.ndarray, hold, picker, rated: list, top: np.ndarray,
          pos_of: np.ndarray, pop: np.ndarray | None, batch: int = 500) -> pd.DataFrame:
    """Списки по баллам keys (люди people × POOL, в порядке part.pos) — теми же правилами и тем же судьёй, что у
    формулы (`filters.pick_top`, `layers.judge_row`); rated — `rated_filters`."""
    from booksengine.model.filters import pick_top
    rows = []
    for s in range(0, len(people), batch):
        ps = people[s:s + batch]
        sc = np.full((len(ps), len(top)), -np.inf, dtype=np.float64)
        np.put_along_axis(sc, part.pos[ps], keys[s:s + batch], 1)
        for u, t in zip(ps, pick_top(sc, top, picker, [rated[u] for u in ps], metrics.K)):
            rows.append(ly.judge_row(hold, int(u), t, pos_of, None, pop))
    return pd.DataFrame(rows)


def _mean(d: pd.DataFrame, m: str) -> float:
    x = d[m].to_numpy(dtype=np.float64)
    x = x[~np.isnan(x)]
    return float(x.mean()) if len(x) else float("nan")


def curve_at(curve: list[dict], hits: float) -> float | None:
    """Качество формулы при том же числе угаданных (линейно между её точками); вне кривой — None."""
    h = np.array([p["hits"] for p in curve])
    q = np.array([p["quality"] for p in curve])
    if not h.min() <= hits <= h.max():
        return None
    o = np.argsort(h)
    return float(np.interp(hits, h[o], q[o]))


def _paired(a: pd.DataFrame, b: pd.DataFrame, m: str) -> dict:
    """Парная разность b − a на тех же людях (по user_id), 95% бутстреп."""
    x = b.set_index("user_id")[m] - a.set_index("user_id")[m]
    return metrics.bootstrap(x.to_numpy(dtype=np.float64), np.random.default_rng(0), 1000)


def _by_bucket(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    out = {}
    for bk in BUCKET_ORDER:
        ua, ub = a[a.bucket == bk], b[b.bucket == bk]
        if len(ua):
            out[bk] = {"quality": _paired(ua, ub, "quality"), "hits": _paired(ua, ub, "hits"),
                       "formula_quality": _mean(ua, "quality"), "quality_new": _mean(ub, "quality")}
    return out


def check(*, clean_dir: Path, split_dir: Path, models_dir: Path, eval_dir: Path, folds: int = FOLDS,
          sizes=SIZES, c_grid=C_GRID, pool: int = POOL, log=print) -> dict:
    """Дешёвая проверка на проверочных людях: перекрёстно по частям, кривая обучения и наборы признаков; сравнение с
    формулой (выдача models/layers и её кривая по весу вкуса) — тем же судьёй и на тех же людях."""
    from booksengine.model.evaluate import load_eval_holdout
    from booksengine.model.filters import ListPicker, work_info
    from booksengine.model.matrix import catalog_works
    ratings_path = clean_dir / "ratings.parquet"
    work_ids = catalog_works(ratings_path)
    layers = ly.Layers.load(models_dir / "layers")
    info = work_info(clean_dir, work_ids)
    hold = load_eval_holdout(ratings_path, split_dir, "val", work_ids)
    top = layers.mix.ease.top_cols
    pos_of = np.full(len(work_ids), -1)
    pos_of[top] = np.arange(len(top))
    pop = ly.popularity(ratings_path, work_ids)
    picker = ListPicker(info)
    # формула: нынешняя выдача, опора порога и кривая по весу вкуса
    t0 = time.perf_counter()
    v = layers.variant
    formula_vs = [ly.Variant(v.crowd, w, v.cutoff, v.als_weight) for w in FORMULA_WEIGHTS]
    per = ly.evaluate(layers, hold, info, list(dict.fromkeys([v, ly.ANCHOR_LIKE, *formula_vs])), pop=pop)
    current, anchor = per[v], per[ly.ANCHOR_LIKE]
    floor = ly.GUARD * _mean(anchor, "hits")
    curve = [{"taste_weight": x.taste_weight, "hits": _mean(per[x], "hits"), "quality": _mean(per[x], "quality")}
             for x in formula_vs]
    log(f"формула: качество {_mean(current, 'quality'):.3f}, угадано {_mean(current, 'hits'):.2f}, порог "
        f"{floor:.2f} ({time.perf_counter() - t0:.0f} с)")
    books, genres = book_table(clean_dir, split_dir / "holdout_users.parquet", work_ids, info, layers.taste.item_bias)
    extra = value_tastes(models_dir)
    part = build(layers, hold, info, books, genres, extra, pool=pool, log=log)
    rated = rated_filters(hold, picker)
    n = part.pos.shape[0]
    fold = np.random.default_rng(SEED).permutation(n) % folds
    order = np.random.default_rng(SEED + 1).permutation(n)       # порядок набора людей в кривой обучения
    sets = {k: c for k, c in SETS.items() if k != "value" or extra}
    runs = [("full", size) for size in sizes] + [(k, sizes[-1]) for k in sets if k != "full"]
    res = {"n_users": int(n), "pool": pool, "floor": floor, "formula": {"label": v.label(),
           "quality": _mean(current, "quality"), "hits": _mean(current, "hits")}, "curve": curve,
           "positives": int((part.rating > 0).sum()), "runs": []}
    for name, size in runs:
        cols = [FEATURES.index(f) for f in sets[name]]
        P, V = np.zeros(n * pool), np.zeros(n * pool)
        t0 = time.perf_counter()
        trees = []
        for k in range(folds):
            test = np.flatnonzero(fold == k)
            train = order[fold[order] != k][:size]
            m = fit(part, train, cols, seed=SEED + k)
            trees.append((m.read.best_iteration, m.value.best_iteration))
            r = (test[:, None] * pool + np.arange(pool)[None, :]).ravel()
            P[r], V[r] = m.predict(part.X[r])
        people = np.arange(n)
        points = []
        for c in c_grid:
            d = judge(Rerank.key(P, V, c).reshape(n, pool), part, people, hold, picker, rated, top, pos_of, pop)
            points.append({"c": c, "per_user": d, "quality": _mean(d, "quality"), "hits": _mean(d, "hits")})
        ok = [p for p in points if p["hits"] >= floor]
        best = max(ok, key=lambda p: p["quality"]) if ok else max(points, key=lambda p: p["hits"])
        d = best["per_user"]
        at = curve_at(curve, best["hits"])
        run = {"set": name, "size": size, "trees": trees, "seconds": round(time.perf_counter() - t0),
               "points": [{k: p[k] for k in ("c", "quality", "hits")} for p in points],
               "chosen_c": best["c"], "passes_floor": bool(ok), "quality": best["quality"], "hits": best["hits"],
               "quality_diff": _paired(current, d, "quality"), "hits_diff": _paired(current, d, "hits"),
               "over_formula_curve": None if at is None else best["quality"] - at,
               "five": _mean(d, "s5"), "low": _mean(d, "s1") + _mean(d, "s2"),
               "formula_five": _mean(current, "s5"), "formula_low": _mean(current, "s1") + _mean(current, "s2"),
               "known": _mean(d, "known"), "formula_known": _mean(current, "known"),
               "buckets": _by_bucket(current, d)}
        if name == "full" and size == sizes[-1]:
            imp = pd.Series(m.read.feature_importance("gain"), index=[FEATURES[i] for i in cols])
            imv = pd.Series(m.value.feature_importance("gain"), index=[FEATURES[i] for i in cols])
            run["importance"] = {"read": (imp / imp.sum()).round(4).sort_values(ascending=False).to_dict(),
                                 "value": (imv / imv.sum()).round(4).sort_values(ascending=False).to_dict()}
        res["runs"].append(run)
        q, over = run["quality_diff"], run["over_formula_curve"]
        log(f"{name}, {size} человек: c = {best['c']:g}, качество {best['quality']:.3f} "
            f"({q['mean']:+.3f} [{q['lo']:+.3f}; {q['hi']:+.3f}]), угадано {best['hits']:.2f}, "
            f"сверх кривой формулы {'—' if over is None else f'{over:+.3f}'}, {run['seconds']} с")
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "rerank_check.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    return res


SET_NAMES = {"scores": "только баллы формулы (толпа, вкус, их места)", "full": "все признаки",
             "value": "все признаки + вкус на шкале судьи"}


def _ci(x: dict) -> str:
    return "—" if x["mean"] is None else f"{x['mean']:+.3f} [{x['lo']:+.3f}; {x['hi']:+.3f}]"


def report(res: dict) -> str:
    f = res["formula"]
    runs = res["runs"]
    full = max((r for r in runs if r["set"] == "full"), key=lambda r: r["size"])
    best = max(runs, key=lambda r: r["quality"] if r["passes_floor"] else -np.inf)
    d = best["quality_diff"]
    sure = d["lo"] is not None and d["lo"] > 0
    curve = [r for r in runs if r["set"] == "full"]
    grows = len(curve) > 1 and curve[-1]["quality_diff"]["mean"] > curve[0]["quality_diff"]["mean"] + 0.005
    verdict = (f"**Бустинг лучше формулы уверенно** ({SET_NAMES[best['set']]}, {best['size']:,} человек обучения): "
               if sure else f"**Бустинг не лучше формулы уверенно** (лучший — {SET_NAMES[best['set']]}): ")
    verdict += f"качество {_ci(d)}, угадано {best['hits']:.2f} против {f['hits']:.2f} (порог {res['floor']:.2f})."
    lines = [f"# Переранжирование бустингом — дешёвая проверка на {res['n_users']:,} проверочных людях", "", verdict,
             "Выигрыш " + ("растёт" if grows else "не растёт") + " с числом людей обучения: "
             + " → ".join(f"{r['quality_diff']['mean']:+.3f} ({r['size']:,})" for r in curve) + ".", "",
             f"Формула («{f['label']}»): качество {f['quality']:.3f}, угадано {f['hits']:.2f}. Кандидаты — первые "
             f"{res['pool']} книг толпы; прочитанных среди кандидатов {res['positives']:,}.", "",
             "| вариант | людей обучения | c | качество | разница с формулой | сверх кривой формулы | угадано | 5★ | 1–2★ |",
             "|---|---|---|---|---|---|---|---|---|",
             f"| формула | | | {f['quality']:.3f} | | | {f['hits']:.2f} | {full['formula_five']:.1%} | "
             f"{full['formula_low']:.1%} |"]
    for r in runs:
        over = "—" if r["over_formula_curve"] is None else f"{r['over_formula_curve']:+.3f}"
        mark = "" if r["passes_floor"] else " (мало угадывает)"
        lines.append(f"| {SET_NAMES[r['set']]}{mark} | {r['size']:,} | {r['chosen_c']:g} | {r['quality']:.3f} | "
                     f"{_ci(r['quality_diff'])} | {over} | {r['hits']:.2f} | {r['five']:.1%} | {r['low']:.1%} |")
    lines += ["", f"По этапам ({SET_NAMES[full['set']]}, {full['size']:,} человек обучения):", "",
              "| оценок | качество формулы | бустинга | разница | угадано: разница |", "|---|---|---|---|---|"]
    for bk, g in full["buckets"].items():
        lines.append(f"| {bk} | {g['formula_quality']:.3f} | {g['quality_new']:.3f} | {_ci(g['quality'])} | "
                     f"{_ci(g['hits'])} |")
    lines += ["", "Кривая формулы (вес вкуса → угадано, качество): "
              + "; ".join(f"{p['taste_weight']:g} → {p['hits']:.2f}, {p['quality']:.3f}" for p in res["curve"]) + "."]
    if "importance" in full:
        for k, name in (("read", "«прочтёт ли»"), ("value", "«ценность, если прочтёт»")):
            imp = list(full["importance"][k].items())[:8]
            lines.append(f"Главные признаки модели {name}: " + ", ".join(f"{a} {b:.0%}" for a, b in imp) + ".")
    lines += ["", "**Как читать.** Бустинг — две модели: «прочтёт ли человек книгу» (p) и «насколько она ему понравится, "
              "если прочтёт» (v, по шкале судьи). Балл книги — p·(v − c): c — ручка смелости, как вес вкуса у формулы "
              "(больше c — угаданные ценнее, но их меньше); выбирается то c, где качество лучше всего, а угадано не "
              "меньше порога. Учатся перекрёстно: люди поделены на 5 частей, модели учатся на четырёх и судятся на "
              "пятой — ни один человек не судится моделью, которая видела его книги. **Людей обучения** — сколько "
              "людей из четырёх частей взято в обучение: если выигрыш растёт с ними, стоит отложить больше людей. "
              "**Разница с формулой** — парно на тех же людях, в скобках 95% интервал. **Сверх кривой формулы** — "
              "качество минус качество формулы с другим весом вкуса при том же числе угаданных: смелее список — "
              "качество растёт и само, сравнивать честно только при равном угаданном. " + ly.HOW_TO_READ]
    return "\n".join(lines) + "\n"


# Большая проверка: отдельные люди обучения бустинга, которых не видели толпа и вкус
RANK_PER_BUCKET = 10_000          # людей обучения на этап — 50 000 всего
RANK_SEED = SEED + 50
LEARN = (5_000, 15_000, 50_000)   # кривая обучения на большой группе


def make_split(*, ratings_path: Path, users_path: Path, split_dir: Path, out_dir: Path,
               per_bucket: int = RANK_PER_BUCKET, seed: int = RANK_SEED) -> dict:
    """Разбиение для бустинга (out_dir): проверочные и тестовые — копия split_dir, плюс группа «rank» — по per_bucket
    человек с каждого этапа из остальных (по стабильному хэшу, как `split`), у каждого спрятано 20% оценок. Толпа и
    вкус учатся без всех трёх групп (`refit`)."""
    import shutil

    from booksengine.model import split
    held = pd.read_parquet(split_dir / "holdout_users.parquet")
    con = duckdb.connect()
    counts = con.execute("SELECT user_id, count(*) AS n FROM read_parquet(?) GROUP BY 1 ORDER BY 1",
                         [str(ratings_path)]).df()
    counts = counts.merge(con.execute("SELECT user_id, external_id FROM read_parquet(?)", [str(users_path)]).df(),
                          on="user_id")
    counts = counts[~counts.user_id.isin(held.user_id)].reset_index(drop=True)
    counts["bucket"] = split.bucket_of(counts.n.to_numpy())
    counts["draw"] = split.hash01(counts.external_id.to_numpy(), seed)
    chosen = (counts[counts.bucket.isin(BUCKET_ORDER)].sort_values("draw").groupby("bucket").head(per_bucket)
              .sort_values("user_id"))
    con.register("ids", chosen[["user_id"]])
    r = con.execute("SELECT r.user_id, r.work_id, r.rating FROM read_parquet(?) r JOIN ids USING (user_id)",
                    [str(ratings_path)]).df()
    total = con.execute("SELECT count(*) FROM read_parquet(?)", [str(ratings_path)]).fetchone()[0]
    con.close()
    inp, hid = split.hide(r, split.HIDDEN_SHARE, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    inp.to_parquet(out_dir / "rank_input.parquet", index=False)
    hid.to_parquet(out_dir / "rank_hidden.parquet", index=False)
    for g in split.GROUPS:
        for kind in ("input", "hidden"):
            shutil.copy(split_dir / f"{g}_{kind}.parquet", out_dir / f"{g}_{kind}.parquet")
    rank = pd.DataFrame({"user_id": chosen.user_id.to_numpy(dtype=held.user_id.dtype), "group": "rank",
                         "n_ratings": chosen.n.to_numpy(), "bucket": chosen.bucket.to_numpy()})
    pd.concat([held, rank], ignore_index=True).to_parquet(out_dir / "holdout_users.parquet", index=False)
    meta = {"source": str(split_dir), "per_bucket": per_bucket, "seed": seed, "rank_users": int(len(rank)),
            "rank_ratings": int(len(r)), "rank_ratings_share": float(len(r) / total),
            "buckets": {b: int((rank.bucket == b).sum()) for b in BUCKET_ORDER}}
    (out_dir / "split.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    return meta


def refit(*, clean_dir: Path, split_dir: Path, models_dir: Path, source_models: Path, log=print) -> None:
    """Все части формулы с настройками source_models — без людей обучения бустинга (split_dir — `make_split`):
    ALS, EASE, смесь, вкус, толпа «ценность», вкусы на шкале судьи. Вариант формулы тот же — это то же самое на
    меньшем числе людей, а не подбор."""
    from booksengine.data import merged
    from booksengine.model.base import read_params
    merged.refit(clean_dir, split_dir, models_dir, source_models, min_user=read_params(source_models / "ease_like")["min_user"],
                 log=log)
    p = json.loads((source_models / "layers" / "params.json").read_text())
    p["components"] = ly.Layers.component_fingerprints(models_dir)
    (models_dir / "layers" / "params.json").write_text(json.dumps(p, ensure_ascii=False, indent=1))
    fit_value_tastes(ratings_path=clean_dir / "ratings.parquet", split_dir=split_dir, models_dir=models_dir, log=log)


def _choose(points: list[dict], floor: float) -> dict:
    """Лучшее качество среди c, угадывающих не меньше порога; если таких нет — больше всех угадывающее."""
    ok = [p for p in points if p["hits"] >= floor]
    return max(ok, key=lambda p: p["quality"]) if ok else max(points, key=lambda p: p["hits"])


def _formula(layers, hold, info, pop) -> tuple[pd.DataFrame, float, list[dict]]:
    """Формула на людях hold: нынешний вариант, порог угаданного (`layers.GUARD` от опоры) и кривая по весу вкуса."""
    v = layers.variant
    vs = [ly.Variant(v.crowd, w, v.cutoff, v.als_weight) for w in FORMULA_WEIGHTS]
    per = ly.evaluate(layers, hold, info, list(dict.fromkeys([v, ly.ANCHOR_LIKE, *vs])), pop=pop)
    curve = [{"taste_weight": x.taste_weight, "hits": _mean(per[x], "hits"), "quality": _mean(per[x], "quality")}
             for x in vs]
    return per[v], ly.GUARD * _mean(per[ly.ANCHOR_LIKE], "hits"), curve


def _side(d: pd.DataFrame, ref: pd.DataFrame, curve: list[dict]) -> dict:
    at = curve_at(curve, _mean(d, "hits"))
    return {"quality": _mean(d, "quality"), "hits": _mean(d, "hits"), "five": _mean(d, "s5"),
            "low": _mean(d, "s1") + _mean(d, "s2"), "known": _mean(d, "known"),
            "quality_diff": _paired(ref, d, "quality"), "hits_diff": _paired(ref, d, "hits"),
            "over_formula_curve": None if at is None else _mean(d, "quality") - at, "buckets": _by_bucket(ref, d)}


def final(*, clean_dir: Path, split_dir: Path, models_dir: Path, prod_models: Path, eval_dir: Path,
          set_name: str = "value", learn=LEARN, c_grid=C_GRID, pool: int = POOL, log=print) -> dict:
    """Большая проверка: бустинг учится на людях «rank» (`make_split`), части формулы — без них (`refit`, models_dir).
    Проверка: кривая обучения и выбор c — на проверочных людях; один замер — на тесте против формулы на тех же частях
    и против нынешней выдачи (prod_models)."""
    from booksengine.model.evaluate import load_eval_holdout
    from booksengine.model.filters import ListPicker, work_info
    from booksengine.model.matrix import catalog_works
    ratings_path = clean_dir / "ratings.parquet"
    work_ids = catalog_works(ratings_path)
    info = work_info(clean_dir, work_ids)
    pop = ly.popularity(ratings_path, work_ids)
    picker = ListPicker(info)
    res = {"set": set_name, "pool": pool}
    # нынешняя выдача на тесте — первой: две модели в памяти разом не держим
    test = load_eval_holdout(ratings_path, split_dir, "test", work_ids)
    prod = ly.Layers.load(prod_models / "layers")
    prod_test = ly.evaluate(prod, test, info, [prod.variant], pop=pop)[prod.variant]
    del prod
    layers = ly.Layers.load(models_dir / "layers")
    if set_name == "value" and not value_tastes(models_dir):
        raise ValueError(f"{models_dir}: нет вкусов на шкале судьи — `rerank refit`")
    extra = value_tastes(models_dir)
    cols = [FEATURES.index(f) for f in SETS[set_name]]
    top = layers.mix.ease.top_cols
    pos_of = np.full(len(work_ids), -1)
    pos_of[top] = np.arange(len(top))
    books, genres = book_table(clean_dir, split_dir / "holdout_users.parquet", work_ids, info, layers.taste.item_bias)
    t0 = time.perf_counter()
    rank = load_eval_holdout(ratings_path, split_dir, "rank", work_ids)
    tp = build(layers, rank, info, books, genres, extra, pool=pool, neg_rate=NEG_RATE, log=log)
    n_rank = len(rank.user_ids)
    del rank
    log(f"люди обучения: {n_rank:,}, строк {len(tp.rating):,}, прочитанных {(tp.rating > 0).sum():,} "
        f"({time.perf_counter() - t0:.0f} с)")
    order = np.random.default_rng(RANK_SEED).permutation(n_rank)
    sizes = [s for s in learn if s < n_rank] + [n_rank]
    models = {}
    for s in sizes:
        t0 = time.perf_counter()
        models[s] = fit(tp, order[:s], cols)
        log(f"бустинг на {s:,} людях: деревьев {models[s].read.best_iteration} и {models[s].value.best_iteration}, "
            f"{time.perf_counter() - t0:.0f} с")
    imp = pd.Series(models[sizes[-1]].read.feature_importance("gain"), index=[FEATURES[i] for i in cols])
    imv = pd.Series(models[sizes[-1]].value.feature_importance("gain"), index=[FEATURES[i] for i in cols])
    res["importance"] = {"read": (imp / imp.sum()).round(4).sort_values(ascending=False).to_dict(),
                         "value": (imv / imv.sum()).round(4).sort_values(ascending=False).to_dict()}
    del tp
    # проверка: кривая обучения и выбор c
    val = load_eval_holdout(ratings_path, split_dir, "val", work_ids)
    f_val, floor, curve = _formula(layers, val, info, pop)
    vp = build(layers, val, info, books, genres, extra, pool=pool, log=log)
    rated = rated_filters(val, picker)
    n = vp.pos.shape[0]
    res["val"] = {"n_users": int(n), "floor": floor, "curve": curve, "formula": _side(f_val, f_val, curve),
                  "learn": []}
    for s in sizes:
        P, V = models[s].predict(vp.X)
        points = []
        for c in c_grid:
            d = judge(Rerank.key(P, V, c).reshape(n, pool), vp, np.arange(n), val, picker, rated, top, pos_of, pop)
            points.append({"c": c, "per_user": d, "quality": _mean(d, "quality"), "hits": _mean(d, "hits")})
        best = _choose(points, floor)
        row = {"size": s, "c": best["c"], "passes_floor": best["hits"] >= floor,
               "points": [{k: p[k] for k in ("c", "quality", "hits")} for p in points]} | _side(best["per_user"], f_val, curve)
        res["val"]["learn"].append(row)
        log(f"проверка, бустинг на {s:,} людях: c = {best['c']:g}, качество {row['quality']:.3f} "
            f"({row['quality_diff']['mean']:+.3f} [{row['quality_diff']['lo']:+.3f}; {row['quality_diff']['hi']:+.3f}])"
            f", угадано {row['hits']:.2f} (порог {floor:.2f})")
    del vp, val, rated
    # тест: один замер бустинга на всех людях обучения с c, выбранным на проверке
    c = res["val"]["learn"][-1]["c"]
    f_test, t_floor, t_curve = _formula(layers, test, info, pop)
    xp = build(layers, test, info, books, genres, extra, pool=pool, log=log)
    P, V = models[sizes[-1]].predict(xp.X)
    d = judge(Rerank.key(P, V, c).reshape(xp.pos.shape[0], pool), xp, np.arange(xp.pos.shape[0]), test, picker,
              rated_filters(test, picker), top, pos_of, pop)
    res["test"] = {"n_users": int(len(test.user_ids)), "c": c, "floor": t_floor, "curve": t_curve,
                   "rerank": _side(d, f_test, t_curve), "formula": _side(f_test, f_test, t_curve),
                   "rerank_vs_prod": _side(d, prod_test, t_curve), "formula_vs_prod": _side(f_test, prod_test, t_curve),
                   "prod": _side(prod_test, prod_test, t_curve)}
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "rerank_final.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    return res


def report_final(res: dict) -> str:
    v, t = res["val"], res["test"]
    r, rp = t["rerank"], t["rerank_vs_prod"]
    sure = lambda x: x["lo"] is not None and x["lo"] > 0
    verdict = (f"**Бустинг лучше нынешней выдачи уверенно**: " if sure(rp["quality_diff"]) else
               f"**Бустинг не лучше нынешней выдачи уверенно**: ")
    verdict += (f"качество на тесте {_ci(rp['quality_diff'])}, угадано {rp['hits']:.2f} против {t['prod']['hits']:.2f} "
                f"(порог {t['floor']:.2f}); против формулы на тех же частях — {_ci(r['quality_diff'])}.")
    lines = [f"# Переранжирование бустингом — большая проверка ({SET_NAMES[res['set']]})", "", verdict, "",
             "Кривая обучения (проверка, c выбирается на ней же):", "",
             "| людей обучения | c | качество | разница с формулой | сверх кривой формулы | угадано |",
             "|---|---|---|---|---|---|",
             f"| формула | | {v['formula']['quality']:.3f} | | | {v['formula']['hits']:.2f} |"]
    for x in v["learn"]:
        over = "—" if x["over_formula_curve"] is None else f"{x['over_formula_curve']:+.3f}"
        lines.append(f"| {x['size']:,}{'' if x['passes_floor'] else ' (мало угадывает)'} | {x['c']:g} | "
                     f"{x['quality']:.3f} | {_ci(x['quality_diff'])} | {over} | {x['hits']:.2f} |")
    lines += ["", f"Тест, {t['n_users']:,} человек, c = {t['c']:g}:", "",
              "| выдача | качество | разница с нынешней | угадано | 5★ | 1–2★ | известность |", "|---|---|---|---|---|---|---|"]
    for name, x in (("нынешняя (формула, все люди обучения)", t["prod"]),
                    ("формула без людей бустинга", t["formula_vs_prod"]), ("бустинг", rp)):
        diff = "" if x is t["prod"] else _ci(x["quality_diff"])
        lines.append(f"| {name} | {x['quality']:.3f} | {diff} | {x['hits']:.2f} | {x['five']:.1%} | {x['low']:.1%} | "
                     f"{x['known']:,.0f} |".replace(",", " "))
    lines += ["", "По этапам на тесте (бустинг против нынешней):", "", "| оценок | нынешняя | бустинг | разница |",
              "|---|---|---|---|"]
    for bk, g in rp["buckets"].items():
        lines.append(f"| {bk} | {g['formula_quality']:.3f} | {g['quality_new']:.3f} | {_ci(g['quality'])} |")
    for k, name in (("read", "«прочтёт ли»"), ("value", "«ценность, если прочтёт»")):
        imp = list(res["importance"][k].items())[:8]
        lines.append("")
        lines.append(f"Главные признаки модели {name}: " + ", ".join(f"{a} {b:.0%}" for a, b in imp) + ".")
    lines += ["", "**Как читать.** Бустинг учится на отдельных людях, которых не видели толпа и вкус; толпа и вкус "
              "для этого переобучены без них («формула без людей бустинга» — сколько стоит их убрать). c выбрано на "
              "проверочных людях, тест судится один раз. " + ly.HOW_TO_READ]
    return "\n".join(lines) + "\n"
