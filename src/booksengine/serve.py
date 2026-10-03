"""`booksengine serve`: выдача для приложения (C# API) — тот же код, что у консольного `recommend`.

Модель меняется почти каждый день, и повтор её на C# отставал бы от консоли (так было со смесью): сервис держит в
памяти `recommend.Engine` домена и отвечает API по HTTP на localhost. Книги — по внешнему id, как в каталоге БД:
`goodreads` — work_id Goodreads, `amazon` — ключ новой книги «автор|название» (`merged.new_work_keys`: work_id новых
книг сдвигается при пересборке базы, ключ — нет).

POST /recommend  {ratings: [{source, id, rating, dnf, title}], top, rank_tops} → списки с шансом и подписями;
                 ranks — состав списков размеров rank_tops (места для стрелок в приложении)
POST /chance     {ratings, work: {source, id}}                         → шанс одной книги (null — не кандидат), в модели ли
POST /match      {rows: [{title_en, author}]}                          → новые книги по английскому названию и автору
GET  /similar?source=&id=                                              → «ценят её — ценят и эти» (связи толпы)
GET  /health                                                           → домен, отпечатки модели и кода
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import scipy.sparse as sp

from booksengine import recommend as rec
from booksengine.data.merged import match_new_works, new_work_keys
from booksengine.model.mix import Mix

GOODREADS, AMAZON = "goodreads", "amazon"
SIMILAR = 10


class Service:
    """Выдача по запросам приложения; один расчёт за раз (`lock`): модель не рассчитана на параллельные вызовы."""

    def __init__(self, engine: rec.Engine, domain: str = ""):
        self.engine, self.domain = engine, domain
        self.keys = new_work_keys(engine.clean_dir)                       # work_id → ключ новой книги
        self.by_key = pd.Series(self.keys.index, index=self.keys.to_numpy())
        model = engine.model
        mix = model if isinstance(model, Mix) else (model.like_mix or model.mix)
        self.top_cols = mix.ease.top_cols                                 # позиция EASE → столбец ядра
        self.links = sp.csr_matrix(mix.ease._B)                           # связи «откуда → куда» по позициям EASE
        self.pos = np.full(len(engine.work_ids), -1)
        self.pos[self.top_cols] = np.arange(len(self.top_cols))
        self.code = rec.code_fingerprint()
        self.lock = threading.Lock()

    # ---- id ----

    def work_id(self, source: str, ext) -> int | None:
        """Внешний id → work_id базы; неизвестная книга — None."""
        if source == GOODREADS:
            try:
                return int(ext)
            except (TypeError, ValueError):
                return None
        if source == AMAZON:
            w = self.by_key.get(str(ext))
            return None if w is None else int(w)
        raise ValueError(f"источник книги — {GOODREADS} или {AMAZON}, а не {source!r}")

    def ref(self, work_id: int) -> dict:
        if work_id in self.keys.index:
            return {"source": AMAZON, "id": self.keys[work_id]}
        return {"source": GOODREADS, "id": str(work_id)}

    def col(self, work_id: int | None) -> int | None:
        ids = self.engine.work_ids
        c = int(np.searchsorted(ids, work_id)) if work_id is not None else len(ids)
        return c if c < len(ids) and ids[c] == work_id else None

    def profile(self, ratings: list[dict]) -> rec.Profile:
        rows = [(self.work_id(r["source"], r["id"]), r.get("rating"), "dnf" if r.get("dnf") else "read",
                 r.get("title") or str(r["id"])) for r in ratings]
        frame = pd.DataFrame(rows, columns=["goodreads_work_id", "rating", "status", "title"])
        frame["goodreads_work_id"] = frame.goodreads_work_id.astype("Int64")
        frame["rating"] = pd.to_numeric(frame.rating)
        return self.engine.profile(frame)

    # ---- ответы ----

    def recommend(self, body: dict) -> dict:
        top, rank_tops = int(body.get("top", 20)), tuple(int(k) for k in body.get("rank_tops", ()))
        if not all(1 <= k <= 100 for k in (top, *rank_tops)):
            raise ValueError("top и rank_tops — от 1 до 100")
        with self.lock:
            res = self.engine.recommend(self.profile(body.get("ratings", [])), top=top, rank_tops=rank_tops)
        sections: list[dict] = []
        for r in res.recs:
            if not sections or sections[-1]["name"] != r.section:
                sections.append({"name": r.section, "note": rec.NEW_NOTE if r.section == rec.NEW else None,
                                 "items": []})
            sections[-1]["items"].append(self.ref(r.work_id) | {
                "title": r.title, "author": r.author, "chance": r.chance, "why": rec.why_text(r)})
        return {"used": res.n_used, "chance_label": res.chance, "legend": rec.LEGEND, "sections": sections,
                "skipped": [{"title": t, "reason": w} for t, w in res.skipped],
                "ranks": {str(k): [self.ref(w) | {"section": sec} for sec, w in v] for k, v in res.ranks.items()}}

    def chance(self, body: dict) -> dict:
        w = body["work"]
        c = self.col(self.work_id(w["source"], w["id"]))
        out = {"chance": None, "chance_label": self.engine.chance.label(),
               "in_model": c is not None and bool(self.pos[c] >= 0)}
        if c is None:
            return out
        with self.lock:
            prof = self.profile(body.get("ratings", []))
            if prof.x.nnz == 0:
                return out
            sc, _ = self.engine.scores(prof)
            pct = self.engine.chance_pct(prof, sc, np.array([c]))[0]
        out["chance"] = None if np.isnan(pct) else int(round(pct * 100))
        return out

    def similar(self, source: str, ext: str, n: int = SIMILAR) -> dict:
        """Книги, которые сильнее всего поднимает эта (связи толпы с плюсом, по убыванию); вне EASE — пусто."""
        c = self.col(self.work_id(source, ext))
        if c is None or self.pos[c] < 0:
            return {"items": []}
        p = self.pos[c]
        a, b = self.links.indptr[p], self.links.indptr[p + 1]
        to, w = self.links.indices[a:b], self.links.data[a:b]
        o = np.argsort(-w, kind="stable")
        best = to[o][w[o] > 0][:n]
        return {"items": [self.ref(int(self.engine.work_ids[self.top_cols[t]])) for t in best.tolist()]}

    def match(self, body: dict) -> dict:
        rows = body.get("rows", [])
        found = match_new_works(self.engine.clean_dir, [r.get("title_en") for r in rows],
                                [r.get("author") for r in rows])
        return {"items": [None if pd.isna(w) else self.ref(int(w)) for w in found]}

    def health(self) -> dict:
        return {"domain": self.domain, "model": self.engine.model_fp, "code": self.code,
                "chance_label": self.engine.chance.label()}


def handler(service: Service) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _run(self, fn) -> None:
            try:
                self._send(200, fn())
            except (ValueError, KeyError, TypeError) as e:
                self._send(400, {"error": f"{type(e).__name__}: {e}"})

        def do_GET(self) -> None:
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            if url.path == "/health":
                self._run(service.health)
            elif url.path == "/similar":
                self._run(lambda: service.similar(q["source"], q["id"]))
            else:
                self._send(404, {"error": "нет такого адреса"})

        def do_POST(self) -> None:
            routes = {"/recommend": service.recommend, "/chance": service.chance, "/match": service.match}
            fn = routes.get(urlparse(self.path).path)
            if fn is None:
                self._send(404, {"error": "нет такого адреса"})
                return
            if self.headers.get("Content-Length") is None:   # chunked http.server не читает — не молчать пустым телом
                self._send(411, {"error": "нужен Content-Length"})
                return
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
            self._run(lambda: fn(body))

        def log_message(self, fmt, *args) -> None:   # без строки на каждый запрос
            pass

    return Handler


def run(*, clean_dir: Path, models_dir: Path, domain: str, host: str = "127.0.0.1", port: int = 5090) -> None:
    t0 = time.monotonic()
    service = Service(rec.Engine(clean_dir, models_dir), domain)
    print(f"Выдача {domain} ({service.engine.model_fp[:12]}) загружена за {time.monotonic() - t0:.0f} с: "
          f"http://{host}:{port}", flush=True)
    ThreadingHTTPServer((host, port), handler(service)).serve_forever()
