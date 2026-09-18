"""FastAPI app. Serves the JSON API under /api and (in production) the built React SPA from frontend/dist.

    uv run uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.schemas import AskRequest, CompareRequest
from config import ROOT, get_settings
from db.conn import get_pool
from evals.metrics import retrieval_metrics
from pipeline.chunking.base import Passage
from pipeline.chunking.strategies import DEFAULT_STRATEGIES, build_chunker
from pipeline.graph import run_pipeline, shutdown
from pipeline.logging_setup import setup_logging
from pipeline.query_log import fetch_query_log, fetch_query_logs
from pipeline.retrieval.expand import EXPANSIONS
from pipeline.retrieval.runner import RETRIEVERS, RetrievalConfig, config_from_dict
from pipeline.retrieval.store import list_strategies
from pipeline.retrieval.transforms import TRANSFORMS
from pipeline.tracing import provider

log = logging.getLogger("api")
DIST = ROOT / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    setup_logging(s.log_level, s.log_dir, "api")
    get_pool()
    # warm the local models + BM25 caches so the first request is not a 60 s stall
    from pipeline.embedder import get_embedder
    from pipeline.reranker import get_reranker
    from pipeline.retrieval.bm25 import get_index

    await run_in_threadpool(get_embedder)
    await run_in_threadpool(get_reranker)  # MedCPT download is ~440 MB on first run
    for st in await run_in_threadpool(list_strategies):
        try:
            await run_in_threadpool(get_index, st["name"])
        except Exception as e:  # noqa: BLE001
            log.warning("[api] bm25 warmup failed for %s: %s", st["name"], e)
    log.info("[api] ready. tracing=%s", provider())
    yield
    shutdown()


app = FastAPI(title="fde-chunking-retrieval-exp", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _cfg(req: AskRequest | dict) -> RetrievalConfig:
    d = req if isinstance(req, dict) else req.model_dump()
    cfg = config_from_dict(d)
    try:
        cfg.validate()
    except AssertionError as e:
        raise HTTPException(400, str(e)) from e
    return cfg


def _gold(qa_id: int | None) -> set[int]:
    if qa_id is None:
        return set()
    with get_pool().connection() as conn, conn.cursor() as cur:
        cur.execute("select relevant_passage_ids from qa_pairs where id = %s", (qa_id,))
        r = cur.fetchone()
    return set(r["relevant_passage_ids"]) if r else set()


# ── meta ──────────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    with get_pool().connection() as conn, conn.cursor() as cur:
        cur.execute("select count(*) n from passages")
        n = cur.fetchone()["n"]
        cur.execute("select pg_size_pretty(pg_database_size(current_database())) as size")
        size = cur.fetchone()["size"]
    return {"ok": True, "passages": n, "db_size": size, "tracing": provider()}


@app.get("/api/options")
def options():
    s = get_settings()
    return {"strategies": list_strategies(), "retrievers": list(RETRIEVERS), "transforms": list(TRANSFORMS),
            "expansions": list(EXPANSIONS), "embedding_model": s.embedding_model, "reranker_model": s.reranker_model,
            "llm_provider": s.llm_provider, "llm_models": s.gemini_model_list if s.llm_provider == "gemini" else [s.deepseek_model],
            "tracing": provider()}


# ── ask / compare ─────────────────────────────────────────────────────────────
@app.post("/api/ask")
async def ask(req: AskRequest):
    cfg = _cfg(req)
    gold = _gold(req.qa_id)
    out = await run_in_threadpool(run_pipeline, req.question, cfg, req.qa_id, "api", not req.generate, None,
                                  req.session_id)
    if gold:
        out["metrics"] = retrieval_metrics([h["passage_id"] for h in out["retrieved"]], gold)
        out["gold_passage_ids"] = sorted(gold)
        for h in out["retrieved"]:
            h["is_gold"] = h["passage_id"] in gold
    return out


@app.post("/api/compare")
async def compare(req: CompareRequest):
    gold = _gold(req.qa_id)
    results = []
    for c in req.configs:
        cfg = _cfg(c)
        out = await run_in_threadpool(run_pipeline, req.question, cfg, req.qa_id, "api", not req.generate)
        if gold:
            out["metrics"] = retrieval_metrics([h["passage_id"] for h in out["retrieved"]], gold)
            for h in out["retrieved"]:
                h["is_gold"] = h["passage_id"] in gold
        out["label"] = cfg.label()
        results.append(out)
    return {"question": req.question, "gold_passage_ids": sorted(gold), "results": results}


# ── corpus & chunk explorer ───────────────────────────────────────────────────
@app.get("/api/questions")
def questions(set_name: str = Query("eval150", alias="set"), q: str | None = None, limit: int = 200):
    with get_pool().connection() as conn, conn.cursor() as cur:
        if set_name == "all":
            cur.execute("select id, question, question_type, cardinality(relevant_passage_ids) n_gold from qa_pairs "
                        "where cardinality(relevant_passage_ids) > 0 and (%s::text is null or question ilike %s::text) order by id limit %s",
                        (q, f"%{q}%" if q else None, limit))
        else:
            cur.execute("""select q.id, q.question, q.question_type, cardinality(q.relevant_passage_ids) n_gold
                           from eval_sets e join qa_pairs q on q.id = e.qa_id where e.name = %s
                           and (%s::text is null or q.question ilike %s::text) order by e.position limit %s""",
                        (set_name, q, f"%{q}%" if q else None, limit))
        return cur.fetchall()


@app.get("/api/questions/{qa_id}")
def question(qa_id: int):
    with get_pool().connection() as conn, conn.cursor() as cur:
        cur.execute("select id, question, answer, relevant_passage_ids, n_gold_total, question_type from qa_pairs where id = %s", (qa_id,))
        r = cur.fetchone()
        if not r:
            raise HTTPException(404, "unknown question")
        cur.execute("select id, left(text, 300) preview from passages where id = any(%s)", (r["relevant_passage_ids"],))
        r["gold_passages"] = cur.fetchall()
        return r


@app.get("/api/passages/{pmid}")
def passage(pmid: int, strategies: str | None = None):
    """Passage text plus how every strategy chunks it (the Chunk Explorer)."""
    with get_pool().connection() as conn, conn.cursor() as cur:
        cur.execute("select id, text, n_words, n_tokens, sentence_offsets, kg_entity_ids from passages where id = %s", (pmid,))
        p = cur.fetchone()
        if not p:
            raise HTTPException(404, "unknown passage")
        cur.execute("select strategy, id, chunk_index, char_start, char_end, n_tokens from chunks where passage_id = %s "
                    "order by strategy, chunk_index", (pmid,))
        chunks: dict[str, list] = {}
        for r in cur.fetchall():
            chunks.setdefault(r["strategy"], []).append(r)
        cur.execute("select id, name, doc_freq from kg_entities where id = any(%s) order by doc_freq", (p["kg_entity_ids"] or [],))
        ents = cur.fetchall()
    return {"id": p["id"], "text": p["text"], "n_words": p["n_words"], "n_tokens": p["n_tokens"],
            "sentence_offsets": p["sentence_offsets"], "chunks": chunks, "entities": ents,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"}


@app.get("/api/passages")
def sample_passages(limit: int = 12, q: str | None = None):
    with get_pool().connection() as conn, conn.cursor() as cur:
        if q:
            cur.execute("select id, left(text, 200) preview, n_tokens from passages where text ilike %s order by id limit %s",
                        (f"%{q}%", limit))
        else:
            cur.execute("select id, left(text, 200) preview, n_tokens from passages where n_tokens between 150 and 400 "
                        "order by random() limit %s", (limit,))
        return cur.fetchall()


@app.post("/api/chunk-preview")
def chunk_preview(body: dict):
    """Chunk arbitrary text with every (non-embedding) strategy — for the explorer's 'paste your own' mode."""
    text = " ".join(str(body.get("text", "")).split())
    if not text:
        raise HTTPException(400, "text required")
    p = Passage(0, text, None)
    out = {}
    for name in DEFAULT_STRATEGIES:
        if name == "semantic":
            continue
        out[name] = [{"chunk_index": c.index, "char_start": c.start, "char_end": c.end, "n_tokens": c.n_tokens}
                     for c in build_chunker(name).chunk(p)]
    return {"text": text, "chunks": out}


# ── knowledge graph ───────────────────────────────────────────────────────────
@app.get("/api/kg/entity")
def kg_entity(name: str, limit: int = 30):
    """Entity node + its neighbourhood (co-occurring entities weighted by shared passages)."""
    with get_pool().connection() as conn, conn.cursor() as cur:
        cur.execute("select id, name, doc_freq, passage_ids from kg_entities where name = %s", (name.lower(),))
        e = cur.fetchone()
        if not e:
            cur.execute("select id, name, doc_freq, passage_ids, similarity(name, %s) sim from kg_entities where name %% %s "
                        "order by sim desc limit 1", (name.lower(), name.lower()))
            e = cur.fetchone()
        if not e:
            raise HTTPException(404, "entity not found")
        pids = e["passage_ids"][:400]
        cur.execute("""select k.id, k.name, k.doc_freq, count(*) shared from passages p, unnest(p.kg_entity_ids) eid
                       join kg_entities k on k.id = eid where p.id = any(%s) and k.id <> %s
                       group by k.id, k.name, k.doc_freq order by shared desc, k.doc_freq asc limit %s""",
                    (pids, e["id"], limit))
        nb = cur.fetchall()
        cur.execute("select id, left(text, 160) preview from passages where id = any(%s) limit 10", (pids[:10],))
        ps = cur.fetchall()
    return {"entity": {"id": e["id"], "name": e["name"], "doc_freq": e["doc_freq"]}, "neighbours": nb, "passages": ps}


@app.get("/api/kg/search")
def kg_search(q: str, limit: int = 10):
    with get_pool().connection() as conn, conn.cursor() as cur:
        cur.execute("select id, name, doc_freq, similarity(name, %s) sim from kg_entities where name %% %s or name ilike %s "
                    "order by sim desc, doc_freq desc limit %s", (q.lower(), q.lower(), f"%{q.lower()}%", limit))
        return cur.fetchall()


# ── logs & evals ──────────────────────────────────────────────────────────────
@app.get("/api/logs")
def logs(limit: int = 50, source: str | None = None, q: str | None = None):
    return fetch_query_logs(limit, source, q)


@app.get("/api/logs/{qid}")
def log_detail(qid: str):
    r = fetch_query_log(qid)
    if not r:
        raise HTTPException(404, "unknown query log")
    return r


@app.get("/api/eval-runs")
def eval_runs(kind: str | None = None):
    with get_pool().connection() as conn, conn.cursor() as cur:
        cur.execute("select id, created_at, finished_at, name, kind, eval_set, n_questions, status, config, summary "
                    "from eval_runs where (%s::text is null or kind = %s::text) order by created_at desc", (kind, kind))
        return cur.fetchall()


@app.get("/api/eval-runs/{run_id}")
def eval_run(run_id: str):
    with get_pool().connection() as conn, conn.cursor() as cur:
        cur.execute("select * from eval_runs where id = %s", (run_id,))
        run = cur.fetchone()
        if not run:
            raise HTTPException(404, "unknown run")
        cur.execute("""select r.qa_id, q.question, q.question_type, r.metrics, r.retrieved, r.answer, r.latency_ms, r.query_log_id
                       from eval_results r join qa_pairs q on q.id = r.qa_id where r.run_id = %s""", (run_id,))
        run["results"] = cur.fetchall()
    return run


# ── SPA (production) ──────────────────────────────────────────────────────────
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str):
        f = DIST / full_path
        return FileResponse(f if f.is_file() else DIST / "index.html")
