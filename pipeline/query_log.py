"""Persist one row per pipeline run to `query_logs` (Postgres) — the "what happened to my question" record.

Insert is best-effort (never fails the request). The API's Query Inspector tab reads this table;
the eval harness links eval_results.query_log_id to it so every eval number is traceable to a full run.
"""

from __future__ import annotations

import json
import logging
import uuid

from db.conn import get_pool

log = logging.getLogger("querylog")


def _compact_hits(hits, keep_text: bool) -> list:
    """Chunk text is reconstructible from chunk_id (chunk_text view); eval runs write thousands of rows,
    so only API/smoke runs keep the text inline (Query Inspector convenience)."""
    if keep_text or not hits:
        return hits or []
    return [{k: v for k, v in h.items() if k not in ("text", "context_text")} for h in hits]


def write_query_log(row: dict) -> str | None:
    qid = row.get("id") or str(uuid.uuid4())
    retrieved = _compact_hits(row.get("retrieved"), keep_text=row.get("source", "api") != "eval")
    try:
        with get_pool().connection() as conn, conn.cursor() as cur:
            cur.execute(
                """insert into query_logs (id, source, question, qa_id, config, stages, retrieved, answer, citations,
                                           metrics, latency_ms, tokens, trace_provider, trace_id, trace_url, error)
                   values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (qid, row.get("source", "api"), row["question"], row.get("qa_id"),
                 json.dumps(row.get("config")), json.dumps(row.get("stages")), json.dumps(retrieved),
                 row.get("answer"), json.dumps(row.get("citations")), json.dumps(row.get("metrics")),
                 row.get("latency_ms"), json.dumps(row.get("tokens")), row.get("trace_provider"),
                 row.get("trace_id"), row.get("trace_url"), row.get("error")))
            conn.commit()
        return qid
    except Exception as e:  # noqa: BLE001
        log.error("[querylog] failed to persist query log: %s", e)
        return None


def write_query_logs_bulk(rows: list[dict]) -> int:
    """Batch insert (one round trip) — used by the eval harness where per-row inserts at ~0.4 s RTT dominate."""
    if not rows:
        return 0
    params = []
    for row in rows:
        retrieved = _compact_hits(row.get("retrieved"), keep_text=row.get("source", "api") != "eval")
        params.append((row.get("id") or str(uuid.uuid4()), row.get("source", "api"), row["question"], row.get("qa_id"),
                       json.dumps(row.get("config")), json.dumps(row.get("stages")), json.dumps(retrieved),
                       row.get("answer"), json.dumps(row.get("citations")), json.dumps(row.get("metrics")),
                       row.get("latency_ms"), json.dumps(row.get("tokens")), row.get("trace_provider"),
                       row.get("trace_id"), row.get("trace_url"), row.get("error")))
    try:
        with get_pool().connection() as conn, conn.cursor() as cur:
            cur.executemany(
                """insert into query_logs (id, source, question, qa_id, config, stages, retrieved, answer, citations,
                                           metrics, latency_ms, tokens, trace_provider, trace_id, trace_url, error)
                   values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""", params)
            conn.commit()
        return len(params)
    except Exception as e:  # noqa: BLE001
        log.error("[querylog] bulk persist failed: %s", e)
        return 0


def fetch_query_logs(limit: int = 50, source: str | None = None, q: str | None = None) -> list[dict]:
    where, args = [], []
    if source:
        where.append("source = %s")
        args.append(source)
    if q:
        where.append("question ilike %s")
        args.append(f"%{q}%")
    sql = ("select id, created_at, source, question, qa_id, config, latency_ms, trace_url, error, "
           "left(answer, 200) as answer_preview, metrics from query_logs")
    if where:
        sql += " where " + " and ".join(where)
    sql += " order by created_at desc limit %s"
    args.append(limit)
    with get_pool().connection() as conn, conn.cursor() as cur:
        cur.execute(sql, args)
        return cur.fetchall()


def fetch_query_log(qid: str) -> dict | None:
    with get_pool().connection() as conn, conn.cursor() as cur:
        cur.execute("select * from query_logs where id = %s", (qid,))
        return cur.fetchone()
