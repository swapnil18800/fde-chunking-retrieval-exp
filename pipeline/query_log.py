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


def write_query_log(row: dict) -> str | None:
    qid = row.get("id") or str(uuid.uuid4())
    try:
        with get_pool().connection() as conn, conn.cursor() as cur:
            cur.execute(
                """insert into query_logs (id, source, question, qa_id, config, stages, retrieved, answer, citations,
                                           metrics, latency_ms, tokens, trace_provider, trace_id, trace_url, error)
                   values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (qid, row.get("source", "api"), row["question"], row.get("qa_id"),
                 json.dumps(row.get("config")), json.dumps(row.get("stages")), json.dumps(row.get("retrieved")),
                 row.get("answer"), json.dumps(row.get("citations")), json.dumps(row.get("metrics")),
                 row.get("latency_ms"), json.dumps(row.get("tokens")), row.get("trace_provider"),
                 row.get("trace_id"), row.get("trace_url"), row.get("error")))
            conn.commit()
        return qid
    except Exception as e:  # noqa: BLE001
        log.error("[querylog] failed to persist query log: %s", e)
        return None


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
