"""Thin DB helpers used by retrievers (sync, pooled)."""

from __future__ import annotations

import numpy as np

from db.conn import get_pool
from pipeline.retrieval.base import Hit


def strategy_info(strategy: str) -> dict | None:
    with get_pool().connection() as conn, conn.cursor() as cur:
        cur.execute("select * from chunk_strategies where name = %s", (strategy,))
        return cur.fetchone()


def list_strategies() -> list[dict]:
    with get_pool().connection() as conn, conn.cursor() as cur:
        cur.execute("select name, description, params, embedding_model, embedding_dim, n_chunks, avg_tokens, built_at "
                    "from chunk_strategies order by n_chunks")
        return cur.fetchall()


def attach_text(hits: list[Hit]) -> list[Hit]:
    ids = [h.chunk_id for h in hits if h.text is None]
    if not ids:
        return hits
    with get_pool().connection() as conn, conn.cursor() as cur:
        cur.execute("select id, text, chunk_index, char_start, char_end from chunk_text where id = any(%s)", (ids,))
        rows = {r["id"]: r for r in cur.fetchall()}
    for h in hits:
        r = rows.get(h.chunk_id)
        if r:
            h.text, h.chunk_index, h.char_start, h.char_end = r["text"], r["chunk_index"], r["char_start"], r["char_end"]
    return hits


def passages_text(passage_ids: list[int]) -> dict[int, str]:
    if not passage_ids:
        return {}
    with get_pool().connection() as conn, conn.cursor() as cur:
        cur.execute("select id, text from passages where id = any(%s)", (passage_ids,))
        return {r["id"]: r["text"] for r in cur.fetchall()}


def neighbours(strategy: str, hits: list[Hit], window: int = 1) -> dict[int, list[dict]]:
    """For each hit: the chunks of the same passage within ±window chunk_index (incl. itself), with text."""
    if not hits:
        return {}
    pids = list({h.passage_id for h in hits})
    with get_pool().connection() as conn, conn.cursor() as cur:
        cur.execute("select id, passage_id, chunk_index, char_start, char_end, text from chunk_text "
                    "where strategy = %s and passage_id = any(%s) order by passage_id, chunk_index", (strategy, pids))
        by_pid: dict[int, list[dict]] = {}
        for r in cur.fetchall():
            by_pid.setdefault(r["passage_id"], []).append(r)
    out = {}
    for h in hits:
        out[h.chunk_id] = [c for c in by_pid.get(h.passage_id, []) if abs(c["chunk_index"] - h.chunk_index) <= window]
    return out


def chunks_for_passages(strategy: str, passage_ids: list[int], with_embeddings: bool = False) -> list[dict]:
    if not passage_ids:
        return []
    cols = "id, passage_id, chunk_index, char_start, char_end" + (", embedding" if with_embeddings else "")
    with get_pool().connection() as conn, conn.cursor() as cur:
        cur.execute(f"select {cols} from chunks where strategy = %s and passage_id = any(%s)", (strategy, passage_ids))
        rows = cur.fetchall()
    if with_embeddings:
        for r in rows:
            r["embedding"] = np.asarray(r["embedding"], dtype=np.float32)
    return rows
