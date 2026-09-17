"""Dense retrieval: pgvector cosine over halfvec embeddings (HNSW where indexed, exact scan otherwise)."""

from __future__ import annotations

import logging

import numpy as np

from db.conn import get_pool
from pipeline.embedder import get_embedder
from pipeline.retrieval.base import Hit, Retriever

log = logging.getLogger("retrieve.dense")


def _lit(v: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v.tolist()) + "]"


class DenseRetriever(Retriever):
    name = "dense"

    def retrieve(self, query: str, strategy: str, k: int, query_vec: np.ndarray | None = None) -> list[Hit]:
        qv = query_vec if query_vec is not None else get_embedder().encode_query(query)
        return self.retrieve_vec(qv, strategy, k)

    def retrieve_vec(self, qv: np.ndarray, strategy: str, k: int) -> list[Hit]:
        lit = _lit(qv)
        with get_pool().connection() as conn, conn.cursor() as cur:
            cur.execute("set local hnsw.ef_search = %s", (max(64, 2 * k),))
            cur.execute(
                """select id, passage_id, chunk_index, char_start, char_end,
                          1 - (embedding <=> %s::halfvec) as score
                   from chunks where strategy = %s
                   order by embedding <=> %s::halfvec limit %s""",
                (lit, strategy, lit, k))
            rows = cur.fetchall()
        return [Hit(r["id"], r["passage_id"], float(r["score"]), i, self.name, r["chunk_index"], r["char_start"],
                    r["char_end"]) for i, r in enumerate(rows, 1)]
