"""Lexical retrieval: BM25 (bm25s, Lucene variant) — one in-process index per chunking strategy.

Indexes are built from the chunk_text view and cached at .cache/bm25/<strategy>/ so the API and the
eval harness start in seconds. Tokenisation: lowercase, English stopwords, Snowball stemming.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import bm25s
import numpy as np
import Stemmer

from config import get_settings
from db.conn import get_pool
from pipeline.retrieval.base import Hit, Retriever

log = logging.getLogger("retrieve.bm25")
_stemmer = Stemmer.Stemmer("english")


def tokenize(texts: list[str], show_progress: bool = False):
    return bm25s.tokenize(texts, stopwords="en", stemmer=_stemmer, show_progress=show_progress)


class BM25Index:
    def __init__(self, strategy: str):
        self.strategy = strategy
        self.dir = get_settings().cache_dir / "bm25" / strategy
        self.retriever: bm25s.BM25 | None = None
        self.meta: np.ndarray | None = None   # columns: chunk_id, passage_id, chunk_index, char_start, char_end

    def load_or_build(self) -> "BM25Index":
        if (self.dir / "meta.npy").exists():
            self.retriever = bm25s.BM25.load(str(self.dir), load_corpus=False)
            self.meta = np.load(self.dir / "meta.npy")
            info = json.loads((self.dir / "info.json").read_text())
            with get_pool().connection() as conn, conn.cursor() as cur:
                cur.execute("select n_chunks from chunk_strategies where name = %s", (self.strategy,))
                row = cur.fetchone()
            if row and row["n_chunks"] == info["n_chunks"]:
                log.info("[bm25] loaded cached index for %s (%d chunks)", self.strategy, info["n_chunks"])
                return self
            log.info("[bm25] cache for %s is stale — rebuilding", self.strategy)
        return self.build()

    def build(self) -> "BM25Index":
        t0 = time.time()
        with get_pool().connection() as conn, conn.cursor() as cur:
            cur.execute("select id, passage_id, chunk_index, char_start, char_end, text from chunk_text "
                        "where strategy = %s order by id", (self.strategy,))
            rows = cur.fetchall()
        if not rows:
            raise RuntimeError(f"no chunks for strategy {self.strategy} — run db/ingestion/build_chunks.py")
        texts = [r["text"] for r in rows]
        self.meta = np.array([[r["id"], r["passage_id"], r["chunk_index"], r["char_start"], r["char_end"]] for r in rows],
                             dtype=np.int64)
        self.retriever = bm25s.BM25(method="lucene", k1=1.2, b=0.75)
        self.retriever.index(tokenize(texts), show_progress=False)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.retriever.save(str(self.dir))
        np.save(self.dir / "meta.npy", self.meta)
        (self.dir / "info.json").write_text(json.dumps({"n_chunks": len(rows), "built": time.time()}))
        log.info("[bm25] built index for %s: %d chunks in %.1fs", self.strategy, len(rows), time.time() - t0)
        return self

    def search(self, query: str, k: int) -> list[Hit]:
        assert self.retriever is not None
        k = min(k, len(self.meta))
        q = tokenize([query])
        idx, scores = self.retriever.retrieve(q, k=k, show_progress=False)
        out = []
        for rank, (i, s) in enumerate(zip(idx[0].tolist(), scores[0].tolist()), 1):
            if s <= 0:
                break
            cid, pid, ci, cs, ce = self.meta[i].tolist()
            out.append(Hit(cid, pid, float(s), rank, "bm25", ci, cs, ce))
        return out


_indexes: dict[str, BM25Index] = {}


def get_index(strategy: str) -> BM25Index:
    if strategy not in _indexes:
        _indexes[strategy] = BM25Index(strategy).load_or_build()
    return _indexes[strategy]


class BM25Retriever(Retriever):
    name = "bm25"

    def retrieve(self, query: str, strategy: str, k: int) -> list[Hit]:
        return get_index(strategy).search(query, k)
