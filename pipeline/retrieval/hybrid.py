"""Hybrid retrieval: dense ∪ BM25 fused with reciprocal-rank fusion (k=60)."""

from __future__ import annotations

from pipeline.retrieval.base import Hit, Retriever, rrf
from pipeline.retrieval.bm25 import BM25Retriever
from pipeline.retrieval.dense import DenseRetriever


class HybridRetriever(Retriever):
    name = "hybrid"

    def __init__(self, candidate_multiplier: int = 3):
        self.dense, self.bm25 = DenseRetriever(), BM25Retriever()
        self.mult = candidate_multiplier

    def retrieve(self, query: str, strategy: str, k: int) -> list[Hit]:
        n = k * self.mult
        lists = {"dense": self.dense.retrieve(query, strategy, n), "bm25": self.bm25.retrieve(query, strategy, n)}
        return rrf(lists, source="hybrid")[:k]
