"""Shared retrieval types. Every retriever returns ranked `Hit`s at CHUNK granularity for one
chunking strategy; the pipeline later attaches text, expands (window/parent), reranks, and cites.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Hit:
    chunk_id: int
    passage_id: int
    score: float
    rank: int
    source: str                              # retriever that produced it (dense / bm25 / rrf / kg / grep / rerank)
    chunk_index: int = 0
    char_start: int = 0
    char_end: int = 0
    text: str | None = None
    retriever_ranks: dict[str, int] = field(default_factory=dict)   # rank per underlying retriever
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"chunk_id": self.chunk_id, "passage_id": self.passage_id, "score": round(self.score, 5),
                "rank": self.rank, "source": self.source, "chunk_index": self.chunk_index,
                "char_start": self.char_start, "char_end": self.char_end, "retriever_ranks": self.retriever_ranks,
                **({"meta": self.meta} if self.meta else {})}


class Retriever(ABC):
    name: str

    @abstractmethod
    def retrieve(self, query: str, strategy: str, k: int) -> list[Hit]: ...


def rrf(ranked_lists: dict[str, list[Hit]], k: int = 60, source: str = "rrf") -> list[Hit]:
    """Reciprocal-rank fusion over several ranked lists keyed by retriever name."""
    fused: dict[int, Hit] = {}
    for name, hits in ranked_lists.items():
        for h in hits:
            f = fused.get(h.chunk_id)
            if f is None:
                f = Hit(h.chunk_id, h.passage_id, 0.0, 0, source, h.chunk_index, h.char_start, h.char_end, h.text)
                fused[h.chunk_id] = f
            f.score += 1.0 / (k + h.rank)
            f.retriever_ranks[name] = h.rank
    out = sorted(fused.values(), key=lambda h: -h.score)
    for i, h in enumerate(out, 1):
        h.rank = i
    return out


def dedupe_by_passage(hits: list[Hit]) -> list[Hit]:
    seen, out = set(), []
    for h in hits:
        if h.passage_id not in seen:
            seen.add(h.passage_id)
            out.append(h)
    return out
