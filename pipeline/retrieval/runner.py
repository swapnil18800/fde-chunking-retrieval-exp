"""One entry point for every retrieval configuration in the experiment matrix.

    cfg = RetrievalConfig(strategy="sentence", retriever="hybrid", transform="none", rerank=True, expansion="window")
    result = run_retrieval("Is RANKL secreted from the cells?", cfg)

Pipeline:  transform(query) → retriever(s) [+RRF] → (rerank top-N) → expand → attach text
Every stage is timed and summarised in `result.stages`, which the API, the query log and the traces
all reuse — so what you see in the UI is exactly what was measured.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field

from pipeline.retrieval.base import Hit, Retriever
from pipeline.retrieval.bm25 import BM25Retriever
from pipeline.retrieval.dense import DenseRetriever
from pipeline.retrieval.expand import EXPANSIONS, expand
from pipeline.retrieval.grep import GrepRetriever
from pipeline.retrieval.hybrid import HybridRetriever
from pipeline.retrieval.kg import KGRetriever
from pipeline.retrieval.store import attach_text
from pipeline.retrieval.transforms import TRANSFORMS, generate_queries, retrieve_multi
from pipeline.tracing import child_span

log = logging.getLogger("retrieve")

RETRIEVERS: dict[str, type[Retriever]] = {
    "bm25": BM25Retriever, "dense": DenseRetriever, "hybrid": HybridRetriever, "grep": GrepRetriever, "kg": KGRetriever,
}
_instances: dict[str, Retriever] = {}


def get_retriever(name: str) -> Retriever:
    if name not in _instances:
        _instances[name] = RETRIEVERS[name]()
    return _instances[name]


@dataclass
class RetrievalConfig:
    strategy: str = "passage"
    retriever: str = "hybrid"
    transform: str = "none"
    rerank: bool = False
    expansion: str = "none"
    top_k: int = 5
    candidate_k: int = 20          # candidates pulled before rerank

    def validate(self) -> None:
        assert self.retriever in RETRIEVERS, f"retriever must be one of {list(RETRIEVERS)}"
        assert self.transform in TRANSFORMS, f"transform must be one of {TRANSFORMS}"
        assert self.expansion in EXPANSIONS, f"expansion must be one of {EXPANSIONS}"

    def label(self) -> str:
        parts = [self.strategy, self.retriever]
        if self.transform != "none":
            parts.append(self.transform)
        if self.rerank:
            parts.append("rerank")
        if self.expansion != "none":
            parts.append(self.expansion)
        return "+".join(parts)


@dataclass
class RetrievalResult:
    hits: list[Hit]
    stages: list[dict] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    hypothetical: str | None = None
    llm_usage: dict | None = None

    def to_dict(self) -> dict:
        return {"hits": [h.to_dict() | {"text": h.text, "context_text": h.meta.get("context_text")} for h in self.hits],
                "stages": self.stages, "queries": self.queries, "hypothetical": self.hypothetical,
                "llm_usage": self.llm_usage}


def _stage(stages: list, name: str, t0: float, **detail) -> None:
    stages.append({"name": name, "ms": int((time.time() - t0) * 1000), **detail})


def run_retrieval(question: str, cfg: RetrievalConfig, callbacks: list | None = None) -> RetrievalResult:
    cfg.validate()
    stages: list[dict] = []
    t0 = time.time()
    # observation names are stable verbs (Langfuse best practice); the variant lives in metadata
    with child_span("transform-query", as_type="chain" if cfg.transform != "none" else "span",
                    input={"question": question}, metadata={"mode": cfg.transform}) as sp:
        q = generate_queries(question, cfg.transform, callbacks)
        if sp is not None:
            sp.update(output={"queries": q["queries"], "hypothetical": q["hypothetical"]})
    _stage(stages, f"transform:{cfg.transform}", t0, n_queries=len(q["queries"]), llm=q["llm"])

    n = cfg.candidate_k if cfg.rerank else cfg.top_k
    t0 = time.time()
    base = get_retriever(cfg.retriever)
    with child_span("retrieve-chunks", as_type="retriever", input={"queries": q["queries"], "k": n},
                    metadata={"retriever": cfg.retriever, "strategy": cfg.strategy}) as sp:
        hits = retrieve_multi(base, q["queries"], cfg.strategy, n)
        if sp is not None:
            sp.update(output=[{"chunk_id": h.chunk_id, "passage_id": h.passage_id, "score": round(h.score, 4)} for h in hits])
    _stage(stages, f"retrieve:{cfg.retriever}", t0, strategy=cfg.strategy, candidates=len(hits),
           passages=len({h.passage_id for h in hits}))

    if cfg.rerank:
        from pipeline.reranker import get_reranker

        t0 = time.time()
        attach_text(hits)
        with child_span("rerank-chunks", as_type="span", input={"candidates": len(hits), "top_k": cfg.top_k},
                        metadata={"model": "MedCPT-Cross-Encoder"}) as sp:
            hits = get_reranker().rerank(question, hits, cfg.top_k)
            if sp is not None:
                sp.update(output=[{"chunk_id": h.chunk_id, "score": round(h.score, 3)} for h in hits])
        _stage(stages, "rerank", t0, kept=len(hits))

    t0 = time.time()
    attach_text(hits)
    hits = expand(hits, cfg.strategy, cfg.expansion)
    _stage(stages, f"expand:{cfg.expansion}", t0, hits=len(hits))
    return RetrievalResult(hits, stages, q["queries"], q["hypothetical"], q["llm"])


def config_from_dict(d: dict) -> RetrievalConfig:
    return RetrievalConfig(**{k: v for k, v in d.items() if k in RetrievalConfig.__dataclass_fields__})


def config_to_dict(cfg: RetrievalConfig) -> dict:
    return asdict(cfg)
