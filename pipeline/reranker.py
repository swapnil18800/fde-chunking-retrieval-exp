"""Cross-encoder reranking with a PubMed-trained model (default: ncbi/MedCPT-Cross-Encoder).

Runs locally on MPS/CPU. Rerank the top-N candidates of any retriever, keep top-k. Scores are raw
logits (higher = more relevant) — not comparable across queries, only within one ranking.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import torch
from sentence_transformers import CrossEncoder

from config import get_settings
from pipeline.embedder import resolve_device
from pipeline.retrieval.base import Hit

log = logging.getLogger("rerank")


class Reranker:
    def __init__(self, model_name: str | None = None, device: str | None = None):
        s = get_settings()
        self.model_name = model_name or s.reranker_model
        self.device = resolve_device(device or s.torch_device)
        # raw logits, not sigmoid: MedCPT's logits saturate a sigmoid at 1.0 and hide the ranking margin
        self.model = CrossEncoder(self.model_name, device=self.device, max_length=512, activation_fn=torch.nn.Identity())
        log.info("[rerank] loaded %s on %s", self.model_name, self.device)

    def rerank(self, query: str, hits: list[Hit], k: int, text_of=lambda h: h.text or "") -> list[Hit]:
        if not hits:
            return hits
        scores = self.model.predict([(query, text_of(h)) for h in hits], batch_size=32, show_progress_bar=False)
        for h, sc in zip(hits, scores):
            h.retriever_ranks.setdefault(h.source, h.rank)
            h.meta["pre_rerank_score"], h.meta["pre_rerank_rank"] = round(h.score, 5), h.rank
            h.score, h.source = float(sc), "rerank"
        out = sorted(hits, key=lambda h: -h.score)[:k]
        for i, h in enumerate(out, 1):
            h.rank = i
        return out


@lru_cache
def get_reranker() -> Reranker:
    return Reranker()
