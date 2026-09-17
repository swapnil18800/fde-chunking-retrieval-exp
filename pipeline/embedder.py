"""Local open-source embeddings via sentence-transformers (Apple MPS when available).

Default model `abhinand/MedEmbed-small-v0.1` = bge-small-en-v1.5 fine-tuned on medical/clinical
retrieval (384-d, ~33M params). Swap with EMBEDDING_MODEL / EMBEDDING_DIM in .env — every chunk set
records the model it was embedded with (chunk_strategies.embedding_model), and retrieval refuses to
mix models.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from config import get_settings

log = logging.getLogger("embed")

# bge-style models accept an optional query instruction; MedEmbed inherits it. Empty = off.
QUERY_INSTRUCTIONS = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
}


def resolve_device(pref: str = "auto") -> str:
    if pref != "auto":
        return pref
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Embedder:
    def __init__(self, model_name: str | None = None, device: str | None = None, batch_size: int | None = None):
        s = get_settings()
        self.model_name = model_name or s.embedding_model
        self.device = resolve_device(device or s.torch_device)
        self.batch_size = batch_size or s.embedding_batch_size
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.query_instruction = QUERY_INSTRUCTIONS.get(self.model_name, "")
        log.info("[embed] loaded %s dim=%d device=%s", self.model_name, self.dim, self.device)

    def encode(self, texts: list[str], is_query: bool = False, show_progress: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if is_query and self.query_instruction:
            texts = [self.query_instruction + t for t in texts]
        vecs = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=show_progress)
        return vecs.astype(np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode([text], is_query=True)[0]


@lru_cache
def get_embedder() -> Embedder:
    return Embedder()
