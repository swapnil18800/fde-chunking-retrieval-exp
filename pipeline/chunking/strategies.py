"""The chunking strategies under test. Register new ones in REGISTRY (bottom of file).

| name             | unit                         | avg tokens | why it is in the matrix                                   |
|------------------|------------------------------|-----------:|-----------------------------------------------------------|
| passage          | whole abstract               |   ~225     | baseline; matches the granularity of the gold labels      |
| fixed_128_32     | 128-token window, 32 overlap |   ~110     | the naive default everyone starts with                     |
| recursive_128_32 | LangChain recursive splitter |   ~100     | sentence/clause-aware version of fixed size                |
| sentence         | one sentence                 |    ~30     | finest granularity; pairs with window/parent expansion     |
| semantic         | embedding-similarity groups  |   ~90      | topic-coherent groups of sentences                         |
"""

from __future__ import annotations

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pipeline.chunking.base import Chunk, Chunker, ChunkerSpec, Passage
from pipeline.tokens import n_tokens, token_char_offsets

# ── 1. passage ────────────────────────────────────────────────────────────────
class PassageChunker(Chunker):
    spec = ChunkerSpec("passage", "Whole abstract as a single chunk (baseline; gold-label granularity).")

    def chunk(self, p: Passage) -> list[Chunk]:
        return self._finalise(p, [(0, len(p.text))])


# ── 2. fixed token windows ────────────────────────────────────────────────────
class FixedTokenChunker(Chunker):
    def __init__(self, size: int = 128, overlap: int = 32):
        self.size, self.overlap = size, overlap
        self.spec = ChunkerSpec(f"fixed_{size}_{overlap}",
                                f"Sliding {size}-token window with {overlap}-token overlap, snapped to word boundaries.",
                                {"size_tokens": size, "overlap_tokens": overlap})

    def chunk(self, p: Passage) -> list[Chunk]:
        offs = token_char_offsets(p.text)
        n = len(offs) - 1
        if n <= self.size:
            return self._finalise(p, [(0, len(p.text))])
        spans, start = [], 0
        step = self.size - self.overlap
        while start < n:
            end = min(start + self.size, n)
            s, e = offs[start], offs[end]
            # snap to word boundaries so we never emit half a token/word
            while s > 0 and not p.text[s - 1].isspace():
                s -= 1
            while e < len(p.text) and not p.text[e].isspace():
                e += 1
            spans.append((s, e))
            if end == n:
                break
            start += step
        return self._finalise(p, spans)


# ── 3. recursive character splitter (LangChain) ───────────────────────────────
class RecursiveChunker(Chunker):
    def __init__(self, size: int = 128, overlap: int = 32):
        self.size, self.overlap = size, overlap
        self.spec = ChunkerSpec(f"recursive_{size}_{overlap}",
                                f"LangChain RecursiveCharacterTextSplitter, {size} tokens / {overlap} overlap, "
                                "separators sentence → clause → word.",
                                {"size_tokens": size, "overlap_tokens": overlap,
                                 "separators": [". ", "? ", "! ", "; ", ", ", " ", ""]})
        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base", chunk_size=size, chunk_overlap=overlap,
            separators=[". ", "? ", "! ", "; ", ", ", " ", ""], keep_separator="end")

    def chunk(self, p: Passage) -> list[Chunk]:
        # LangChain's add_start_index mixes token overlap with char offsets and returns -1 on
        # overlapping chunks, so we locate each piece ourselves (pieces are emitted in order).
        spans, cursor = [], 0
        for piece in self.splitter.split_text(p.text):
            start = p.text.find(piece, cursor)
            if start < 0:
                start = p.text.find(piece)
            spans.append((start, start + len(piece)))
            cursor = start + 1
        return self._finalise(p, spans)


# ── 4. sentence ───────────────────────────────────────────────────────────────
class SentenceChunker(Chunker):
    requires_sentences = True

    def __init__(self, min_tokens: int = 6):
        self.min_tokens = min_tokens
        self.spec = ChunkerSpec("sentence", "One scispaCy sentence per chunk; fragments under "
                                f"{min_tokens} tokens are merged into the previous sentence.",
                                {"min_tokens": min_tokens})

    def chunk(self, p: Passage) -> list[Chunk]:
        spans = []
        for s, e in p.sentences():
            if spans and n_tokens(p.text[s:e]) < self.min_tokens:
                spans[-1] = (spans[-1][0], e)
            else:
                spans.append((s, e))
        return self._finalise(p, spans)


# ── 5. semantic (embedding breakpoints) ───────────────────────────────────────
class SemanticChunker(Chunker):
    """Group consecutive sentences; start a new chunk where cosine distance between adjacent
    sentence embeddings exceeds `threshold` (calibrated once on a corpus sample: the distance
    percentile given by `breakpoint_percentile`), or when the group would exceed `max_tokens`.
    Equivalent in spirit to LangChain's SemanticChunker, with a global rather than per-document
    threshold because abstracts are too short (≈6 sentences) for per-document percentiles.
    """

    requires_sentences = True
    requires_embedder = True

    def __init__(self, embedder, threshold: float | None = None, breakpoint_percentile: int = 75,
                 max_tokens: int = 256, min_tokens: int = 6):
        self.embedder = embedder
        self.threshold = threshold
        self.pct, self.max_tokens, self.min_tokens = breakpoint_percentile, max_tokens, min_tokens
        self.spec = ChunkerSpec("semantic", "Adjacent-sentence embedding breakpoints (global threshold = "
                                f"p{breakpoint_percentile} of adjacent distances), max {max_tokens} tokens.",
                                {"breakpoint_percentile": breakpoint_percentile, "max_tokens": max_tokens,
                                 "threshold": threshold, "embedding_model": embedder.model_name})

    def calibrate(self, sample: list[Passage]) -> float:
        dists = []
        for grp in self._embed_sentences(sample):
            for a, b in zip(grp, grp[1:]):
                dists.append(1.0 - float(np.dot(a, b)))
        self.threshold = float(np.percentile(dists, self.pct)) if dists else 0.5
        self.spec.params["threshold"] = round(self.threshold, 4)
        return self.threshold

    def _embed_sentences(self, passages: list[Passage]) -> list[np.ndarray]:
        flat, owners = [], []
        for i, p in enumerate(passages):
            for s, e in p.sentences():
                flat.append(p.text[s:e])
                owners.append(i)
        vecs = self.embedder.encode(flat)
        out: list[list] = [[] for _ in passages]
        for o, v in zip(owners, vecs):
            out[o].append(v)
        return [np.asarray(g) for g in out]

    def chunk(self, p: Passage) -> list[Chunk]:
        return self.chunk_batch([p])[0]

    def chunk_batch(self, passages: list[Passage]) -> list[list[Chunk]]:
        assert self.threshold is not None, "call calibrate() first"
        results = []
        for p, vecs in zip(passages, self._embed_sentences(passages)):
            sents = p.sentences()
            if len(sents) <= 1:
                results.append(self._finalise(p, [(0, len(p.text))]))
                continue
            spans, cur_s, cur_e = [], sents[0][0], sents[0][1]
            cur_tok = n_tokens(p.text[cur_s:cur_e])
            for i in range(1, len(sents)):
                s, e = sents[i]
                dist = 1.0 - float(np.dot(vecs[i - 1], vecs[i]))
                t = n_tokens(p.text[s:e])
                if (dist > self.threshold and cur_tok >= self.min_tokens) or cur_tok + t > self.max_tokens:
                    spans.append((cur_s, cur_e))
                    cur_s, cur_e, cur_tok = s, e, t
                else:
                    cur_e, cur_tok = e, cur_tok + t
            spans.append((cur_s, cur_e))
            results.append(self._finalise(p, spans))
        return results


# ── registry ──────────────────────────────────────────────────────────────────
def build_chunker(name: str, embedder=None) -> Chunker:
    if name == "passage":
        return PassageChunker()
    if name.startswith("fixed_"):
        _, size, overlap = name.split("_")
        return FixedTokenChunker(int(size), int(overlap))
    if name.startswith("recursive_"):
        _, size, overlap = name.split("_")
        return RecursiveChunker(int(size), int(overlap))
    if name == "sentence":
        return SentenceChunker()
    if name == "semantic":
        assert embedder is not None, "semantic chunker needs an embedder"
        return SemanticChunker(embedder)
    raise ValueError(f"unknown chunking strategy: {name}")


DEFAULT_STRATEGIES = ["passage", "fixed_128_32", "recursive_128_32", "sentence", "semantic"]
