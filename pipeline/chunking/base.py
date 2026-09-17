"""Chunker interface. A chunk is a (start, end) char span into the *normalised passage text*
(see db/ingestion/load_corpus.py) plus an optional `prefix` that is prepended only at embed
time (contextual strategies). Text is never duplicated in the DB — see the chunk_text view.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from pipeline.tokens import n_tokens

_SENT_FALLBACK = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


@dataclass(frozen=True)
class Passage:
    id: int
    text: str
    sentence_offsets: list[int] | None = None

    def sentences(self) -> list[tuple[int, int]]:
        """(start, end) char spans. Uses scispaCy offsets when present, regex fallback otherwise."""
        if self.sentence_offsets:
            starts = [o for o in self.sentence_offsets if 0 <= o < len(self.text)]
            if not starts or starts[0] != 0:
                starts = [0] + starts
            return [(s, e) for s, e in zip(starts, starts[1:] + [len(self.text)]) if e > s]
        spans, pos = [], 0
        for m in _SENT_FALLBACK.finditer(self.text):
            spans.append((pos, m.start()))
            pos = m.end()
        spans.append((pos, len(self.text)))
        return [(s, e) for s, e in spans if e > s]


@dataclass
class Chunk:
    passage_id: int
    index: int
    start: int
    end: int
    prefix: str | None = None
    n_tokens: int = 0

    def text(self, passage_text: str) -> str:
        return passage_text[self.start:self.end]

    def embed_text(self, passage_text: str) -> str:
        t = self.text(passage_text)
        return f"{self.prefix} {t}" if self.prefix else t


@dataclass
class ChunkerSpec:
    name: str
    description: str
    params: dict = field(default_factory=dict)


class Chunker(ABC):
    spec: ChunkerSpec
    requires_sentences: bool = False   # needs passages.sentence_offsets (preprocess_nlp.py)
    requires_embedder: bool = False    # needs an Embedder at chunk time (semantic)

    @abstractmethod
    def chunk(self, p: Passage) -> list[Chunk]: ...

    def chunk_batch(self, passages: list[Passage]) -> list[list[Chunk]]:
        """Override for strategies that benefit from batching (e.g. embedding many sentences at once)."""
        return [self.chunk(p) for p in passages]

    @staticmethod
    def _finalise(p: Passage, spans: list[tuple[int, int]], prefixes: list[str | None] | None = None) -> list[Chunk]:
        out = []
        for i, (s, e) in enumerate(spans):
            s, e = _trim(p.text, s, e)
            if e <= s:
                continue
            out.append(Chunk(p.id, len(out), s, e, prefixes[i] if prefixes else None, n_tokens(p.text[s:e])))
        return out


def _trim(text: str, s: int, e: int) -> tuple[int, int]:
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e
