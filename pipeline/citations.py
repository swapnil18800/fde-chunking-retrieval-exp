"""Turn `[n]` markers in the answer into resolvable citations with three levels of provenance:

    citation → chunk (exact char span)  → parent passage (full abstract) → PubMed (PMID URL)

The frontend uses `char_start/char_end` (and the wider `context_*` span when expansion was used) to
highlight the cited span inside the parent passage.
"""

from __future__ import annotations

import re

from pipeline.retrieval.base import Hit
from pipeline.retrieval.expand import context_text

_CITE = re.compile(r"\[(\d{1,2})\]")
PUBMED = "https://pubmed.ncbi.nlm.nih.gov/{pmid}/"


def numbered_context(hits: list[Hit]) -> list[dict]:
    return [{"n": i, "pmid": h.passage_id, "chunk_id": h.chunk_id, "text": context_text(h)} for i, h in enumerate(hits, 1)]


def build_citations(answer: str, hits: list[Hit]) -> tuple[list[dict], list[int]]:
    """Returns (citations, invalid_numbers). Each citation carries chunk + parent + url."""
    used = sorted({int(n) for n in _CITE.findall(answer)})
    cites, invalid = [], []
    for n in used:
        if 1 <= n <= len(hits):
            h = hits[n - 1]
            cites.append({
                "n": n, "chunk_id": h.chunk_id, "passage_id": h.passage_id, "pmid": h.passage_id,
                "url": PUBMED.format(pmid=h.passage_id),
                "chunk_text": h.text, "char_start": h.char_start, "char_end": h.char_end,
                "context_start": h.meta.get("context_start", h.char_start),
                "context_end": h.meta.get("context_end", h.char_end),
                "score": round(h.score, 4), "source": h.source, "retriever_ranks": h.retriever_ranks,
            })
        else:
            invalid.append(n)
    return cites, invalid


def strip_invalid(answer: str, invalid: list[int]) -> str:
    for n in invalid:
        answer = answer.replace(f"[{n}]", "")
    return re.sub(r"\s{2,}", " ", answer).strip()
