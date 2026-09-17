"""Post-retrieval context expansion — what the generator actually sees for each hit.

    none    – the chunk itself
    window  – the chunk plus its ±1 neighbours in the same passage (sentence-window retrieval)
    parent  – the whole parent passage (small-to-big / parent-document retrieval); hits from the
              same passage collapse into one
"""

from __future__ import annotations

from pipeline.retrieval.base import Hit, dedupe_by_passage
from pipeline.retrieval.store import neighbours, passages_text

EXPANSIONS = ("none", "window", "parent")


def expand(hits: list[Hit], strategy: str, mode: str) -> list[Hit]:
    if mode == "none" or not hits:
        return hits
    if mode == "window":
        nb = neighbours(strategy, hits, window=1)
        for h in hits:
            group = nb.get(h.chunk_id) or []
            if group:
                h.meta["context_start"], h.meta["context_end"] = group[0]["char_start"], group[-1]["char_end"]
                h.meta["context_text"] = " ".join(c["text"] for c in group)
                h.meta["expansion"] = "window"
        return hits
    if mode == "parent":
        hits = dedupe_by_passage(hits)
        texts = passages_text([h.passage_id for h in hits])
        for i, h in enumerate(hits, 1):
            t = texts.get(h.passage_id, "")
            h.meta["context_start"], h.meta["context_end"], h.meta["context_text"] = 0, len(t), t
            h.meta["expansion"] = "parent"
            h.rank = i
        return hits
    raise ValueError(f"unknown expansion {mode}")


def context_text(h: Hit) -> str:
    return h.meta.get("context_text") or h.text or ""
