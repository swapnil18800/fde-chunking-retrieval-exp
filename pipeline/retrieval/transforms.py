"""LLM query transforms that wrap any base retriever (all optional, all free-tier Gemini):

    none         – the raw question
    hyde         – Hypothetical Document Embeddings: the LLM drafts a plausible abstract-style answer;
                   we retrieve with the hypothetical text AND the question, fused by RRF
    multi_query  – 3 paraphrases from different angles (synonyms, mechanism, clinical) + original → RRF
    decompose    – multi-hop style: split into 2-4 atomic sub-questions (MultiHop-RAG), retrieve each,
                   fuse by RRF. Helps list-type BioASQ questions whose gold spans many passages.
"""

from __future__ import annotations

import json
import logging
import re

from pipeline.llm import get_llm
from pipeline.retrieval.base import Hit, Retriever, rrf

log = logging.getLogger("retrieve.transform")
TRANSFORMS = ("none", "hyde", "multi_query", "decompose")

HYDE_PROMPT = """You are a biomedical researcher. Write a short PubMed-style abstract passage (3-5 sentences, \
no headings, no citations) that would directly answer the question below. Be specific: name genes, \
proteins, drugs, diseases, mechanisms. Do not say you are unsure; write the most plausible answer.

Question: {q}"""

MULTI_PROMPT = """Generate 3 alternative search queries for retrieving PubMed abstracts that answer the question. \
Vary vocabulary (synonyms, official gene/drug names, abbreviations), and angle (mechanism / clinical / \
epidemiology). Return ONLY a JSON array of 3 strings.

Question: {q}"""

DECOMP_PROMPT = """Break the biomedical question into 2-4 atomic sub-questions that each target one fact needed \
to answer it (entities, mechanisms, associations). If the question is already atomic, return it alone. \
Return ONLY a JSON array of strings.

Question: {q}"""


def _json_list(text: str) -> list[str]:
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
        return [str(x).strip() for x in arr if str(x).strip()]
    except json.JSONDecodeError:
        return []


def generate_queries(question: str, mode: str, callbacks: list | None = None) -> dict:
    """Returns {"queries": [...], "hypothetical": str|None, "llm": {...}} — original question always included."""
    out = {"queries": [question], "hypothetical": None, "llm": None}
    if mode == "none":
        return out
    llm = get_llm("transform")
    cfg = {"callbacks": callbacks or [], "run_name": "rewrite-query", "metadata": {"mode": mode}}
    if mode == "hyde":
        r = llm.chat(HYDE_PROMPT.format(q=question), config=cfg)
        out["hypothetical"] = r.text
        out["queries"] = [question, r.text]
    elif mode == "multi_query":
        r = llm.chat(MULTI_PROMPT.format(q=question), config=cfg)
        out["queries"] = [question] + _json_list(r.text)[:3]
    elif mode == "decompose":
        r = llm.chat(DECOMP_PROMPT.format(q=question), config=cfg)
        subs = _json_list(r.text)[:4]
        out["queries"] = [question] + [s for s in subs if s.lower() != question.lower()]
    else:
        raise ValueError(f"unknown transform {mode}")
    out["llm"] = {"model": r.model, "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
                  "latency_ms": r.latency_ms}
    return out


def retrieve_multi(base: Retriever, queries: list[str], strategy: str, k: int, per_query_k: int | None = None) -> list[Hit]:
    """Run the base retriever once per query and fuse with RRF (single query → plain result)."""
    if len(queries) == 1:
        return base.retrieve(queries[0], strategy, k)
    n = per_query_k or k * 2
    lists = {f"q{i}": base.retrieve(q, strategy, n) for i, q in enumerate(queries)}
    fused = rrf(lists, source=base.name)
    for h in fused:
        h.meta["fused_from"] = len(queries)
    return fused[:k]
