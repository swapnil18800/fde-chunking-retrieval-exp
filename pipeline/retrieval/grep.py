"""Grep retrieval: literal term matching, the way an agent with a `grep` tool would search a corpus.

1. Pull search terms from the question: scispaCy entities (multi-word kept intact) + remaining content
   words (stop/question words removed, length >= 3). No embeddings, no LLM.
2. Regex-scan `passages.text` per term (word-boundary, case-insensitive) and score a passage by the
   sum of idf-like weights of the distinct terms it matches:  w(term) = log(N / df_term).
3. Map the top passages to chunks of the requested strategy, picking the chunk(s) with most term hits.
Strength: exact gene/drug names. Weakness: synonyms, morphology ("secreted" vs "secretion") — the
stemmed BM25 baseline covers those, which is exactly the contrast we want to measure.
"""

from __future__ import annotations

import logging
import math
import re
from functools import lru_cache

from db.conn import get_pool
from pipeline.nlp import load_sci_nlp
from pipeline.retrieval.base import Hit, Retriever

log = logging.getLogger("retrieve.grep")
STOP = set("""a an the of in on at to for from by with and or is are was were be been being do does did can could
should would will what which who whom whose where when why how list name describe explain define role known about into
between among as that this these those there their it its than then also any all some such used use using
associated involved related regarding concerning versus vs mendelian multifactorial""".split())
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9\-']{2,}")


def extract_terms(question: str, max_terms: int = 8) -> list[str]:
    nlp = load_sci_nlp(disable=("parser", "lemmatizer"))
    doc = nlp(question)
    terms: list[str] = []
    covered: set[int] = set()
    for e in doc.ents:
        t = e.text.strip(" ?.,;:()").lower()
        if len(t) >= 3 and t not in STOP:
            terms.append(t)
            covered.update(range(e.start, e.end))
    for tok in doc:
        if tok.i in covered:
            continue
        w = tok.text.lower().strip("?.,;:()")
        if _WORD.fullmatch(w) and w not in STOP and not tok.is_stop:
            terms.append(w)
    seen, out = set(), []
    for t in terms:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out[:max_terms]


class GrepRetriever(Retriever):
    name = "grep"

    def __init__(self, passage_candidates: int = 50):
        self.passage_candidates = passage_candidates

    @staticmethod
    @lru_cache(maxsize=2048)
    def _score_passages(query: str, passage_candidates: int) -> tuple[tuple[str, ...], tuple, dict]:
        """Strategy-independent part (terms → passage scores). Cached so the eval matrix pays the
        passage scan once per question, not once per chunking strategy."""
        terms = extract_terms(query)
        if not terms:
            return (), (), {}
        with get_pool().connection() as conn, conn.cursor() as cur:
            cur.execute("select count(*) n from passages")
            n_total = cur.fetchone()["n"]
            # ONE scan: passages containing any term (word-boundary, case-insensitive); per-term
            # matching + df happen in Python over the (much smaller) matched set.
            alt = r"\m(" + "|".join(re.escape(t) for t in terms) + r")\M"
            cur.execute("select id, text from passages where text ~* %s", (alt,))
            rows = cur.fetchall()
        pats = {t: re.compile(r"\b" + re.escape(t) + r"\b", re.I) for t in terms}
        matches: dict[str, set[int]] = {t: set() for t in terms}
        for r in rows:
            for t, pat in pats.items():
                if pat.search(r["text"]):
                    matches[t].add(r["id"])
        matches = {t: ids for t, ids in matches.items() if ids}
        if not matches:
            return tuple(terms), (), {}
        weights = {t: math.log(n_total / len(ids)) for t, ids in matches.items()}
        score: dict[int, float] = {}
        hit_terms: dict[int, list[str]] = {}
        for t, ids in matches.items():
            for pid in ids:
                score[pid] = score.get(pid, 0.0) + weights[t]
                hit_terms.setdefault(pid, []).append(t)
        top = tuple(sorted(score.items(), key=lambda kv: -kv[1])[:passage_candidates])
        return tuple(terms), top, {pid: hit_terms[pid] for pid, _ in top}

    def retrieve(self, query: str, strategy: str, k: int) -> list[Hit]:
        terms, top, hit_terms = self._score_passages(query, self.passage_candidates)
        if not top:
            return []
        pids = [pid for pid, _ in top]
        with get_pool().connection() as conn, conn.cursor() as cur:
            # best chunk per passage = the one containing the most matched terms (ties → earlier chunk)
            cur.execute("select id, passage_id, chunk_index, char_start, char_end, text from chunk_text "
                        "where strategy = %s and passage_id = any(%s)", (strategy, pids))
            by_pid: dict[int, list[dict]] = {}
            for r in cur.fetchall():
                by_pid.setdefault(r["passage_id"], []).append(r)
        hits = []
        for pid, sc in top:
            chunks = by_pid.get(pid, [])
            if not chunks:
                continue
            best, best_n = None, -1
            for c in sorted(chunks, key=lambda c: c["chunk_index"]):
                low = c["text"].lower()
                n = sum(1 for t in hit_terms[pid] if t in low)
                if n > best_n:
                    best, best_n = c, n
            hits.append(Hit(best["id"], pid, sc, len(hits) + 1, self.name, best["chunk_index"], best["char_start"],
                            best["char_end"], best["text"], meta={"terms": hit_terms[pid], "chunk_term_hits": best_n}))
            if len(hits) >= k:
                break
        log.debug("[grep] terms=%s -> %d hits", terms, len(hits))
        for h in hits:
            h.meta["query_terms"] = list(terms)
        return hits
