"""Retrieval metrics at PASSAGE level (gold labels are PMIDs). Chunk hits are collapsed to their parent
passage in rank order before scoring, so strategies with many chunks per passage are not rewarded for
retrieving the same passage repeatedly.

    recall@k      |gold ∩ top-k| / |gold|            — did we surface the evidence?
    precision@k   |gold ∩ top-k| / k                 — how much noise does the generator see?
    hit@k         1 if any gold in top-k              — "at least one right passage"
    mrr           1 / rank of first gold              — how high is the first good passage?
    ndcg@k        DCG@k / IDCG@k, binary gains        — rank-sensitive recall
    gold_coverage_chunks   for the retrieved gold passages: fraction of retrieved chunks that are gold
"""

from __future__ import annotations

import math


def collapse_to_passages(ranked_passage_ids: list[int]) -> list[int]:
    seen, out = set(), []
    for p in ranked_passage_ids:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def retrieval_metrics(ranked_passage_ids: list[int], gold: set[int], ks: tuple[int, ...] = (1, 3, 5, 10, 20)) -> dict:
    ranked = collapse_to_passages(ranked_passage_ids)
    out: dict = {"n_gold": len(gold), "n_retrieved_passages": len(ranked)}
    if not gold:
        return out
    first = next((i for i, p in enumerate(ranked, 1) if p in gold), None)
    out["mrr"] = 1.0 / first if first else 0.0
    out["first_gold_rank"] = first
    for k in ks:
        top = ranked[:k]
        hits = sum(1 for p in top if p in gold)
        out[f"recall@{k}"] = hits / len(gold)
        out[f"precision@{k}"] = hits / k
        out[f"hit@{k}"] = 1.0 if hits else 0.0
        dcg = sum(1.0 / math.log2(i + 1) for i, p in enumerate(top, 1) if p in gold)
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
        out[f"ndcg@{k}"] = dcg / idcg if idcg else 0.0
    return out


def aggregate(rows: list[dict]) -> dict:
    """Mean of every numeric metric across questions (None-safe)."""
    keys = sorted({k for r in rows for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
    agg = {}
    for k in keys:
        vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
        if vals:
            agg[k] = sum(vals) / len(vals)
    agg["n"] = len(rows)
    return agg
