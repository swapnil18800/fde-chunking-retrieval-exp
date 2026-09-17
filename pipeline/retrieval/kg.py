"""Knowledge-graph retrieval over the entity–passage graph built by preprocess_nlp.py.

    question ──scispaCy──▶ query entities ──exact/trigram match──▶ seed entity nodes
    seeds ──passage_ids──▶ candidate passages, scored Σ idf(entity)
    (1-hop) top candidate passages ──kg_entity_ids──▶ co-occurring entities ──▶ more passages, damped
    top passages ──▶ chunk of the requested strategy with highest cosine to the query (dense tie-break)

Scores are graph-structural (which entities connect question and passage), so the result is
explainable: every hit carries the entity path that produced it.
"""

from __future__ import annotations

import logging
import math
from collections import Counter

import numpy as np

from db.conn import get_pool
from pipeline.embedder import get_embedder
from pipeline.retrieval.base import Hit, Retriever
from pipeline.retrieval.grep import extract_terms

log = logging.getLogger("retrieve.kg")


def _vec(v) -> np.ndarray:
    """pgvector returns HalfVector objects; numpy wants floats."""
    return v.to_numpy().astype(np.float32) if hasattr(v, "to_numpy") else np.asarray(v, dtype=np.float32)


class KGRetriever(Retriever):
    name = "kg"

    def __init__(self, hop: bool = True, seeds_per_term: int = 3, hop_entities: int = 15, hop_damping: float = 0.3,
                 passage_candidates: int = 40, similarity_floor: float = 0.55):
        self.hop, self.seeds_per_term, self.hop_entities = hop, seeds_per_term, hop_entities
        self.damping, self.passage_candidates, self.sim_floor = hop_damping, passage_candidates, similarity_floor

    def _match_entities(self, cur, terms: list[str]) -> list[dict]:
        found: dict[int, dict] = {}
        for t in terms:
            cur.execute("select id, name, doc_freq, passage_ids from kg_entities where name = %s", (t,))
            rows = cur.fetchall()
            if not rows:
                cur.execute("""select id, name, doc_freq, passage_ids, similarity(name, %s) as sim from kg_entities
                               where name %% %s order by sim desc limit %s""", (t, t, self.seeds_per_term))
                rows = [r for r in cur.fetchall() if r["sim"] >= self.sim_floor]
            for r in rows:
                r = dict(r)
                r["query_term"] = t
                found[r["id"]] = r
        return list(found.values())

    def retrieve(self, query: str, strategy: str, k: int) -> list[Hit]:
        terms = extract_terms(query)
        if not terms:
            return []
        with get_pool().connection() as conn, conn.cursor() as cur:
            cur.execute("select count(*) n from passages")
            n_total = cur.fetchone()["n"]
            seeds = self._match_entities(cur, terms)
            if not seeds:
                return []
            score: dict[int, float] = {}
            path: dict[int, list[str]] = {}
            for e in seeds:
                w = math.log(n_total / max(e["doc_freq"], 1))
                for pid in e["passage_ids"]:
                    score[pid] = score.get(pid, 0.0) + w
                    path.setdefault(pid, []).append(e["name"])
            n_seed_passages = len(score)
            if self.hop and score:
                seed_ids = {e["id"] for e in seeds}
                top_pids = [pid for pid, _ in sorted(score.items(), key=lambda kv: -kv[1])[: self.passage_candidates]]
                cur.execute("select id, kg_entity_ids from passages where id = any(%s)", (top_pids,))
                co = Counter()
                for r in cur.fetchall():
                    for eid in r["kg_entity_ids"] or []:
                        if eid not in seed_ids:
                            co[eid] += 1
                hop_ids = [eid for eid, _ in co.most_common(self.hop_entities)]
                if hop_ids:
                    cur.execute("select id, name, doc_freq, passage_ids from kg_entities where id = any(%s)", (hop_ids,))
                    for e in cur.fetchall():
                        w = self.damping * math.log(n_total / max(e["doc_freq"], 1)) * (co[e["id"]] / len(top_pids))
                        for pid in e["passage_ids"]:
                            score[pid] = score.get(pid, 0.0) + w
                            path.setdefault(pid, []).append(f"~{e['name']}")
            top = sorted(score.items(), key=lambda kv: -kv[1])[: self.passage_candidates]
            pids = [pid for pid, _ in top]
            cur.execute("select id, passage_id, chunk_index, char_start, char_end, embedding from chunks "
                        "where strategy = %s and passage_id = any(%s)", (strategy, pids))
            by_pid: dict[int, list[dict]] = {}
            for r in cur.fetchall():
                by_pid.setdefault(r["passage_id"], []).append(r)
        qv = get_embedder().encode_query(query)
        hits = []
        for pid, sc in top:
            chunks = by_pid.get(pid)
            if not chunks:
                continue
            best = max(chunks, key=lambda c: float(np.dot(_vec(c["embedding"]), qv)))
            hits.append(Hit(best["id"], pid, sc, len(hits) + 1, self.name, best["chunk_index"], best["char_start"],
                            best["char_end"], meta={"entities": sorted(set(path[pid]))[:8]}))
            if len(hits) >= k:
                break
        log.debug("[kg] terms=%s seeds=%d seed_passages=%d -> %d hits", terms, len(seeds), n_seed_passages, len(hits))
        for h in hits:
            h.meta["seed_entities"] = [e["name"] for e in seeds][:10]
        return hits
