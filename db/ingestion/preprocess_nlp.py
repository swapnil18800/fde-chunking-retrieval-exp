"""One spaCy pass over the corpus: sentence offsets (for sentence-aware chunkers) + knowledge graph.

    uv run python db/ingestion/preprocess_nlp.py              # skips passages that already have offsets
    uv run python db/ingestion/preprocess_nlp.py --replace    # recompute everything, rebuild kg_*

Knowledge graph = entity–passage bipartite graph (`kg_mentions`) + entity co-occurrence
edges (`kg_edges`, weight = #passages where both appear). Entities are scispaCy `en_core_sci_sm`
surface forms, lower-cased, noise-filtered (pipeline/nlp.py), kept when 2 <= doc_freq <= 5% of corpus.
Edges are kept when weight >= 2. No UMLS linking (kept deliberately simple and free).
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rich.progress import track  # noqa: E402

from config import get_settings  # noqa: E402
from db.conn import connect  # noqa: E402
from pipeline.logging_setup import get_logger, setup_logging  # noqa: E402
from pipeline.nlp import entities, load_sci_nlp, sentence_offsets  # noqa: E402

MAX_DOC_FREQ_FRAC = 0.05
MIN_DOC_FREQ = 2
MIN_EDGE_WEIGHT = 2
MAX_ENTS_PER_PASSAGE_FOR_EDGES = 25


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replace", action="store_true")
    ap.add_argument("--n-process", type=int, default=4)
    args = ap.parse_args()
    s = get_settings()
    setup_logging(s.log_level, s.log_dir, "ingest")
    log = get_logger("ingest.nlp")
    nlp = load_sci_nlp(disable=("lemmatizer",))

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("select id, text from passages order by id")
            rows = cur.fetchall()
            cur.execute("select count(*) n from passages where sentence_offsets is not null")
            done = cur.fetchone()["n"]
        if done == len(rows) and not args.replace:
            log.info("[nlp] all %d passages already preprocessed — skip (use --replace)", done)
            return
        n_total = len(rows)
        log.info("[nlp] processing %d passages with %s (n_process=%d)", n_total, nlp.meta["name"], args.n_process)

        offsets: list[tuple[int, list[int]]] = []
        mentions: dict[str, dict[int, int]] = defaultdict(dict)  # entity -> {passage_id: n}
        per_passage: dict[int, list[str]] = {}
        texts = (r["text"] for r in rows)
        ids = [r["id"] for r in rows]
        docs = nlp.pipe(texts, batch_size=256, n_process=args.n_process)
        for pid, doc in track(zip(ids, docs), total=n_total, description="spaCy"):
            offsets.append((pid, sentence_offsets(doc)))
            ents = entities(doc)
            per_passage[pid] = list(ents)
            for e, n in ents.items():
                mentions[e][pid] = n

        # ── sentence offsets → passages ───────────────────────────────────────
        with conn.cursor() as cur:
            cur.execute("create temp table tmp_off (id bigint, off int[]) on commit drop")
            with cur.copy("copy tmp_off (id, off) from stdin") as cp:
                for pid, off in offsets:
                    cp.write_row((pid, off))
            cur.execute("update passages p set sentence_offsets = t.off from tmp_off t where t.id = p.id")
        conn.commit()
        n_sent = sum(len(o) for _, o in offsets)
        log.info("[nlp] sentence offsets written: %d sentences, %.1f per passage", n_sent, n_sent / n_total)

        # ── knowledge graph ───────────────────────────────────────────────────
        max_df = int(n_total * MAX_DOC_FREQ_FRAC)
        keep = {e: pm for e, pm in mentions.items() if MIN_DOC_FREQ <= len(pm) <= max_df}
        log.info("[nlp] entities: %d raw -> %d kept (2 <= df <= %d)", len(mentions), len(keep), max_df)
        with conn.cursor() as cur:
            cur.execute("truncate kg_edges, kg_mentions, kg_entities restart identity")
            with cur.copy("copy kg_entities (name, doc_freq) from stdin") as cp:
                for e, pm in keep.items():
                    cp.write_row((e, len(pm)))
            cur.execute("select id, name from kg_entities")
            eid = {r["name"]: r["id"] for r in cur.fetchall()}
            with cur.copy("copy kg_mentions (entity_id, passage_id, n) from stdin") as cp:
                for e, pm in keep.items():
                    for pid, n in pm.items():
                        cp.write_row((eid[e], pid, n))
            # co-occurrence edges (undirected, stored once with src < dst)
            edges: Counter = Counter()
            for pid, ents in per_passage.items():
                kept = sorted((eid[e] for e in ents if e in eid))
                if len(kept) > MAX_ENTS_PER_PASSAGE_FOR_EDGES:  # prefer rarer entities
                    kept = sorted(kept, key=lambda i: 0)[:MAX_ENTS_PER_PASSAGE_FOR_EDGES]
                for a, b in combinations(kept, 2):
                    edges[(a, b)] += 1
            strong = [(a, b, w) for (a, b), w in edges.items() if w >= MIN_EDGE_WEIGHT]
            with cur.copy("copy kg_edges (src, dst, weight) from stdin") as cp:
                for a, b, w in strong:
                    cp.write_row((a, b, w))
            cur.execute("create extension if not exists pg_trgm")
            cur.execute("create index if not exists kg_entities_name_trgm on kg_entities using gin (name gin_trgm_ops)")
            cur.execute("select pg_size_pretty(pg_database_size(current_database())) as size")
            size = cur.fetchone()["size"]
        conn.commit()
        n_m = sum(len(pm) for pm in keep.values())
        log.info("[nlp] kg built: %d entities, %d mentions, %d edges (of %d pairs, w>=%d). db=%s",
                 len(keep), n_m, len(strong), len(edges), MIN_EDGE_WEIGHT, size)
        top = sorted(keep.items(), key=lambda kv: -len(kv[1]))[:15]
        log.info("[nlp] most frequent kept entities: %s", [(e, len(pm)) for e, pm in top])


if __name__ == "__main__":
    main()
