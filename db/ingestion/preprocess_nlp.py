"""One spaCy pass over the corpus: sentence offsets (for sentence-aware chunkers) + knowledge graph.

    uv run python db/ingestion/preprocess_nlp.py              # skips passages that already have offsets
    uv run python db/ingestion/preprocess_nlp.py --replace    # recompute everything, rebuild kg_*

Knowledge graph = entity–passage bipartite graph stored as two array columns (kg_entities.passage_ids,
passages.kg_entity_ids) — ~25 MB instead of ~180 MB as row tables. Entity co-occurrence (1-hop
expansion) is derived at query time. Entities are scispaCy `en_core_sci_sm` surface forms,
lower-cased, noise-filtered (pipeline/nlp.py), df-capped. No UMLS linking (deliberately simple and free).
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rich.progress import track  # noqa: E402

from config import get_settings  # noqa: E402
from db.conn import connect  # noqa: E402
from pipeline.logging_setup import get_logger, setup_logging  # noqa: E402
from pipeline.nlp import entities, load_sci_nlp, sentence_offsets  # noqa: E402

MAX_DOC_FREQ_FRAC = 0.025      # multi-word entities: drop if in > 2.5% of passages
MAX_DF_SINGLE_WORD = 300       # single words ("mechanism", "reduced") are mostly generic — tighter cap
MIN_DOC_FREQ = 2


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

        # ── knowledge graph (array form) ──────────────────────────────────────
        max_df = int(n_total * MAX_DOC_FREQ_FRAC)
        keep = {e: pm for e, pm in mentions.items()
                if MIN_DOC_FREQ <= len(pm) <= (max_df if " " in e else MAX_DF_SINGLE_WORD)}
        log.info("[nlp] entities: %d raw -> %d kept (df in [%d, %d], single-word <= %d)", len(mentions), len(keep),
                 MIN_DOC_FREQ, max_df, MAX_DF_SINGLE_WORD)
        with conn.cursor() as cur:
            cur.execute("truncate kg_entities restart identity")
            with cur.copy("copy kg_entities (name, doc_freq, passage_ids) from stdin") as cp:
                for e, pm in keep.items():
                    cp.write_row((e, len(pm), sorted(pm)))
            cur.execute("select id, name from kg_entities")
            eid = {r["name"]: r["id"] for r in cur.fetchall()}
            cur.execute("create temp table tmp_pe (id bigint, ents int[]) on commit drop")
            with cur.copy("copy tmp_pe (id, ents) from stdin") as cp:
                for pid, ents in per_passage.items():
                    cp.write_row((pid, sorted(eid[e] for e in ents if e in eid)))
            cur.execute("update passages p set kg_entity_ids = t.ents from tmp_pe t where t.id = p.id")
            cur.execute("create extension if not exists pg_trgm")
            cur.execute("create index if not exists kg_entities_name_trgm on kg_entities using gin (name gin_trgm_ops)")
            cur.execute("select pg_size_pretty(pg_database_size(current_database())) as size")
            size = cur.fetchone()["size"]
        conn.commit()
        n_m = sum(len(pm) for pm in keep.values())
        log.info("[nlp] kg built: %d entities, %d entity-passage links. db=%s", len(keep), n_m, size)
        top = sorted(keep.items(), key=lambda kv: -len(kv[1]))[:15]
        log.info("[nlp] most frequent kept entities: %s", [(e, len(pm)) for e, pm in top])


if __name__ == "__main__":
    main()
