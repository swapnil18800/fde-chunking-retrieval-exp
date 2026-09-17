"""Shrink the corpus to a fixed-size subset so all chunk sets + HNSW fit the Supabase free tier (500 MB).

    uv run python db/ingestion/subsample_corpus.py --target 20000 --seed 42

Composition (deterministic):
  1. every gold passage of every question in the `eval150` set (so retrieval metrics stay exact),
  2. gold passages of *other* BioASQ questions, sampled — topically close "hard" distractors,
  3. random remaining passages until --target.
Everything not selected is DELETED from `passages` (chunks cascade). `qa_pairs.relevant_passage_ids`
is filtered to surviving ids (original count kept in `n_gold_total`), KG arrays are filtered, and the
recipe is recorded in `corpus_meta`. Re-run load_corpus.py --replace to get the full 40k back.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import get_settings  # noqa: E402
from db.conn import connect  # noqa: E402
from pipeline.logging_setup import get_logger, setup_logging  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--protect-set", default="eval150")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()
    s = get_settings()
    setup_logging(s.log_level, s.log_dir, "ingest")
    log = get_logger("ingest.subsample")
    rng = random.Random(args.seed)

    with connect() as conn, conn.cursor() as cur:
        cur.execute("select id from passages")
        all_ids = [r["id"] for r in cur.fetchall()]
        cur.execute("""select distinct p from qa_pairs q join eval_sets e on e.qa_id = q.id and e.name = %s,
                       unnest(q.relevant_passage_ids) p""", (args.protect_set,))
        protected = {r["p"] for r in cur.fetchall()}
        cur.execute("select distinct p from qa_pairs, unnest(relevant_passage_ids) p")
        gold_all = {r["p"] for r in cur.fetchall()} - protected
        if len(all_ids) <= args.target:
            log.info("[subsample] corpus already has %d <= %d passages — nothing to do", len(all_ids), args.target)
            return
        keep = set(protected)
        gold_pool = sorted(gold_all)
        rng.shuffle(gold_pool)
        n_gold_extra = min(len(gold_pool), max(0, (args.target - len(keep)) // 2))
        keep.update(gold_pool[:n_gold_extra])
        rest = [i for i in all_ids if i not in keep]
        rng.shuffle(rest)
        keep.update(rest[: max(0, args.target - len(keep))])
        drop = [i for i in all_ids if i not in keep]
        log.info("[subsample] keep %d = %d protected gold (%s) + %d other gold + %d random; drop %d",
                 len(keep), len(protected), args.protect_set, n_gold_extra, len(keep) - len(protected) - n_gold_extra,
                 len(drop))
        if not args.yes and input("Delete these passages? type 'yes': ") != "yes":
            sys.exit(1)

        cur.execute("set statement_timeout = '30min'")  # Supabase default (2 min) is too short for the array rewrites
        cur.execute("alter table qa_pairs add column if not exists n_gold_total int")
        cur.execute("update qa_pairs set n_gold_total = cardinality(relevant_passage_ids) where n_gold_total is null")
        cur.execute("create temp table tmp_drop (id bigint primary key) on commit drop")
        with cur.copy("copy tmp_drop (id) from stdin") as cp:
            for i in drop:
                cp.write_row((i,))
        cur.execute("delete from chunks where passage_id in (select id from tmp_drop)")
        cur.execute("delete from passages where id in (select id from tmp_drop)")
        # array rewrites as set-based LEFT JOINs (per-row sub-selects / NOT IN blow past the statement timeout)
        cur.execute("""update qa_pairs q set relevant_passage_ids = coalesce(t.ids, '{}') from (
                         select q.id, array_agg(p order by p) filter (where ps.id is not null) ids
                         from qa_pairs q, unnest(q.relevant_passage_ids) p left join passages ps on ps.id = p
                         group by q.id) t where t.id = q.id""")
        cur.execute("""update kg_entities e set passage_ids = coalesce(t.ids, '{}') from (
                         select e.id, array_agg(p order by p) filter (where ps.id is not null) ids
                         from kg_entities e, unnest(e.passage_ids) p left join passages ps on ps.id = p
                         group by e.id) t where t.id = e.id""")
        cur.execute("delete from kg_entities where cardinality(passage_ids) < 2")
        cur.execute("update kg_entities set doc_freq = cardinality(passage_ids)")
        cur.execute("""update passages p set kg_entity_ids = coalesce(t.ids, '{}') from (
                         select p.id, array_agg(e order by e) filter (where k.id is not null) ids
                         from passages p, unnest(p.kg_entity_ids) e left join kg_entities k on k.id = e
                         group by p.id) t where t.id = p.id""")
        cur.execute("update chunk_strategies set n_chunks = (select count(*) from chunks c where c.strategy = name)")
        cur.execute("""create table if not exists corpus_meta (key text primary key, value jsonb, updated_at timestamptz default now())""")
        cur.execute("""insert into corpus_meta (key, value) values ('subsample', %s)
                       on conflict (key) do update set value = excluded.value, updated_at = now()""",
                    (json.dumps({"target": args.target, "seed": args.seed, "protect_set": args.protect_set,
                                 "kept": len(keep), "protected_gold": len(protected), "other_gold": n_gold_extra,
                                 "dropped": len(drop), "source_total": len(all_ids)}),))
        conn.commit()
    with connect() as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            for t in ("passages", "kg_entities", "chunks", "qa_pairs"):
                cur.execute(f"vacuum full {t}")
            cur.execute("select count(*) n from passages")
            n = cur.fetchone()["n"]
            cur.execute("select count(*) n from qa_pairs where cardinality(relevant_passage_ids) = 0")
            empty = cur.fetchone()["n"]
            cur.execute("select count(*) n, sum(cardinality(passage_ids)) m from kg_entities")
            kg = cur.fetchone()
            cur.execute("select pg_size_pretty(pg_database_size(current_database())) as size")
            size = cur.fetchone()["size"]
    log.info("[subsample] done: %d passages, %d QA now have no gold in corpus (excluded from evals), "
             "kg=%d entities/%d mentions, db=%s", n, empty, kg["n"], kg["m"], size)


if __name__ == "__main__":
    main()
