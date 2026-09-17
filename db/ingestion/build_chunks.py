"""Chunk the corpus with one or more strategies, embed every chunk locally, and store in Supabase.

    uv run python db/ingestion/build_chunks.py --strategies passage fixed_128_32 recursive_128_32
    uv run python db/ingestion/build_chunks.py --strategies sentence semantic     # need preprocess_nlp.py first
    uv run python db/ingestion/build_chunks.py --all --replace
    uv run python db/ingestion/build_chunks.py --strategies passage --limit 500   # quick dev run

Idempotent per strategy: an existing strategy is skipped unless --replace. Embeddings are written
as halfvec via COPY in blocks of --block passages. Progress + throughput go to logs/ingest.jsonl.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
from rich.progress import Progress  # noqa: E402

from config import get_settings  # noqa: E402
from db.conn import connect  # noqa: E402
from pipeline.chunking.base import Passage  # noqa: E402
from pipeline.chunking.strategies import DEFAULT_STRATEGIES, build_chunker  # noqa: E402
from pipeline.embedder import Embedder  # noqa: E402
from pipeline.logging_setup import get_logger, setup_logging  # noqa: E402


# Free-tier budget (500 MB): HNSW on the four smaller sets; `sentence` (~240k chunks) is exact-scanned.
NO_INDEX = {"sentence"}


def create_hnsw(conn, name: str, log) -> None:
    idx = f"chunks_hnsw_{name}"
    t0 = time.time()
    with conn.cursor() as cur:
        cur.execute("set maintenance_work_mem = '256MB'")
        cur.execute(f"""create index if not exists {idx} on chunks using hnsw (embedding halfvec_cosine_ops)
                        with (m = 16, ef_construction = 96) where strategy = %s""", (name,))
        cur.execute("select pg_size_pretty(pg_relation_size(%s)) as size", (idx,))
        size = cur.fetchone()["size"]
    conn.commit()
    log.info("[chunk] hnsw index %s built in %.0fs (%s)", idx, time.time() - t0, size)


def halfvec_literal(v: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.5f}" for x in v.tolist()) + "]"


def fetch_passages(cur, limit: int | None) -> list[Passage]:
    cur.execute("select id, text, sentence_offsets from passages order by id" + (f" limit {int(limit)}" if limit else ""))
    return [Passage(r["id"], r["text"], r["sentence_offsets"]) for r in cur.fetchall()]


def build_one(conn, name: str, passages: list[Passage], embedder: Embedder, block: int, replace: bool, log,
              no_index: bool = False) -> None:
    with conn.cursor() as cur:
        cur.execute("select n_chunks from chunk_strategies where name = %s", (name,))
        row = cur.fetchone()
        if row and not replace:
            log.info("[chunk] %s already built (%d chunks) — skip (use --replace)", name, row["n_chunks"])
            return
        if row:
            cur.execute("delete from chunk_strategies where name = %s", (name,))  # cascades to chunks
            conn.commit()
    chunker = build_chunker(name, embedder)
    if chunker.requires_sentences and sum(p.sentence_offsets is None for p in passages) > len(passages) * 0.05:
        log.warning("[chunk] %s wants sentence offsets but most passages lack them — run preprocess_nlp.py first", name)
    if name == "semantic":
        thr = chunker.calibrate(passages[:: max(1, len(passages) // 1500)][:1500])
        log.info("[chunk] semantic threshold calibrated: %.4f (p%d of adjacent-sentence distances)", thr, chunker.pct)
    with conn.cursor() as cur:
        cur.execute("""insert into chunk_strategies (name, description, params, embedding_model, embedding_dim, n_chunks)
                       values (%s, %s, %s, %s, %s, 0)""",
                    (name, chunker.spec.description, json.dumps(chunker.spec.params), embedder.model_name, embedder.dim))
    conn.commit()

    n_chunks, tok_sum, t0 = 0, 0, time.time()
    with Progress() as prog:
        task = prog.add_task(f"[{name}]", total=len(passages))
        for i in range(0, len(passages), block):
            batch = passages[i:i + block]
            chunk_lists = chunker.chunk_batch(batch)
            flat = [(p, c) for p, cl in zip(batch, chunk_lists) for c in cl]
            vecs = embedder.encode([c.embed_text(p.text) for p, c in flat])
            with conn.cursor() as cur, cur.copy(
                "copy chunks (strategy, passage_id, chunk_index, char_start, char_end, n_tokens, prefix, embedding) from stdin"
            ) as cp:
                for (p, c), v in zip(flat, vecs):
                    cp.write_row((name, p.id, c.index, c.start, c.end, c.n_tokens, c.prefix, halfvec_literal(v)))
            conn.commit()
            n_chunks += len(flat)
            tok_sum += sum(c.n_tokens for _, c in flat)
            prog.update(task, advance=len(batch))
    dt = time.time() - t0
    with conn.cursor() as cur:
        cur.execute("update chunk_strategies set n_chunks = %s, avg_tokens = %s, built_at = now() where name = %s",
                    (n_chunks, tok_sum / max(n_chunks, 1), name))
        cur.execute("select pg_size_pretty(pg_database_size(current_database())) as size")
        size = cur.fetchone()["size"]
    conn.commit()
    log.info("[chunk] %s: %d chunks (%.2f/passage, avg %.0f tokens) in %.0fs (%.0f chunks/s). db=%s",
             name, n_chunks, n_chunks / len(passages), tok_sum / max(n_chunks, 1), dt, n_chunks / dt, size,
             extra={"ctx_strategy": name, "ctx_n_chunks": n_chunks, "ctx_seconds": round(dt, 1)})
    if name not in NO_INDEX and not no_index:
        create_hnsw(conn, name, log)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", nargs="*", default=[])
    ap.add_argument("--all", action="store_true", help=f"build {DEFAULT_STRATEGIES}")
    ap.add_argument("--replace", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="only first N passages (dev)")
    ap.add_argument("--block", type=int, default=512, help="passages per embed/COPY block")
    ap.add_argument("--no-index", action="store_true", help="skip HNSW creation (e.g. for --limit dev runs)")
    args = ap.parse_args()
    names = DEFAULT_STRATEGIES if args.all else args.strategies
    if not names:
        ap.error("give --strategies ... or --all")
    s = get_settings()
    setup_logging(s.log_level, s.log_dir, "ingest")
    log = get_logger("ingest.chunks")
    embedder = Embedder()
    with connect() as conn:
        with conn.cursor() as cur:
            passages = fetch_passages(cur, args.limit)
        log.info("[chunk] %d passages loaded; strategies=%s", len(passages), names)
        for name in names:
            build_one(conn, name, passages, embedder, args.block, args.replace, log, args.no_index)


if __name__ == "__main__":
    main()
