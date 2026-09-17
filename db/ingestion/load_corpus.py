"""Load rag-mini-bioasq into Supabase: `passages` (40,221 PubMed abstracts) and `qa_pairs` (4,719).

    uv run python db/ingestion/load_corpus.py            # skips tables that are already populated
    uv run python db/ingestion/load_corpus.py --replace  # truncate + reload

Normalisation: passages in the HF dump are hard-wrapped at ~80 chars ("...\\n"). We collapse all
whitespace runs to a single space so sentence splitting and offsets are stable. Chunk offsets
everywhere in this repo refer to THIS normalised text.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import tiktoken  # noqa: E402
from datasets import load_dataset  # noqa: E402
from rich.progress import track  # noqa: E402

from config import get_settings  # noqa: E402
from db.conn import connect  # noqa: E402
from pipeline.logging_setup import get_logger, setup_logging  # noqa: E402

DATASET = "rag-datasets/rag-mini-bioasq"
_WS = re.compile(r"\s+")
_YESNO = re.compile(r"^(is|are|does|do|can|has|have|was|were|should|could|did|will)\b", re.I)
_LIST = re.compile(r"^(list|which|what are|name)\b", re.I)


def normalise(text: str) -> str:
    return _WS.sub(" ", text).strip()


def question_type(q: str) -> str:
    if _YESNO.match(q):
        return "yesno"
    if _LIST.match(q):
        return "list"
    if len(q.split()) > 14 or q.lower().startswith(("describe", "what is the role", "what is known", "how")):
        return "summary"
    return "factoid"


def load_passages(conn, enc, replace: bool, log) -> None:
    with conn.cursor() as cur:
        cur.execute("select count(*) as n from passages")
        existing = cur.fetchone()["n"]
        if existing and not replace:
            log.info("[ingest] passages already loaded (%d) — skip (use --replace)", existing)
            return
        if replace:
            cur.execute("truncate passages cascade")
    ds = load_dataset(DATASET, "text-corpus", split="passages")
    rows = []
    for r in track(ds, description="normalising passages", total=len(ds)):
        t = normalise(r["passage"])
        if not t:
            continue
        rows.append((int(r["id"]), t, len(t), len(t.split()), len(enc.encode(t))))
    with conn.cursor() as cur, cur.copy("copy passages (id, text, n_chars, n_words, n_tokens) from stdin") as cp:
        for row in rows:
            cp.write_row(row)
    conn.commit()
    log.info("[ingest] passages loaded: %d (dropped %d empty)", len(rows), len(ds) - len(rows))


def load_qa(conn, replace: bool, log) -> None:
    with conn.cursor() as cur:
        cur.execute("select count(*) as n from qa_pairs")
        existing = cur.fetchone()["n"]
        if existing and not replace:
            log.info("[ingest] qa_pairs already loaded (%d) — skip (use --replace)", existing)
            return
        if replace:
            cur.execute("truncate qa_pairs cascade")
        cur.execute("select id from passages")
        known = {r["id"] for r in cur.fetchall()}
    ds = load_dataset(DATASET, "question-answer-passages", split="test")
    rows, missing = [], 0
    for r in ds:
        ids = [int(x) for x in ast.literal_eval(r["relevant_passage_ids"])]
        kept = [i for i in ids if i in known]
        missing += len(ids) - len(kept)
        q = normalise(r["question"])
        rows.append((int(r["id"]), q, normalise(r["answer"]), kept, question_type(q)))
    with conn.cursor() as cur, cur.copy(
        "copy qa_pairs (id, question, answer, relevant_passage_ids, question_type) from stdin"
    ) as cp:
        for row in rows:
            cp.write_row(row)
    conn.commit()
    log.info("[ingest] qa_pairs loaded: %d (gold ids not in corpus: %d)", len(rows), missing)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replace", action="store_true")
    args = ap.parse_args()
    s = get_settings()
    setup_logging(s.log_level, s.log_dir, "ingest")
    log = get_logger("ingest.corpus")
    enc = tiktoken.get_encoding("cl100k_base")
    with connect() as conn:
        load_passages(conn, enc, args.replace, log)
        load_qa(conn, args.replace, log)
        with conn.cursor() as cur:
            cur.execute("""select count(*) n, round(avg(n_words)) w, round(avg(n_tokens)) t,
                                  min(n_tokens) mn, max(n_tokens) mx from passages""")
            log.info("[ingest] passage stats: %s", dict(cur.fetchone()))
            cur.execute("select question_type, count(*) n from qa_pairs group by 1 order by 2 desc")
            log.info("[ingest] question types: %s", {r["question_type"]: r["n"] for r in cur.fetchall()})
            cur.execute("select pg_size_pretty(pg_database_size(current_database())) as size")
            log.info("[ingest] db size: %s", cur.fetchone()["size"])


if __name__ == "__main__":
    main()
