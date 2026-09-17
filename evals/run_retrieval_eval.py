"""Tier-1 evaluation: retrieval quality over the chunking x retrieval matrix — no LLM, no cost.

    uv run python evals/run_retrieval_eval.py --set smoke5 --strategies passage --retrievers bm25 dense
    uv run python evals/run_retrieval_eval.py --set eval150 --matrix                # 5 strategies x 5 retrievers
    uv run python evals/run_retrieval_eval.py --set eval150 --matrix --rerank       # + cross-encoder on every cell
    uv run python evals/run_retrieval_eval.py --set eval150 --strategies sentence --retrievers hybrid \\
        --transforms hyde multi_query decompose                                     # LLM query transforms (Gemini)

For every (question, config) the ranked passage list is scored against BioASQ gold PMIDs
(evals/metrics.py). Results go to eval_runs/eval_results (DB) and evals/results/<timestamp>_<name>/
(summary.json, per_config.csv, report.md). Every question run is also a `query_logs` row (source=eval).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from rich.progress import Progress  # noqa: E402

from config import get_settings  # noqa: E402
from db.conn import connect  # noqa: E402
from evals.metrics import aggregate, retrieval_metrics  # noqa: E402
from pipeline.graph import run_pipeline, shutdown  # noqa: E402
from pipeline.logging_setup import get_logger, setup_logging  # noqa: E402
from pipeline.retrieval.runner import RETRIEVERS, RetrievalConfig  # noqa: E402

RESULTS = Path(__file__).parent / "results"
KEY_METRICS = ["recall@5", "recall@10", "precision@5", "mrr", "ndcg@10", "hit@5"]


def load_set(conn, name: str) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute("""select q.id, q.question, q.answer, q.relevant_passage_ids, q.question_type from eval_sets e
                       join qa_pairs q on q.id = e.qa_id where e.name = %s and cardinality(q.relevant_passage_ids) > 0
                       order by e.position""", (name,))
        return cur.fetchall()


def built_strategies(conn) -> list[str]:
    with conn.cursor() as cur:
        cur.execute("select name from chunk_strategies where n_chunks > 0 order by n_chunks")
        return [r["name"] for r in cur.fetchall()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="smoke5")
    ap.add_argument("--name", default=None)
    ap.add_argument("--strategies", nargs="*", default=None)
    ap.add_argument("--retrievers", nargs="*", default=None)
    ap.add_argument("--transforms", nargs="*", default=["none"])
    ap.add_argument("--expansions", nargs="*", default=["none"])
    ap.add_argument("--matrix", action="store_true", help="all built strategies x all retrievers")
    ap.add_argument("--rerank", action="store_true", help="also run every cell with reranking")
    ap.add_argument("--rerank-only", action="store_true")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--candidate-k", type=int, default=30)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    s = get_settings()
    setup_logging(s.log_level, s.log_dir, "eval")
    log = get_logger("eval.retrieval")

    with connect() as conn:
        questions = load_set(conn, args.set)[: args.limit]
        strategies = args.strategies or (built_strategies(conn) if args.matrix else ["passage"])
    retrievers = args.retrievers or (list(RETRIEVERS) if args.matrix else ["hybrid"])
    rerank_opts = [True] if args.rerank_only else ([False, True] if args.rerank else [False])
    configs = [RetrievalConfig(strategy=st, retriever=r, transform=t, rerank=rr, expansion=ex,
                               top_k=args.top_k, candidate_k=args.candidate_k)
               for st in strategies for r in retrievers for t in args.transforms for rr in rerank_opts
               for ex in args.expansions]
    name = args.name or f"{args.set}_{'matrix' if args.matrix else '+'.join(strategies)}"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = RESULTS / f"{stamp}_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("[eval] %s: %d questions x %d configs -> %s", name, len(questions), len(configs), out_dir)

    with connect() as conn, conn.cursor() as cur:
        cur.execute("""insert into eval_runs (name, kind, eval_set, config, n_questions) values (%s, 'retrieval', %s, %s, %s)
                       returning id""", (name, args.set, json.dumps({"configs": [c.label() for c in configs],
                                                                      "top_k": args.top_k, "candidate_k": args.candidate_k}),
                                          len(questions)))
        run_id = cur.fetchone()["id"]
        conn.commit()

    per_q_rows, summaries = [], []
    with Progress() as prog:
        task = prog.add_task("configs", total=len(configs) * len(questions))
        for cfg in configs:
            rows, t0 = [], time.time()
            for q in questions:
                gold = set(q["relevant_passage_ids"])
                res = run_pipeline(q["question"], cfg, qa_id=q["id"], source="eval", skip_generation=True)
                ranked = [h["passage_id"] for h in res.get("retrieved", [])]
                m = retrieval_metrics(ranked, gold)
                m["latency_ms"] = res["latency_ms"]
                m["error"] = res.get("error")
                rows.append(m)
                per_q_rows.append({"config": cfg.label(), "qa_id": q["id"], "question_type": q["question_type"],
                                   "query_log_id": res["id"], **m})
                with connect() as conn, conn.cursor() as cur:
                    cur.execute("""insert into eval_results (run_id, qa_id, metrics, retrieved, latency_ms, query_log_id)
                                   values (%s, %s, %s, %s, %s, %s) on conflict do nothing""",
                                (run_id, q["id"], json.dumps({"config": cfg.label(), **m}), json.dumps(ranked),
                                 res["latency_ms"], res["id"]))
                    conn.commit()
                prog.update(task, advance=1)
            agg = aggregate(rows)
            agg.update(config=cfg.label(), strategy=cfg.strategy, retriever=cfg.retriever, transform=cfg.transform,
                       rerank=cfg.rerank, expansion=cfg.expansion, errors=sum(1 for r in rows if r.get("error")),
                       seconds=round(time.time() - t0, 1))
            summaries.append(agg)
            log.info("[eval] %-45s recall@5=%.3f recall@10=%.3f mrr=%.3f ndcg@10=%.3f  (%.1fs, %d err)",
                     cfg.label(), agg.get("recall@5", 0), agg.get("recall@10", 0), agg.get("mrr", 0),
                     agg.get("ndcg@10", 0), agg["seconds"], agg["errors"])

    df = pd.DataFrame(summaries)
    df.to_csv(out_dir / "per_config.csv", index=False)
    pd.DataFrame(per_q_rows).to_csv(out_dir / "per_question.csv", index=False)
    summary = {"run_id": str(run_id), "name": name, "eval_set": args.set, "n_questions": len(questions),
               "top_k": args.top_k, "candidate_k": args.candidate_k, "created": stamp, "configs": summaries}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    (out_dir / "report.md").write_text(render_report(summary, df))
    with connect() as conn, conn.cursor() as cur:
        cur.execute("update eval_runs set summary = %s, status = 'done', finished_at = now() where id = %s",
                    (json.dumps({"configs": summaries}, default=str), run_id))
        conn.commit()
    shutdown()
    log.info("[eval] done -> %s", out_dir / "report.md")


def render_report(summary: dict, df: pd.DataFrame) -> str:
    cols = ["config"] + [c for c in KEY_METRICS if c in df.columns] + ["latency_ms", "errors"]
    table = df.sort_values("recall@10", ascending=False)[cols].round(3).to_markdown(index=False)
    lines = [f"# Retrieval eval — {summary['name']}", "",
             f"- set: `{summary['eval_set']}` ({summary['n_questions']} questions) · top_k={summary['top_k']} · "
             f"candidate_k={summary['candidate_k']} · run_id `{summary['run_id']}` · {summary['created']}", "",
             "Metrics are passage-level (chunk hits collapsed to parent PMID). Sorted by recall@10.", "", table, ""]
    if {"strategy", "retriever"} <= set(df.columns) and df["strategy"].nunique() > 1 and df["retriever"].nunique() > 1:
        base = df[(~df["rerank"]) & (df["transform"] == "none") & (df["expansion"] == "none")]
        if len(base):
            pv = base.pivot_table(index="strategy", columns="retriever", values="recall@10").round(3)
            lines += ["## recall@10 — chunking × retriever (no rerank)", "", pv.to_markdown(), ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
