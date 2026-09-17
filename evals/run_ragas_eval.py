"""Tier-2 evaluation: end-to-end answer quality with RAGAS (LLM-judged) on the best retrieval configs.

    uv run python evals/run_ragas_eval.py --set smoke5 --configs passage+hybrid            # harness check
    uv run python evals/run_ragas_eval.py --set eval150 --configs sentence+hybrid+rerank+window passage+hybrid \\
        --limit 50

Config labels follow RetrievalConfig.label(): strategy+retriever[+transform][+rerank][+expansion].
For every question the full pipeline runs (Gemini generates a cited answer), then RAGAS scores:

    faithfulness              answer claims supported by retrieved context      (LLM judge)
    answer_relevancy          answer addresses the question                     (LLM + local embeddings)
    context_precision (id)    retrieved passage ids vs gold ids, rank-aware     (no LLM — exact)
    context_recall (id)       gold passage ids found in retrieved ids           (no LLM — exact)
    factual_correctness       answer vs BioASQ reference answer (F1 of claims)  (LLM judge)

Judge = GEMINI_JUDGE_MODEL (free tier; 15 RPM → run_config max_workers=2, generous timeout).
Results: eval_runs/eval_results (DB) + evals/results/<stamp>_ragas_<name>/ (summary.json, per_question.csv, report.md).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evals.ragas_compat  # noqa: F401, E402  (must precede ragas imports)
import pandas as pd  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402
from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import (  # noqa: E402
    FactualCorrectness, Faithfulness, IDBasedContextPrecision, IDBasedContextRecall, ResponseRelevancy,
)

from config import get_settings  # noqa: E402
from db.conn import connect, get_pool  # noqa: E402
from evals.metrics import aggregate, retrieval_metrics  # noqa: E402
from evals.run_retrieval_eval import load_set  # noqa: E402
from pipeline.graph import run_pipeline, shutdown  # noqa: E402
from pipeline.llm import judge_chat_model  # noqa: E402
from pipeline.logging_setup import get_logger, setup_logging  # noqa: E402
from pipeline.retrieval.runner import RetrievalConfig  # noqa: E402

RESULTS = Path(__file__).parent / "results"
RAGAS_COLS = ["faithfulness", "answer_relevancy", "id_based_context_precision", "id_based_context_recall",
              "factual_correctness(mode=f1)"]


class LocalEmbeddings(Embeddings):
    """LangChain Embeddings adapter over our sentence-transformers Embedder (no API cost)."""

    def __init__(self):
        from pipeline.embedder import get_embedder

        self.e = get_embedder()

    def embed_documents(self, texts):
        return self.e.encode(texts).tolist()

    def embed_query(self, text):
        return self.e.encode_query(text).tolist()


def parse_label(label: str, top_k: int, candidate_k: int) -> RetrievalConfig:
    parts = label.split("+")
    cfg = RetrievalConfig(strategy=parts[0], retriever=parts[1], top_k=top_k, candidate_k=candidate_k)
    for p in parts[2:]:
        if p == "rerank":
            cfg.rerank = True
        elif p in ("window", "parent"):
            cfg.expansion = p
        elif p in ("hyde", "multi_query", "decompose"):
            cfg.transform = p
        else:
            raise ValueError(f"unknown config part {p!r} in {label}")
    cfg.validate()
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="smoke5")
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--candidate-k", type=int, default=30)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=3, help="ragas concurrency (free tier: ~1 per Gemini model in rotation)")
    ap.add_argument("--judge", default=None, choices=[None, "gemini", "deepseek"],
                    help="judge provider (default LLM_PROVIDER). gemini = round-robin over GEMINI_MODELS")
    args = ap.parse_args()
    s = get_settings()
    setup_logging(s.log_level, s.log_dir, "eval")
    log = get_logger("eval.ragas")

    with connect() as conn:
        questions = load_set(conn, args.set)[: args.limit]
    configs = [parse_label(c, args.top_k, args.candidate_k) for c in args.configs]
    name = args.name or f"ragas_{args.set}_{len(configs)}cfg"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = RESULTS / f"{stamp}_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("[ragas] %s: %d questions x %d configs -> %s", name, len(questions), len(configs), out_dir)

    judge_name = args.judge or s.llm_provider
    judge = LangchainLLMWrapper(judge_chat_model(judge_name))
    emb = LangchainEmbeddingsWrapper(LocalEmbeddings())
    # strictness=1: Gemini's OpenAI-compatible endpoint rejects n>1 ("Multiple candidates is not enabled")
    metrics = [Faithfulness(llm=judge), ResponseRelevancy(llm=judge, embeddings=emb, strictness=1),
               IDBasedContextPrecision(), IDBasedContextRecall(), FactualCorrectness(llm=judge)]
    run_config = RunConfig(max_workers=args.workers, timeout=180, max_retries=6, max_wait=60)

    with connect() as conn, conn.cursor() as cur:
        cur.execute("""insert into eval_runs (name, kind, eval_set, config, n_questions) values (%s, 'ragas', %s, %s, %s)
                       returning id""", (name, args.set, json.dumps({"configs": [c.label() for c in configs],
                                                                      "top_k": args.top_k, "judge": judge_name}),
                                          len(questions)))
        run_id = cur.fetchone()["id"]
        conn.commit()

    summaries, per_q = [], []
    for cfg in configs:
        t0 = time.time()
        samples, runs = [], []
        for q in questions:
            out = run_pipeline(q["question"], cfg, qa_id=q["id"], source="eval")
            gold = [str(p) for p in q["relevant_passage_ids"]]
            ctx_ids = [str(h["passage_id"]) for h in out["retrieved"]]
            ctx = [h.get("context_text") or h.get("text") or "" for h in out["retrieved"]]
            samples.append(SingleTurnSample(user_input=q["question"], response=out["answer"] or "(no answer)",
                                            retrieved_contexts=ctx or ["(nothing retrieved)"],
                                            retrieved_context_ids=ctx_ids or ["-1"], reference_context_ids=gold,
                                            reference=q["answer"]))
            runs.append(out)
        log.info("[ragas] %s: pipeline done for %d questions in %.0fs — scoring with %s judge", cfg.label(), len(runs),
                 time.time() - t0, judge_name)
        result = evaluate(EvaluationDataset(samples=samples), metrics=metrics, run_config=run_config,
                          show_progress=True, raise_exceptions=False, batch_size=4)
        df = result.to_pandas()
        rows = []
        for q, out, (_, r) in zip(questions, runs, df.iterrows()):
            scores = {c: (None if pd.isna(r.get(c)) else float(r.get(c))) for c in RAGAS_COLS if c in df.columns}
            rm = retrieval_metrics([h["passage_id"] for h in out["retrieved"]], set(q["relevant_passage_ids"]))
            row = {**scores, "recall@5": rm.get("recall@5"), "mrr": rm.get("mrr"), "latency_ms": out["latency_ms"],
                   "n_citations": len(out["citations"]), "error": out.get("error")}
            rows.append(row)
            per_q.append({"config": cfg.label(), "qa_id": q["id"], "question_type": q["question_type"],
                          "question": q["question"], "answer": out["answer"], "reference": q["answer"],
                          "query_log_id": out["id"], "trace_url": out.get("trace_url"), **row})
            with get_pool().connection() as conn, conn.cursor() as cur:
                cur.execute("""insert into eval_results (run_id, config, qa_id, metrics, retrieved, answer, latency_ms, query_log_id)
                               values (%s, %s, %s, %s, %s, %s, %s, %s) on conflict do nothing""",
                            (run_id, cfg.label(), q["id"], json.dumps({"config": cfg.label(), **row}),
                             json.dumps([h["passage_id"] for h in out["retrieved"]]), out["answer"], out["latency_ms"], out["id"]))
                conn.commit()
        agg = aggregate(rows)
        agg.update(config=cfg.label(), seconds=round(time.time() - t0, 1), errors=sum(1 for r in rows if r.get("error")))
        summaries.append(agg)
        log.info("[ragas] %-45s %s", cfg.label(), {k: round(v, 3) for k, v in agg.items() if isinstance(v, float)})

    sdf = pd.DataFrame(summaries)
    sdf.to_csv(out_dir / "per_config.csv", index=False)
    pd.DataFrame(per_q).to_csv(out_dir / "per_question.csv", index=False)
    summary = {"run_id": str(run_id), "name": name, "eval_set": args.set, "n_questions": len(questions),
               "judge": judge_name, "top_k": args.top_k, "created": stamp, "configs": summaries}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    cols = ["config"] + [c for c in RAGAS_COLS + ["recall@5", "mrr", "latency_ms", "errors"] if c in sdf.columns]
    (out_dir / "report.md").write_text(
        f"# RAGAS eval — {name}\n\n- set `{args.set}` ({len(questions)} q) · judge `{judge_name}` · "
        f"top_k={args.top_k} · run `{run_id}` · {stamp}\n\n" + sdf[cols].round(3).to_markdown(index=False) + "\n")
    with connect() as conn, conn.cursor() as cur:
        cur.execute("update eval_runs set summary = %s, status = 'done', finished_at = now() where id = %s",
                    (json.dumps({"configs": summaries}, default=str), run_id))
        conn.commit()
    shutdown()
    log.info("[ragas] done -> %s", out_dir / "report.md")


if __name__ == "__main__":
    main()
