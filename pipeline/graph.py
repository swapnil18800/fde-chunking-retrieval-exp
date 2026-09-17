"""LangGraph pipeline: retrieve → generate → cite → log. One `run()` call = one traced, logged question.

    from pipeline.graph import run_pipeline
    out = run_pipeline("Is RANKL secreted from the cells?", RetrievalConfig(strategy="sentence", retriever="hybrid"))

State flows through 4 nodes; every node appends a timed entry to `stages` so the API, query_logs and
the trace all describe the same run. Tracing (Langfuse/LangSmith/off) is initialised at import.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from pipeline.citations import build_citations, numbered_context, strip_invalid
from pipeline.llm import get_llm
from pipeline.prompts import ANSWER_SYSTEM, ANSWER_USER, format_context
from pipeline.query_log import write_query_log
from pipeline.retrieval.base import Hit
from pipeline.retrieval.runner import RetrievalConfig, config_to_dict, run_retrieval
from pipeline.tracing import flush, init_tracing, langchain_callbacks, trace_span, update_span

log = logging.getLogger("graph")
init_tracing()


class RAGState(TypedDict, total=False):
    run_id: str
    question: str
    qa_id: int | None
    source: str
    config: dict
    callbacks: list
    hits: list[Hit]
    retrieval: dict
    stages: list[dict]
    answer: str
    citations: list[dict]
    tokens: dict
    error: str | None
    skip_generation: bool


def node_retrieve(state: RAGState) -> dict:
    cfg = RetrievalConfig(**state["config"])
    res = run_retrieval(state["question"], cfg, state.get("callbacks"))
    return {"hits": res.hits, "retrieval": res.to_dict(), "stages": state.get("stages", []) + res.stages}


def node_generate(state: RAGState) -> dict:
    if state.get("skip_generation"):
        return {"answer": "", "tokens": {}, "stages": state["stages"] + [{"name": "generate", "ms": 0, "skipped": True}]}
    t0 = time.time()
    ctx = numbered_context(state["hits"])
    if not ctx:
        return {"answer": "The retrieved passages do not answer this question (no passages retrieved).",
                "tokens": {}, "stages": state["stages"] + [{"name": "generate", "ms": 0, "no_context": True}]}
    prompt = ANSWER_USER.format(question=state["question"], context=format_context(ctx))
    r = get_llm("generate").chat(prompt, system=ANSWER_SYSTEM,
                                 config={"callbacks": state.get("callbacks") or [], "run_name": "generate-answer"})
    tokens = {"model": r.model, "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
              "attempts": r.attempts}
    return {"answer": r.text, "tokens": tokens,
            "stages": state["stages"] + [{"name": "generate", "ms": int((time.time() - t0) * 1000), **tokens,
                                          "context_chunks": len(ctx)}]}


def node_cite(state: RAGState) -> dict:
    t0 = time.time()
    cites, invalid = build_citations(state.get("answer", ""), state["hits"])
    answer = strip_invalid(state.get("answer", ""), invalid)
    return {"answer": answer, "citations": cites,
            "stages": state["stages"] + [{"name": "cite", "ms": int((time.time() - t0) * 1000),
                                          "citations": len(cites), "invalid_markers": invalid}]}


def build_graph():
    g = StateGraph(RAGState)
    g.add_node("retrieve-context", node_retrieve)
    g.add_node("generate-answer", node_generate)
    g.add_node("cite-sources", node_cite)
    g.add_edge(START, "retrieve-context")
    g.add_edge("retrieve-context", "generate-answer")
    g.add_edge("generate-answer", "cite-sources")
    g.add_edge("cite-sources", END)
    return g.compile()


_graph = None


def graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_pipeline(question: str, cfg: RetrievalConfig, qa_id: int | None = None, source: str = "api",
                 skip_generation: bool = False, metrics: dict | None = None, session_id: str | None = None,
                 log_query: bool = True) -> dict[str, Any]:
    """Execute the graph for one question. Returns a JSON-serialisable dict (also what gets logged)."""
    run_id = str(uuid.uuid4())
    t0 = time.time()
    cfg_d = config_to_dict(cfg)
    tags = [source, cfg.strategy, cfg.retriever] + (["rerank"] if cfg.rerank else [])
    out: dict[str, Any] = {"id": run_id, "question": question, "qa_id": qa_id, "source": source, "config": cfg_d}
    with trace_span("rag-answer", input={"question": question, "config": cfg_d}, session_id=session_id, tags=tags,
                    metadata={"qa_id": qa_id, "run_id": run_id}) as (ref, span):
        callbacks = langchain_callbacks()
        try:
            state = graph().invoke(
                {"run_id": run_id, "question": question, "qa_id": qa_id, "source": source, "config": cfg_d,
                 "callbacks": callbacks, "stages": [], "skip_generation": skip_generation},
                config={"callbacks": callbacks, "run_name": "rag-graph", "tags": tags,
                        "metadata": {"run_id": run_id, "qa_id": qa_id, **cfg_d}})
            out.update(answer=state.get("answer", ""), citations=state.get("citations", []),
                       retrieved=[h.to_dict() | {"text": h.text, "context_text": h.meta.get("context_text")}
                                  for h in state["hits"]],
                       stages=state["stages"], tokens=state.get("tokens", {}), queries=state["retrieval"]["queries"],
                       hypothetical=state["retrieval"].get("hypothetical"), error=None)
            update_span(span, output={"answer": out["answer"], "n_citations": len(out["citations"]),
                                      "retrieved_pmids": [h["passage_id"] for h in out["retrieved"]]})
        except Exception as e:  # noqa: BLE001
            log.exception("[graph] pipeline failed for %r", question[:80])
            out.update(answer="", citations=[], retrieved=[], stages=[], tokens={}, error=f"{type(e).__name__}: {e}")
            update_span(span, output={"error": out["error"]}, level="ERROR")
        out["latency_ms"] = int((time.time() - t0) * 1000)
        out["trace_provider"], out["trace_id"], out["trace_url"] = ref.provider, ref.trace_id, ref.url
    out["metrics"] = metrics
    if log_query:
        write_query_log(out)
    log.info("[graph] %s | %s | %dms | %d hits | %d cites | trace=%s%s", source, cfg.label(), out["latency_ms"],
             len(out.get("retrieved", [])), len(out.get("citations", [])), ref.provider,
             f" | ERROR {out['error']}" if out.get("error") else "",
             extra={"ctx_run_id": run_id, "ctx_config": cfg.label(), "ctx_latency_ms": out["latency_ms"]})
    return out


def shutdown() -> None:
    flush()
