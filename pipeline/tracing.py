"""Tracing switch: TRACING_PROVIDER = langfuse | langsmith | off.

* langfuse  – Langfuse Cloud via the LangChain CallbackHandler (LangGraph nodes, LLM generations
              with model + token usage) plus `@observe`-style manual spans for retrievers.
* langsmith – native LangChain tracing (env-driven). Used once to validate the harness; credits are
              limited, so leave it off afterwards.
* off       – no exporter; the pipeline still writes query_logs to Postgres and logs/*.jsonl.

`init_tracing()` must run BEFORE any LangChain/OpenAI client is created (import order matters for
the auto-instrumentation) — pipeline/graph.py calls it at import time.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass

from config import get_settings

log = logging.getLogger("tracing")
_state: dict = {"provider": None, "langfuse": None}


@dataclass
class TraceRef:
    provider: str
    trace_id: str | None = None
    url: str | None = None


def init_tracing() -> str:
    s = get_settings()
    if _state["provider"] is not None:
        return _state["provider"]
    provider = (s.tracing_provider or "off").lower()
    if provider == "langfuse":
        if not (s.langfuse_public_key and s.langfuse_secret_key):
            log.warning("[tracing] langfuse selected but keys missing — tracing off")
            provider = "off"
        else:
            os.environ.setdefault("LANGFUSE_PUBLIC_KEY", s.langfuse_public_key)
            os.environ.setdefault("LANGFUSE_SECRET_KEY", s.langfuse_secret_key)
            os.environ.setdefault("LANGFUSE_BASE_URL", s.langfuse_base_url)
            from langfuse import get_client

            _state["langfuse"] = get_client()
            ok = _state["langfuse"].auth_check()
            log.info("[tracing] langfuse auth_check=%s host=%s", ok, s.langfuse_base_url)
            if not ok:
                provider = "off"
    elif provider == "langsmith":
        if not s.langsmith_api_key:
            log.warning("[tracing] langsmith selected but LANGSMITH_API_KEY missing — tracing off")
            provider = "off"
        else:
            os.environ["LANGSMITH_TRACING"] = "true"
            os.environ["LANGSMITH_API_KEY"] = s.langsmith_api_key
            os.environ["LANGSMITH_ENDPOINT"] = s.langsmith_endpoint
            os.environ["LANGSMITH_PROJECT"] = s.langsmith_project
            log.info("[tracing] langsmith project=%s", s.langsmith_project)
    else:
        provider = "off"
        os.environ["LANGSMITH_TRACING"] = "false"
    _state["provider"] = provider
    return provider


def provider() -> str:
    return _state["provider"] or init_tracing()


def langchain_callbacks() -> list:
    """Callbacks to attach to graph.invoke(config=...). LangSmith is env-driven → none needed."""
    if provider() == "langfuse":
        from langfuse.langchain import CallbackHandler

        return [CallbackHandler()]
    return []


@contextmanager
def trace_span(name: str, input: dict | None = None, session_id: str | None = None, tags: list[str] | None = None,
               metadata: dict | None = None):
    """Root observation for one pipeline run (Langfuse SDK v4 semantics).

    The root span *is* the trace: its name/input/output become the trace's. Trace-level attributes
    (session_id, tags, metadata) are propagated to every nested observation — including the ones the
    LangChain CallbackHandler creates for LangGraph nodes and LLM generations — via propagate_attributes.
    Yields (TraceRef, span). For langsmith the root run is created by LangGraph itself.
    """
    p = provider()
    ref = TraceRef(p)
    if p == "langfuse":
        from langfuse import propagate_attributes

        lf = _state["langfuse"]
        meta = {k: str(v) for k, v in (metadata or {}).items() if v is not None}
        with lf.start_as_current_observation(as_type="chain", name=name, input=input, metadata=meta) as span:
            with propagate_attributes(session_id=session_id, tags=tags or [], metadata=meta, trace_name=name):
                ref.trace_id = lf.get_current_trace_id()
                ref.url = lf.get_trace_url(trace_id=ref.trace_id)
                span.set_trace_io(input=input)
                yield ref, span
        return
    if p == "langsmith":
        s = get_settings()
        ref.url = f"https://smith.langchain.com/o/-/projects/p/{s.langsmith_project}"
    yield ref, None


def update_span(span, output=None, **kwargs) -> None:
    """Set the root span's output (also becomes the trace output) and any extra attributes."""
    if span is not None:
        span.update(output=output, **kwargs)
        if output is not None:
            span.set_trace_io(output=output)


@contextmanager
def child_span(name: str, as_type: str = "span", input=None, metadata: dict | None = None):
    """Nested observation (retriever / reranker / tool) under the current trace. No-op unless langfuse."""
    if provider() == "langfuse":
        lf = _state["langfuse"]
        with lf.start_as_current_observation(as_type=as_type, name=name, input=input, metadata=metadata) as obs:
            yield obs
    else:
        yield None


def flush() -> None:
    if provider() == "langfuse" and _state["langfuse"] is not None:
        _state["langfuse"].flush()
