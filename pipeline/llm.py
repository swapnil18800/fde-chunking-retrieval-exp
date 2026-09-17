"""LLM factory. One OpenAI-compatible client class serves both providers:

* **gemini** (default, free tier): Gemini's OpenAI-compatible endpoint. Free-tier quota is
  15 RPM *per model*, so we round-robin across GEMINI_MODELS and honour 429 `retryDelay`.
* **deepseek** (fallback, paid): only when `LLM_PROVIDER=deepseek` or `role="judge"` is asked for
  and Gemini keeps failing. Keep credits for when quality is imperative.

Every call goes through `chat()` so token usage is captured uniformly (returned in LLMResult and
picked up by Langfuse/LangSmith through the LangChain callbacks in pipeline/tracing.py).
"""

from __future__ import annotations

import itertools
import logging
import re
import threading
import time
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import RateLimitError, APIStatusError

from config import get_settings

log = logging.getLogger("llm")
_RETRY_DELAY = re.compile(r"retry(?:Delay|.{0,20}in)\D*(\d+(?:\.\d+)?)\s*s", re.I)


@dataclass
class LLMResult:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    attempts: int = 1
    meta: dict = field(default_factory=dict)


class LLMClient:
    """Thread-safe round-robin over a list of models on one OpenAI-compatible base URL."""

    def __init__(self, provider: str | None = None, models: list[str] | None = None, max_tokens: int | None = None,
                 temperature: float | None = None):
        s = get_settings()
        self.provider = provider or s.llm_provider
        if self.provider == "gemini":
            self.base_url, self.api_key = s.gemini_base_url, s.gemini_api_key
            self.models = models or s.gemini_model_list
        elif self.provider == "deepseek":
            self.base_url, self.api_key = s.deepseek_base_url, s.deepseek_api_key
            self.models = models or [s.deepseek_model]
        else:
            raise ValueError(f"unknown LLM provider {self.provider}")
        self.max_tokens = max_tokens or s.llm_max_tokens
        self.temperature = s.llm_temperature if temperature is None else temperature
        self._cycle = itertools.cycle(self.models)
        self._lock = threading.Lock()
        self._clients: dict[str, BaseChatModel] = {}

    def _next_model(self) -> str:
        with self._lock:
            return next(self._cycle)

    def model(self, name: str | None = None) -> BaseChatModel:
        name = name or self._next_model()
        if name not in self._clients:
            self._clients[name] = ChatOpenAI(model=name, base_url=self.base_url, api_key=self.api_key,
                                             max_tokens=self.max_tokens, temperature=self.temperature,
                                             timeout=90, max_retries=0)
        return self._clients[name]

    def chat(self, prompt: str, system: str | None = None, model: str | None = None, max_attempts: int = 6,
             config: dict | None = None) -> LLMResult:
        msgs = ([SystemMessage(system)] if system else []) + [HumanMessage(prompt)]
        last_err: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            name = model or self._next_model()
            t0 = time.time()
            try:
                resp = self.model(name).invoke(msgs, config=config or {})
                usage = resp.usage_metadata or {}
                text = resp.content if isinstance(resp.content, str) else "".join(
                    b.get("text", "") for b in resp.content if isinstance(b, dict))
                return LLMResult(text=text.strip(), model=name, prompt_tokens=usage.get("input_tokens", 0),
                                 completion_tokens=usage.get("output_tokens", 0),
                                 latency_ms=int((time.time() - t0) * 1000), attempts=attempt)
            except RateLimitError as e:
                last_err = e
                m = _RETRY_DELAY.search(str(e))
                wait = min(float(m.group(1)) if m else 8.0 * attempt, 60.0)
                log.warning("[llm] 429 on %s (attempt %d) — sleeping %.0fs then rotating model", name, attempt, wait)
                time.sleep(wait if len(self.models) == 1 else min(wait, 5.0))
            except APIStatusError as e:
                last_err = e
                log.warning("[llm] %s on %s (attempt %d): %s", e.status_code, name, attempt, str(e)[:160])
                time.sleep(min(2.0 * attempt, 10.0))
            except Exception as e:  # network / timeout
                last_err = e
                log.warning("[llm] error on %s (attempt %d): %s", name, attempt, str(e)[:160])
                time.sleep(min(2.0 * attempt, 10.0))
        raise RuntimeError(f"LLM call failed after {max_attempts} attempts: {last_err}")


_clients: dict[str, LLMClient] = {}


def get_llm(role: str = "generate") -> LLMClient:
    """role: generate | judge | transform. Judge uses the stronger Gemini model (or DeepSeek if selected)."""
    s = get_settings()
    if role not in _clients:
        if role == "judge":
            if s.llm_provider == "deepseek":
                _clients[role] = LLMClient("deepseek", max_tokens=4096)
            else:
                _clients[role] = LLMClient("gemini", models=[s.gemini_judge_model], max_tokens=4096)
        elif role == "transform":  # short outputs: query rewrites, HyDE, entity extraction
            _clients[role] = LLMClient(max_tokens=512)
        else:
            _clients[role] = LLMClient()
    return _clients[role]


# ── LangChain-native round-robin model (for RAGAS / any framework that holds ONE model object) ──
from typing import Any, List, Optional  # noqa: E402

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun  # noqa: E402
from langchain_core.messages import BaseMessage  # noqa: E402
from langchain_core.outputs import ChatResult  # noqa: E402


class RoundRobinChat(BaseChatModel):
    """A BaseChatModel that spreads calls across several OpenAI-compatible models on the same key.

    Gemini free-tier quotas are *per model* (15 RPM, RPD per model), so a judge that rotates over
    GEMINI_MODELS gets N× the throughput. On 429/5xx it waits briefly and tries the next model.
    Only the fields LangChain needs are declared; everything else is delegated to ChatOpenAI.
    """

    client: Any  # LLMClient
    max_attempts: int = 6

    @property
    def _llm_type(self) -> str:
        return "round-robin-openai-compatible"

    @property
    def _identifying_params(self) -> dict:
        return {"models": self.client.models, "provider": self.client.provider}

    def _pick(self) -> BaseChatModel:
        return self.client.model()

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
                  run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> ChatResult:
        kwargs.pop("n", None)  # Gemini's OpenAI endpoint rejects n>1
        last: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            m = self._pick()
            try:
                return m._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            except Exception as e:  # noqa: BLE001
                last = e
                wait = 3.0 * attempt if "429" in str(e) else 1.5 * attempt
                log.warning("[llm] judge call failed on %s (attempt %d): %s — retrying in %.0fs",
                            getattr(m, "model_name", "?"), attempt, str(e)[:120], wait)
                time.sleep(min(wait, 30.0))
        raise RuntimeError(f"RoundRobinChat exhausted retries: {last}")

    async def _agenerate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
                         run_manager: Optional[AsyncCallbackManagerForLLMRun] = None, **kwargs: Any) -> ChatResult:
        import asyncio

        kwargs.pop("n", None)
        last: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            m = self._pick()
            try:
                return await m._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
            except Exception as e:  # noqa: BLE001
                last = e
                wait = 3.0 * attempt if "429" in str(e) else 1.5 * attempt
                log.warning("[llm] judge call failed on %s (attempt %d): %s — retrying in %.0fs",
                            getattr(m, "model_name", "?"), attempt, str(e)[:120], wait)
                await asyncio.sleep(min(wait, 30.0))
        raise RuntimeError(f"RoundRobinChat exhausted retries: {last}")


def judge_chat_model(provider: str | None = None) -> BaseChatModel:
    """LangChain model for RAGAS. gemini → round-robin over all GEMINI_MODELS (+ judge model); deepseek → single."""
    s = get_settings()
    provider = provider or s.llm_provider
    if provider == "deepseek":
        return LLMClient("deepseek", max_tokens=4096).model()
    models = list(dict.fromkeys([s.gemini_judge_model, *s.gemini_model_list]))
    return RoundRobinChat(client=LLMClient("gemini", models=models, max_tokens=4096))
