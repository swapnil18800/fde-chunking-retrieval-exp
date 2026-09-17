"""Import shim: ragas 0.4.x still imports `langchain_community.chat_models.vertexai`, which was removed in
langchain-community 0.4. We only ever pass OpenAI-compatible LangChain models, so a stub class is enough.
Import this module BEFORE anything from `ragas`.
"""

from __future__ import annotations

import sys
import types

try:  # pragma: no cover
    import langchain_community.chat_models.vertexai  # noqa: F401
except ImportError:
    _stub = types.ModuleType("langchain_community.chat_models.vertexai")

    class ChatVertexAI:  # noqa: D401 - placeholder only used for isinstance checks inside ragas
        pass

    _stub.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _stub
