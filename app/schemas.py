"""Pydantic request/response models for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    strategy: str = "passage"
    retriever: str = "hybrid"
    transform: str = "none"
    rerank: bool = False
    expansion: str = "none"
    top_k: int = Field(default=5, ge=1, le=20)
    candidate_k: int = Field(default=20, ge=1, le=100)
    qa_id: int | None = None
    generate: bool = True
    session_id: str | None = None


class CompareRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    configs: list[dict] = Field(min_length=1, max_length=6)
    qa_id: int | None = None
    generate: bool = False
