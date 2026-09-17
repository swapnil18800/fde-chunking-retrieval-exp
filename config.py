"""Central configuration. Every env var is declared here — never read os.getenv elsewhere."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    # Database
    database_url: str
    supabase_url: str = ""
    supabase_publishable_key: str = ""
    db_pool_min: int = 1
    db_pool_max: int = 8

    # LLM
    llm_provider: str = "gemini"  # gemini | deepseek
    gemini_api_key: str = ""
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    gemini_models: str = "gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-3.5-flash"
    gemini_judge_model: str = "gemini-3.5-flash"
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    llm_max_tokens: int = 2048
    llm_temperature: float = 0.0

    # Local models
    embedding_model: str = "abhinand/MedEmbed-small-v0.1"
    embedding_dim: int = 384
    embedding_batch_size: int = 128
    reranker_model: str = "ncbi/MedCPT-Cross-Encoder"
    torch_device: str = "auto"

    # Tracing
    tracing_provider: str = "off"  # langfuse | langsmith | off
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_base_url: str = "https://cloud.langfuse.com"
    langsmith_api_key: str = ""
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_project: str = "fde-chunking-retrieval-exp"

    # App
    log_level: str = "INFO"
    log_dir: Path = Field(default=ROOT / "logs")
    cache_dir: Path = Field(default=ROOT / ".cache")
    api_port: int = 8000

    @property
    def gemini_model_list(self) -> list[str]:
        return [m.strip() for m in self.gemini_models.split(",") if m.strip()]


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.log_dir.mkdir(parents=True, exist_ok=True)
    s.cache_dir.mkdir(parents=True, exist_ok=True)
    return s
