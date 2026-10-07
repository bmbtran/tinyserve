"""Request models for the OpenAI-compatible subset tinyserve implements.

Unknown fields are ignored (clients like `openai` send extras)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StreamOptions(BaseModel):
    include_usage: bool = False


class _Common(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str | None = None
    max_tokens: int | None = None
    temperature: float = 1.0  # OpenAI default
    top_p: float = 1.0
    top_k: int = -1
    stream: bool = False
    stream_options: StreamOptions | None = None
    ignore_eos: bool = False  # vLLM extension, used by fixed-length benchmarks
    seed: int | None = None
    stop_token_ids: list[int] = Field(default_factory=list)


class CompletionRequest(_Common):
    prompt: str | list[int]
    max_tokens: int | None = 16  # OpenAI default for completions


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: str
    content: str


class ChatCompletionRequest(_Common):
    messages: list[ChatMessage]
    max_completion_tokens: int | None = None
    chat_template_kwargs: dict[str, Any] = Field(default_factory=lambda: {"enable_thinking": False})
