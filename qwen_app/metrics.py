from __future__ import annotations

from dataclasses import dataclass
import asyncio
import time
from typing import Any


def approx_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text) / 4))


@dataclass
class Metrics:
    request_id: str
    model: str
    first_token_ms: float | None
    total_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    completion_tps: float | None
    total_tps: float | None
    created_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "model": self.model,
            "first_token_ms": self.first_token_ms,
            "total_ms": self.total_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "completion_tps": self.completion_tps,
            "total_tps": self.total_tps,
            "created_at": self.created_at,
        }


class MetricsBuffer:
    def __init__(self, size: int) -> None:
        self._size = max(1, size)
        self._items: list[Metrics] = []

    def add(self, metrics: Metrics) -> None:
        self._items.append(metrics)
        if len(self._items) > self._size:
            self._items = self._items[-self._size :]

    def list(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self._items]


class TokenCounter:
    def __init__(self, model: str) -> None:
        self._model = model
        self._tokenizer: Any | None = None
        self._load_failed = False
        self._lock = asyncio.Lock()

    async def _ensure_tokenizer(self) -> Any | None:
        if self._tokenizer is not None or self._load_failed:
            return self._tokenizer
        async with self._lock:
            if self._tokenizer is not None or self._load_failed:
                return self._tokenizer

            def _load() -> Any:
                from transformers import AutoTokenizer

                return AutoTokenizer.from_pretrained(self._model)

            try:
                self._tokenizer = await asyncio.to_thread(_load)
            except Exception:
                self._load_failed = True
                self._tokenizer = None
        return self._tokenizer

    async def count_messages(self, messages: list[dict[str, Any]]) -> int:
        tokenizer = await self._ensure_tokenizer()
        if tokenizer is None:
            text = "\n".join(
                f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                for msg in messages
            )
            return approx_tokens(text)
        try:
            if hasattr(tokenizer, "apply_chat_template"):
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                text = "\n".join(
                    f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                    for msg in messages
                )
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            text = "\n".join(
                f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                for msg in messages
            )
            return approx_tokens(text)

    async def count_text(self, text: str) -> int:
        tokenizer = await self._ensure_tokenizer()
        if tokenizer is None:
            return approx_tokens(text)
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            return approx_tokens(text)


def build_metrics(
    request_id: str,
    model: str,
    start_time: float,
    end_time: float,
    first_token_time: float | None,
    prompt_tokens: int | None,
    completion_tokens: int | None,
) -> Metrics:
    total_ms = (end_time - start_time) * 1000
    first_token_ms = (
        (first_token_time - start_time) * 1000 if first_token_time is not None else None
    )
    total_tokens = (
        prompt_tokens + completion_tokens
        if prompt_tokens is not None and completion_tokens is not None
        else None
    )
    completion_window = end_time - (first_token_time or start_time)
    completion_tps = (
        completion_tokens / completion_window
        if completion_tokens is not None and completion_window > 0
        else None
    )
    total_tps = (
        total_tokens / (end_time - start_time)
        if total_tokens is not None and end_time > start_time
        else None
    )
    return Metrics(
        request_id=request_id,
        model=model,
        first_token_ms=first_token_ms,
        total_ms=total_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        completion_tps=completion_tps,
        total_tps=total_tps,
        created_at=time.time(),
    )
