from __future__ import annotations

from dataclasses import dataclass
import os
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse


@dataclass(frozen=True)
class Settings:
    upstream_url: str
    proxy_url: str
    model: str
    history_path: Path
    timeout: float
    temperature: float
    max_tokens: int
    metrics_buffer: int


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value is not None and value.strip() else default


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    history_raw = _env("QWEN_HISTORY_PATH", "~/.qwen_tui/history.json")
    history_path = Path(os.path.expanduser(history_raw))
    return Settings(
        upstream_url=_env("QWEN_UPSTREAM_URL", "http://localhost:8000"),
        proxy_url=_env("QWEN_PROXY_URL", "http://localhost:9000"),
        model=_env("QWEN_MODEL", "Qwen/Qwen3.5-4B"),
        history_path=history_path,
        timeout=float(_env("QWEN_TIMEOUT", "120")),
        temperature=float(_env("QWEN_TEMPERATURE", "0.7")),
        max_tokens=int(_env("QWEN_MAX_TOKENS", "512")),
        metrics_buffer=int(_env("QWEN_METRICS_BUFFER", "50")),
    )


def proxy_host_port(proxy_url: str) -> tuple[str, int]:
    parsed = urlparse(proxy_url)
    host = parsed.hostname or "127.0.0.1"
    if parsed.port:
        return host, parsed.port
    if parsed.scheme == "https":
        return host, 443
    return host, 80
