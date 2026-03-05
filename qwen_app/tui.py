from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import httpx
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Input, RichLog, Static
from textual import on

from qwen_app.config import get_settings
from qwen_app.history import HistoryStore
from qwen_app.sse import iter_sse_messages


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class ChatApp(App):
    CSS = """
    #log {
        height: 1fr;
    }
    #live {
        height: auto;
    }
    #status {
        height: auto;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self.settings = get_settings()
        self.history_store = HistoryStore(self.settings.history_path)
        self.messages: list[dict[str, Any]] = []
        self.client: httpx.AsyncClient | None = None

    def compose(self) -> ComposeResult:
        with Vertical():
            yield RichLog(id="log", highlight=False, markup=True)
            yield Static("", id="live")
            yield Static("", id="status")
            yield Input(placeholder="Type a message and press Enter", id="input")

    async def on_mount(self) -> None:
        self.messages = self.history_store.load()
        self.client = httpx.AsyncClient(
            base_url=self.settings.proxy_url,
            timeout=self.settings.timeout,
        )
        self._render_history()

    async def on_unmount(self) -> None:
        if self.client:
            await self.client.aclose()

    def _render_history(self) -> None:
        log = self.query_one("#log", RichLog)
        log.clear()
        for message in self.messages:
            role = message.get("role", "assistant")
            content = message.get("content", "")
            log.write(self._format_message(role, content))

    def _format_message(self, role: str, content: str) -> str:
        if role == "user":
            return f"[bold cyan]You:[/bold cyan] {content}"
        if role == "assistant":
            return f"[bold green]Assistant:[/bold green] {content}"
        return f"[bold yellow]{role.title()}:[/bold yellow] {content}"

    def _set_status(self, text: str) -> None:
        status = self.query_one("#status", Static)
        status.update(text)

    def _set_live(self, text: str) -> None:
        live = self.query_one("#live", Static)
        live.update(text)

    def _set_input_enabled(self, enabled: bool) -> None:
        input_widget = self.query_one("#input", Input)
        input_widget.disabled = not enabled

    @on(Input.Submitted)
    async def handle_submit(self, event: Input.Submitted) -> None:
        content = event.value.strip()
        event.input.value = ""
        if not content:
            return

        user_message = {"role": "user", "content": content, "timestamp": _timestamp()}
        self.history_store.append(self.messages, user_message)
        log = self.query_one("#log", RichLog)
        log.write(self._format_message("user", content))
        await self._stream_response()

    async def _stream_response(self) -> None:
        if self.client is None:
            return
        self._set_input_enabled(False)
        self._set_live("[bold green]Assistant:[/bold green] ")
        self._set_status("Streaming...")

        payload_messages = [
            {"role": msg.get("role", "user"), "content": msg.get("content", "")}
            for msg in self.messages
        ]
        payload = {
            "model": self.settings.model,
            "messages": payload_messages,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
            "stream": True,
        }

        assistant_text = ""
        reasoning_text = ""
        try:
            async with self.client.stream(
                "POST", "/v1/chat/completions", json=payload
            ) as response:
                response.raise_for_status()
                async for message in iter_sse_messages(response.aiter_bytes()):
                    if message.event == "error":
                        self._set_status(f"Error: {message.data}")
                        continue
                    if message.event == "metrics":
                        self._handle_metrics(message.data)
                        continue
                    if message.data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(message.data)
                    except json.JSONDecodeError:
                        continue
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content") or ""
                    reasoning = (
                        delta.get("reasoning_content") or delta.get("reasoning") or ""
                    )
                    if reasoning:
                        reasoning_text += reasoning
                    if content:
                        assistant_text += content

                    live_text = assistant_text or reasoning_text
                    if live_text:
                        self._set_live(
                            f"[bold green]Assistant:[/bold green] {live_text}"
                        )
        except httpx.HTTPError as exc:
            self._set_status(f"Error: {exc}")
        finally:
            self._set_live("")
            final_text = assistant_text or reasoning_text
            if final_text:
                assistant_message = {
                    "role": "assistant",
                    "content": final_text,
                    "timestamp": _timestamp(),
                }
                self.history_store.append(self.messages, assistant_message)
                log = self.query_one("#log", RichLog)
                log.write(self._format_message("assistant", final_text))
            self._set_input_enabled(True)

    def _handle_metrics(self, data: str) -> None:
        try:
            metrics = json.loads(data)
        except json.JSONDecodeError:
            return
        first_token = metrics.get("first_token_ms")
        total_ms = metrics.get("total_ms")
        completion_tps = metrics.get("completion_tps")
        total_tps = metrics.get("total_tps")
        prompt_tokens = metrics.get("prompt_tokens")
        completion_tokens = metrics.get("completion_tokens")

        parts: list[str] = []
        if isinstance(first_token, (int, float)):
            parts.append(f"first_token_ms={first_token:.0f}")
        if isinstance(total_ms, (int, float)):
            parts.append(f"total_ms={total_ms:.0f}")
        if isinstance(completion_tps, (int, float)):
            parts.append(f"completion_tps={completion_tps:.2f}")
        if isinstance(total_tps, (int, float)):
            parts.append(f"total_tps={total_tps:.2f}")
        if prompt_tokens is not None:
            parts.append(f"prompt={prompt_tokens}")
        if completion_tokens is not None:
            parts.append(f"completion={completion_tokens}")

        self._set_status(" ".join(parts) if parts else "metrics unavailable")


def main() -> None:
    ChatApp().run()


if __name__ == "__main__":
    main()
