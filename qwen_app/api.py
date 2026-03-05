from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from qwen_app.config import get_settings, proxy_host_port
from qwen_app.metrics import MetricsBuffer, TokenCounter, build_metrics
from qwen_app.sse import format_sse, iter_sse_messages


def _sanitize_messages(messages: Any) -> list[dict[str, str]]:
    if not isinstance(messages, list):
        return []
    sanitized: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        sanitized.append(
            {
                "role": str(message.get("role", "user")),
                "content": str(message.get("content", "")),
            }
        )
    return sanitized


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.token_counter = TokenCounter(settings.model)
    app.state.metrics_buffer = MetricsBuffer(settings.metrics_buffer)
    app.state.client = httpx.AsyncClient(
        base_url=settings.upstream_url,
        timeout=settings.timeout,
    )
    yield
    await app.state.client.aclose()


def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan, title="Qwen Proxy")

    @app.get("/health")
    async def health() -> dict[str, str]:
        client: httpx.AsyncClient = app.state.client
        try:
            response = await client.get("/v1/models")
            response.raise_for_status()
        except httpx.HTTPError:
            return {"status": "down"}
        return {"status": "ok"}

    @app.get("/diag/last")
    async def diag_last() -> list[dict[str, Any]]:
        buffer: MetricsBuffer = app.state.metrics_buffer
        return buffer.list()

    @app.get("/v1/models")
    async def models() -> Response:
        client: httpx.AsyncClient = app.state.client
        response = await client.get("/v1/models")
        return Response(
            content=response.content,
            status_code=response.status_code,
            media_type=response.headers.get("content-type", "application/json"),
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        settings = app.state.settings
        client: httpx.AsyncClient = app.state.client
        token_counter: TokenCounter = app.state.token_counter
        metrics_buffer: MetricsBuffer = app.state.metrics_buffer

        payload = await request.json()
        stream = bool(payload.get("stream"))
        messages = _sanitize_messages(payload.get("messages"))
        payload["messages"] = messages
        request_id = str(uuid.uuid4())
        start_time = time.monotonic()

        if stream:
            payload["stream"] = True
            request_obj = client.build_request(
                "POST", "/v1/chat/completions", json=payload
            )

            async def event_generator():
                completion_text = ""
                reasoning_text = ""
                combined_text = ""
                first_token_time: float | None = None
                response: httpx.Response | None = None
                try:
                    response = await client.send(request_obj, stream=True)
                    if response.status_code >= 400:
                        content = await response.aread()
                        error_payload = json.dumps(
                            {
                                "status": response.status_code,
                                "body": content.decode("utf-8", errors="replace"),
                            }
                        )
                        yield format_sse(error_payload, event="error")
                    else:
                        async for message in iter_sse_messages(response.aiter_bytes()):
                            if message.data == "[DONE]":
                                break
                            if message.data:
                                try:
                                    chunk = json.loads(message.data)
                                    delta = chunk.get("choices", [{}])[0].get(
                                        "delta", {}
                                    )
                                    content = delta.get("content") or ""
                                    reasoning = (
                                        delta.get("reasoning_content")
                                        or delta.get("reasoning")
                                        or ""
                                    )
                                    if content or reasoning:
                                        if first_token_time is None:
                                            first_token_time = time.monotonic()
                                        completion_text += content
                                        reasoning_text += reasoning
                                        combined_text += content + reasoning
                                except json.JSONDecodeError:
                                    pass
                            yield format_sse(message.data, event=message.event)

                        prompt_tokens = await token_counter.count_messages(messages)
                        completion_tokens = await token_counter.count_text(
                            combined_text or completion_text or reasoning_text
                        )
                        end_time = time.monotonic()
                        metrics = build_metrics(
                            request_id=request_id,
                            model=payload.get("model", settings.model),
                            start_time=start_time,
                            end_time=end_time,
                            first_token_time=first_token_time,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                        )
                        metrics_buffer.add(metrics)
                        yield format_sse(json.dumps(metrics.to_dict()), event="metrics")
                except Exception as exc:
                    error_payload = json.dumps(
                        {"error": f"{type(exc).__name__}: {exc}"}
                    )
                    yield format_sse(error_payload, event="error")
                finally:
                    if response is not None:
                        await response.aclose()
                    yield format_sse("[DONE]")

            headers = {"X-Request-Id": request_id}
            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers=headers,
            )

        response = await client.post("/v1/chat/completions", json=payload)
        if response.status_code >= 400:
            return Response(
                content=response.content,
                status_code=response.status_code,
                media_type=response.headers.get("content-type", "application/json"),
            )

        data = response.json()
        end_time = time.monotonic()
        usage = data.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")

        if prompt_tokens is None:
            prompt_tokens = await token_counter.count_messages(messages)
        if completion_tokens is None:
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            completion_tokens = await token_counter.count_text(content)

        metrics = build_metrics(
            request_id=request_id,
            model=payload.get("model", settings.model),
            start_time=start_time,
            end_time=end_time,
            first_token_time=start_time,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        metrics_buffer.add(metrics)

        headers = {
            "X-Request-Id": request_id,
            "X-Metrics": json.dumps(metrics.to_dict()),
        }
        return JSONResponse(content=data, headers=headers)

    return app


app = create_app()


def main() -> None:
    import uvicorn

    settings = get_settings()
    host, port = proxy_host_port(settings.proxy_url)
    uvicorn.run("qwen_app.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
