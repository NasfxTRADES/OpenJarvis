"""Provider streams must never park the server's event loop.

The Google, OpenAI and Anthropic SDKs all stream synchronously. Draining one of
those iterators inside an async generator stops ``/health`` from answering for
as long as the provider holds the socket, so Render marks the instance unhealthy
and restarts it — which is the 502 the phone saw mid-conversation.

These tests pin the two paths: Gemini's native async client is preferred, and a
sync-only client is pulled chunk by chunk in a worker thread so the loop keeps
running.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import Any

from openjarvis.engine.cloud import _google_chunks, _iterate_off_loop


def _chunk(text: str) -> Any:
    return SimpleNamespace(text=text)


async def _collect(stream: Any) -> list[str]:
    return [chunk.text async for chunk in stream]


class _AsyncModels:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate_content_stream(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)

        async def stream() -> Any:
            for text in ("Good evening", ", sir."):
                yield _chunk(text)

        return stream()


class _AsyncClient:
    def __init__(self) -> None:
        # The sync surface is left unset: using it would fail the test.
        self.models = None
        self.aio = SimpleNamespace(models=_AsyncModels())


class _SyncModels:
    def __init__(self, texts: list[str]) -> None:
        self.texts = texts
        self.calls: list[dict[str, Any]] = []

    def generate_content_stream(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        for text in self.texts:
            yield _chunk(text)


class _SyncClient:
    def __init__(self, texts: list[str]) -> None:
        self.models = _SyncModels(texts)


class _SlowStream:
    """A blocking stream that takes real time, like a provider socket."""

    def __init__(self, count: int, delay: float = 0.05) -> None:
        self.remaining = count
        self.delay = delay

    def __iter__(self) -> _SlowStream:
        return self

    def __next__(self) -> int:
        if self.remaining <= 0:
            raise StopIteration
        self.remaining -= 1
        time.sleep(self.delay)
        return self.remaining


def test_google_stream_uses_the_native_async_client() -> None:
    client = _AsyncClient()

    texts = asyncio.run(
        _collect(
            _google_chunks(
                client,
                model="gemini-3.8-flash",
                contents=[],
                config=None,
                aio_models=client.aio.models,
            )
        )
    )

    assert texts == ["Good evening", ", sir."]
    assert client.aio.models.calls[0]["model"] == "gemini-3.8-flash"


def test_google_stream_falls_back_to_a_sync_client() -> None:
    client = _SyncClient(["Hello", " there"])

    texts = asyncio.run(
        _collect(
            _google_chunks(
                client, model="gemini-2.5-flash", contents=[], config=None
            )
        )
    )

    assert texts == ["Hello", " there"]
    assert client.models.calls[0]["model"] == "gemini-2.5-flash"


def test_a_blocking_stream_leaves_the_event_loop_running() -> None:
    async def scenario() -> int:
        ticks = 0

        async def ticker() -> None:
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        task = asyncio.create_task(ticker())
        try:
            async for _ in _iterate_off_loop(_SlowStream(3)):
                pass
        finally:
            task.cancel()
        return ticks

    # A blocking ``for`` would never let the ticker run, so this would be 0.
    assert asyncio.run(scenario()) >= 3
