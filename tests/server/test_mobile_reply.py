"""Voice replies stream from the engine instead of waiting for the agent."""

from __future__ import annotations

import types
from collections.abc import AsyncIterator, Sequence
from typing import Any, Dict, List

from fastapi import FastAPI
from fastapi.testclient import TestClient

from openjarvis.core.types import Message, Role
from openjarvis.server.mobile_routes import router


class _FakeStreamingEngine:
    def __init__(self) -> None:
        self.seen: List[Dict[str, Any]] = []

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        self.seen.append(
            {
                "messages": list(messages),
                "model": model,
                "max_tokens": max_tokens,
            }
        )
        for token in ("Good ", "evening, ", "sir."):
            yield token


def _client(engine: Any, model: str = "google/gemini-2.5-flash") -> TestClient:
    app = FastAPI()
    app.state.engine = engine
    app.state.model = model
    app.state.mobile_runtime = types.SimpleNamespace(ensure_started=lambda: None)
    app.include_router(router)  # the router already owns /v1/mobile
    return TestClient(app)


def test_reply_streams_tokens_without_waiting_for_the_agent() -> None:
    engine = _FakeStreamingEngine()
    client = _client(engine)

    with client.stream(
        "POST",
        "/v1/mobile/reply",
        json={"messages": [{"role": "user", "content": "status report"}]},
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    # Every token arrives as its own frame, plus the role frame and [DONE].
    assert body.count("data: ") == 5
    assert "Good " in body and "evening, " in body and "[DONE]" in body


def test_reply_sends_the_spoken_persona_and_a_short_token_budget() -> None:
    engine = _FakeStreamingEngine()

    with _client(engine).stream(
        "POST",
        "/v1/mobile/reply",
        json={"messages": [{"role": "user", "content": "hello"}]},
    ) as response:
        "".join(response.iter_text())

    sent = engine.seen[0]
    assert sent["messages"][0].role == Role.SYSTEM
    assert "read aloud" in sent["messages"][0].text
    assert sent["messages"][-1].text == "hello"
    assert sent["messages"][-1].role == Role.USER
    assert sent["max_tokens"] == 320


def test_reply_reports_a_missing_model() -> None:
    response = _client(None, model="").post(
        "/v1/mobile/reply", json={"messages": [{"role": "user", "content": "hi"}]}
    )

    assert response.status_code == 503


def test_reply_caps_the_requested_token_budget() -> None:
    engine = _FakeStreamingEngine()

    with _client(engine).stream(
        "POST",
        "/v1/mobile/reply",
        json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 5000},
    ) as response:
        "".join(response.iter_text())

    assert engine.seen[0]["max_tokens"] == 700
