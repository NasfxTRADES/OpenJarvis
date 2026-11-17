"""Tests for the free-provider failover chain."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any, Dict, List

import pytest

from openjarvis.core.types import Message, Role
from openjarvis.engine._base import EngineConnectionError
from openjarvis.engine._stubs import InferenceEngine
from openjarvis.engine.failover import (
    FailoverEngine,
    ProviderSlot,
    is_quota_error,
)

MESSAGES = [Message(role=Role.USER, content="hello")]


class FakeEngine(InferenceEngine):
    """Engine that either answers or raises a canned error."""

    engine_id = "fake"

    def __init__(
        self,
        *,
        content: str = "ok",
        error: Exception | None = None,
        tokens: Sequence[str] = ("he", "llo"),
    ) -> None:
        self.content = content
        self.error = error
        self.tokens = list(tokens)
        self.calls = 0

    def generate(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return {
            "content": self.content,
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "model": model,
        }

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        self.calls += 1
        if self.error is not None:
            raise self.error
        for token in self.tokens:
            yield token

    def list_models(self) -> List[str]:
        return []

    def health(self) -> bool:
        return True


def _chain() -> tuple[FailoverEngine, FakeEngine, FakeEngine]:
    first = FakeEngine(
        content="from-groq", error=Exception("Error code: 429 - rate limit exceeded")
    )
    second = FakeEngine(content="from-gemini")
    engine = FailoverEngine(
        [
            ProviderSlot("groq", "Groq", first, "groq/openai/gpt-oss-120b"),
            ProviderSlot("gemini", "Google Gemini", second, "google/gemini-3.8-flash"),
        ]
    )
    return engine, first, second


def test_generate_uses_first_ready_provider() -> None:
    first = FakeEngine(content="first")
    second = FakeEngine(content="second")
    engine = FailoverEngine(
        [
            ProviderSlot("groq", "Groq", first, "groq/m1"),
            ProviderSlot("gemini", "Google Gemini", second, "google/m2"),
        ]
    )
    result = engine.generate(MESSAGES, model="groq/m1")
    assert result["content"] == "first"
    assert result["provider"] == "groq"
    assert second.calls == 0


def test_generate_switches_when_free_tier_is_spent() -> None:
    engine, first, second = _chain()
    result = engine.generate(MESSAGES, model="groq/openai/gpt-oss-120b")
    assert result["content"] == "from-gemini"
    assert result["provider"] == "gemini"
    assert first.calls == 1 and second.calls == 1
    assert result["failover_errors"]

    status = {entry["provider"]: entry for entry in engine.describe()}
    assert status["groq"]["ready"] is False
    assert status["groq"]["cooldown_seconds"] > 0
    assert status["gemini"]["ready"] is True

    # The next request goes straight to the provider that still has room.
    again = engine.generate(MESSAGES, model="groq/openai/gpt-oss-120b")
    assert again["provider"] == "gemini"
    assert first.calls == 1


def test_generate_reports_when_every_provider_fails() -> None:
    engine = FailoverEngine(
        [
            ProviderSlot(
                "groq", "Groq", FakeEngine(error=Exception("429 rate limit")), "groq/m1"
            ),
            ProviderSlot(
                "gemini",
                "Google Gemini",
                FakeEngine(error=Exception("quota")),
                "google/m2",
            ),
        ]
    )
    with pytest.raises(EngineConnectionError, match="Every configured AI provider"):
        engine.generate(MESSAGES, model="")


def test_non_availability_errors_also_fail_over() -> None:
    first = FakeEngine(error=ConnectionError("connection reset"))
    second = FakeEngine(content="second")
    engine = FailoverEngine(
        [
            ProviderSlot("groq", "Groq", first, "groq/m1"),
            ProviderSlot("gemini", "Google Gemini", second, "google/m2"),
        ]
    )
    assert engine.generate(MESSAGES, model="")["content"] == "second"


def test_context_length_errors_are_not_retried_elsewhere() -> None:
    boom = EngineConnectionError("This model's maximum context length is 8192 tokens")
    first = FakeEngine(error=boom)
    second = FakeEngine(content="second")
    engine = FailoverEngine(
        [
            ProviderSlot("groq", "Groq", first, "groq/m1"),
            ProviderSlot("gemini", "Google Gemini", second, "google/m2"),
        ]
    )
    with pytest.raises(EngineConnectionError):
        engine.generate(MESSAGES, model="")
    assert second.calls == 0


@pytest.mark.asyncio
async def test_stream_fails_over_before_any_output() -> None:
    engine, first, second = _chain()
    chunks = [chunk async for chunk in engine.stream(MESSAGES, model="")]
    assert "".join(chunks) == "hello"
    assert first.calls == 1 and second.calls == 1


@pytest.mark.asyncio
async def test_stream_uses_the_healthy_provider_after_switch() -> None:
    engine, _, _ = _chain()
    engine.generate(MESSAGES, model="")  # burns Groq's allowance
    chunks = [chunk async for chunk in engine.stream(MESSAGES, model="")]
    assert "".join(chunks) == "hello"


def test_previous_provider_returns_after_cooldown() -> None:
    first = FakeEngine(error=Exception("429 rate limit"))
    second = FakeEngine(content="second")
    engine = FailoverEngine(
        [
            ProviderSlot("groq", "Groq", first, "groq/m1"),
            ProviderSlot("gemini", "Google Gemini", second, "google/m2"),
        ],
        quota_cooldown_seconds=0,
    )
    engine.generate(MESSAGES, model="")
    first.error = None
    first.content = "back-on-groq"
    assert engine.generate(MESSAGES, model="")["content"] == "back-on-groq"


def test_is_quota_error_matches_vendor_wordings() -> None:
    assert is_quota_error(Exception("Error code: 429 - rate limit reached"))
    assert is_quota_error(Exception("RESOURCE_EXHAUSTED: quota exceeded"))
    assert is_quota_error(Exception("You exceeded your current quota, check your plan"))
    assert not is_quota_error(Exception("Invalid API key provided"))
    assert not is_quota_error(
        EngineConnectionError("maximum context length exceeded for this model")
    )


def test_model_ids_and_providers_dedupe() -> None:
    shared = FakeEngine()
    engine = FailoverEngine(
        [
            ProviderSlot("groq", "Groq", shared, "groq/m1"),
            ProviderSlot("groq", "Groq", shared, "groq/m2"),
            ProviderSlot("gemini", "Google Gemini", shared, "google/m3"),
        ]
    )
    assert engine.model_ids() == ["groq/m1", "groq/m2", "google/m3"]
    assert engine.providers() == ["groq", "gemini"]
    assert engine.primary_model == "groq/m1"
    assert engine.can_serve("anything")
