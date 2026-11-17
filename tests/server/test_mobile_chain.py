"""The mobile runtime adopts the free-provider chain when no provider is set."""

from __future__ import annotations

import types
from collections.abc import AsyncIterator, Sequence
from typing import Any, Dict, List

import pytest

from openjarvis.core.types import Message
from openjarvis.engine._stubs import InferenceEngine
from openjarvis.engine.failover import FailoverEngine, ProviderSlot


class _EchoEngine(InferenceEngine):
    engine_id = "echo"

    def generate(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return {"content": "hi", "usage": {}}

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        yield "hi"

    def list_models(self) -> List[str]:
        return []

    def health(self) -> bool:
        return True


def _runtime(pytest_config: pytest.MonkeyPatch, home: str) -> Any:
    pytest_config.setenv("OPENJARVIS_HOME", home)
    from openjarvis.server.mobile_routes import MobileRuntime

    app = types.SimpleNamespace(
        state=types.SimpleNamespace(
            engine=None, model="", engine_name="", bus=None, agent=None
        )
    )
    return MobileRuntime(app), app


def _chain() -> FailoverEngine:
    return FailoverEngine(
        [
            ProviderSlot("groq", "Groq", _EchoEngine(), "groq/openai/gpt-oss-120b"),
            ProviderSlot(
                "gemini", "Google Gemini", _EchoEngine(), "google/gemini-3.8-flash"
            ),
        ]
    )


def test_status_reports_the_free_chain(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, app = _runtime(monkeypatch, str(tmp_path))
    try:
        runtime._activate_chain(_chain())
        status = runtime.status()
        assert status["provider"] == "auto"
        assert status["configured"] is True
        assert status["active_provider"] == "Groq"
        assert [entry["provider"] for entry in status["chain"]] == ["groq", "gemini"]
        assert app.state.model == "groq/openai/gpt-oss-120b"
        assert app.state.engine_name == "auto"
    finally:
        runtime.stop()


def test_saved_single_provider_is_not_overridden(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, app = _runtime(monkeypatch, str(tmp_path))
    try:
        from openjarvis.server.mobile_routes import _save_settings

        _save_settings({"provider": "gemini", "model": "google/gemini-3.8-flash"})
        runtime._activate_chain(_chain())
        assert app.state.engine is None
        _save_settings({})
    finally:
        runtime.stop()
