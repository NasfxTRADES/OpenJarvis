"""Gemini ids and request configs must match the model family in use.

The chain names models the way the rest of the codebase does
(``google/gemini-2.5-flash``), but the Google SDK only accepts the bare id and
answers 404 for anything else — which looked like every Gemini model being
down at once.

Gemini 3 also changed how thinking is controlled: the old token budget is
deprecated there and can produce ``400 INVALID_ARGUMENT``, while the sampling
parameters were dropped at the same time.
"""

from __future__ import annotations

import pytest

from openjarvis.engine.cloud import (
    _google_config,
    _google_model_id,
    _google_thinking_config,
    _is_google_model,
)


class _FakeTypes:
    """Stand-in for google.genai.types; only config objects are used."""

    class ThinkingConfig:
        def __init__(
            self, thinking_budget: int | None = None, thinking_level: str | None = None
        ) -> None:
            self.thinking_budget = thinking_budget
            self.thinking_level = thinking_level

    class GenerateContentConfig:
        def __init__(self, **kwargs: object) -> None:
            self.temperature = kwargs.get("temperature")
            self.max_output_tokens = kwargs.get("max_output_tokens")
            self.thinking_config = None


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("google/gemini-2.5-flash", "gemini-2.5-flash"),
        ("google/gemini-2.5-flash-lite", "gemini-2.5-flash-lite"),
        ("models/gemini-2.5-pro", "gemini-2.5-pro"),
        ("gemini-2.0-flash", "gemini-2.0-flash"),
        ("google/models/gemini-3-pro", "gemini-3-pro"),
    ],
)
def test_google_model_id_strips_the_prefix(model: str, expected: str) -> None:
    assert _google_model_id(model) == expected


def test_gemini_2_5_flash_still_uses_a_zero_thinking_budget() -> None:
    # Unset, Gemini spends seconds reasoning before the first spoken word.
    config = _google_thinking_config("google/gemini-2.5-flash", _FakeTypes)
    assert config is not None
    assert config.thinking_budget == 0
    assert config.thinking_level is None

    lite = _google_thinking_config("google/gemini-2.5-flash-lite", _FakeTypes)
    assert lite is not None and lite.thinking_budget == 0


def test_gemini_3_uses_a_level_because_the_budget_is_rejected() -> None:
    config = _google_thinking_config("google/gemini-3.8-flash", _FakeTypes)
    assert config is not None
    assert config.thinking_level == "low"
    # Sending both fields is an error, so the deprecated one stays unset.
    assert config.thinking_budget is None


def test_gemini_3_config_drops_the_deprecated_sampling_parameter() -> None:
    config = _google_config(
        "google/gemini-3.8-flash", _FakeTypes, temperature=0.65, max_tokens=320
    )
    assert config.temperature is None
    assert config.max_output_tokens == 320
    assert config.thinking_config.thinking_level == "low"

    older = _google_config(
        "google/gemini-2.5-flash", _FakeTypes, temperature=0.65, max_tokens=320
    )
    assert older.temperature == 0.65
    assert older.thinking_config.thinking_budget == 0


def test_thinking_config_is_left_alone_where_unsupported() -> None:
    for model in ("google/gemini-2.0-flash", "google/gemini-2.5-pro"):
        assert _google_thinking_config(model, _FakeTypes) is None

    class _OldTypes:
        pass

    assert _google_thinking_config("google/gemini-2.5-flash", _OldTypes) is None


def test_google_models_still_route_to_the_google_client() -> None:
    # The routing predicate has to keep accepting the prefixed form the chain
    # uses, even though the SDK call is made with the stripped id.
    assert _is_google_model("google/gemini-2.5-flash") is True
    assert _is_google_model("gemini-2.5-pro") is True
    assert _is_google_model("openrouter/google/gemini-2.5-pro") is False
