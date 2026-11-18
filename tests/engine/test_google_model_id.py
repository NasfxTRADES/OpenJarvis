"""Gemini ids must reach the Google client without their provider prefix.

The chain names models the way the rest of the codebase does
(``google/gemini-2.5-flash``), but the Google SDK only accepts the bare id and
answers 404 for anything else — which looked like every Gemini model being
down at once.
"""

from __future__ import annotations

import pytest

from openjarvis.engine.cloud import _google_model_id, _is_google_model


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


def test_google_models_still_route_to_the_google_client() -> None:
    # The routing predicate has to keep accepting the prefixed form the chain
    # uses, even though the SDK call is made with the stripped id.
    assert _is_google_model("google/gemini-2.5-flash") is True
    assert _is_google_model("gemini-2.5-pro") is True
    assert _is_google_model("openrouter/google/gemini-2.5-pro") is False
