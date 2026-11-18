"""The mobile app asks the server to speak replies with the configured voice."""

from __future__ import annotations

from typing import Any, List

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from openjarvis.server.mobile_routes import router
from openjarvis.speech.tts import TTSResult


class _FakeVoice:
    backend_id = "fish_audio"

    def __init__(self) -> None:
        self.calls: List[dict[str, Any]] = []

    def synthesize(
        self,
        text: str,
        *,
        voice_id: str = "",
        speed: float = 1.0,
        output_format: str = "mp3",
    ) -> TTSResult:
        self.calls.append(
            {
                "text": text,
                "voice_id": voice_id,
                "format": output_format,
                "speed": speed,
            }
        )
        return TTSResult(
            audio=b"ID3fake-audio",
            format=output_format,
            voice_id=voice_id,
            sample_rate=44100,
        )

    def health(self) -> bool:
        return True


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(router)  # the router already owns /v1/mobile
    return TestClient(app)


def test_speech_returns_audio_from_the_voice_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    voice = _FakeVoice()
    monkeypatch.setattr(
        "openjarvis.speech._tts_discovery.get_tts_backend",
        lambda preferred="": voice,
    )

    response = _client().post(
        "/v1/mobile/speech", json={"text": "Good evening, operator."}
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    assert response.headers["cache-control"] == "no-store"
    assert response.content == b"ID3fake-audio"
    assert voice.calls == [
        {
            "text": "Good evening, operator.",
            "voice_id": "",
            "format": "mp3",
            "speed": 1.12,
        }
    ]


def test_speech_forwards_a_requested_voice(monkeypatch: pytest.MonkeyPatch) -> None:
    voice = _FakeVoice()
    monkeypatch.setattr(
        "openjarvis.speech._tts_discovery.get_tts_backend",
        lambda preferred="": voice,
    )

    _client().post(
        "/v1/mobile/speech",
        json={"text": "Understood.", "voice_id": "80b49b8aebaa4ada87b19ff5ac94122b"},
    )

    assert voice.calls[0]["voice_id"] == "80b49b8aebaa4ada87b19ff5ac94122b"


def test_speech_clamps_a_requested_speed(monkeypatch: pytest.MonkeyPatch) -> None:
    voice = _FakeVoice()
    monkeypatch.setattr(
        "openjarvis.speech._tts_discovery.get_tts_backend",
        lambda preferred="": voice,
    )

    _client().post("/v1/mobile/speech", json={"text": "Faster.", "speed": 3.0})

    assert voice.calls[0]["speed"] == 2.0


def test_speech_explains_a_missing_voice(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "openjarvis.speech._tts_discovery.get_tts_backend",
        lambda preferred="": None,
    )

    response = _client().post("/v1/mobile/speech", json={"text": "Hello"})

    assert response.status_code == 503
    assert "FISH_AUDIO_API_KEY" in response.json()["detail"]


def test_speech_rejects_blank_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "openjarvis.speech._tts_discovery.get_tts_backend",
        lambda preferred="": _FakeVoice(),
    )

    response = _client().post("/v1/mobile/speech", json={"text": "   "})

    assert response.status_code == 400
