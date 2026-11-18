"""Fish Audio text-to-speech backend.

Uses the Fish Audio REST API for voice synthesis, including cloned voices that
are addressed by the model ID Fish Audio assigns to them.
Requires a FISH_AUDIO_API_KEY environment variable or explicit ``api_key``.
"""

from __future__ import annotations

import io
import os
import re
import wave
from collections.abc import Callable, Iterator
from typing import Any, List, Optional, TypeVar

import httpx

from openjarvis.core.registry import TTSRegistry
from openjarvis.speech.tts import TTSBackend, TTSResult

_FISH_API_BASE = "https://api.fish.audio"

# ``s2.1-pro-free`` is the free developer tier of the same model. An
# unrecognised ``model`` header silently falls back to the paid ``s2.1-pro``,
# so the free tier is pinned explicitly rather than left to the default.
_DEFAULT_MODEL = "s2.1-pro-free"

# Kokoro-style IDs ("bm_george", "pf_dora"). SpeechConfig.voice_id defaults to
# a Kokoro voice, so it can reach this backend; Fish Audio would not know it.
_KOKORO_VOICE_RE = re.compile(r"^[a-z][fm]_[a-z]+$")

_PCM_SAMPLE_RATE = 24000
_OUTPUT_FORMATS = ("mp3", "wav", "pcm", "opus")
_ResponseT = TypeVar("_ResponseT")


def _normalize_api_key(value: str) -> str:
    key = value.strip()
    if any(not 33 <= ord(char) <= 126 for char in key):
        # Reject unsupported header characters without echoing the credential.
        raise ValueError("Fish Audio API key contains unsupported characters")
    return key


def _request(
    request: Callable[..., httpx.Response],
    path: str,
    api_key: str,
    read_response: Callable[[httpx.Response], _ResponseT],
    extra_headers: Optional[dict[str, str]] = None,
    **kwargs: Any,
) -> _ResponseT:
    """Keep provider request/response details out of errors and their tracebacks."""
    key = _normalize_api_key(api_key)
    if not key:
        raise RuntimeError("FISH_AUDIO_API_KEY not set")
    headers = {"Authorization": f"Bearer {key}", **(extra_headers or {})}
    try:
        response = request(f"{_FISH_API_BASE}{path}", headers=headers, **kwargs)
        response.raise_for_status()
        # Reading audio/JSON belongs inside the same boundary as sending:
        # transport and decode errors can contain credential-bearing details.
        return read_response(response)
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"Fish Audio request failed (HTTP {exc.response.status_code})"
        ) from None
    except Exception as exc:
        # Preserve the error category, never its message, headers, URL, or body.
        raise RuntimeError(
            f"Fish Audio request failed ({type(exc).__name__})"
        ) from None


def _voice_ids(response: httpx.Response) -> List[str]:
    payload = response.json()
    items = payload.get("items", []) if isinstance(payload, dict) else None
    if not isinstance(items, list) or any(
        not isinstance(item, dict) or not isinstance(item.get("_id"), str)
        for item in items
    ):
        raise ValueError("Invalid voice list")
    return [item["_id"] for item in items]


def _pcm_to_wav(pcm: bytes, sample_rate: int = _PCM_SAMPLE_RATE) -> bytes:
    """Wrap raw 16-bit mono PCM in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def _fish_payload(
    text: str,
    voice_id: str,
    output_format: str,
    speed: float,
    sample_rate: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "text": text,
        "format": output_format,
        "normalize": True,
    }
    if output_format == "mp3":
        payload["mp3_bitrate"] = 128
        payload["sample_rate"] = 44100
    elif output_format == "opus":
        payload["sample_rate"] = 48000
    else:
        payload["sample_rate"] = sample_rate
    if voice_id:
        # Without a reference the account's default voice is used.
        payload["reference_id"] = voice_id
    if speed != 1.0:
        # Fish Audio accepts speed in [0.5, 2.0].
        payload["prosody"] = {"speed": max(0.5, min(2.0, speed))}
    return payload


def _fish_synthesize(
    api_key: str,
    text: str,
    voice_id: str = "",
    model: str = _DEFAULT_MODEL,
    output_format: str = "mp3",
    speed: float = 1.0,
    sample_rate: int = _PCM_SAMPLE_RATE,
) -> bytes:
    """Call the Fish Audio TTS API and return raw audio bytes."""
    return _request(
        httpx.post,
        "/v1/tts",
        api_key,
        lambda response: response.content,
        extra_headers={
            "Content-Type": "application/json",
            "model": model or _DEFAULT_MODEL,
        },
        json=_fish_payload(text, voice_id, output_format, speed, sample_rate),
        timeout=120.0,
    )


@TTSRegistry.register("fish_audio")
class FishAudioTTSBackend(TTSBackend):
    """Fish Audio TTS backend — cloned and library voices."""

    backend_id = "fish_audio"

    def __init__(
        self, *, api_key: str = "", model: str = "", voice_id: str = ""
    ) -> None:
        self._api_key = _normalize_api_key(
            api_key
            or os.environ.get("FISH_AUDIO_API_KEY", "")
            or os.environ.get("FISH_API_KEY", "")
        )
        self._model = (
            model
            or os.environ.get("FISH_AUDIO_MODEL", "")
            or os.environ.get("FISH_MODEL", "")
            or _DEFAULT_MODEL
        )
        self._voice_id = (
            voice_id
            or os.environ.get("FISH_AUDIO_VOICE_ID", "")
            or os.environ.get("FISH_VOICE_ID", "")
        )

    def synthesize(
        self,
        text: str,
        *,
        voice_id: str = "",
        speed: float = 1.0,
        output_format: str = "mp3",
    ) -> TTSResult:
        if not self._api_key:
            raise RuntimeError("FISH_AUDIO_API_KEY not set")

        if _KOKORO_VOICE_RE.match(voice_id):
            voice_id = ""
        voice_id = voice_id or self._voice_id
        fmt = output_format if output_format in _OUTPUT_FORMATS else "mp3"
        sample_rate = (
            44100 if fmt == "mp3" else 48000 if fmt == "opus" else _PCM_SAMPLE_RATE
        )

        audio = _fish_synthesize(
            self._api_key,
            text,
            voice_id=voice_id,
            model=self._model,
            output_format="pcm" if fmt == "wav" else fmt,
            speed=speed,
            sample_rate=sample_rate,
        )
        if fmt == "wav":
            # PCM is requested so the container is always well-formed; the API's
            # own WAV stream is fine too but this keeps one code path.
            audio = _pcm_to_wav(audio, sample_rate)

        return TTSResult(
            audio=audio,
            format=fmt,
            voice_id=voice_id,
            sample_rate=sample_rate,
            metadata={"backend": "fish_audio", "model": self._model},
        )

    def stream_synthesize(
        self,
        text: str,
        *,
        voice_id: str = "",
        speed: float = 1.0,
        output_format: str = "mp3",
    ) -> Iterator[bytes]:
        """Yield audio as Fish Audio generates it.

        Waiting for the whole clip is most of the delay between a reply and the
        first spoken word, so the HTTP body is passed straight through.
        """
        if not self._api_key:
            raise RuntimeError("FISH_AUDIO_API_KEY not set")
        if _KOKORO_VOICE_RE.match(voice_id):
            voice_id = ""
        voice_id = voice_id or self._voice_id
        fmt = "mp3" if output_format not in _OUTPUT_FORMATS else output_format
        key = _normalize_api_key(self._api_key)
        try:
            with httpx.stream(
                "POST",
                f"{_FISH_API_BASE}/v1/tts",
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                    "model": self._model or _DEFAULT_MODEL,
                },
                json=_fish_payload(text, voice_id, fmt, speed, 44100),
                timeout=120.0,
            ) as response:
                response.raise_for_status()
                yield from response.iter_bytes()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"Fish Audio request failed (HTTP {exc.response.status_code})"
            ) from None
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Fish Audio request failed ({type(exc).__name__})"
            ) from None

    def available_voices(self) -> List[str]:
        if not self._api_key:
            return []
        return _request(
            httpx.get,
            "/model",
            self._api_key,
            _voice_ids,
            params={"page_size": 100},
            timeout=30.0,
        )

    def health(self) -> bool:
        return bool(self._api_key)
