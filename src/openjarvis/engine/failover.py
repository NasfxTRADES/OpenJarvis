"""Free-provider failover.

One engine in front of every provider that is configured: when the provider in
use runs out of its free allowance (HTTP 429 / quota errors) or is otherwise
unavailable, the next one answers instead. Nothing is retried against the same
provider until its cooldown expires, so the assistant keeps replying as long as
*one* configured provider still has room.

The chain is built from whichever keys are present — ``GROQ_API_KEY``,
``CEREBRAS_API_KEY``, ``GEMINI_API_KEY``, ``MISTRAL_API_KEY``,
``OPENROUTER_API_KEY``, ``XAI_API_KEY``, ``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``
— and can be reordered with ``OPENJARVIS_FREE_CHAIN="groq,gemini,openrouter"``.
Free tiers are ordered first so a paid key is only spent once every free
allowance is gone.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from openjarvis.core.types import Message
from openjarvis.engine._base import (
    EngineConnectionError,
    looks_like_context_length_error,
)
from openjarvis.engine._stubs import InferenceEngine, StreamChunk

logger = logging.getLogger(__name__)

__all__ = [
    "FREE_PROVIDERS",
    "FailoverEngine",
    "ProviderSlot",
    "build_free_failover_engine",
    "is_quota_error",
]

# Ordered best-first. Every entry here has a free tier except ``openai`` and
# ``claude``, which are last so they are only used once the free ones are spent.
FREE_PROVIDERS = (
    "groq",
    "cerebras",
    "gemini",
    "mistral",
    "openrouter",
    "grok",
    "openai",
    "claude",
)

# Preferred models per provider, best first. These are only the starting point:
# where a provider publishes its catalogue the live list wins, because free-tier
# model IDs are retired regularly (Groq's Llama models, for one).
PREFERRED_MODELS: Dict[str, tuple[str, ...]] = {
    "groq": (
        "groq/openai/gpt-oss-120b",
        "groq/openai/gpt-oss-20b",
        "groq/qwen/qwen3.8-27b",
    ),
    "cerebras": (
        "cerebras/llama3.1-8b",
        "cerebras/gpt-oss-120b",
        "cerebras/qwen-3-235b-a22b-instruct-2507",
    ),
    "gemini": (
        "google/gemini-3.8-flash",
        "google/gemini-3.5-flash-lite",
        "google/gemini-2.5-flash",
    ),
    "mistral": (
        "mistral/mistral-small-latest",
        "mistral/open-mistral-nemo",
        "mistral/mistral-medium-latest",
    ),
    "openrouter": (
        "openrouter/openrouter/free",
        "openrouter/google/gemma-4-31b-it:free",
        "openrouter/inclusionai/ling-3.2-flash:free",
    ),
    "grok": ("grok/grok-4-fast", "grok/grok-4", "grok/grok-3-mini"),
    "openai": ("gpt-4o-mini", "gpt-4o"),
    "claude": ("claude-haiku-4-5", "claude-sonnet-4-6"),
}

PROVIDER_LABELS: Dict[str, str] = {
    "groq": "Groq",
    "cerebras": "Cerebras",
    "gemini": "Google Gemini",
    "mistral": "Mistral",
    "openrouter": "OpenRouter",
    "grok": "Grok (xAI)",
    "openai": "ChatGPT (OpenAI)",
    "claude": "Claude (Anthropic)",
}

# Environment variables that can hold each provider's key, first match wins.
PROVIDER_KEYS: Dict[str, tuple[str, ...]] = {
    "groq": ("GROQ_API_KEY",),
    "cerebras": ("CEREBRAS_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "mistral": ("MISTRAL_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
    "grok": ("XAI_API_KEY", "GROK_API_KEY"),
    "openai": ("OPENAI_API_KEY",),
    "claude": ("ANTHROPIC_API_KEY",),
}

_MODEL_CATALOG_URLS: Dict[str, str] = {
    "groq": "https://api.groq.com/openai/v1/models",
    "cerebras": "https://api.cerebras.ai/v1/models",
    "mistral": "https://api.mistral.ai/v1/models",
    "openrouter": "https://openrouter.ai/api/v1/models",
    "grok": "https://api.x.ai/v1/models",
}

# Generous quota/rate-limit markers. Matched case-insensitively against the
# provider's error text because every vendor words this differently.
_QUOTA_MARKERS = (
    "rate limit",
    "rate_limit",
    "ratelimit",
    "too many requests",
    "quota",
    "resource_exhausted",
    "resource has been exhausted",
    "insufficient_quota",
    "insufficient credit",
    "out of credit",
    "billing",
    "402",
    "429",
    "daily limit",
    "tokens per day",
    "requests per day",
)

def _error_text(exc: BaseException) -> str:
    parts = [str(exc)]
    for attr in ("status_code", "code", "message"):
        value = getattr(exc, attr, None)
        if value is not None:
            parts.append(str(value))
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        parts.append(str(body.get("message") or body))
    response = getattr(exc, "response", None)
    if response is not None:
        parts.append(str(getattr(response, "status_code", "")))
        try:
            parts.append(str(response.text))
        except Exception:  # pragma: no cover - response may not expose text
            pass
    return " ".join(parts).lower()


def is_quota_error(exc: BaseException) -> bool:
    """True when *exc* says the account is out of allowance, not broken.

    A context-window overflow also contains "exceeded" in most vendors'
    wording, so it is explicitly excluded: routing it to another provider
    would just repeat the same failure.
    """
    text = _error_text(exc)
    if looks_like_context_length_error(text):
        return False
    return any(marker in text for marker in _QUOTA_MARKERS)


@dataclass
class ProviderSlot:
    """One model on one provider, in failover order."""

    provider: str
    label: str
    engine: InferenceEngine
    model: str

    def to_dict(self) -> Dict[str, Any]:
        return {"provider": self.provider, "label": self.label, "model": self.model}


@dataclass
class _SlotState:
    cooldown_until: float = 0.0
    last_error: str = ""
    failures: int = 0


class FailoverEngine(InferenceEngine):
    """Serves requests from the first provider with capacity left."""

    engine_id = "failover"
    is_cloud = True

    def __init__(
        self,
        slots: Sequence[ProviderSlot],
        *,
        cooldown_seconds: float = 60.0,
        quota_cooldown_seconds: float = 3600.0,
    ) -> None:
        if not slots:
            raise ValueError("FailoverEngine needs at least one provider")
        self._slots: List[ProviderSlot] = list(slots)
        self._states: Dict[int, _SlotState] = {
            index: _SlotState() for index in range(len(slots))
        }
        self._lock = threading.Lock()
        self.cooldown_seconds = cooldown_seconds
        self.quota_cooldown_seconds = quota_cooldown_seconds
        self.switches = 0
        self.last_provider = ""

    # -- Introspection -------------------------------------------------------

    @property
    def slots(self) -> List[ProviderSlot]:
        return list(self._slots)

    @property
    def primary_model(self) -> str:
        return self._slots[0].model

    def model_ids(self) -> List[str]:
        seen: Dict[str, None] = {}
        for slot in self._slots:
            seen.setdefault(slot.model, None)
        return list(seen)

    def providers(self) -> List[str]:
        seen: Dict[str, None] = {}
        for slot in self._slots:
            seen.setdefault(slot.provider, None)
        return list(seen)

    def describe(self) -> List[Dict[str, Any]]:
        """Per-slot status for the phone app: what is ready, what is cooling."""
        now = time.time()
        out: List[Dict[str, Any]] = []
        for index, slot in enumerate(self._slots):
            state = self._states.get(index) or _SlotState()
            remaining = max(0, int(round(state.cooldown_until - now)))
            out.append(
                {
                    **slot.to_dict(),
                    "ready": remaining == 0,
                    "cooldown_seconds": remaining,
                    "last_error": state.last_error,
                }
            )
        return out

    def active_label(self) -> str:
        for index, slot in enumerate(self._slots):
            if self._states[index].cooldown_until <= time.time():
                return slot.label
        return self._slots[0].label

    # -- Cooldowns -----------------------------------------------------------

    def _ready(self, index: int) -> bool:
        return self._states[index].cooldown_until <= time.time()

    def _order(self, requested_model: str) -> List[int]:
        """Slot indexes to try: ready ones first, requested model on top.

        Cooling-down slots come last rather than being dropped, so a request
        still reaches a provider whose cooldown has not expired yet instead of
        failing outright ("never run out" beats "never repeat").
        """
        now = time.time()
        ready = [
            index
            for index in range(len(self._slots))
            if self._states[index].cooldown_until <= now
        ]
        if requested_model:
            exact = [
                index
                for index in ready
                if self._slots[index].model == requested_model
            ]
            ready = exact + [index for index in ready if index not in exact]
        cooling = [index for index in range(len(self._slots)) if index not in ready]
        cooling.sort(key=lambda index: self._states[index].cooldown_until)
        return ready + cooling

    def _record_failure(self, index: int, exc: BaseException) -> None:
        state = self._states[index]
        quota = is_quota_error(exc)
        delay = self.quota_cooldown_seconds if quota else self.cooldown_seconds
        state.cooldown_until = time.time() + delay
        state.last_error = str(exc)[:400]
        state.failures += 1
        if quota:
            logger.warning(
                "%s (%s) is out of allowance; switching to the next free provider "
                "for %d minutes",
                self._slots[index].label,
                self._slots[index].model,
                int(delay // 60),
            )
        else:
            logger.warning(
                "%s (%s) failed: %s — trying the next provider",
                self._slots[index].label,
                self._slots[index].model,
                str(exc)[:200],
            )

    def _record_success(self, index: int) -> None:
        state = self._states[index]
        state.cooldown_until = 0.0
        state.last_error = ""
        state.failures = 0
        if self.last_provider and self.last_provider != self._slots[index].label:
            self.switches += 1
        self.last_provider = self._slots[index].label

    @staticmethod
    def _should_fail_over(exc: BaseException) -> bool:
        # A too-long conversation fails the same way everywhere; report it.
        return not looks_like_context_length_error(str(exc))

    def _reset_if_all_cooling(self) -> None:
        """Give a fresh chance to every slot when the whole chain is blocked.

        Free tiers reset daily; without this, a long-running server would keep
        returning "all providers busy" hours after the allowance came back.
        """
        now = time.time()
        with self._lock:
            if all(state.cooldown_until > now for state in self._states.values()):
                for state in self._states.values():
                    state.cooldown_until = 0.0

    # -- Inference -----------------------------------------------------------

    def generate(
        self,
        messages: Sequence[Message],
        *,
        model: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        errors: List[str] = []
        order = self._order(model)
        for index in order:
            slot = self._slots[index]
            try:
                result = slot.engine.generate(
                    messages,
                    model=slot.model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
            except Exception as exc:  # noqa: BLE001 - any provider error fails over
                if not self._should_fail_over(exc):
                    raise
                self._record_failure(index, exc)
                errors.append(f"{slot.label}: {exc}")
                continue
            if errors:
                logger.info(
                    "Answered with %s after %d failed provider(s)",
                    slot.label,
                    len(errors),
                )
            self._record_success(index)
            out = dict(result)
            out["provider"] = slot.provider
            out["model"] = slot.model
            if errors:
                out["failover_errors"] = errors
            return out
        self._reset_if_all_cooling()
        raise EngineConnectionError(
            "Every configured AI provider refused the request: " + "; ".join(errors)
        )

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        model: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        errors: List[str] = []
        for index in self._order(model):
            slot = self._slots[index]
            agen = slot.engine.stream(
                messages,
                model=slot.model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            try:
                first = await anext(agen)
            except StopAsyncIteration:
                self._record_success(index)
                return
            except Exception as exc:  # noqa: BLE001 - fail over before any output
                if not self._should_fail_over(exc):
                    raise
                self._record_failure(index, exc)
                errors.append(f"{slot.label}: {exc}")
                continue
            self._record_success(index)
            if errors:
                logger.info(
                    "Streaming from %s after %d failed provider(s)",
                    slot.label,
                    len(errors),
                )
            yield first
            try:
                async for token in agen:
                    yield token
            except Exception as exc:  # noqa: BLE001 - output already started
                logger.warning("%s failed mid-stream: %s", slot.label, exc)
                self._record_failure(index, exc)
                yield f"\n\n[{slot.label} stopped early — {exc}]"
            return
        self._reset_if_all_cooling()
        raise EngineConnectionError(
            "Every configured AI provider refused the request: " + "; ".join(errors)
        )

    async def stream_full(
        self,
        messages: Sequence[Message],
        *,
        model: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        errors: List[str] = []
        for index in self._order(model):
            slot = self._slots[index]
            agen = slot.engine.stream_full(
                messages,
                model=slot.model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            try:
                first = await anext(agen)
            except StopAsyncIteration:
                self._record_success(index)
                return
            except Exception as exc:  # noqa: BLE001 - fail over before any output
                if not self._should_fail_over(exc):
                    raise
                self._record_failure(index, exc)
                errors.append(f"{slot.label}: {exc}")
                continue
            self._record_success(index)
            yield first
            try:
                async for chunk in agen:
                    yield chunk
            except Exception as exc:  # noqa: BLE001 - output already started
                logger.warning("%s failed mid-stream: %s", slot.label, exc)
                self._record_failure(index, exc)
                yield StreamChunk(content=f"\n\n[{slot.label} stopped early — {exc}]")
            return
        self._reset_if_all_cooling()
        raise EngineConnectionError(
            "Every configured AI provider refused the request: " + "; ".join(errors)
        )

    # -- InferenceEngine surface --------------------------------------------

    def list_models(self) -> List[str]:
        return self.model_ids()

    def health(self) -> bool:
        return bool(self._slots)

    def can_serve(self, model: str) -> bool:
        # The chain ignores the requested ID and uses each slot's own model.
        return bool(self._slots)

    def close(self) -> None:
        engines = {id(slot.engine): slot.engine for slot in self._slots}
        for engine in engines.values():
            close = getattr(engine, "close", None)
            if callable(close):
                close()


# ---------------------------------------------------------------------------
# Building the chain
# ---------------------------------------------------------------------------


def _env_key(provider: str) -> str:
    for name in PROVIDER_KEYS.get(provider, ()):
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    return ""


def _chain_order() -> List[str]:
    raw = (os.environ.get("OPENJARVIS_FREE_CHAIN") or "").strip()
    if not raw:
        return list(FREE_PROVIDERS)
    wanted = [part.strip().lower() for part in raw.split(",") if part.strip()]
    unknown = [name for name in wanted if name not in FREE_PROVIDERS]
    if unknown:
        logger.warning(
            "Ignoring unknown providers in OPENJARVIS_FREE_CHAIN: %s", unknown
        )
    order = [name for name in wanted if name in FREE_PROVIDERS]
    # Keep providers the user did not name, after the ones they did.
    order.extend(name for name in FREE_PROVIDERS if name not in order)
    return order


def _http_model_ids(url: str, key: str, provider: str) -> List[str]:
    """Fetch a provider's model catalogue; [] when it cannot be read."""
    import httpx

    headers = {"Authorization": f"Bearer {key}"}
    try:
        resp = httpx.get(url, headers=headers, timeout=10.0)
    except Exception as exc:  # noqa: BLE001 - catalogue is optional
        logger.debug("%s model list unavailable: %s", provider, exc)
        return []
    if resp.status_code >= 400:
        logger.debug("%s model list returned HTTP %s", provider, resp.status_code)
        return []
    try:
        payload = resp.json()
    except Exception:  # noqa: BLE001 - non-JSON body
        return []
    entries = payload.get("data") or payload.get("models") or []
    ids: List[str] = []
    for entry in entries:
        if isinstance(entry, str):
            ids.append(entry)
            continue
        model_id = str(entry.get("id") or entry.get("name") or "")
        model_id = model_id.removeprefix("models/")
        if model_id:
            ids.append(model_id)
    return ids


def _order_by_preference(found: Sequence[str], preferred: Sequence[str]) -> List[str]:
    """Put preferred models first, then anything else the provider offers."""
    ordered = [m for m in preferred if m in found]
    ordered.extend(m for m in found if m not in ordered and m not in preferred)
    return ordered


def _gemini_models(key: str) -> List[str]:
    """List Gemini chat models for *key*, free Flash models first."""
    import httpx

    from openjarvis.core.credentials import gemini_api_base

    preferred = list(PREFERRED_MODELS["gemini"])
    try:
        resp = httpx.get(
            f"{gemini_api_base()}/v1beta/models",
            headers={"x-goog-api-key": key},
            timeout=10.0,
        )
    except Exception as exc:  # noqa: BLE001 - catalogue is optional
        logger.debug("Gemini model list unavailable: %s", exc)
        return preferred
    if resp.status_code >= 400:
        logger.debug("Gemini model list returned HTTP %s", resp.status_code)
        return preferred
    names: List[str] = []
    for entry in resp.json().get("models", []):
        name = str(entry.get("name", "")).removeprefix("models/")
        if not name or "gemini" not in name:
            continue
        methods = entry.get("supportedGenerationMethods") or []
        if methods and "generateContent" not in methods:
            continue
        if any(bad in name for bad in ("embedding", "image", "tts", "audio")):
            continue
        names.append(f"google/{name}")
    if not names:
        return preferred
    return _order_by_preference(names, preferred)


def _openrouter_free_models(key: str) -> List[str]:
    """Free (zero-priced) text models on OpenRouter, newest first."""
    import httpx

    from openjarvis.core.credentials import openrouter_api_base

    try:
        resp = httpx.get(f"{openrouter_api_base()}/models", timeout=10.0)
    except Exception as exc:  # noqa: BLE001 - catalogue is optional
        logger.debug("OpenRouter model list unavailable: %s", exc)
        return list(PREFERRED_MODELS["openrouter"])
    if resp.status_code >= 400:
        return list(PREFERRED_MODELS["openrouter"])

    def _zero(value: Any) -> bool:
        try:
            return float(value) == 0.0
        except (TypeError, ValueError):
            return False

    free: List[tuple[int, str]] = []
    for entry in resp.json().get("data", []):
        pricing = entry.get("pricing") or {}
        if not (_zero(pricing.get("prompt")) and _zero(pricing.get("completion"))):
            continue
        architecture = entry.get("architecture") or {}
        if "text" not in (architecture.get("output_modalities") or ["text"]):
            continue
        if "text" not in (architecture.get("input_modalities") or ["text"]):
            continue
        if {"audio", "image"} & set(architecture.get("output_modalities") or []):
            continue
        model_id = str(entry.get("id", ""))
        if model_id:
            created = entry.get("created")
            free.append((int(created) if isinstance(created, int) else 0, model_id))
    free.sort(key=lambda item: item[0], reverse=True)
    free.sort(key=lambda item: 0 if item[1] == "openrouter/free" else 1)
    models = [f"openrouter/{model_id}" for _, model_id in free]
    if not any(model.endswith("openrouter/free") for model in models):
        models.insert(0, "openrouter/openrouter/free")
    return models or list(PREFERRED_MODELS["openrouter"])


def discover_models(provider: str, key: str) -> List[str]:
    """Models to try for *provider*, falling back to the preferred list."""
    preferred = list(PREFERRED_MODELS.get(provider, ()))
    try:
        if provider == "gemini":
            return _gemini_models(key)
        if provider == "openrouter":
            return _openrouter_free_models(key)
        url = _MODEL_CATALOG_URLS.get(provider)
        if not url:
            return preferred
        found = _http_model_ids(url, key, provider)
        if not found:
            return preferred
        ordered = _order_by_preference(found, preferred)
        # Keep catalogue noise (audio/vision/embedding models) out of the chain.
        return [
            model
            for model in ordered
            if not any(
                bad in model.lower()
                for bad in ("whisper", "tts", "embed", "guard", "moderation", "audio")
            )
        ] or preferred
    except Exception as exc:  # noqa: BLE001 - discovery must never break startup
        logger.debug("Model discovery failed for %s: %s", provider, exc)
        return preferred


def build_free_failover_engine(
    *,
    providers: Sequence[str] | None = None,
    max_models_per_provider: int = 3,
    discover: Any = None,
) -> Optional[FailoverEngine]:
    """Build a failover engine from every provider key this machine has.

    Returns ``None`` when no provider key is configured, so callers can fall
    back to telling the user how to connect one.
    """
    from openjarvis.core.credentials import load_provider_key_files

    load_provider_key_files()
    order = list(providers) if providers else _chain_order()
    discover_fn = discover or discover_models

    try:
        from openjarvis.engine.cloud import CloudEngine

        engine: InferenceEngine = CloudEngine()
    except Exception as exc:  # noqa: BLE001 - optional cloud extras
        logger.warning("Cloud engine unavailable: %s", exc)
        return None

    slots: List[ProviderSlot] = []
    for provider in order:
        if provider not in PREFERRED_MODELS:
            continue
        key = _env_key(provider)
        if not key:
            continue
        try:
            models = discover_fn(provider, key)
        except Exception as exc:  # noqa: BLE001 - one provider must not break the rest
            logger.debug("Discovery failed for %s: %s", provider, exc)
            models = list(PREFERRED_MODELS[provider])
        added = 0
        for model in models:
            if added >= max_models_per_provider:
                break
            if not engine.can_serve(model):
                continue
            slots.append(
                ProviderSlot(
                    provider=provider,
                    label=PROVIDER_LABELS.get(provider, provider.title()),
                    engine=engine,
                    model=model,
                )
            )
            added += 1

    if not slots:
        close = getattr(engine, "close", None)
        if callable(close):
            close()
        return None
    logger.info(
        "Free provider chain: %s",
        " → ".join(f"{slot.label}/{slot.model}" for slot in slots),
    )
    return FailoverEngine(slots)
