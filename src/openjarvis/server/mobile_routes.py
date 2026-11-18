"""Mobile assistant runtime and JSON API.

Backs the installable phone web app served at ``/m``: a chat surface plus
recurring automation ("do X every morning") that runs through the same agent
and tool stack as the desktop UI.

The runtime is deliberately independent of ``jarvis serve``: it owns its task
scheduler (SQLite under ``$OPENJARVIS_HOME/mobile``) and can build a working
agent at request time, so a phone can be pointed at a fresh server, connected
to a free model provider, and start scheduling tasks without a restart.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from openjarvis.core.credentials import (
    gemini_api_base,
    gemini_key_source,
    load_gemini_key_file,
    load_openrouter_key_file,
    load_provider_key_files,
    normalize_gemini_key,
    normalize_openrouter_key,
    openrouter_api_base,
    openrouter_key_source,
    probe_gemini_generation,
)
from openjarvis.core.paths import get_config_dir
from openjarvis.core.types import Message
from openjarvis.engine._stubs import InferenceEngine
from openjarvis.engine.failover import FailoverEngine, build_free_failover_engine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/mobile", tags=["mobile"])

# Tools exposed to the phone assistant. Everything here is either read-only or
# a local action; shell/code execution stays out of the default mobile agent.
MOBILE_TOOL_ALLOWLIST: tuple[str, ...] = (
    "think",
    "calculator",
    "web_search",
    "http_request",
    "weather",
    "fx_macro",
    "file_read",
    "file_write",
    "schedule_task",
    "list_scheduled_tasks",
    "pause_scheduled_task",
    "resume_scheduled_task",
    "cancel_scheduled_task",
)

# Used only when the live OpenRouter catalogue cannot be fetched. The first
# entry is OpenRouter's own free-model router, so the default keeps working as
# individual free models come and go.
# Gemini's free tier covers the Flash models (per model, per project); Pro and
# preview models need a billing account, so they are listed after these.
GEMINI_MODEL_PREFERENCE: tuple[str, ...] = (
    "google/gemini-2.5-flash",
    "google/gemini-2.5-flash-lite",
    "google/gemini-2.0-flash",
)

FALLBACK_FREE_MODELS: tuple[str, ...] = (
    "openrouter/openrouter/free",
    "openrouter/google/gemma-4-31b-it:free",
    "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
    "openrouter/inclusionai/ling-3.2-flash:free",
)

FREE_ROUTER_MODEL = "openrouter/openrouter/free"

_NEED_SETUP_MESSAGE = (
    "No AI model is connected yet. Open Settings in the Jarvis app and connect "
    "a free provider (OpenRouter free models or Ollama running on this "
    "computer), then try again."
)


def _mobile_dir() -> Path:
    path = get_config_dir() / "mobile"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _settings_path() -> Path:
    return _mobile_dir() / "settings.json"


def load_settings() -> dict[str, Any]:
    try:
        return json.loads(_settings_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_settings(data: dict[str, Any]) -> None:
    path = _settings_path()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _cloud_keys_path() -> Path:
    return get_config_dir() / "cloud-keys.env"


def _persist_cloud_key(name: str, value: str) -> None:
    """Store a provider key where the rest of OpenJarvis already looks for it."""
    path = _cloud_keys_path()
    existing: list[str] = []
    if path.exists():
        existing = [
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith(f"{name}=")
        ]
    existing.append(f"{name}={value}")
    path.write_text("\n".join(existing) + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass


class UnconfiguredEngine(InferenceEngine):
    """Placeholder that lets the phone connect a provider after startup."""

    engine_id = "unconfigured"
    is_cloud = False
    # Chat routes use this to fail loudly instead of streaming an empty reply.
    is_placeholder = True

    def generate(
        self,
        messages: Sequence[Message],
        *,
        model: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {
            "content": _NEED_SETUP_MESSAGE,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        model: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        yield _NEED_SETUP_MESSAGE

    def list_models(self) -> list[str]:
        return []

    def health(self) -> bool:
        return False

    def can_serve(self, model: str) -> bool:
        return False


def build_mobile_agent(
    engine: InferenceEngine,
    model: str,
    scheduler: Any = None,
    *,
    bus: Any = None,
    max_turns: int = 8,
) -> Any:
    """Build the tool-using agent the phone talks to.

    Reuses the registered tools and, when a scheduler is available, hands the
    scheduling tools a live scheduler so "every morning at 8" works from chat.
    """
    import openjarvis.agents  # noqa: F401  (registers agent classes)
    import openjarvis.scheduler.tools  # noqa: F401
    import openjarvis.tools  # noqa: F401  (registers tool classes)
    from openjarvis.agents.native_react import NativeReActAgent
    from openjarvis.core.config import load_config
    from openjarvis.core.registry import ToolRegistry
    from openjarvis.tools._paths import file_tool_kwargs

    try:
        config = load_config()
    except Exception:  # pragma: no cover - config is optional for tool binding
        config = None

    tools: list[Any] = []
    for name in MOBILE_TOOL_ALLOWLIST:
        if not ToolRegistry.contains(name):
            continue
        try:
            tool = ToolRegistry.get(name)
            instance = tool(**file_tool_kwargs(name, config))
        except Exception as exc:
            logger.debug("Mobile tool %s unavailable: %s", name, exc)
            continue
        if scheduler is not None and hasattr(instance, "_scheduler"):
            instance._scheduler = scheduler
        tools.append(instance)

    return NativeReActAgent(
        engine,
        model,
        tools=tools,
        bus=bus,
        max_turns=max_turns,
        agent_id="mobile",
    )


class _AgentRunner:
    """``TaskScheduler.system`` adapter that always uses the live agent."""

    def __init__(self, runtime: "MobileRuntime") -> None:
        self._runtime = runtime

    def ask(self, prompt: str, **kwargs: Any) -> str:
        agent = getattr(self._runtime.app.state, "agent", None)
        if agent is None:
            raise RuntimeError(
                "No assistant agent is configured. Connect a model in Settings first."
            )
        result = agent.run(prompt)
        return str(getattr(result, "content", result))


class MobileRuntime:
    """Owns the mobile scheduler and model configuration for an app instance."""

    def __init__(self, app: Any, *, config: Any = None, api_key: str = "") -> None:
        self.app = app
        self.config = config
        self.api_key = api_key
        self._scheduler: Any = None
        self._lock = threading.Lock()
        self._jobs: dict[str, float] = {}
        self._auto_attempted = False
        # Key files on the server mean the phone never has to paste one.
        load_provider_key_files()

    # -- Lifecycle -----------------------------------------------------------

    def ensure_started(self) -> None:
        """Start the scheduler and agent on first use (cheap to install)."""
        if self._scheduler is not None and getattr(self.app.state, "agent", None):
            return
        with self._lock:
            if self._scheduler is None:
                self.start()
        self._ensure_agent()

    def _ensure_agent(self) -> None:
        """Attach the mobile agent when the server was started without one.

        ``jarvis serve`` can run with a cloud engine and no configured agent;
        the phone app still needs a tool-using agent for chat and automations.
        """
        if getattr(self.app.state, "agent", None) is not None:
            return
        engine = getattr(self.app.state, "engine", None)
        model = getattr(self.app.state, "model", "") or ""
        if engine is None or isinstance(engine, UnconfiguredEngine) or not model:
            # First use with no provider chosen: build the free-provider chain
            # in the background so this request is not held up by discovery.
            self.start_auto()
            if getattr(self.app.state, "agent", None) is not None:
                return
            engine = getattr(self.app.state, "engine", None)
            model = getattr(self.app.state, "model", "") or ""
            if engine is None or isinstance(engine, UnconfiguredEngine) or not model:
                return
        try:
            agent = build_mobile_agent(
                engine, model, self._scheduler, bus=getattr(self.app.state, "bus", None)
            )
        except Exception as exc:
            logger.warning("Mobile agent build failed: %s", exc)
            return
        self.app.state.agent = agent
        self.app.state.agent_name = "mobile"
        self._wire_scheduler_tools()

    def start_auto(self) -> None:
        """Build the free-provider failover chain once, in the background.

        Every key on the server joins the chain in free-first order, so a
        provider whose allowance runs out is simply skipped until its cooldown
        expires instead of the assistant going silent.
        """
        if self._auto_attempted:
            return
        self._auto_attempted = True

        def _build() -> None:
            try:
                engine = build_free_failover_engine()
            except Exception as exc:  # pragma: no cover - optional extras
                logger.warning("Free provider chain unavailable: %s", exc)
                return
            if engine is None:
                return
            try:
                self._activate_chain(engine)
            except Exception:
                logger.exception("Activating the free provider chain failed")

        threading.Thread(target=_build, name="free-provider-chain", daemon=True).start()

    def _activate_chain(self, engine: FailoverEngine) -> None:
        """Make the failover chain the live engine, keeping a saved choice."""
        settings = load_settings()
        if (settings.get("provider") or "") not in ("", "auto"):
            # The user picked one provider in Settings; do not override it.
            return
        self.app.state.engine = engine
        self.app.state.engine_name = "auto"
        self.app.state.model = engine.primary_model
        try:
            self.app.state.agent = build_mobile_agent(
                engine,
                engine.primary_model,
                self._scheduler,
                bus=getattr(self.app.state, "bus", None),
            )
            self.app.state.agent_name = "mobile"
        except Exception:
            logger.exception("Could not build the mobile agent for the free chain")
        self._wire_scheduler_tools()
        settings.update(
            {
                "provider": "auto",
                "model": engine.primary_model,
                "models": engine.model_ids(),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )
        _save_settings(settings)

    def start(self) -> None:
        try:
            from openjarvis.scheduler.scheduler import TaskScheduler
            from openjarvis.scheduler.store import SchedulerStore

            store = SchedulerStore(_mobile_dir() / "tasks.db")
            self._scheduler = TaskScheduler(
                store,
                _AgentRunner(self),
                poll_interval=30,
                bus=getattr(self.app.state, "bus", None),
            )
            self._scheduler.start()
        except Exception as exc:
            logger.warning("Mobile scheduler unavailable: %s", exc)
            self._scheduler = None
            return
        self._wire_scheduler_tools()

    def stop(self) -> None:
        if self._scheduler is not None:
            try:
                self._scheduler.stop()
            except Exception:
                logger.debug("Mobile scheduler stop failed", exc_info=True)
            self._scheduler = None

    def _wire_scheduler_tools(self) -> None:
        agent = getattr(self.app.state, "agent", None)
        for tool in getattr(agent, "_tools", []) or []:
            if hasattr(tool, "_scheduler"):
                tool._scheduler = self._scheduler

    # -- Status / configuration ---------------------------------------------

    def _engine_label(self) -> str:
        return getattr(self.app.state, "engine_name", "") or ""

    def status(self) -> dict[str, Any]:
        settings = load_settings()
        engine = getattr(self.app.state, "engine", None)
        model = getattr(self.app.state, "model", "") or ""
        configured = bool(model) and not isinstance(engine, UnconfiguredEngine)
        models = settings.get("models") or []
        chain: list[dict[str, Any]] = []
        active_provider = ""
        if isinstance(engine, FailoverEngine):
            chain = engine.describe()
            active_provider = engine.active_label()
        return {
            "configured": configured,
            "provider": settings.get("provider", "") or self._engine_label(),
            "model": model,
            "models": models,
            "chain": chain,
            "active_provider": active_provider,
            "voice": self.voice_status(),
            "speech_out": self.tts_status(),
            "tasks": len(self.list_tasks()),
            "free_models": list(FALLBACK_FREE_MODELS),
            "key_source": openrouter_key_source(),
            "gemini_key_source": gemini_key_source(),
            "scheduler": self._scheduler is not None,
            "server_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    def tts_status(self) -> dict[str, Any]:
        """Report whether replies can be spoken out loud on this server."""
        try:
            from openjarvis.speech._tts_discovery import get_tts_backend

            backend = get_tts_backend("fish_audio")
        except Exception as exc:  # pragma: no cover - import/env failures
            return {"available": False, "backend": "", "reason": str(exc)}
        if backend is None:
            return {
                "available": False,
                "backend": "",
                "reason": (
                    "No voice is configured on the server. Add FISH_AUDIO_API_KEY "
                    "in the server's environment to hear Jarvis speak."
                ),
            }
        return {"available": True, "backend": backend.backend_id, "reason": ""}

    def voice_status(self) -> dict[str, Any]:
        """Report whether hold-to-talk can transcribe on this server."""
        backend = getattr(self.app.state, "speech_backend", None)
        if backend is None:
            return {
                "available": False,
                "backend": "",
                "reason": (
                    "Voice transcription is not enabled on the server. Start "
                    "Jarvis from a terminal running: uv sync --extra speech"
                ),
            }
        try:
            available = bool(backend.health())
        except Exception as exc:
            return {
                "available": False,
                "backend": backend.backend_id,
                "reason": str(exc),
            }
        reason = ""
        if not available:
            last_error = getattr(backend, "last_error", None)
            if callable(last_error):
                reason = str(last_error() or "")
        return {"available": available, "backend": backend.backend_id, "reason": reason}

    async def configure(
        self,
        *,
        provider: str,
        api_key: str = "",
        model: str = "",
        base_url: str = "",
    ) -> dict[str, Any]:
        provider = (provider or "").strip().lower()
        if provider == "auto":
            return await self._configure_auto()
        if provider == "openrouter":
            return await self._configure_openrouter(api_key=api_key, model=model)
        if provider == "gemini":
            return await self._configure_gemini(api_key=api_key, model=model)
        if provider == "ollama":
            return await self._configure_ollama(model=model, base_url=base_url)
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider!r}")

    async def _configure_auto(self) -> dict[str, Any]:
        """Use every free provider key on this server, with automatic failover."""
        import asyncio

        engine = await asyncio.to_thread(build_free_failover_engine)
        if engine is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No AI provider key is configured on this server yet. Add "
                    "GROQ_API_KEY (console.groq.com/keys), GEMINI_API_KEY "
                    "(aistudio.google.com/apikey) or OPENROUTER_API_KEY "
                    "(openrouter.ai/keys) — each is free — and try again."
                ),
            )
        self._activate(
            engine,
            engine.primary_model,
            provider="auto",
            models=engine.model_ids(),
        )
        return {
            "configured": True,
            "provider": "auto",
            "model": engine.primary_model,
            "models": engine.model_ids(),
            "chain": engine.describe(),
        }

    async def _configure_openrouter(
        self, *, api_key: str, model: str
    ) -> dict[str, Any]:
        # No key typed in? Fall back to one the server loaded from a file, so a
        # phone can connect without anybody pasting a secret.
        if not (api_key or "").strip():
            load_openrouter_key_file()
        key = normalize_openrouter_key(
            api_key or os.environ.get("OPENROUTER_API_KEY", "")
        )
        if not key:
            raise HTTPException(
                status_code=400,
                detail=(
                    "An OpenRouter API key is required (free to create at "
                    "openrouter.ai/keys)."
                ),
            )

        free_models = await self._openrouter_free_models(key)
        chosen = model.strip()
        if not chosen or not chosen.startswith("openrouter/"):
            if chosen and "/" in chosen:
                chosen = f"openrouter/{chosen}"
            else:
                chosen = free_models[0] if free_models else FREE_ROUTER_MODEL

        _persist_cloud_key("OPENROUTER_API_KEY", key)
        os.environ["OPENROUTER_API_KEY"] = key

        try:
            from openjarvis.engine.cloud import CloudEngine

            engine = CloudEngine()
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Cloud engine unavailable: {exc}"
            ) from exc
        if not engine.health() or not engine.can_serve(chosen):
            raise HTTPException(
                status_code=400,
                detail="OpenRouter rejected the key or the model is unavailable.",
            )

        self._activate(engine, chosen, provider="openrouter", models=free_models)
        return {
            "configured": True,
            "provider": "openrouter",
            "model": chosen,
            "models": free_models,
        }

    async def _configure_gemini(
        self, *, api_key: str, model: str
    ) -> dict[str, Any]:
        # No key typed in? Fall back to one the server loaded from a file.
        if not (api_key or "").strip():
            load_gemini_key_file()
        key = normalize_gemini_key(
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or ""
        )
        if not key:
            raise HTTPException(
                status_code=400,
                detail=(
                    "A Gemini API key is required (free to create at "
                    "aistudio.google.com/apikey)."
                ),
            )

        models = await self._gemini_models(key)
        chosen = model.strip()
        if not chosen or not chosen.startswith("google/"):
            if chosen and "/" in chosen:
                chosen = f"google/{chosen}"
            else:
                chosen = next(iter(models), GEMINI_MODEL_PREFERENCE[0])
        if chosen not in models:
            models = [chosen, *models]

        # A key that lists models can still be refused for generation (the
        # known AQ authorization-key case), so prove it can answer first.
        import asyncio

        ok, reason = await asyncio.to_thread(probe_gemini_generation, key, chosen)
        if not ok:
            raise HTTPException(
                status_code=400,
                detail=f"Gemini cannot answer with this key: {reason}.",
            )

        _persist_cloud_key("GEMINI_API_KEY", key)
        os.environ["GEMINI_API_KEY"] = key

        try:
            from openjarvis.engine.cloud import CloudEngine

            engine = CloudEngine()
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Cloud engine unavailable: {exc}"
            ) from exc
        if not engine.health() or not engine.can_serve(chosen):
            raise HTTPException(
                status_code=501,
                detail=(
                    "Gemini support needs the Google client: "
                    "uv sync --extra inference-google"
                ),
            )

        self._activate(engine, chosen, provider="gemini", models=list(models))
        return {
            "configured": True,
            "provider": "gemini",
            "model": chosen,
            "models": list(models),
        }

    async def _gemini_models(self, api_key: str) -> list[str]:
        import httpx

        url = f"{gemini_api_base()}/v1beta/models"
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.get(url, headers={"x-goog-api-key": api_key})
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Could not reach Gemini: {exc}",
            ) from exc
        if resp.status_code in (401, 403):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Google rejected that API key. Create a new one at "
                    "aistudio.google.com/apikey."
                ),
            )
        if resp.status_code >= 400:
            raise HTTPException(
                status_code=502,
                detail=f"Gemini returned HTTP {resp.status_code}.",
            )

        entries = resp.json().get("models", [])
        available: dict[str, None] = {}
        for entry in entries:
            name = str(entry.get("name", "")).removeprefix("models/")
            if not name or "gemini" not in name:
                continue
            methods = entry.get("supportedGenerationMethods") or []
            if methods and "generateContent" not in methods:
                continue
            # Embedding, image and TTS models cannot hold a conversation.
            if any(bad in name for bad in ("embedding", "image", "tts", "audio")):
                continue
            available[f"google/{name}"] = None

        # Free-tier Flash models first; anything else the key can reach after.
        ordered = [m for m in GEMINI_MODEL_PREFERENCE if m in available]
        ordered.extend(m for m in available if m not in ordered)
        return ordered or list(GEMINI_MODEL_PREFERENCE)

    async def _configure_ollama(self, *, model: str, base_url: str) -> dict[str, Any]:
        host = (base_url or os.environ.get("OLLAMA_HOST") or "").strip()
        try:
            from openjarvis.engine.ollama import OllamaEngine

            engine = OllamaEngine(host=host) if host else OllamaEngine()
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Ollama engine unavailable: {exc}"
            ) from exc

        import asyncio

        models = await asyncio.to_thread(engine.list_models)
        chosen = model.strip() or (models[0] if models else "")
        if not chosen:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Ollama is not answering. Start it on the computer running "
                    "Jarvis (ollama serve) and pull a model first."
                ),
            )
        self._activate(engine, chosen, provider="ollama", models=list(models))
        return {
            "configured": True,
            "provider": "ollama",
            "model": chosen,
            "models": list(models),
        }

    def _activate(
        self,
        engine: InferenceEngine,
        model: str,
        *,
        provider: str,
        models: list[str],
    ) -> None:
        agent = build_mobile_agent(
            engine,
            model,
            self._scheduler,
            bus=getattr(self.app.state, "bus", None),
        )
        with self._lock:
            self.app.state.engine = engine
            self.app.state.engine_name = provider
            self.app.state.model = model
            self.app.state.agent = agent
            self.app.state.agent_name = "mobile"
            self._wire_scheduler_tools()
            settings = load_settings()
            settings.update(
                {
                    "provider": provider,
                    "model": model,
                    "models": models,
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            )
            _save_settings(settings)

    async def _openrouter_free_models(self, api_key: str) -> list[str]:
        import httpx

        headers = {"Authorization": f"Bearer {api_key}"}
        base = openrouter_api_base()
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                # The public model catalogue answers without authentication, so
                # it cannot prove a key works. This endpoint requires a real
                # key: 401 here means the key is invalid, disabled, or revoked.
                auth_resp = await client.get(f"{base}/key", headers=headers)
                if auth_resp.status_code in (401, 403):
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "OpenRouter rejected that API key. Copy it again "
                            "from openrouter.ai/keys."
                        ),
                    )
                if auth_resp.status_code >= 400:
                    raise HTTPException(
                        status_code=502,
                        detail=(
                            "Could not verify the key with OpenRouter "
                            f"(HTTP {auth_resp.status_code})."
                        ),
                    )
                resp = await client.get(f"{base}/models", headers=headers)
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Could not reach OpenRouter: {exc}",
            ) from exc
        if resp.status_code >= 400:
            raise HTTPException(
                status_code=502,
                detail=f"OpenRouter returned HTTP {resp.status_code}.",
            )
        payload = resp.json().get("data", [])
        free: list[tuple[int, str]] = []
        for entry in payload:
            pricing = entry.get("pricing") or {}

            def _is_zero(value: Any) -> bool:
                try:
                    return float(value) == 0.0
                except (TypeError, ValueError):
                    return False

            prompt_price = pricing.get("prompt")
            completion_price = pricing.get("completion")
            if not (_is_zero(prompt_price) and _is_zero(completion_price)):
                continue
            # Chat only: skip models that emit audio/images (e.g. music models)
            # and anything that cannot take text input.
            architecture = entry.get("architecture") or {}
            outputs = architecture.get("output_modalities") or ["text"]
            inputs = architecture.get("input_modalities") or ["text"]
            if "text" not in outputs or "text" not in inputs:
                continue
            if "audio" in outputs or "image" in outputs:
                continue
            model_id = str(entry.get("id", ""))
            if not model_id:
                continue
            created = entry.get("created")
            free.append((int(created) if isinstance(created, int) else 0, model_id))

        # Newest free models first, with OpenRouter's free router pinned on top.
        free.sort(key=lambda item: item[0], reverse=True)
        models = [
            f"openrouter/{model_id}"
            for _, model_id in free
            if model_id != "openrouter/free"
        ]
        if any(model_id == "openrouter/free" for _, model_id in free):
            models.insert(0, FREE_ROUTER_MODEL)
        return models or list(FALLBACK_FREE_MODELS)

    # -- Tasks ---------------------------------------------------------------

    def _require_scheduler(self) -> Any:
        self.ensure_started()
        if self._scheduler is None:
            raise HTTPException(status_code=503, detail="Task scheduler unavailable.")
        return self._scheduler

    def list_tasks(self) -> list[dict[str, Any]]:
        scheduler = self._require_scheduler()
        tasks = [self._serialize(task) for task in scheduler.list_tasks()]
        tasks.sort(key=lambda t: (t.get("status") != "active", t.get("next_run") or ""))
        return tasks

    def get_task(self, task_id: str) -> dict[str, Any]:
        scheduler = self._require_scheduler()
        for task in scheduler.list_tasks():
            if task.id == task_id:
                return self._serialize(task)
        raise HTTPException(status_code=404, detail="Task not found.")

    def create_task(
        self,
        *,
        prompt: str,
        schedule_type: str,
        schedule_value: str,
        title: str = "",
        timezone: str = "",
    ) -> dict[str, Any]:
        scheduler = self._require_scheduler()
        prompt = prompt.strip()
        if not prompt:
            raise HTTPException(
                status_code=400, detail="Describe what Jarvis should do."
            )
        if schedule_type not in {"cron", "interval", "once"}:
            raise HTTPException(status_code=400, detail="Invalid schedule type.")
        if not schedule_value.strip():
            raise HTTPException(status_code=400, detail="Invalid schedule value.")
        if schedule_type == "interval":
            try:
                if float(schedule_value) < 60:
                    raise ValueError
            except ValueError as exc:
                raise HTTPException(
                    status_code=400, detail="Intervals must be at least 60 seconds."
                ) from exc
        metadata: dict[str, Any] = {}
        if title.strip():
            metadata["title"] = title.strip()
        if timezone:
            metadata["timezone"] = timezone
        task = scheduler.create_task(
            prompt=prompt,
            schedule_type=schedule_type,
            schedule_value=schedule_value,
            metadata=metadata,
            agent="mobile",
        )
        return self._serialize(task)

    def set_task_status(self, task_id: str, status: str) -> dict[str, Any]:
        scheduler = self._require_scheduler()
        try:
            if status == "paused":
                scheduler.pause_task(task_id)
            elif status == "active":
                scheduler.resume_task(task_id)
            else:
                raise HTTPException(
                    status_code=400, detail="Status must be paused or active."
                )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Task not found.") from exc
        return self.get_task(task_id)

    def delete_task(self, task_id: str) -> None:
        scheduler = self._require_scheduler()
        try:
            scheduler.delete_task(task_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Task not found.") from exc

    def runs(self, task_id: str, limit: int = 10) -> list[dict[str, Any]]:
        scheduler = self._require_scheduler()
        try:
            return scheduler.get_run_logs(task_id, limit=limit)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Task not found.") from exc

    def run_now(self, task_id: str) -> dict[str, Any]:
        """Start a manual run in the background; the phone polls for results."""
        scheduler = self._require_scheduler()
        self.get_task(task_id)  # 404s for unknown ids before spawning a thread
        with self._lock:
            if task_id in self._jobs and self._jobs[task_id] > time.time() - 600:
                return {"status": "running", "already_running": True}
            self._jobs[task_id] = time.time()

        def _run() -> None:
            try:
                scheduler.run_task_now(task_id)
            except Exception:
                logger.exception("Manual run of task %s failed", task_id)
            finally:
                with self._lock:
                    self._jobs.pop(task_id, None)

        threading.Thread(
            target=_run, name=f"mobile-task-{task_id}", daemon=True
        ).start()
        return {"status": "running"}

    def is_running(self, task_id: str) -> bool:
        with self._lock:
            return task_id in self._jobs

    @staticmethod
    def _serialize(task: Any) -> dict[str, Any]:
        data = task.to_dict()
        metadata = data.get("metadata") or {}
        data["title"] = metadata.get("title") or _summarize(data.get("prompt", ""))
        data["timezone"] = metadata.get("timezone", "")
        return data


def _summarize(text: str, limit: int = 60) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


# ---------------------------------------------------------------------------
# Request models + routes
# ---------------------------------------------------------------------------


class MobileSetupRequest(BaseModel):
    provider: str
    api_key: str = ""
    model: str = ""
    base_url: str = ""


class MobileTaskCreate(BaseModel):
    prompt: str
    schedule_type: str
    schedule_value: str
    title: str = ""
    timezone: str = ""


class MobileTaskPatch(BaseModel):
    status: str


class MobileSpeechRequest(BaseModel):
    text: str
    voice_id: str = ""
    format: str = "mp3"


def _runtime(request: Request) -> MobileRuntime:
    runtime = getattr(request.app.state, "mobile_runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="Mobile runtime unavailable.")
    return runtime


@router.get("/status")
async def mobile_status(request: Request) -> dict[str, Any]:
    return _runtime(request).status()


@router.post("/speech")
async def mobile_speech(body: MobileSpeechRequest, request: Request) -> Response:
    """Speak a reply with the configured TTS voice and return the audio bytes."""
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Nothing to speak.")
    if len(text) > 4000:
        text = text[:4000]

    from openjarvis.speech._tts_discovery import get_tts_backend

    backend = await asyncio.to_thread(get_tts_backend, "fish_audio")
    if backend is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No voice is configured on the server. Add FISH_AUDIO_API_KEY to "
                "the server's environment, then try again."
            ),
        )
    try:
        result = await asyncio.to_thread(
            backend.synthesize,
            text,
            voice_id=body.voice_id,
            output_format=body.format,
        )
    except Exception as exc:
        logger.warning("Voice synthesis failed: %s", exc)
        raise HTTPException(
            status_code=502, detail=f"Voice synthesis failed: {exc}"
        ) from exc

    media_type = {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "opus": "audio/ogg",
    }.get(result.format, "audio/mpeg")
    return Response(
        content=result.audio,
        media_type=media_type,
        headers={"Cache-Control": "no-store"},
    )


@router.post("/setup")
async def mobile_setup(body: MobileSetupRequest, request: Request) -> dict[str, Any]:
    return await _runtime(request).configure(
        provider=body.provider,
        api_key=body.api_key,
        model=body.model,
        base_url=body.base_url,
    )


@router.get("/tasks")
async def mobile_list_tasks(request: Request) -> dict[str, Any]:
    runtime = _runtime(request)
    tasks = runtime.list_tasks()
    for task in tasks:
        runs = runtime.runs(task["id"], limit=1)
        task["last_run_result"] = runs[0] if runs else None
        task["running"] = runtime.is_running(task["id"])
    return {"tasks": tasks}


@router.post("/tasks")
async def mobile_create_task(
    body: MobileTaskCreate, request: Request
) -> dict[str, Any]:
    task = _runtime(request).create_task(
        prompt=body.prompt,
        schedule_type=body.schedule_type,
        schedule_value=body.schedule_value,
        title=body.title,
        timezone=body.timezone,
    )
    return {"task": task}


@router.patch("/tasks/{task_id}")
async def mobile_update_task(
    task_id: str, body: MobileTaskPatch, request: Request
) -> dict[str, Any]:
    return {"task": _runtime(request).set_task_status(task_id, body.status)}


@router.delete("/tasks/{task_id}")
async def mobile_delete_task(task_id: str, request: Request) -> dict[str, Any]:
    _runtime(request).delete_task(task_id)
    return {"deleted": True}


@router.post("/tasks/{task_id}/run")
async def mobile_run_task(task_id: str, request: Request) -> dict[str, Any]:
    return _runtime(request).run_now(task_id)


@router.get("/tasks/{task_id}/runs")
async def mobile_task_runs(
    task_id: str, request: Request, limit: int = 10
) -> dict[str, Any]:
    return {"runs": _runtime(request).runs(task_id, limit=limit)}


def install_mobile_runtime(app: Any, *, config: Any = None, api_key: str = "") -> None:
    """Attach the mobile runtime; its scheduler starts on first use."""
    runtime = MobileRuntime(app, config=config, api_key=api_key)
    app.state.mobile_runtime = runtime
    # No provider chosen at startup? Build the free-provider chain now rather
    # than making the phone's first request wait on model discovery.
    if getattr(app.state, "engine", None) is None or isinstance(
        getattr(app.state, "engine", None), UnconfiguredEngine
    ):
        runtime.start_auto()

    @app.on_event("shutdown")
    async def _shutdown_mobile_runtime() -> None:
        runtime.stop()


__all__ = [
    "FALLBACK_FREE_MODELS",
    "MOBILE_TOOL_ALLOWLIST",
    "MobileRuntime",
    "UnconfiguredEngine",
    "build_mobile_agent",
    "install_mobile_runtime",
    "load_settings",
    "router",
]
