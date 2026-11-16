"""Credential persistence for tools and channels.

Stores credentials in ~/.openjarvis/credentials.toml with 0o600 permissions.
Thread-safe writes via lock. Sets os.environ on save for immediate effect.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

import tomlkit

from openjarvis.core.paths import get_config_dir
from openjarvis.security.file_utils import secure_write_text

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

_LOCK = threading.RLock()


def _default_path() -> Path:
    """Resolve the credentials file under the OpenJarvis root (env-aware)."""
    return get_config_dir() / "credentials.toml"


TOOL_CREDENTIALS: dict[str, list[str]] = {
    "web_search": [
        "TAVILY_API_KEY",
        "YOUDOTCOM_API_KEY",
        "SERPLY_API_KEY",
        "FIRECRAWL_API_KEY",
    ],
    "get_weather": ["OPENWEATHERMAP_API_KEY"],
    "fx_macro_data": ["FXMACRODATA_API_KEY"],
    "image_generate": ["OPENAI_API_KEY"],
    "slack": ["SLACK_BOT_TOKEN", "SLACK_APP_TOKEN"],
    "telegram": ["TELEGRAM_BOT_TOKEN"],
    "discord": ["DISCORD_BOT_TOKEN"],
    "email": ["EMAIL_USERNAME", "EMAIL_PASSWORD"],
    "whatsapp": ["WHATSAPP_ACCESS_TOKEN", "WHATSAPP_PHONE_NUMBER_ID"],
    "signal": ["SIGNAL_CLI_PATH"],
    "google_chat": ["GOOGLE_CHAT_WEBHOOK_URL"],
    "teams": ["TEAMS_WEBHOOK_URL"],
    "bluebubbles": ["BLUEBUBBLES_SERVER_URL", "BLUEBUBBLES_PASSWORD"],
    "line": ["LINE_CHANNEL_ACCESS_TOKEN", "LINE_CHANNEL_SECRET"],
    "viber": ["VIBER_AUTH_TOKEN"],
    "messenger": ["MESSENGER_PAGE_ACCESS_TOKEN", "MESSENGER_VERIFY_TOKEN"],
    "reddit": [
        "REDDIT_CLIENT_ID",
        "REDDIT_CLIENT_SECRET",
        "REDDIT_USERNAME",
        "REDDIT_PASSWORD",
    ],
    "mastodon": ["MASTODON_ACCESS_TOKEN", "MASTODON_API_BASE_URL"],
    "twitch": ["TWITCH_TOKEN", "TWITCH_CHANNEL"],
    "matrix": ["MATRIX_HOMESERVER", "MATRIX_ACCESS_TOKEN"],
    "mattermost": ["MATTERMOST_URL", "MATTERMOST_TOKEN"],
    "zulip": ["ZULIP_EMAIL", "ZULIP_API_KEY", "ZULIP_SITE"],
    "rocketchat": ["ROCKETCHAT_URL", "ROCKETCHAT_USER_ID", "ROCKETCHAT_AUTH_TOKEN"],
    "xmpp": ["XMPP_JID", "XMPP_PASSWORD"],
    "feishu": ["FEISHU_APP_ID", "FEISHU_APP_SECRET"],
    "nostr": ["NOSTR_PRIVATE_KEY"],
}


# Keys that unlock or upgrade a tool but are not prerequisites for it. Every
# key in TOOL_CREDENTIALS is otherwise treated as required — by the Settings UI,
# by ``jarvis doctor``, and by anything else reading credential status — which
# leaves no way to describe a tool that also works without one. ``web_search``
# is the first such tool: the You.com keyless tier serves it with no key at all,
# and every listed key only raises limits, changes the engine, or improves
# result quality.
OPTIONAL_TOOL_CREDENTIALS: dict[str, frozenset[str]] = {
    "web_search": frozenset(
        {
            "TAVILY_API_KEY",
            "YOUDOTCOM_API_KEY",
            "SERPLY_API_KEY",
            "FIRECRAWL_API_KEY",
        }
    ),
    # Recent USD releases, the USD calendar and every catalogue are keyless;
    # the key unlocks other currencies, full history and FX rates.
    "fx_macro_data": frozenset({"FXMACRODATA_API_KEY"}),
}


def is_credential_optional(tool_name: str, key: str) -> bool:
    """Return whether ``key`` is an upgrade for ``tool_name`` rather than required."""
    return key in OPTIONAL_TOOL_CREDENTIALS.get(tool_name, frozenset())


def get_required_credentials(tool_name: str) -> list[str]:
    """Return only the keys ``tool_name`` cannot run without."""
    optional = OPTIONAL_TOOL_CREDENTIALS.get(tool_name, frozenset())
    return [k for k in TOOL_CREDENTIALS.get(tool_name, []) if k not in optional]


def load_credentials(path: Path | None = None) -> dict[str, dict[str, str]]:
    """Load credentials, preserving malformed input as a recoverable backup."""
    p = Path(path) if path else _default_path()
    with _LOCK:
        if not p.exists():
            return {}
        try:
            with open(p, "rb") as f:
                return tomllib.load(f)
        except tomllib.TOMLDecodeError:
            backup = p.with_name(f"{p.name}.corrupt-{time.time_ns()}")
            try:
                os.replace(p, backup)
                os.chmod(backup, 0o600)
            except OSError:
                logger.exception(
                    "Could not preserve malformed credential file %s",
                    p,
                )
                raise
            logger.warning(
                "Malformed credential file moved to %s; starting with an empty store",
                backup,
            )
            return {}


def _validate_credential_key(tool_name: str, key: str) -> None:
    allowed = TOOL_CREDENTIALS.get(tool_name, [])
    if key not in allowed:
        raise ValueError(f"Unknown credential key '{key}' for tool '{tool_name}'")


def _write_credentials(creds: dict[str, dict[str, str]], path: Path) -> None:
    document = tomlkit.document()
    for section, kvs in creds.items():
        table = tomlkit.table()
        for key, value in kvs.items():
            table.add(key, value)
        document.add(section, table)
    secure_write_text(path, tomlkit.dumps(document), mode=0o600, encoding="utf-8")


def save_credential(
    tool_name: str,
    key: str,
    value: str,
    *,
    path: Path | None = None,
) -> None:
    """Save a single credential key, validate, write file, and set os.environ."""
    _validate_credential_key(tool_name, key)
    stripped = value.strip()
    if not stripped:
        raise ValueError("Credential value must not be empty")

    p = Path(path) if path else _default_path()
    with _LOCK:
        creds = load_credentials(path=p)
        if tool_name not in creds:
            creds[tool_name] = {}
        creds[tool_name][key] = stripped
        _write_credentials(creds, p)

    os.environ[key] = stripped


def delete_credential(
    tool_name: str,
    key: str,
    *,
    path: Path | None = None,
) -> None:
    """Delete a persisted credential and remove it from the running process."""
    _validate_credential_key(tool_name, key)
    p = Path(path) if path else _default_path()
    with _LOCK:
        creds = load_credentials(path=p)
        tool_creds = creds.get(tool_name)
        if tool_creds is not None:
            tool_creds.pop(key, None)
            if not tool_creds:
                creds.pop(tool_name, None)
            _write_credentials(creds, p)

    os.environ.pop(key, None)


def get_credential_status(tool_name: str) -> dict[str, bool]:
    """Return {KEY: bool} for each declared key indicating if set in env.

    Includes optional keys; use :func:`is_credential_optional` to tell whether a
    missing key actually blocks the tool.
    """
    keys = TOOL_CREDENTIALS.get(tool_name, [])
    return {k: bool(os.environ.get(k)) for k in keys}


def inject_credentials(path: Path | None = None) -> None:
    """Load credentials.toml and inject into os.environ. Call at server startup."""
    creds = load_credentials(path=path)
    for _tool, kvs in creds.items():
        for k, v in kvs.items():
            if k not in os.environ:
                os.environ[k] = v


OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
_OPENROUTER_KEY_FILENAME = "openrouter.key"


def openrouter_api_base() -> str:
    """API base for OpenRouter, overridable with ``$OPENROUTER_BASE_URL``."""
    base = os.environ.get("OPENROUTER_BASE_URL") or OPENROUTER_DEFAULT_BASE_URL
    return base.rstrip("/")

# Where the OpenRouter key came from, for status output. Never holds the key.
_openrouter_key_source = ""


def _read_openrouter_key(path: Path) -> str:
    """Return the key in ``path``, accepting a bare key or a KEY=value line."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" in line:
            name, _, value = line.partition("=")
            if name.strip() != OPENROUTER_KEY_ENV:
                continue
            line = value.strip()
        return line.strip().strip("'\"")
    return ""


def load_openrouter_key_file() -> str:
    """Load the OpenRouter key from a file so it never has to be pasted in.

    ``$OPENROUTER_API_KEY_FILE`` wins, then ``<openjarvis home>/openrouter.key``,
    then a key saved earlier in ``cloud-keys.env``. A key already present in the
    environment is left alone. Returns where the key came from (``""`` when
    nothing was found); the key itself is never returned or logged.
    """
    global _openrouter_key_source
    if os.environ.get(OPENROUTER_KEY_ENV):
        if not _openrouter_key_source:
            _openrouter_key_source = "environment"
        return _openrouter_key_source

    candidates: list[Path] = []
    explicit = os.environ.get("OPENROUTER_API_KEY_FILE", "").strip()
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.append(get_config_dir() / _OPENROUTER_KEY_FILENAME)
    candidates.append(get_config_dir() / "cloud-keys.env")

    for path in candidates:
        if not path.is_file():
            continue
        key = _read_openrouter_key(path)
        if not key:
            continue
        os.environ[OPENROUTER_KEY_ENV] = key
        _openrouter_key_source = f"file:{path}"
        logger.info("Loaded OpenRouter key from %s", path)
        return _openrouter_key_source
    return ""


def openrouter_key_source() -> str:
    """Return ``""``, ``"environment"`` or ``"file:<path>"`` (never the key)."""
    return _openrouter_key_source


def verify_openrouter_key(key: str | None = None) -> tuple[bool, str]:
    """Check a key against OpenRouter. Returns ``(ok, reason)``.

    OpenRouter's public model catalogue answers without authentication, so this
    calls an endpoint that requires a real key.
    """
    value = (key or os.environ.get(OPENROUTER_KEY_ENV, "")).strip()
    if not value:
        return False, "no key"
    try:
        import httpx
    except ImportError:  # pragma: no cover - httpx ships with the server extra
        return False, "httpx is not installed"
    try:
        resp = httpx.get(
            f"{openrouter_api_base()}/key",
            headers={"Authorization": f"Bearer {value}"},
            timeout=15.0,
        )
    except Exception as exc:  # noqa: BLE001 - offline or DNS failure
        return False, f"could not reach OpenRouter ({exc.__class__.__name__})"
    if resp.status_code in (401, 403):
        return False, "rejected by OpenRouter"
    if resp.status_code >= 400:
        return False, f"HTTP {resp.status_code}"
    return True, "ok"


def get_tool_credential(
    tool_name: str,
    key: str,
    *,
    path: Path | None = None,
) -> str | None:
    """Read a single credential without polluting ``os.environ``.

    Falls back to ``os.environ`` if the key is not in credentials.toml,
    for backward compatibility with Docker env var workflows.
    """
    creds = load_credentials(path=path)
    tool_creds = creds.get(tool_name, {})
    value = tool_creds.get(key)
    if value is not None:
        return value
    return os.environ.get(key) or None
