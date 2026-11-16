"""``jarvis mobile`` — phone app + API for a free personal assistant.

Serves the installable web app at ``/m`` with a tool-using agent, a persistent
task scheduler, and hold-to-talk voice input. Works over your Wi-Fi with no
account, or against any free OpenAI-compatible provider you connect from the
phone's Settings tab.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess

import click
from rich.console import Console

from openjarvis.cli._banner import print_banner
from openjarvis.core.events import EventBus
from openjarvis.core.paths import get_config_dir

# Server modules (fastapi/uvicorn) are imported inside the command so the rest
# of the CLI still works without the optional ``server`` extra.


def _lan_address() -> str:
    """Best-effort LAN IP so the phone can reach this machine over Wi-Fi."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return str(sock.getsockname()[0])
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def _resolve_engine(console: Console, model_override: str | None):
    """Return (engine, model, engine_name) for the current saved settings."""
    from openjarvis.server.mobile_routes import (
        FALLBACK_FREE_MODELS,
        UnconfiguredEngine,
        load_settings,
    )

    settings = load_settings()
    provider = str(settings.get("provider", "") or "")
    saved_model = model_override or str(settings.get("model", "") or "")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")

    if provider == "openrouter" or (not provider and openrouter_key):
        if openrouter_key:
            try:
                from openjarvis.engine.cloud import CloudEngine

                engine = CloudEngine()
                model = saved_model or FALLBACK_FREE_MODELS[0]
                if engine.health() and engine.can_serve(model):
                    return engine, model, "cloud"
                console.print(
                    "[yellow]Saved OpenRouter model is unavailable; "
                    "reconnect it in the phone app's Settings tab.[/yellow]"
                )
            except Exception as exc:
                console.print(f"[yellow]Cloud engine unavailable: {exc}[/yellow]")
        return UnconfiguredEngine(), "", "unconfigured"

    if provider == "ollama" or not provider:
        try:
            from openjarvis.engine.ollama import OllamaEngine

            engine = OllamaEngine()
            models = engine.list_models()
            if models:
                model = saved_model if saved_model in models else models[0]
                return engine, model, "ollama"
        except Exception:
            pass

    return UnconfiguredEngine(), "", "unconfigured"


def _resolve_speech(config, console: Console):
    """Return a local/cloud speech-to-text backend, if one is installed."""
    try:
        from openjarvis.speech._discovery import get_speech_backend

        backend = get_speech_backend(config)
    except Exception as exc:
        console.print(f"[yellow]Voice unavailable: {exc}[/yellow]")
        return None
    if backend is None:
        console.print(
            "  Voice:  [yellow]off[/yellow] "
            "(install with: uv sync --extra speech)"
        )
    else:
        console.print(f"  Voice:  [cyan]{backend.backend_id}[/cyan]")
    return backend


def _ensure_tls_certificates(lan_ip: str) -> tuple[str, str] | None:
    """Create a self-signed certificate so the phone can use its microphone.

    iOS only allows microphone capture in a secure context, which excludes a
    plain ``http://192.168.x.x`` LAN address. The certificate is generated once
    and reused across restarts.
    """
    openssl = shutil.which("openssl")
    if not openssl:
        return None
    tls_dir = get_config_dir() / "mobile" / "tls"
    tls_dir.mkdir(parents=True, exist_ok=True)
    cert = tls_dir / "cert.pem"
    key = tls_dir / "key.pem"
    if cert.exists() and key.exists():
        return str(cert), str(key)

    config = tls_dir / "openssl.cnf"
    config.write_text(
        "[req]\n"
        "distinguished_name = dn\n"
        "x509_extensions = v3_req\n"
        "prompt = no\n"
        "[dn]\n"
        "CN = Jarvis Local\n"
        "[v3_req]\n"
        f"subjectAltName = IP:{lan_ip}, IP:127.0.0.1, DNS:localhost\n"
        "basicConstraints = CA:TRUE\n",
        encoding="utf-8",
    )
    try:
        subprocess.run(
            [
                openssl,
                "req",
                "-x509",
                "-newkey",
                "rsa:2048",
                "-nodes",
                "-keyout",
                str(key),
                "-out",
                str(cert),
                "-days",
                "825",
                "-config",
                str(config),
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    try:
        key.chmod(0o600)
    except OSError:
        pass
    return str(cert), str(key)


@click.command()
@click.option("--host", default="0.0.0.0", show_default=True, help="Bind address.")
@click.option("--port", default=8787, type=int, show_default=True, help="Port.")
@click.option(
    "-m",
    "--model",
    "model_name",
    default=None,
    help="Override the saved model (e.g. openrouter/openrouter/free).",
)
@click.option(
    "--openrouter-key",
    "openrouter_key",
    default=None,
    help="Connect a free OpenRouter key without using the phone app.",
)
@click.option(
    "--access-key",
    "access_key",
    default=None,
    help="Key the phone must present. Generated and remembered if omitted.",
)
@click.option(
    "--tls",
    "use_tls",
    is_flag=True,
    default=False,
    help="Serve HTTPS with a self-signed certificate (needed for voice on the LAN).",
)
@click.pass_context
def mobile(
    ctx: click.Context,
    host: str,
    port: int,
    model_name: str | None,
    openrouter_key: str | None,
    access_key: str | None,
    use_tls: bool,
) -> None:
    """Start the OpenJarvis phone app (chat, voice, task automation)."""
    print_banner(quiet=(ctx.obj or {}).get("quiet", False))
    console = Console(stderr=True)

    try:
        import uvicorn  # noqa: F401
        from fastapi import FastAPI  # noqa: F401
    except ImportError:
        console.print(
            "[red bold]Server dependencies not installed.[/red bold]\n\n"
            "Install the server extra:\n  [cyan]uv sync --extra server[/cyan]"
        )
        raise SystemExit(1)

    from openjarvis.core.credentials import (
        load_openrouter_key_file,
        verify_openrouter_key,
    )
    from openjarvis.server.mobile_routes import (
        UnconfiguredEngine,
        _save_settings,
        build_mobile_agent,
        load_settings,
    )

    if openrouter_key:
        os.environ["OPENROUTER_API_KEY"] = openrouter_key.strip()

    # A key file lets the phone connect without anybody pasting a secret.
    key_source = load_openrouter_key_file()
    key_note = ""
    if key_source.startswith("file:"):
        valid, reason = verify_openrouter_key()
        key_note = "verified" if valid else reason
        if not valid:
            # A rejected key is worse than none: the phone would show a live
            # assistant whose every message fails. Start unconfigured instead
            # and let Settings report the problem.
            os.environ.pop("OPENROUTER_API_KEY", None)

    settings = load_settings()
    key = (access_key or os.environ.get("OPENJARVIS_API_KEY") or "").strip()
    if not key:
        key = str(settings.get("access_key", "") or "")
    if not key:
        from openjarvis.server.auth_middleware import generate_api_key

        key = generate_api_key()
    settings["access_key"] = key
    _save_settings(settings)
    os.environ["OPENJARVIS_API_KEY"] = key

    bus = EventBus(record_history=False)
    engine, model, engine_name = _resolve_engine(console, model_name)

    from openjarvis.core.config import load_config

    try:
        config = load_config()
    except Exception:
        config = None

    speech_backend = _resolve_speech(config, console) if config is not None else None

    agent = None
    if not isinstance(engine, UnconfiguredEngine):
        agent = build_mobile_agent(engine, model, None, bus=bus)

    from openjarvis.server.app import create_app

    app = create_app(
        engine,
        model,
        agent=agent,
        bus=bus,
        engine_name=engine_name,
        agent_name="mobile" if agent is not None else "",
        config=config,
        speech_backend=speech_backend,
        api_key=key,
    )
    # Start the scheduler now (not lazily) so automations fire even when the
    # phone is closed, and hand it to the agent so "every morning at 8" works
    # from chat as well as from the Tasks tab.
    runtime = getattr(app.state, "mobile_runtime", None)
    if runtime is not None:
        runtime.start()
        if agent is not None:
            runtime._wire_scheduler_tools()

    from openjarvis.server.auth_middleware import check_bind_safety

    check_bind_safety(host, api_key=key)

    lan_ip = _lan_address()
    scheme = "http"
    cert_file = key_file = None
    if use_tls:
        certificates = _ensure_tls_certificates(lan_ip)
        if certificates is None:
            console.print(
                "[yellow bold]--tls requested but no certificate could be created. "
                "Install OpenSSL or keep using http (voice needs https).[/yellow bold]"
            )
        else:
            cert_file, key_file = certificates
            scheme = "https"

    lan_url = f"{scheme}://{lan_ip}:{port}/m"
    model_label = model or "not connected yet (use the app Settings tab)"
    key_line = ""
    if key_source.startswith("file:"):
        path = key_source[len("file:") :]
        key_line = f"  Key:        [cyan]{path}[/cyan] ({key_note})\n"
    elif key_source == "environment":
        key_line = "  Key:        [cyan]from the environment[/cyan]\n"
    console.print(
        "[green]OpenJarvis mobile is ready[/green]\n"
        f"  Phone URL:  [cyan bold]{lan_url}[/cyan bold]\n"
        f"  Access key: [cyan]{key}[/cyan]\n"
        f"{key_line}"
        f"  Model:      [cyan]{model_label}[/cyan]\n\n"
        "On your iPhone: open the URL in Safari, tap Share, then "
        "'Add to Home Screen'.\n"
        "The computer must stay awake; both devices need the same Wi-Fi."
    )
    if use_tls and cert_file:
        console.print(
            "  Cert:       [yellow]self-signed[/yellow] — on the iPhone tap "
            "'Show Details' then 'visit this website' once.\n"
            "  Voice:      hold the microphone button to talk."
        )
    elif speech_backend is not None:
        console.print(
            "  Voice:      [yellow]needs https[/yellow] — restart with "
            "[cyan]--tls[/cyan] to use the microphone over Wi-Fi."
        )

    from openjarvis.server.daemon import run_server

    run_server(
        app,
        host=host,
        port=port,
        log_level="info",
        ssl_certfile=cert_file,
        ssl_keyfile=key_file,
    )


__all__ = ["mobile"]
