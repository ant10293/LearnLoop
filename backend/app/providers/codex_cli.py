"""ChatGPT plan transport through the local Codex CLI.

Analysis calls use `codex exec` in a read-only sandbox. The ChatGPT login,
including Plus, is the plan. Those calls are recorded unpriced. The API key
path stays on OpenAIProvider. The child environment drops API credentials so a
key in the lab process cannot switch the run onto API billing.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from app.providers.claude_code import login_url_from, subscription_env
from app.providers.cli_account import link_status, mask_account
from app.providers.types import (
    AuthenticationError,
    GenerationRequest,
    ProviderError,
    ProviderResponse,
    UnsupportedMode,
    Usage,
    provider_timeout,
)
from app.services.storage import data_path, ensure_data_dirs

TRANSPORTS = ("api", "codex")
_SETTINGS = "codex_cli.json"
_API_ENV = (
    "CODEX_API_KEY",
    "OPENAI_API_BASE",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
)
_SIGNED_OUT = (
    "not logged",
    "please log in",
    "sign in with chatgpt",
    "unauthorized",
    "authentication",
)
_EFFORT = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
_login_proc: asyncio.subprocess.Process | None = None
_login_task: asyncio.Task[None] | None = None
_login_url = ""
_login_lock = asyncio.Lock()


def load_codex() -> dict[str, str]:
    ensure_data_dirs()
    path = data_path(_SETTINGS)
    binary = "codex"
    transport = "api"
    if path.is_file():
        try:
            raw = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError, UnicodeError):
            raw = {}
        if isinstance(raw, dict):
            if isinstance(raw.get("binary"), str) and raw["binary"].strip():
                binary = raw["binary"].strip()
            if raw.get("transport") in TRANSPORTS:
                transport = str(raw["transport"])
    return {"binary": binary, "transport": transport}


def save_codex(binary: str, transport: str) -> dict[str, str]:
    binary = (binary or "").strip() or "codex"
    if transport not in TRANSPORTS:
        raise ValueError("transport must be api or codex")
    if any(char in binary for char in "\n\r\0"):
        raise ValueError("executable path is invalid")
    resolved = resolve_binary(binary)
    if binary != "codex" and resolved is None:
        raise ValueError(f"Codex executable not found: {binary}")
    if transport == "codex" and resolved is None:
        raise ValueError(f"Codex executable not found: {binary}")
    ensure_data_dirs()
    payload = {"binary": binary, "transport": transport}
    data_path(_SETTINGS).write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def resolve_binary(binary: str) -> str | None:
    candidate = Path(binary).expanduser()
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate.resolve())
    return shutil.which(binary)


def child_env() -> dict[str, str]:
    """Use the ChatGPT login stored by `codex login`, not an API key."""
    env = subscription_env()
    for name in _API_ENV:
        env.pop(name, None)
    return env


def _empty_status(binary: str, transport: str, error: str = "") -> dict:
    return link_status(binary=binary, transport=transport, error=error)


def auth_file() -> Path:
    home = os.environ.get("CODEX_HOME", "").strip()
    root = Path(home).expanduser() if home else Path.home() / ".codex"
    return root / "auth.json"


def _jwt_payload(token: str) -> dict | None:
    segment = token.split(".", 2)[1]
    padded = segment + "=" * (-len(segment) % 4)
    try:
        decoded = base64.urlsafe_b64decode(padded)
        payload = json.loads(decoded)
    except (ValueError, json.JSONDecodeError, UnicodeError):
        return None
    return payload if isinstance(payload, dict) else None


def chatgpt_identity(path: Path | None = None) -> tuple[str, str]:
    """Plan and masked email from the local ChatGPT login.

    `codex login status` only names the auth mode. The plan and address are
    claims on the id token the CLI already stored. The token itself is not
    returned or sent anywhere.
    """
    file = path or auth_file()
    try:
        raw = json.loads(file.read_text())
    except (OSError, json.JSONDecodeError, UnicodeError):
        return "", ""
    tokens = raw.get("tokens") if isinstance(raw, dict) else None
    token = str(tokens.get("id_token") or "") if isinstance(tokens, dict) else ""
    if token.count(".") < 2:
        return "", ""
    payload = _jwt_payload(token)
    if payload is None:
        return "", ""
    email = str(payload.get("email") or "")
    auth = payload.get("https://api.openai.com/auth")
    plan = ""
    if isinstance(auth, dict):
        plan = str(auth.get("chatgpt_plan_type") or "").strip()
    return plan, mask_account(email)


def parse_debug_models(stdout: str) -> list[dict]:
    """Read `codex debug models`. Hidden rows are not offered in the picker."""
    start = stdout.find("{")
    if start < 0:
        return []
    try:
        payload = json.loads(stdout[start:])
    except json.JSONDecodeError:
        return []
    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        return []
    records: list[dict] = []
    seen: set[str] = set()
    for item in models:
        if not isinstance(item, dict):
            continue
        if str(item.get("visibility") or "list") != "list":
            continue
        model_id = str(item.get("slug") or item.get("id") or "").strip()
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        modalities = item.get("input_modalities") or []
        if not isinstance(modalities, list):
            modalities = []
        efforts: list[str] = []
        for level in item.get("supported_reasoning_levels") or []:
            if isinstance(level, dict) and level.get("effort"):
                efforts.append(str(level["effort"]))
            elif isinstance(level, str) and level:
                efforts.append(level)
        context = item.get("context_window")
        records.append(
            {
                "model_id": model_id,
                "label": str(item.get("display_name") or "").strip(),
                "image": "image" in modalities,
                "native_video": "video" in modalities,
                "context_tokens": context if isinstance(context, int) else None,
                "reasoning_efforts": efforts,
                "supports_temperature": False,
            }
        )
    return records


def model_records_sync() -> list[dict]:
    """Models the signed-in ChatGPT account can run. Empty when Codex is not the transport."""
    settings = load_codex()
    if settings["transport"] != "codex":
        return []
    resolved = resolve_binary(settings["binary"])
    if resolved is None:
        return []
    try:
        completed = _run_text([resolved, "debug", "models"], 30)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if completed.returncode != 0:
        return []
    return parse_debug_models(completed.stdout or "")


def _run_text(argv: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=child_env(),
    )


def auth_method_from(text: str) -> str:
    """Read the one-line `codex login status` report.

    The CLI prints one of a few fixed sentences. An API-key login is not the
    ChatGPT plan, so it is reported separately and is not treated as linked.
    """
    lowered = " ".join((text or "").lower().split())
    if "not logged" in lowered:
        return ""
    if "api key" in lowered:
        return "api_key"
    if "agent identity" in lowered:
        return "agent_identity"
    if "chatgpt" in lowered:
        return "chatgpt"
    return ""


def probe_sync() -> dict:
    settings = load_codex()
    binary = settings["binary"]
    transport = settings["transport"]
    resolved = resolve_binary(binary)
    if resolved is None:
        return _empty_status(binary, transport, f"Codex executable not found: {binary}")
    try:
        version = _run_text([resolved, "--version"], 15)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return _empty_status(binary, transport, str(exc))
    version_text = (version.stdout or version.stderr or "").strip().splitlines()
    version_line = version_text[0][:80] if version_text else ""
    if version.returncode != 0:
        detail = (version.stderr or version.stdout or "codex --version failed").strip()
        return _empty_status(binary, transport, detail[:500])
    try:
        auth = _run_text([resolved, "login", "status"], 20)
    except (OSError, subprocess.TimeoutExpired) as exc:
        status = _empty_status(binary, transport, str(exc))
        status["installed"] = True
        status["version"] = version_line
        return status
    text = "\n".join(part for part in (auth.stdout, auth.stderr) if part)
    method = auth_method_from(text)
    logged_in = auth.returncode == 0 and method == "chatgpt"
    error = ""
    if method == "api_key":
        error = "Codex CLI is signed in with an API key. Link ChatGPT to use the plan."
    elif method == "agent_identity":
        error = "Codex CLI is signed in with Agent Identity. Link ChatGPT to use the plan."
    elif not logged_in:
        error = "Codex CLI is installed and not signed in."
    subscription, account = chatgpt_identity() if logged_in else ("", "")
    return link_status(
        binary=binary,
        transport=transport,
        installed=True,
        version=version_line,
        logged_in=logged_in,
        auth_method=method if logged_in else "",
        subscription=subscription,
        account=account,
        error=error,
    )


async def probe() -> dict:
    return await asyncio.to_thread(probe_sync)


async def stop_login() -> None:
    global _login_proc, _login_task, _login_url
    proc = _login_proc
    task = _login_task
    _login_proc = None
    _login_task = None
    _login_url = ""
    if proc is not None and proc.returncode is None:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        await proc.wait()
    if task is not None and not task.done():
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def _collect_login(proc: asyncio.subprocess.Process, bucket: bytearray) -> None:
    async def pump(stream: asyncio.StreamReader | None) -> None:
        if stream is None:
            return
        while True:
            chunk = await stream.read(4096)
            if not chunk:
                return
            bucket.extend(chunk)

    await asyncio.gather(pump(proc.stdout), pump(proc.stderr))
    if proc.returncode is None:
        await proc.wait()


async def begin_login(force: bool = False) -> dict:
    """Start `codex login` and keep the local callback server alive."""
    global _login_proc, _login_task, _login_url
    status = await probe()
    if not status["installed"]:
        raise ProviderError(status["error"] or "Codex executable not found")
    if status["transport"] != "codex":
        raise ProviderError("Switch OpenAI runs to the ChatGPT plan before linking.")
    if status["logged_in"] and not force:
        return status
    async with _login_lock:
        if not force and _login_proc is not None and _login_proc.returncode is None and _login_url:
            status["login_started"] = True
            status["login_url"] = _login_url
            status["logged_in"] = False
            status["error"] = ""
            return status
        await stop_login()
        resolved = resolve_binary(status["binary"])
        if resolved is None:
            raise ProviderError(status["error"] or "Codex executable not found")
        env = child_env()
        if force:
            try:
                await asyncio.to_thread(_run_text, [resolved, "logout"], 20)
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise ProviderError(str(exc)) from exc
        try:
            proc = await asyncio.create_subprocess_exec(
                resolved,
                "login",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        except OSError as exc:
            raise ProviderError(str(exc)) from exc
        bucket = bytearray()
        _login_proc = proc
        _login_task = asyncio.create_task(_collect_login(proc, bucket))
        deadline = time.monotonic() + 15
        url = ""
        while time.monotonic() < deadline:
            url = login_url_from(bucket.decode(errors="replace"))
            if url:
                break
            if proc.returncode is not None:
                await asyncio.sleep(0.05)
                url = login_url_from(bucket.decode(errors="replace"))
                break
            await asyncio.sleep(0.05)
        _login_url = url
        if proc.returncode is not None and not url:
            detail = bucket.decode(errors="replace").strip()
            raise ProviderError(
                detail[-500:] or "Codex login exited before a sign-in link was shown"
            )
        status["logged_in"] = False
        status["login_started"] = True
        status["login_url"] = url
        status["error"] = "" if url else "Codex did not show a sign-in link."
        return status


def build_argv(
    binary: str,
    model_id: str,
    workspace: Path,
    schema_path: Path,
    last_message: Path,
    images: list[Path],
    effort: str | None = None,
) -> list[str]:
    argv = [
        binary,
        "exec",
        "--json",
        "--ephemeral",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--color",
        "never",
        "--model",
        model_id,
        "--cd",
        str(workspace),
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(last_message),
    ]
    if effort:
        argv.extend(["--config", f"model_reasoning_effort={effort}"])
    for image in images:
        argv.extend(["--image", str(image)])
    argv.append("-")
    return argv


_SCHEMA_STRIP = {
    "title",
    "default",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "examples",
}


def strict_output_schema(schema: dict) -> dict:
    """Codex sends this schema as a strict response format.

    Every object needs ``additionalProperties: false`` and every property listed
    in ``required``. Numeric bounds and defaults are not accepted. A free-form
    map cannot be expressed that way, so it becomes an empty object.
    """
    converted = _strict_node(schema)
    return converted if isinstance(converted, dict) else {"type": "object", "additionalProperties": False}


def _strict_node(node: object) -> object:
    if isinstance(node, list):
        return [_strict_node(item) for item in node]
    if not isinstance(node, dict):
        return node
    if "$ref" in node:
        return {"$ref": node["$ref"]}
    cleaned: dict = {}
    for key, value in node.items():
        if key in _SCHEMA_STRIP:
            continue
        cleaned[key] = _strict_node(value)
    is_object = cleaned.get("type") == "object" or "properties" in cleaned or "additionalProperties" in cleaned
    if not is_object:
        return cleaned
    properties = cleaned.get("properties")
    if not isinstance(properties, dict) or cleaned.get("additionalProperties") not in (None, False):
        properties = {}
    cleaned["type"] = "object"
    cleaned["properties"] = properties
    cleaned["required"] = list(properties)
    cleaned["additionalProperties"] = False
    return cleaned


def failure_message(stdout: str, stderr: str) -> str:
    """Prefer the JSON turn error over the CLI's stdin progress line."""
    failure = _failure(parse_events(stdout))
    if failure:
        return failure[:2000]
    detail = (stderr or "").strip()
    if detail and not _status_line(detail):
        return detail[:2000]
    text = (stdout or "").strip()
    if text and not text.startswith("{"):
        return text[:2000]
    return detail[:2000] or "Codex CLI failed"


def _status_line(text: str) -> bool:
    lowered = " ".join(text.lower().split())
    return lowered.startswith("reading prompt from stdin")


def build_prompt(request: GenerationRequest, frames: list[tuple[float, str]]) -> str:
    lines = [
        request.system_prompt.strip(),
        "",
        "Return one JSON object and no other text. Do not create, edit, or delete files.",
        "The object must match the JSON schema supplied for this run.",
        "",
    ]
    if frames:
        lines.append(
            "Images are attached in this order. Each number is seconds from the start of the clip."
        )
        for timestamp, name in frames:
            lines.append(f"- {timestamp:.3f}s {name}")
    else:
        lines.append("This call has no images.")
    lines.extend(["", request.user_prompt.strip(), ""])
    return "\n".join(lines)


def parse_events(stdout: str) -> list[dict]:
    events: list[dict] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def _thread_id(events: list[dict]) -> str:
    for event in events:
        if event.get("type") == "thread.started":
            return str(event.get("thread_id") or "")
    return ""


def _usage(events: list[dict]) -> dict:
    found: dict = {}
    for event in events:
        if event.get("type") == "turn.completed" and isinstance(event.get("usage"), dict):
            found = event["usage"]
    return found


def _failure(events: list[dict]) -> str:
    for event in reversed(events):
        kind = str(event.get("type") or "")
        if kind == "turn.failed":
            error = event.get("error")
            if isinstance(error, dict):
                return str(error.get("message") or error.get("detail") or "Codex turn failed")
            if error:
                return str(error)
            return "Codex turn failed"
        if kind == "error":
            return str(event.get("message") or event.get("error") or "Codex failed")
    return ""


def _agent_text(events: list[dict]) -> str:
    text = ""
    for event in events:
        if event.get("type") not in {"item.completed", "item.started"}:
            continue
        item = event.get("item")
        if not isinstance(item, dict) or item.get("type") != "agent_message":
            continue
        message = item.get("text")
        if isinstance(message, str) and message.strip():
            text = message
    return text


def _signed_out(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in _SIGNED_OUT)


def response_from_exec(
    stdout: str,
    last_message: str,
    request: GenerationRequest,
    latency_ms: int,
    version: str,
) -> ProviderResponse:
    events = parse_events(stdout)
    text = last_message.strip() or _agent_text(events)
    failure = _failure(events)
    if failure and not text:
        message = failure[:2000]
        if _signed_out(message):
            raise AuthenticationError(message)
        raise ProviderError(message)
    if not text:
        raise ProviderError("Codex CLI returned an empty analysis")
    usage = _usage(events)
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    thread_id = _thread_id(events)
    return ProviderResponse(
        text=text,
        raw={
            "thread_id": thread_id,
            "model": request.model_id,
            "usage": usage,
            "billing": "chatgpt_subscription",
        },
        usage=Usage(input_tokens, output_tokens, 0.0, False),
        request_id=thread_id,
        latency_ms=latency_ms,
        model_id=request.model_id,
        sdk_version=version,
    )


def _transport(request: GenerationRequest) -> str:
    raw = ""
    if isinstance(request.metadata, dict):
        raw = str(request.metadata.get("transport") or "")
    if raw in TRANSPORTS:
        return raw
    return load_codex()["transport"]


async def _execute(
    argv: list[str],
    env: dict[str, str],
    cwd: str,
    stdin: bytes,
    timeout: float,
) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
        cwd=cwd,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(stdin), timeout)
    except TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        await proc.wait()
        raise ProviderError(f"Codex CLI timed out after {timeout:.0f}s") from None
    return proc.returncode or 0, out.decode(errors="replace"), err.decode(errors="replace")


async def complete(request: GenerationRequest) -> ProviderResponse:
    settings = load_codex()
    resolved = resolve_binary(settings["binary"])
    if resolved is None:
        raise AuthenticationError(
            f"Codex executable not found: {settings['binary']}. "
            "Set the executable on the Providers page."
        )
    if _transport(request) != "codex":
        raise ProviderError("Codex completion was requested without the ChatGPT transport.")
    if request.modality == "native_video" or (request.video_path and not request.frames):
        raise UnsupportedMode(
            "Codex is asked to read attached frames. Native video was not substituted for a video file."
        )
    if request.temperature is not None:
        raise UnsupportedMode(
            "Codex does not accept temperature. Leave it unset, or switch OpenAI runs to the API key."
        )
    if any(char in request.model_id for char in "\n\r\0"):
        raise ProviderError("model id is invalid")
    effort = (request.reasoning_effort or "").strip()
    if effort and not _EFFORT.match(effort):
        raise ProviderError("reasoning effort is invalid")
    started = time.perf_counter()
    env = child_env()
    version = ""
    try:
        checked = await asyncio.to_thread(_run_text, [resolved, "--version"], 15)
        if checked.returncode == 0 and (checked.stdout or checked.stderr).strip():
            version = (checked.stdout or checked.stderr).strip().splitlines()[0][:80]
    except (OSError, subprocess.TimeoutExpired):
        version = ""
    schema = request.json_schema or {"type": "object"}
    with tempfile.TemporaryDirectory(prefix="codex-cli-") as directory:
        root = Path(directory)
        frame_dir = root / "frames"
        frame_dir.mkdir()
        listed: list[tuple[float, str]] = []
        images: list[Path] = []
        for index, frame in enumerate(request.frames):
            if not frame.path.is_file():
                raise ProviderError(f"frame file is missing: {frame.path.name}")
            name = f"frames/{index:03d}_{frame.timestamp:.3f}{frame.path.suffix.lower() or '.jpg'}"
            target = root / name
            shutil.copyfile(frame.path, target)
            listed.append((frame.timestamp, name))
            images.append(target)
        schema_path = root / "schema.json"
        schema_path.write_text(json.dumps(strict_output_schema(schema)), encoding="utf-8")
        last_path = root / "last.txt"
        argv = build_argv(
            resolved,
            request.model_id,
            root,
            schema_path,
            last_path,
            images,
            effort or None,
        )
        try:
            code, out, err = await _execute(
                argv,
                env,
                directory,
                build_prompt(request, listed).encode(),
                provider_timeout(request.timeout_s),
            )
        except OSError as exc:
            raise ProviderError(str(exc)) from exc
        last_message = last_path.read_text(encoding="utf-8") if last_path.is_file() else ""
    latency_ms = int((time.perf_counter() - started) * 1000)
    if code != 0:
        detail = "\n".join(part for part in (err, out) if part)
        if _signed_out(detail):
            raise AuthenticationError((err or out or detail).strip()[-800:])
        try:
            return response_from_exec(out, last_message, request, latency_ms, version)
        except ProviderError as exc:
            if "empty analysis" not in str(exc):
                raise
            raise ProviderError(failure_message(out, err)) from exc
    return response_from_exec(out, last_message, request, latency_ms, version)
