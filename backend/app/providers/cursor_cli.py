"""Cursor CLI transport.

Analysis calls go through the local `agent` executable. Ask mode can read the
frames and is not given permission to edit files. When that answer is not one
JSON object, a second call may edit answer.json in the temporary workspace
until it parses. Linked-account usage is the Cursor plan, so those calls are
recorded unpriced. An API key is a separate transport and is passed only in
the child environment.
"""

from __future__ import annotations

import asyncio
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

TRANSPORTS = ("login", "api_key")
_SETTINGS = "cursor_cli.json"
_SIGNED_OUT = (
    "not authenticated",
    "not logged",
    "no authentication",
    "please log in",
    "run agent login",
    "run `agent login`",
)
_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,80}$")
_MODEL_LINE = re.compile(
    r"^(?P<id>[A-Za-z0-9][A-Za-z0-9._:-]{0,80})(?:\s+-\s+(?P<label>.+))?$"
)
_MODEL_HEADER = {"model", "models", "id", "available"}
_MODEL_CACHE_S = 60.0
_model_cache: dict[str, tuple[float, list[tuple[str, str]]]] = {}
# stream-json writes thinking deltas, tool calls, and (with partial output) text
# as they happen. Those bytes reset this. A quiet CLI is killed; one that keeps
# streaming is stopped only by provider_timeout.
IDLE_TIMEOUT_S = 120
_OUTPUT_POLL_S = 0.25
_PROMPT = (
    "Read instructions.md and every frame file it names. "
    "Return only the JSON object requested there."
)
_PROMPT_VIDEO = (
    "Read instructions.md and watch the video file it names from the first moment through the last. "
    "Return only the JSON object requested there."
)
_PROMPT_VIDEO_AND_FRAMES = (
    "Read instructions.md. Watch the video file from the first moment through the last, "
    "and read every frame file it names. "
    "Return only the JSON object requested there."
)
_ANSWER = "answer.json"
_REPAIR_PROMPT = (
    "answer.json is not one JSON object. Edit answer.json in place until it is. "
    "Use the schema in instructions.md. Do not edit, delete, or add any other file."
)
_login_proc: asyncio.subprocess.Process | None = None
_login_task: asyncio.Task[None] | None = None
_login_url = ""
_login_lock = asyncio.Lock()


def load_cursor_cli() -> dict[str, str]:
    ensure_data_dirs()
    path = data_path(_SETTINGS)
    binary = "agent"
    transport = "login"
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


def save_cursor_cli(binary: str, transport: str) -> dict[str, str]:
    binary = (binary or "").strip() or "agent"
    if transport not in TRANSPORTS:
        raise ValueError("transport must be login or api_key")
    if any(char in binary for char in "\n\r\0"):
        raise ValueError("executable path is invalid")
    resolved = resolve_binary(binary)
    if resolved is None:
        raise ValueError(f"Cursor CLI executable not found: {binary}")
    ensure_data_dirs()
    payload = {"binary": binary, "transport": transport}
    data_path(_SETTINGS).write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def resolve_binary(binary: str) -> str | None:
    candidate = Path(binary).expanduser()
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate.resolve())
    return shutil.which(binary)


def child_env(transport: str, secret: str = "") -> dict[str, str]:
    """Use the plan login, or the stored key, and drop the editor sandbox proxy."""
    env = subscription_env()
    env.pop("CURSOR_API_KEY", None)
    if transport == "api_key" and secret:
        env["CURSOR_API_KEY"] = secret
    return env


def _empty_status(binary: str, transport: str, error: str = "") -> dict:
    return link_status(binary=binary, transport=transport, error=error)


def _run_text(argv: list[str], env: dict[str, str], timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


def _account_from(text: str) -> str:
    match = re.search(r"[\w.+-]+@[\w.-]+\.\w+", text)
    if not match:
        return ""
    return mask_account(match.group(0))


def _subscription_tier(stdout: str) -> str:
    """Read the plan name from `agent about --format json`."""
    try:
        payload = json.loads(stdout or "")
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        return str(payload.get("subscriptionTier") or "").strip()
    for line in (stdout or "").splitlines():
        if line.lower().startswith("subscription tier"):
            _, _, value = line.partition("Tier")
            return value.strip()
    return ""


def _signed_out(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in _SIGNED_OUT)


def probe_sync(secret: str = "") -> dict:
    settings = load_cursor_cli()
    binary = settings["binary"]
    transport = settings["transport"]
    resolved = resolve_binary(binary)
    if resolved is None:
        return _empty_status(binary, transport, f"Cursor CLI executable not found: {binary}")
    if transport == "api_key" and not secret:
        status = _empty_status(binary, transport, "No API key is configured for Cursor.")
        status["installed"] = True
        return status
    env = child_env(transport, secret)
    try:
        version = _run_text([resolved, "--version"], env, 15)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return _empty_status(binary, transport, str(exc))
    version_text = (version.stdout or version.stderr or "").strip().splitlines()
    version_line = version_text[0][:80] if version_text else ""
    if version.returncode != 0:
        detail = (version.stderr or version.stdout or "agent --version failed").strip()
        return _empty_status(binary, transport, detail[:500])
    try:
        auth = _run_text([resolved, "status"], env, 20)
    except (OSError, subprocess.TimeoutExpired) as exc:
        status = _empty_status(binary, transport, str(exc))
        status["installed"] = True
        status["version"] = version_line
        return status
    text = "\n".join(part for part in (auth.stdout, auth.stderr) if part)
    logged_in = auth.returncode == 0 and not _signed_out(text)
    error = ""
    if not logged_in:
        error = "Cursor CLI is installed and not signed in."
        if transport == "api_key":
            error = "Cursor CLI rejected the API key."
    subscription = ""
    if logged_in:
        try:
            about = _run_text([resolved, "about", "--format", "json"], env, 20)
        except (OSError, subprocess.TimeoutExpired):
            about = None
        if about is not None and about.returncode == 0:
            subscription = _subscription_tier(about.stdout or "")
    return link_status(
        binary=binary,
        transport=transport,
        installed=True,
        version=version_line,
        logged_in=logged_in,
        auth_method=transport if logged_in else "",
        subscription=subscription,
        account=_account_from(text) if logged_in else "",
        error=error,
    )


async def probe(secret: str = "") -> dict:
    return await asyncio.to_thread(probe_sync, secret)


def _clean_label(label: str) -> str:
    return re.sub(r"\s+", " ", label).strip()


def parse_model_records(stdout: str) -> list[tuple[str, str]]:
    """Read `agent --list-models` output.

    Current builds print ``id - Display name``. A bare id, or a JSON list, is
    accepted as well. The header line is not a model.
    """
    text = stdout.strip()
    if not text:
        return []
    if text[0] in "[{":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, list):
            return _records_from_json(payload)
        if isinstance(payload, dict):
            raw = payload.get("models") or payload.get("data") or []
            if isinstance(raw, list):
                return _records_from_json(raw)
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line in text.splitlines():
        match = _MODEL_LINE.match(line.strip())
        if not match:
            continue
        token = match.group("id")
        label = _clean_label(match.group("label") or "")
        if not label and token.lower() in _MODEL_HEADER:
            continue
        if token in seen:
            continue
        seen.add(token)
        found.append((token, label))
    return found


def parse_model_list(stdout: str) -> list[str]:
    return [model_id for model_id, _label in parse_model_records(stdout)]


def _records_from_json(raw: list) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for item in raw:
        label = ""
        if isinstance(item, str):
            token = item.strip()
        elif isinstance(item, dict):
            token = str(item.get("id") or item.get("model") or "").strip()
            label = str(item.get("label") or item.get("name") or item.get("display_name") or "")
        else:
            continue
        if token and _MODEL_ID.match(token) and token not in seen:
            seen.add(token)
            found.append((token, _clean_label(label)))
    return found


def _fetch_model_records(secret: str = "") -> list[tuple[str, str]]:
    settings = load_cursor_cli()
    resolved = resolve_binary(settings["binary"])
    if resolved is None:
        raise ProviderError(f"Cursor CLI executable not found: {settings['binary']}")
    if settings["transport"] == "api_key" and not secret:
        raise AuthenticationError("No API key is configured for Cursor.")
    cache_key = f"{resolved}|{settings['transport']}|{secret[-4:] if secret else ''}"
    cached = _model_cache.get(cache_key)
    now = time.monotonic()
    if cached is not None and cached[0] > now:
        return cached[1]
    try:
        completed = _run_text(
            [resolved, "--list-models"],
            child_env(settings["transport"], secret),
            30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProviderError(str(exc)) from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "agent --list-models failed").strip()
        if _signed_out(detail):
            raise AuthenticationError(detail[:500])
        raise ProviderError(detail[:500])
    records = parse_model_records(completed.stdout)
    _model_cache[cache_key] = (now + _MODEL_CACHE_S, records)
    return records


def list_models_sync(secret: str = "") -> list[str]:
    return [model_id for model_id, _label in _fetch_model_records(secret)]


def account_model_options(catalog: list[dict], secret: str = "") -> list[dict]:
    """Replace the static catalog with the models this Cursor account can run.

    The catalog object is returned unchanged when the CLI is missing or not signed in.
    """
    from app.providers.model_list import overlay_options

    try:
        records = _fetch_model_records(secret)
    except (ProviderError, OSError):
        return catalog
    if not records:
        return catalog
    return overlay_options(
        "cursor",
        catalog,
        [{"model_id": model_id, "label": label} for model_id, label in records],
    )


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
    """Start `agent login` and keep it alive until the browser finishes."""
    global _login_proc, _login_task, _login_url
    status = await probe()
    if not status["installed"]:
        raise ProviderError(status["error"] or "Cursor CLI executable not found")
    if status["transport"] != "login":
        raise ProviderError("Switch Cursor runs to the Cursor account before linking.")
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
            raise ProviderError(status["error"] or "Cursor CLI executable not found")
        env = child_env("login")
        env["NO_OPEN_BROWSER"] = "1"
        if force:
            try:
                await asyncio.to_thread(_run_text, [resolved, "logout"], env, 20)
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
            raise ProviderError(detail[-500:] or "Cursor CLI login exited before a sign-in link was shown")
        status["logged_in"] = False
        status["login_started"] = True
        status["login_url"] = url
        status["error"] = "" if url else "Cursor CLI did not show a sign-in link."
        return status


def build_argv(binary: str, model_id: str, workspace: Path, *, video: bool = False, frames: bool = False) -> list[str]:
    if video and frames:
        prompt = _PROMPT_VIDEO_AND_FRAMES
    elif video:
        prompt = _PROMPT_VIDEO
    else:
        prompt = _PROMPT
    return [
        binary,
        "--print",
        "--output-format",
        "stream-json",
        "--stream-partial-output",
        "--mode",
        "ask",
        "--sandbox",
        "enabled",
        "--trust",
        "--workspace",
        str(workspace),
        "--model",
        model_id,
        prompt,
    ]


def build_repair_argv(binary: str, model_id: str, workspace: Path, error: str) -> list[str]:
    """Edit pass. Ask mode cannot write, so this call is allowed to change answer.json."""
    detail = " ".join((error or "").split())[:400]
    prompt = _REPAIR_PROMPT if not detail else f"{_REPAIR_PROMPT} Parser error: {detail}"
    return [
        binary,
        "--print",
        "--output-format",
        "stream-json",
        "--stream-partial-output",
        "--sandbox",
        "enabled",
        "--trust",
        "--workspace",
        str(workspace),
        "--model",
        model_id,
        prompt,
    ]


def write_instructions(
    directory: Path,
    request: GenerationRequest,
    frames: list[tuple[float, str]],
    video_name: str = "",
) -> None:
    schema = json.dumps(request.json_schema or {"type": "object"}, indent=2)
    lines = [
        request.system_prompt.strip(),
        "",
        "Return one JSON object and no other text. Do not create, edit, or delete files.",
        "The object must match this JSON schema:",
        schema,
        "",
    ]
    if video_name:
        lines.append(f"Video file: {video_name}")
        lines.append("This file is the whole recording. Watch it from the first moment through the last.")
    if frames:
        lines.append("Frames. Each path is one moment of that recording. Together they run from the start through the end.")
        for timestamp, name in frames:
            lines.append(f"- {timestamp:.3f}s {name}")
    elif not video_name:
        lines.append("This call has no frames.")
    lines.extend(["", request.user_prompt.strip(), ""])
    (directory / "instructions.md").write_text("\n".join(lines), encoding="utf-8")


def stream_answer(stdout: str) -> str:
    """Read the answer from a stream-json transcript.

    A parsed result event wins. When that final line is truncated, the assistant
    text deltas are the same answer and are joined instead.
    """
    result = ""
    deltas: list[str] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "result" and isinstance(event.get("result"), str):
            result = event["result"]
            continue
        if event.get("type") != "assistant":
            continue
        message = event.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                deltas.append(part["text"])
    if result.strip():
        return result
    return "".join(deltas)


def _json_failure(text: str) -> str | None:
    from app.pipeline.parsing import ParseFailure, extract_json_object

    try:
        extract_json_object(text)
    except ParseFailure as exc:
        return str(exc)
    return None


def _snapshot_workspace(root: Path) -> dict[str, bytes]:
    saved: dict[str, bytes] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.name != _ANSWER:
            saved[path.relative_to(root).as_posix()] = path.read_bytes()
    return saved


def _restore_workspace(root: Path, saved: dict[str, bytes]) -> None:
    for rel, data in saved.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file() or path.read_bytes() != data:
            path.write_bytes(data)
    for path in list(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel not in saved and path.name != _ANSWER:
            path.unlink()


def _payload_with_text(stdout: str, text: str) -> dict:
    try:
        payload = dict(parse_result(stdout))
    except ProviderError:
        payload = {"type": "result", "subtype": "success", "is_error": False}
    payload["result"] = text
    payload["is_error"] = False
    payload["subtype"] = payload.get("subtype") if payload.get("subtype") not in {"error", "error_during_execution"} else "success"
    if not payload.get("subtype"):
        payload["subtype"] = "success"
    return payload


def parse_result(stdout: str) -> dict:
    text = stdout.strip()
    if not text:
        raise ProviderError("Cursor CLI returned no output")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
        for line in reversed(text.splitlines()):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and (candidate.get("type") == "result" or "result" in candidate):
                payload = candidate
                break
        if payload is None:
            raise ProviderError(text[-800:])
    if not isinstance(payload, dict):
        raise ProviderError("Cursor CLI returned an unexpected payload")
    return payload


def response_from_payload(
    payload: dict,
    request: GenerationRequest,
    latency_ms: int,
    version: str,
) -> ProviderResponse:
    if payload.get("is_error") or payload.get("subtype") in {"error", "error_during_execution"}:
        detail = str(payload.get("result") or payload.get("error") or "Cursor CLI failed")
        raise ProviderError(detail[:800])
    text = payload.get("result")
    if not isinstance(text, str) or not text.strip():
        raise ProviderError("Cursor CLI returned an empty result")
    return ProviderResponse(
        text=text,
        raw={
            "session_id": str(payload.get("session_id") or ""),
            "request_id": str(payload.get("request_id") or ""),
            "model": request.model_id,
            "duration_ms": payload.get("duration_ms"),
            "billing": "cursor_subscription" if _transport(request) == "login" else "cursor_api_key",
        },
        usage=Usage(0, 0, 0.0, False),
        request_id=str(payload.get("request_id") or payload.get("session_id") or ""),
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
    return load_cursor_cli()["transport"]


async def _execute(
    argv: list[str],
    env: dict[str, str],
    cwd: str,
    timeout: float,
    idle_timeout: float = IDLE_TIMEOUT_S,
) -> tuple[int, str, str]:
    """Run the agent until it exits, goes quiet, or hits the hard ceiling.

    ``timeout`` is wall-clock time from process start. ``idle_timeout`` resets
    whenever stdout or stderr receives bytes. Thinking deltas and tool calls
    count, so a long analysis stays up. A slow trickle still dies at ``timeout``.
    """
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
        cwd=cwd,
    )
    assert proc.stdout is not None and proc.stderr is not None
    out_parts: list[bytes] = []
    err_parts: list[bytes] = []
    started = time.monotonic()
    last_output = started

    async def copy_output(stream: asyncio.StreamReader, parts: list[bytes]) -> None:
        nonlocal last_output
        while True:
            chunk = await stream.read(65536)
            if not chunk:
                return
            parts.append(chunk)
            last_output = time.monotonic()

    copying = [
        asyncio.create_task(copy_output(proc.stdout, out_parts)),
        asyncio.create_task(copy_output(proc.stderr, err_parts)),
    ]
    try:
        while True:
            now = time.monotonic()
            if now - started >= timeout:
                raise ProviderError(f"Cursor CLI timed out after {timeout:.0f}s")
            if now - last_output >= idle_timeout:
                raise ProviderError(f"Cursor CLI produced no output for {idle_timeout:.0f}s")
            for task in copying:
                if task.done() and not task.cancelled():
                    exc = task.exception()
                    if exc is not None:
                        raise exc
            if all(task.done() for task in copying):
                remaining_run = timeout - (time.monotonic() - started)
                try:
                    await asyncio.wait_for(proc.wait(), timeout=max(remaining_run, 0))
                except TimeoutError:
                    raise ProviderError(f"Cursor CLI timed out after {timeout:.0f}s") from None
                break
            remaining = min(
                timeout - (now - started),
                idle_timeout - (now - last_output),
                _OUTPUT_POLL_S,
            )
            await asyncio.wait(
                set(copying),
                timeout=max(remaining, 0),
                return_when=asyncio.ALL_COMPLETED,
            )
    finally:
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()
        for task in copying:
            if not task.done():
                task.cancel()
        await asyncio.gather(*copying, return_exceptions=True)
    return proc.returncode or 0, b"".join(out_parts).decode(errors="replace"), b"".join(err_parts).decode(errors="replace")


async def complete(request: GenerationRequest, secret: str = "") -> ProviderResponse:
    settings = load_cursor_cli()
    resolved = resolve_binary(settings["binary"])
    if resolved is None:
        raise AuthenticationError(
            f"Cursor CLI executable not found: {settings['binary']}. "
            "Set the executable on the Providers page."
        )
    transport = _transport(request)
    sending_video = request.modality == "native_video" or (request.video_path is not None and not request.frames)
    if request.temperature is not None:
        raise UnsupportedMode("The Cursor CLI does not accept temperature. Leave it unset.")
    if request.reasoning_effort:
        raise UnsupportedMode("The Cursor CLI does not accept a reasoning effort. Leave it unset.")
    if any(char in request.model_id for char in "\n\r\0"):
        raise ProviderError("model id is invalid")
    if transport == "api_key" and not secret:
        raise AuthenticationError("No API key is configured for Cursor.")
    if transport == "api_key" and any(char in secret for char in "\n\r\0"):
        raise ProviderError("API key is invalid")
    started = time.perf_counter()
    env = child_env(transport, secret)
    version = ""
    try:
        checked = await asyncio.to_thread(_run_text, [resolved, "--version"], env, 15)
        if checked.returncode == 0 and checked.stdout.strip():
            version = checked.stdout.strip().splitlines()[0][:80]
    except (OSError, subprocess.TimeoutExpired):
        version = ""
    with tempfile.TemporaryDirectory(prefix="cursor-cli-") as directory:
        root = Path(directory)
        frame_dir = root / "frames"
        frame_dir.mkdir()
        listed: list[tuple[float, str]] = []
        video_name = ""
        if sending_video:
            from app.providers.catalog import cursor_accepts_video

            if not cursor_accepts_video(request.model_id):
                raise UnsupportedMode(
                    f"{request.model_id} does not support native video. "
                    "The Cursor CLI was not given frames instead."
                )
            if request.video_path is None or not request.video_path.is_file():
                raise ProviderError("native video modality requires a video file")
            suffix = request.video_path.suffix.lower()
            if suffix not in {".mp4", ".mov", ".webm", ".mkv"}:
                suffix = ".mp4"
            video_name = f"clip{suffix}"
            shutil.copyfile(request.video_path, root / video_name)
        for index, frame in enumerate(request.frames):
            if not frame.path.is_file():
                raise ProviderError(f"frame file is missing: {frame.path.name}")
            name = f"frames/{index:03d}_{frame.timestamp:.3f}{frame.path.suffix.lower() or '.jpg'}"
            target = root / name
            shutil.copyfile(frame.path, target)
            listed.append((frame.timestamp, name))
        write_instructions(root, request, listed, video_name)
        argv = build_argv(resolved, request.model_id, root, video=bool(video_name), frames=bool(listed))
        try:
            code, out, err = await _execute(argv, env, directory, provider_timeout(request.timeout_s))
        except OSError as exc:
            raise ProviderError(str(exc)) from exc
        if code != 0 and _signed_out(err or out):
            raise AuthenticationError((err or out).strip()[-800:])
        text = stream_answer(out)
        failure = _json_failure(text) if request.json_schema is not None else None
        if failure and text.strip():
            answer = root / _ANSWER
            answer.write_text(text, encoding="utf-8")
            saved = _snapshot_workspace(root)
            repair_out = ""
            try:
                _repair_code, repair_out, _repair_err = await _execute(
                    build_repair_argv(resolved, request.model_id, root, failure),
                    env,
                    directory,
                    provider_timeout(request.timeout_s),
                )
            except (OSError, ProviderError):
                repair_out = ""
            finally:
                _restore_workspace(root, saved)
            file_text = answer.read_text(encoding="utf-8") if answer.is_file() else ""
            if file_text.strip() and _json_failure(file_text) is None:
                text = file_text
            else:
                repaired = stream_answer(repair_out)
                if repaired.strip() and _json_failure(repaired) is None:
                    text = repaired
        if request.json_schema is not None and _json_failure(text):
            raise ProviderError("Cursor CLI did not return a JSON object")
        if not text.strip():
            detail = (err or out or "Cursor CLI returned no output").strip()
            raise ProviderError(detail[-800:] or "Cursor CLI returned no output")
        payload = _payload_with_text(out, text)
    latency_ms = int((time.perf_counter() - started) * 1000)
    return response_from_payload(payload, request, latency_ms, version)
