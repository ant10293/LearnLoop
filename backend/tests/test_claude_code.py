"""Claude Code subscription settings stay beside the Anthropic API key."""

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("LEARNLOOP_WORKER_ENABLED", "0")
os.environ.setdefault("LEARNLOOP_AUTO_SEED", "0")

FAKE = """#!/usr/bin/env python3
import os
import sys
import time

mode = os.environ.get("CLAUDE_FAKE", "out")
args = sys.argv[1:]
if args[:1] == ["--version"]:
    print("9.9.9 (Claude Code)")
elif args[:2] == ["auth", "status"]:
    logged = mode == "in"
    print(
        '{"loggedIn": %s, "authMethod": "claude.ai", "subscriptionType": "pro", '
        '"email": "ada@example.com"}' % ("true" if logged else "false")
    )
    if not logged:
        sys.exit(1)
elif args[:2] == ["auth", "login"]:
    print("Open https://claude.ai/oauth/authorize?code=local", flush=True)
    time.sleep(60)
else:
    sys.stderr.write("unexpected %s\\n" % " ".join(args))
    sys.exit(1)
"""


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("LEARNLOOP_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LEARNLOOP_DATABASE_URL", f"sqlite:///{tmp_path}/lab.db")
    monkeypatch.setenv("LEARNLOOP_WORKER_ENABLED", "0")
    monkeypatch.setenv("LEARNLOOP_AUTO_SEED", "0")
    from app.config import settings
    from app.db import rebind_engine

    settings.data_dir = tmp_path
    settings.database_url = f"sqlite:///{tmp_path}/lab.db"
    settings.worker_enabled = False
    settings.auto_seed = False
    settings.ingest_inline = True
    rebind_engine(settings.sqlalchemy_url)
    from app.main import create_app, init_database

    init_database()
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def _fake(path: Path) -> None:
    path.write_text(FAKE)
    path.chmod(0o755)


def _stop_login() -> None:
    from app.providers import claude_code

    proc = claude_code._login_proc
    claude_code._login_proc = None
    if proc is not None and proc.returncode is None:
        proc.kill()


def test_catalog_lists_current_claude_models_ahead_of_older_ids():
    from app.providers.catalog import lookup, models_for
    from app.providers.types import estimate_cost

    models = models_for("anthropic")
    assert models[0].model_id == "claude-sonnet-5-5"
    ids = [item.model_id for item in models]
    assert ids.index("claude-opus-5-5") < ids.index("claude-sonnet-4-5")
    assert "claude-fable-5-1" in ids
    assert "claude-haiku-4-5" in ids
    sonnet = lookup("anthropic", "claude-sonnet-5-5")
    assert sonnet.native_video is False
    assert sonnet.image is True
    assert sonnet.supports_temperature is False
    assert sonnet.reasoning_efforts == ("low", "medium", "high", "xhigh", "max")
    assert "xhigh" not in lookup("anthropic", "claude-sonnet-4-6").reasoning_efforts
    assert "max" in lookup("anthropic", "claude-opus-4-6").reasoning_efforts
    assert lookup("anthropic", "claude-opus-4-5-20251101").reasoning_efforts == ("low", "medium", "high")
    assert lookup("anthropic", "claude-sonnet-4-5").reasoning_efforts == ()
    cost, known = estimate_cost(sonnet, 1_000_000, 1_000_000)
    assert known is True
    assert cost == 12.0


def test_subscription_env_hides_the_api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "token")
    from app.providers.claude_code import subscription_env

    env = subscription_env()
    assert "ANTHROPIC_API_KEY" not in env
    assert "ANTHROPIC_AUTH_TOKEN" not in env


def test_subscription_env_drops_a_loopback_proxy(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:62763")
    monkeypatch.setenv("https_proxy", "http://127.0.0.1:62763")
    monkeypatch.setenv("ALL_PROXY", "socks5://localhost:62762")
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example.com:8080")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    from app.providers.claude_code import subscription_env

    env = subscription_env()
    assert "HTTPS_PROXY" not in env
    assert "https_proxy" not in env
    assert "ALL_PROXY" not in env
    assert env["HTTP_PROXY"] == "http://proxy.example.com:8080"
    assert env.get("NO_PROXY") != "*"


def test_subscription_env_bypasses_the_sandbox_proxy_when_that_is_the_only_one(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:62763")
    monkeypatch.setenv("__CURSOR_SANDBOX_ENV_RESTORE", "builtin")
    for name in (
        "ALL_PROXY", "HTTP_PROXY", "http_proxy", "all_proxy", "NO_PROXY", "no_proxy",
        "GIT_HTTP_PROXY", "GIT_HTTPS_PROXY", "SOCKS_PROXY", "SOCKS5_PROXY",
    ):
        monkeypatch.delenv(name, raising=False)
    from app.providers.claude_code import subscription_env

    env = subscription_env()
    assert "HTTPS_PROXY" not in env
    assert "__CURSOR_SANDBOX_ENV_RESTORE" not in env
    assert env["NO_PROXY"] == "*"
    assert env["no_proxy"] == "*"


def test_claude_code_uses_the_model_output_maximum(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_MAX_OUTPUT_TOKENS", "32000")
    from app.providers.claude_code import generation_env

    assert generation_env()["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "128000"


def test_cli_result_is_unpriced_plan_usage(tmp_path: Path):
    from app.providers.claude_code import (
        build_argv,
        build_stdin,
        parse_cli_payload,
        response_from_payload,
    )
    from app.providers.types import FrameInput, GenerationRequest

    frame = tmp_path / "frame.png"
    frame.write_bytes(b"\x89PNG\r\n\x1a\nrest")
    request = GenerationRequest(
        model_id="claude-sonnet-5-5",
        system_prompt="system",
        user_prompt="look",
        frames=[FrameInput(1.5, frame)],
        json_schema={"type": "object"},
        metadata={"transport": "claude_code"},
    )
    argv = build_argv("/usr/bin/claude", request.model_id, tmp_path / "system.txt", "{}")
    assert "--model" in argv and "claude-sonnet-5-5" in argv
    assert argv[argv.index("--input-format") + 1] == "stream-json"
    assert argv[argv.index("--output-format") + 1] == "stream-json"
    assert "--verbose" in argv
    assert "--include-partial-messages" in argv
    assert "--effort" not in argv
    assert argv[argv.index("--max-turns") + 1] == "1"
    assert argv[argv.index("--tools") + 1] == ""
    with_effort = build_argv("/usr/bin/claude", request.model_id, tmp_path / "system.txt", "{}", "xhigh")
    assert with_effort[with_effort.index("--effort") + 1] == "xhigh"
    message = json.loads(build_stdin(request))
    blocks = message["message"]["content"]
    assert blocks[0]["text"] == "Frame timestamp 1.500 seconds."
    assert blocks[1]["source"]["media_type"] == "image/png"
    assert blocks[2]["text"] == "look"
    streamed = "\n".join(
        [
            json.dumps({"type": "system", "subtype": "init"}),
            json.dumps(
                {
                    "type": "stream_event",
                    "event": {"type": "content_block_delta", "delta": {"type": "thinking_delta"}},
                }
            ),
            json.dumps({"type": "assistant", "message": {"content": []}}),
            json.dumps({"type": "result", "subtype": "success", "structured_output": {"objects": []}}),
        ]
    )
    assert parse_cli_payload(streamed)["structured_output"] == {"objects": []}
    response = response_from_payload(
        {
            "type": "result",
            "subtype": "success",
            "structured_output": {"objects": []},
            "session_id": "sess",
            "usage": {"input_tokens": 10, "output_tokens": 4},
            "total_cost_usd": 0.2,
        },
        request,
        30,
        "9.9.9",
    )
    assert json.loads(response.text) == {"objects": []}
    assert response.usage.input_tokens == 10
    assert response.usage.price_known is False
    assert response.usage.cost_usd == 0
    assert response.raw["billing"] == "claude_code_subscription"
    assert "total_cost_usd" not in response.raw


def test_claude_code_transport_skips_the_messages_api():
    from app.providers.anthropic_provider import AnthropicProvider
    from app.providers.types import GenerationRequest, ProviderResponse, Usage

    called: dict = {}

    async def fake(request):
        called["model"] = request.model_id
        return ProviderResponse(text="{}", raw={}, usage=Usage(price_known=False), model_id=request.model_id)

    provider = AnthropicProvider()
    request = GenerationRequest(
        model_id="claude-opus-5-5",
        system_prompt="s",
        user_prompt="u",
        metadata={"transport": "claude_code"},
    )

    async def run():
        from app.providers import anthropic_provider

        original = anthropic_provider.complete
        anthropic_provider.complete = fake
        try:
            return await provider.generate("", request)
        finally:
            anthropic_provider.complete = original

    response = asyncio.run(run())
    assert called["model"] == "claude-opus-5-5"
    assert response.text == "{}"


def test_provider_page_saves_claude_code_without_removing_api_settings(client: TestClient, tmp_path: Path, monkeypatch):
    fake = tmp_path / "claude"
    _fake(fake)
    monkeypatch.setenv("CLAUDE_FAKE", "in")
    listed = client.get("/api/providers")
    assert listed.status_code == 200
    anthropic = next(item for item in listed.json() if item["id"] == "anthropic")
    assert anthropic["claude_code"] == {"binary": "claude", "transport": "api"}
    assert any(model["model_id"] == "claude-sonnet-5-5" for model in anthropic["models"])

    rejected = client.put(
        "/api/providers/anthropic/claude-code",
        json={"binary": str(tmp_path / "missing"), "transport": "claude_code"},
    )
    assert rejected.status_code == 400

    saved = client.put(
        "/api/providers/anthropic/claude-code",
        json={"binary": str(fake), "transport": "claude_code"},
    )
    assert saved.status_code == 200
    assert saved.json()["transport"] == "claude_code"

    status = client.get("/api/providers/anthropic/claude-code")
    assert status.status_code == 200
    body = status.json()
    assert body["logged_in"] is True
    assert body["subscription"] == "pro"
    assert body["account"] == "a…@example.com"
    assert "ada@example.com" not in json.dumps(body)

    key = client.post("/api/providers/anthropic/credentials", json={"api_key": "sk-ant-test-key"})
    assert key.status_code == 200
    again = next(item for item in client.get("/api/providers").json() if item["id"] == "anthropic")
    assert again["configured"] is True
    assert again["claude_code"]["transport"] == "claude_code"
    assert again["hint"]


def test_signed_out_status_is_a_sentence_not_the_json_blob(client: TestClient, tmp_path: Path, monkeypatch):
    from app.providers.claude_code import login_url_from

    wrapped = (
        "If the browser didn't open, visit: \x1b]8;;https://claude.com/cai/oauth/authorize?code=true&state=abc\x07"
        "https://claude.com/cai/oauth/authorize?code=true&state=abc\x1b]8;;\x07"
    )
    assert login_url_from(wrapped) == "https://claude.com/cai/oauth/authorize?code=true&state=abc"
    fake = tmp_path / "claude"
    _fake(fake)
    monkeypatch.setenv("CLAUDE_FAKE", "out")
    assert client.put(
        "/api/providers/anthropic/claude-code",
        json={"binary": str(fake), "transport": "claude_code"},
    ).status_code == 200
    body = client.get("/api/providers/anthropic/claude-code").json()
    assert body["logged_in"] is False
    assert body["error"] == "Claude Code is installed and not signed in."
    assert "loggedIn" not in body["error"]


def test_link_starts_sign_in_when_claude_code_is_signed_out(client: TestClient, tmp_path: Path, monkeypatch):
    fake = tmp_path / "claude"
    _fake(fake)
    monkeypatch.setenv("CLAUDE_FAKE", "out")
    assert client.put(
        "/api/providers/anthropic/claude-code",
        json={"binary": str(fake), "transport": "api"},
    ).status_code == 200
    try:
        linked = client.post("/api/providers/anthropic/claude-code/link", json={})
        assert linked.status_code == 200
        body = linked.json()
        assert body["logged_in"] is False
        assert body["login_started"] is True
        assert body["login_url"] == "https://claude.ai/oauth/authorize?code=local"
    finally:
        _stop_login()


def test_new_run_records_the_claude_code_transport(client: TestClient, tmp_path: Path):
    fake = tmp_path / "claude"
    _fake(fake)
    assert client.put(
        "/api/providers/anthropic/claude-code",
        json={"binary": str(fake), "transport": "claude_code"},
    ).status_code == 200
    from app.db import SessionLocal
    from app.models.media import Project, VideoAsset
    from app.providers.types import UnsupportedMode
    from app.services.experiments import create_run

    session = SessionLocal()
    project = Project(name="lab")
    session.add(project)
    session.flush()
    video = VideoAsset(
        project_id=project.id,
        original_filename="clip.mp4",
        stored_path="originals/clip.mp4",
        checksum_sha256="abc",
    )
    session.add(video)
    session.commit()
    run = create_run(
        session,
        {
            "video_id": video.id,
            "provider_id": "anthropic",
            "model_id": "claude-sonnet-5-5",
            "modality": "sampled_frames",
        },
        enqueue=False,
    )
    assert run.config_json["anthropic_transport"] == "claude_code"
    with pytest.raises(UnsupportedMode, match="Claude Code does not accept temperature"):
        create_run(
            session,
            {
                "video_id": video.id,
                "provider_id": "anthropic",
                "model_id": "claude-sonnet-4-5",
                "modality": "sampled_frames",
                "temperature": 0.2,
            },
            enqueue=False,
        )
    session.close()


def test_claude_code_uses_the_global_provider_timeout():
    from app.providers.types import PROVIDER_TIMEOUT_S, provider_timeout

    assert PROVIDER_TIMEOUT_S == 30 * 60
    assert provider_timeout(180) == 1800
    assert provider_timeout(600) == 1800
    assert provider_timeout(2400) == 2400


def _child(path: Path, body: str) -> list[str]:
    path.write_text("import sys, time\n" + body)
    return [sys.executable, str(path)]


async def test_claude_code_stays_alive_while_output_continues(tmp_path: Path):
    from app.providers.claude_code import run_process

    argv = _child(
        tmp_path / "stream.py",
        """
data = sys.stdin.read()
sys.stdout.write("first %s\\n" % data.strip())
sys.stdout.flush()
time.sleep(0.5)
sys.stderr.write("mid\\n")
sys.stderr.flush()
time.sleep(0.5)
sys.stdout.write("last\\n")
sys.stdout.flush()
""",
    )
    code, out, err = await run_process(argv, b"ping\n", os.environ.copy(), 8, str(tmp_path), idle_timeout=0.8)
    assert code == 0
    assert "first ping" in out
    assert "last" in out
    assert "mid" in err


async def test_claude_code_kills_a_quiet_process(tmp_path: Path):
    from app.providers.claude_code import run_process
    from app.providers.types import ProviderError

    argv = _child(
        tmp_path / "quiet.py",
        """
sys.stdout.write("started\\n")
sys.stdout.flush()
time.sleep(6)
""",
    )
    with pytest.raises(ProviderError, match="no output for 1s"):
        await run_process(argv, b"", os.environ.copy(), 8, str(tmp_path), idle_timeout=1)


async def test_claude_code_kills_a_slow_trickle_at_the_hard_ceiling(tmp_path: Path):
    from app.providers.claude_code import run_process
    from app.providers.types import ProviderError

    argv = _child(
        tmp_path / "trickle.py",
        """
deadline = time.monotonic() + 8
while time.monotonic() < deadline:
    sys.stdout.write("x\\n")
    sys.stdout.flush()
    time.sleep(0.05)
""",
    )
    with pytest.raises(ProviderError, match="timed out after 1s"):
        await run_process(argv, b"", os.environ.copy(), 1, str(tmp_path), idle_timeout=5)
