"""ChatGPT plan linking stays beside the OpenAI API key."""

import asyncio
import base64
import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("LEARNLOOP_WORKER_ENABLED", "0")
os.environ.setdefault("LEARNLOOP_AUTO_SEED", "0")

FAKE = """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
log = os.environ.get("CODEX_FAKE_LOG", "")
if args[:1] == ["--version"]:
    print("codex-cli 0.99.0")
    raise SystemExit(0)
if args[:2] == ["login", "status"]:
    mode = os.environ.get("CODEX_FAKE", "out")
    if mode == "in":
        print("Logged in using ChatGPT")
        raise SystemExit(0)
    if mode == "api":
        print("Logged in using an API key")
        raise SystemExit(0)
    print("Not logged in")
    raise SystemExit(1)
if args[:1] == ["logout"]:
    raise SystemExit(0)
if args[:1] == ["login"]:
    print("Open https://auth.openai.com/oauth/authorize?code=local", flush=True)
    raise SystemExit(0)
if args[:2] == ["debug", "models"]:
    print(json.dumps({"models": [
        {
            "slug": "gpt-6.1-sol",
            "display_name": "GPT-6.1-Sol",
            "visibility": "list",
            "input_modalities": ["text", "image"],
            "context_window": 500000,
            "supported_reasoning_levels": [{"effort": "low"}, {"effort": "high"}],
        },
        {"slug": "gpt-reserve", "display_name": "GPT-Reserve", "visibility": "hide"},
    ]}))
    raise SystemExit(0)
if args[:1] == ["exec"]:
    prompt = sys.stdin.read()
    images = []
    last = ""
    for index, arg in enumerate(args):
        if arg == "--image" and index + 1 < len(args):
            images.append(args[index + 1])
        if arg == "--output-last-message" and index + 1 < len(args):
            last = args[index + 1]
    schema = ""
    if "--output-schema" in args:
        schema = Path(args[args.index("--output-schema") + 1]).read_text()
    if log:
        Path(log).write_text(json.dumps({
            "args": args,
            "cwd": os.getcwd(),
            "key": os.environ.get("OPENAI_API_KEY", ""),
            "prompt": prompt,
            "images": images,
            "schema": schema,
        }))
    if os.environ.get("CODEX_FAKE_EXEC") == "auth":
        sys.stderr.write("Not logged in. Run codex login.\\n")
        raise SystemExit(1)
    if last:
        Path(last).write_text('{"ok": true}')
    print(json.dumps({"type": "thread.started", "thread_id": "thread-1"}))
    print(json.dumps({
        "type": "turn.completed",
        "usage": {"input_tokens": 11, "output_tokens": 4},
    }))
    raise SystemExit(0)
sys.stderr.write("unexpected %s\\n" % " ".join(args))
raise SystemExit(1)
"""


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("LEARNLOOP_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LEARNLOOP_DATABASE_URL", f"sqlite:///{tmp_path}/lab.db")
    monkeypatch.setenv("LEARNLOOP_WORKER_ENABLED", "0")
    monkeypatch.setenv("LEARNLOOP_AUTO_SEED", "0")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    home = tmp_path / "codex-home"
    home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(home))
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
    _stop_login()


def _fake(path: Path) -> None:
    path.write_text(FAKE)
    path.chmod(0o755)


def _chatgpt_login(home: Path, email: str = "ada@example.com", plan: str = "plus") -> None:
    def segment(payload: dict) -> str:
        raw = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
        return raw

    token = ".".join(
        [
            segment({"alg": "none"}),
            segment({"email": email, "https://api.openai.com/auth": {"chatgpt_plan_type": plan}}),
            "sig",
        ]
    )
    (home / "auth.json").write_text(json.dumps({"tokens": {"id_token": token}}), encoding="utf-8")


def _stop_login() -> None:
    from app.providers import codex_cli

    proc = codex_cli._login_proc
    task = codex_cli._login_task
    codex_cli._login_proc = None
    codex_cli._login_task = None
    codex_cli._login_url = ""
    if proc is not None and proc.returncode is None:
        proc.kill()
    if task is not None and not task.done():
        task.cancel()


def test_provider_page_saves_codex_without_removing_api_settings(
    client: TestClient, tmp_path: Path, monkeypatch
):
    listed = client.get("/api/providers")
    assert listed.status_code == 200
    openai = next(item for item in listed.json() if item["id"] == "openai")
    assert openai["codex"] == {"binary": "codex", "transport": "api"}

    rejected = client.put(
        "/api/providers/openai/codex",
        json={"binary": str(tmp_path / "missing"), "transport": "codex"},
    )
    assert rejected.status_code == 400

    fake = tmp_path / "codex"
    _fake(fake)
    monkeypatch.setenv("CODEX_FAKE", "in")
    _chatgpt_login(tmp_path / "codex-home")
    saved = client.put(
        "/api/providers/openai/codex",
        json={"binary": str(fake), "transport": "codex"},
    )
    assert saved.status_code == 200
    assert saved.json() == {"binary": str(fake), "transport": "codex"}

    body = client.get("/api/providers/openai/codex").json()
    assert body["logged_in"] is True
    assert body["auth_method"] == "chatgpt"
    assert body["subscription"] == "plus"
    assert body["account"] == "a…@example.com"
    assert "ada@example.com" not in json.dumps(body)
    assert body["version"] == "codex-cli 0.99.0"

    validated = client.post("/api/providers/openai/validate")
    assert validated.status_code == 200
    again = next(item for item in client.get("/api/providers").json() if item["id"] == "openai")
    assert again["validated"] is True
    assert again["configured"] is False
    assert again["codex"]["transport"] == "codex"


def test_api_key_login_is_not_the_chatgpt_plan(client: TestClient, tmp_path: Path, monkeypatch):
    fake = tmp_path / "codex"
    _fake(fake)
    monkeypatch.setenv("CODEX_FAKE", "api")
    assert (
        client.put(
            "/api/providers/openai/codex",
            json={"binary": str(fake), "transport": "codex"},
        ).status_code
        == 200
    )
    body = client.get("/api/providers/openai/codex").json()
    assert body["logged_in"] is False
    assert body["error"] == "Codex CLI is signed in with an API key. Link ChatGPT to use the plan."
    assert client.post("/api/providers/openai/validate").status_code == 502


def test_signed_out_status_and_login_link(client: TestClient, tmp_path: Path, monkeypatch):
    fake = tmp_path / "codex"
    _fake(fake)
    monkeypatch.setenv("CODEX_FAKE", "out")
    assert (
        client.put(
            "/api/providers/openai/codex",
            json={"binary": str(fake), "transport": "api"},
        ).status_code
        == 200
    )
    assert client.post("/api/providers/openai/codex/link", json={}).status_code == 400
    assert (
        client.put(
            "/api/providers/openai/codex",
            json={"binary": str(fake), "transport": "codex"},
        ).status_code
        == 200
    )
    body = client.get("/api/providers/openai/codex").json()
    assert body["logged_in"] is False
    assert body["error"] == "Codex CLI is installed and not signed in."

    linked = client.post("/api/providers/openai/codex/link", json={})
    assert linked.status_code == 200
    started = linked.json()
    assert started["login_started"] is True
    assert started["login_url"] == "https://auth.openai.com/oauth/authorize?code=local"
    assert started["logged_in"] is False


def test_api_transport_still_requires_a_key(client: TestClient, tmp_path: Path):
    fake = tmp_path / "codex"
    _fake(fake)
    assert (
        client.put(
            "/api/providers/openai/codex",
            json={"binary": str(fake), "transport": "api"},
        ).status_code
        == 200
    )
    refused = client.post("/api/providers/openai/validate")
    assert refused.status_code == 400


def test_new_run_records_the_codex_transport(client: TestClient, tmp_path: Path):
    fake = tmp_path / "codex"
    _fake(fake)
    assert (
        client.put(
            "/api/providers/openai/codex",
            json={"binary": str(fake), "transport": "codex"},
        ).status_code
        == 200
    )
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
            "provider_id": "openai",
            "model_id": "gpt-4.1",
            "modality": "sampled_frames",
        },
        enqueue=False,
    )
    assert run.config_json["openai_transport"] == "codex"
    with pytest.raises(UnsupportedMode, match="Codex does not accept temperature"):
        create_run(
            session,
            {
                "video_id": video.id,
                "provider_id": "openai",
                "model_id": "gpt-4.1",
                "modality": "sampled_frames",
                "temperature": 0.2,
            },
            enqueue=False,
        )
    with pytest.raises(UnsupportedMode, match="native video"):
        create_run(
            session,
            {
                "video_id": video.id,
                "provider_id": "openai",
                "model_id": "gpt-4.1",
                "modality": "native_video",
            },
            enqueue=False,
        )
    session.close()


def test_codex_transport_lists_the_account_models(client: TestClient, tmp_path: Path, monkeypatch):
    fake = tmp_path / "codex"
    _fake(fake)
    monkeypatch.setenv("CODEX_FAKE", "in")
    assert (
        client.put(
            "/api/providers/openai/codex",
            json={"binary": str(fake), "transport": "codex"},
        ).status_code
        == 200
    )
    openai = next(item for item in client.get("/api/providers").json() if item["id"] == "openai")
    ids = [model["model_id"] for model in openai["models"]]
    assert ids == ["gpt-6.1-sol"]
    listed = openai["models"][0]
    assert listed["label"] == "GPT-6.1-Sol"
    assert listed["reasoning_efforts"] == ["low", "high"]
    assert listed["context_tokens"] == 500000
    assert listed["image"] is True
    assert listed["native_video"] is False
    assert listed["supports_temperature"] is False
    assert listed["price_known"] is False
    assert "gpt-4.1" not in ids
    assert "gpt-reserve" not in ids

    from app.providers.codex_cli import parse_debug_models

    assert parse_debug_models("not json") == []


def test_api_key_login_ignores_the_chatgpt_token(client: TestClient, tmp_path: Path, monkeypatch):
    fake = tmp_path / "codex"
    _fake(fake)
    monkeypatch.setenv("CODEX_FAKE", "api")
    _chatgpt_login(tmp_path / "codex-home", email="ada@example.com", plan="plus")
    assert (
        client.put(
            "/api/providers/openai/codex",
            json={"binary": str(fake), "transport": "codex"},
        ).status_code
        == 200
    )
    body = client.get("/api/providers/openai/codex").json()
    assert body["logged_in"] is False
    assert body["subscription"] == ""
    assert body["account"] == ""


def test_chatgpt_identity_masks_the_email_and_drops_a_broken_token(tmp_path: Path):
    from app.providers.codex_cli import chatgpt_identity

    home = tmp_path / "auth.json"
    _chatgpt_login(tmp_path, email="ada@example.com", plan="plus")
    assert chatgpt_identity(home) == ("plus", "a…@example.com")
    home.write_text("{", encoding="utf-8")
    assert chatgpt_identity(home) == ("", "")


def test_login_status_sentences():
    from app.providers.codex_cli import auth_method_from

    assert auth_method_from("Logged in using ChatGPT") == "chatgpt"
    assert auth_method_from("Logged in using an API key") == "api_key"
    assert auth_method_from("Logged in using Agent Identity") == "agent_identity"
    assert auth_method_from("Not logged in") == ""


def test_codex_transport_skips_the_responses_api():
    from app.providers.openai_provider import OpenAIProvider
    from app.providers.types import GenerationRequest, ProviderResponse, Usage

    called: dict = {}

    async def fake(request):
        called["model"] = request.model_id
        return ProviderResponse(
            text="{}", raw={}, usage=Usage(price_known=False), model_id=request.model_id
        )

    provider = OpenAIProvider()
    request = GenerationRequest(
        model_id="gpt-4.1",
        system_prompt="s",
        user_prompt="u",
        metadata={"transport": "codex"},
    )

    async def run():
        from app.providers import openai_provider

        original = openai_provider.complete
        openai_provider.complete = fake
        try:
            return await provider.generate("", request)
        finally:
            openai_provider.complete = original

    response = asyncio.run(run())
    assert called["model"] == "gpt-4.1"
    assert response.text == "{}"


def test_completion_attaches_frames_in_read_only_mode_without_the_api_key(
    tmp_path: Path, monkeypatch
):
    from app.providers.codex_cli import complete, save_codex
    from app.providers.types import FrameInput, GenerationRequest

    fake = tmp_path / "codex"
    _fake(fake)
    log = tmp_path / "log.json"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret")
    monkeypatch.setenv("CODEX_FAKE_LOG", str(log))
    monkeypatch.setenv("LEARNLOOP_DATA_DIR", str(tmp_path / "data"))
    from app.config import settings

    settings.data_dir = tmp_path / "data"
    save_codex(str(fake), "codex")
    frame = tmp_path / "shot.jpg"
    frame.write_bytes(b"\xff\xd8\xff\xd9")

    response = asyncio.run(
        complete(
            GenerationRequest(
                model_id="gpt-4.1",
                system_prompt="Describe the clip.",
                user_prompt="Where is the bolt?",
                frames=[FrameInput(1.5, frame)],
                json_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}},
                reasoning_effort="high",
                metadata={"transport": "codex"},
            )
        )
    )
    assert response.text == '{"ok": true}'
    assert response.raw["billing"] == "chatgpt_subscription"
    assert response.request_id == "thread-1"
    assert response.usage.input_tokens == 11
    assert response.usage.output_tokens == 4
    assert response.usage.price_known is False
    assert response.usage.cost_usd == 0
    recorded = json.loads(log.read_text())
    args = recorded["args"]
    assert args[args.index("--sandbox") + 1] == "read-only"
    assert args[args.index("--model") + 1] == "gpt-4.1"
    assert args[args.index("--config") + 1] == "model_reasoning_effort=high"
    assert "--yolo" not in args
    assert "--full-auto" not in args
    assert "danger-full-access" not in args
    assert "workspace-write" not in args
    assert "sk-test-secret" not in json.dumps(args)
    assert recorded["key"] == ""
    assert "Where is the bolt?" in recorded["prompt"]
    assert "Describe the clip." in recorded["prompt"]
    assert "1.500s" in recorded["prompt"]
    assert recorded["images"]
    sent = json.loads(recorded["schema"])
    assert sent["type"] == "object"
    assert sent["additionalProperties"] is False
    assert sent["required"] == ["ok"]
    assert Path(recorded["cwd"]).resolve() != Path.cwd().resolve()
    assert Path(recorded["images"][0]).resolve().is_relative_to(Path(recorded["cwd"]).resolve())


def test_codex_call_uses_the_global_thirty_minute_timeout(tmp_path: Path, monkeypatch):
    from app.providers.codex_cli import complete, save_codex
    from app.providers.types import FrameInput, GenerationRequest, ProviderError

    fake = tmp_path / "codex"
    _fake(fake)
    monkeypatch.setenv("LEARNLOOP_DATA_DIR", str(tmp_path / "data"))
    from app.config import settings

    settings.data_dir = tmp_path / "data"
    save_codex(str(fake), "codex")
    frame = tmp_path / "shot.jpg"
    frame.write_bytes(b"\xff\xd8\xff\xd9")
    seen: dict[str, float] = {}

    async def capture(argv, env, cwd, stdin, timeout):
        seen["timeout"] = timeout
        raise ProviderError("stop after recording the deadline")

    monkeypatch.setattr("app.providers.codex_cli._execute", capture)
    with pytest.raises(ProviderError, match="stop after recording"):
        asyncio.run(
            complete(
                GenerationRequest(
                    model_id="gpt-6-astra",
                    system_prompt="Describe the clip.",
                    user_prompt="What is in the scene?",
                    frames=[FrameInput(1.5, frame)],
                    timeout_s=180,
                    metadata={"transport": "codex"},
                )
            )
        )
    assert seen["timeout"] == 1800


def test_output_schema_is_strict_enough_for_codex():
    from app.providers.codex_cli import strict_output_schema
    from app.schemas.analysis import AnalysisDocument

    schema = strict_output_schema(AnalysisDocument.model_json_schema())

    def objects(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                yield node
            for value in node.values():
                yield from objects(value)
        elif isinstance(node, list):
            for value in node:
                yield from objects(value)

    found = list(objects(schema))
    assert found
    assert all(item.get("additionalProperties") is False for item in found)
    assert all(item.get("required") == list(item.get("properties") or {}) for item in found)
    dumped = json.dumps(schema)
    assert "exclusiveMinimum" not in dumped
    assert '"default"' not in dumped


def test_a_turn_error_is_not_replaced_by_the_stdin_status_line():
    from app.providers.codex_cli import failure_message

    stdout = "\n".join(
        [
            json.dumps({"type": "thread.started", "thread_id": "thread-1"}),
            json.dumps(
                {
                    "type": "turn.failed",
                    "error": {"message": "Invalid schema for response_format"},
                }
            ),
        ]
    )
    assert failure_message(stdout, "Reading prompt from stdin...") == (
        "Invalid schema for response_format"
    )


def test_signed_out_completion_is_an_authentication_error(tmp_path: Path, monkeypatch):
    from app.providers.codex_cli import complete, save_codex
    from app.providers.types import AuthenticationError, GenerationRequest

    fake = tmp_path / "codex"
    _fake(fake)
    monkeypatch.setenv("CODEX_FAKE_EXEC", "auth")
    monkeypatch.setenv("LEARNLOOP_DATA_DIR", str(tmp_path / "data"))
    from app.config import settings

    settings.data_dir = tmp_path / "data"
    save_codex(str(fake), "codex")
    with pytest.raises(AuthenticationError, match="Not logged in"):
        asyncio.run(
            complete(
                GenerationRequest(
                    model_id="gpt-4.1",
                    system_prompt="s",
                    user_prompt="u",
                    metadata={"transport": "codex"},
                )
            )
        )
