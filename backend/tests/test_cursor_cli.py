"""Cursor CLI account linking stays beside the optional API key."""

import json
import os
import sys
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
log = os.environ.get("CURSOR_FAKE_LOG", "")
if args[:1] == ["--version"]:
    print("1.2.3")
    raise SystemExit(0)
if args[:1] == ["status"]:
    key = os.environ.get("CURSOR_API_KEY", "")
    mode = os.environ.get("CURSOR_FAKE", "out")
    if mode == "in" or key:
        who = "key@example.com" if key else "ada@example.com"
        print("Logged in as %s" % who)
        raise SystemExit(0)
    print("Not authenticated. Run agent login.")
    raise SystemExit(1)
if args[:1] == ["about"]:
    key = os.environ.get("CURSOR_API_KEY", "")
    mode = os.environ.get("CURSOR_FAKE", "out")
    if mode == "in" or key:
        print(json.dumps({"subscriptionTier": "Pro+", "userEmail": "key@example.com" if key else "ada@example.com"}))
        raise SystemExit(0)
    raise SystemExit(1)
if args[:1] == ["logout"]:
    raise SystemExit(0)
if args[:1] == ["login"]:
    print("Open https://cursor.com/loginDeepControl?token=local", flush=True)
    raise SystemExit(0)
if args[:1] == ["--list-models"]:
    print("composer-2.5\\nauto")
    raise SystemExit(0)
if "--print" in args:
    instructions = Path("instructions.md").read_text() if Path("instructions.md").is_file() else ""
    frames = sorted(path.name for path in Path("frames").glob("*")) if Path("frames").is_dir() else []
    prompt = args[-1] if args else ""
    if log:
        Path(log).write_text(json.dumps({
            "args": args,
            "cwd": os.getcwd(),
            "key": os.environ.get("CURSOR_API_KEY", ""),
            "instructions": instructions,
            "frames": frames,
            "prompt": prompt,
        }))
    if prompt.startswith("answer.json"):
        shot = next(Path("frames").glob("*"), None) if Path("frames").is_dir() else None
        if shot is not None:
            shot.write_bytes(b"edited")
        Path("scratch.txt").write_text("nope")
        Path("answer.json").write_text('{"ok": true}')
    result = '{"a": 1}{"b": 2}' if os.environ.get("CURSOR_FAKE_BROKEN") and not prompt.startswith("answer.json") else '{"ok": true}'
    print(json.dumps({
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": result,
        "session_id": "sess-1",
        "request_id": "req-1",
        "duration_ms": 12,
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
    monkeypatch.delenv("CURSOR_API_KEY", raising=False)
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


def _stop_login() -> None:
    from app.providers import cursor_cli

    proc = cursor_cli._login_proc
    task = cursor_cli._login_task
    cursor_cli._login_proc = None
    cursor_cli._login_task = None
    cursor_cli._login_url = ""
    if proc is not None and proc.returncode is None:
        proc.kill()
    if task is not None and not task.done():
        task.cancel()


def test_provider_page_saves_the_cursor_cli(client: TestClient, tmp_path: Path, monkeypatch):
    listed = client.get("/api/providers")
    assert listed.status_code == 200
    cursor = next(item for item in listed.json() if item["id"] == "cursor")
    assert cursor["cursor_cli"] == {"binary": "agent", "transport": "login"}
    ids = [model["model_id"] for model in cursor["models"]]
    assert "composer-2.5" in ids
    assert all(model["supports_temperature"] is False for model in cursor["models"])
    for model in cursor["models"]:
        token = model["model_id"].lower().removeprefix("cursor-")
        expects_video = token.startswith("gemini") or token.startswith("muse-")
        assert model["native_video"] is expects_video

    rejected = client.put(
        "/api/providers/cursor/cli",
        json={"binary": str(tmp_path / "missing"), "transport": "login"},
    )
    assert rejected.status_code == 400

    fake = tmp_path / "agent"
    _fake(fake)
    monkeypatch.setenv("CURSOR_FAKE", "in")
    saved = client.put(
        "/api/providers/cursor/cli",
        json={"binary": str(fake), "transport": "login"},
    )
    assert saved.status_code == 200
    assert saved.json() == {"binary": str(fake), "transport": "login"}

    body = client.get("/api/providers/cursor/cli").json()
    assert body["logged_in"] is True
    assert body["account"] == "a…@example.com"
    assert body["subscription"] == "Pro+"
    assert body["auth_method"] == "login"
    assert "ada@example.com" not in json.dumps(body)

    validated = client.post("/api/providers/cursor/validate")
    assert validated.status_code == 200
    again = next(item for item in client.get("/api/providers").json() if item["id"] == "cursor")
    assert again["validated"] is True
    assert again["configured"] is False


def test_signed_out_status_and_login_link(client: TestClient, tmp_path: Path, monkeypatch):
    fake = tmp_path / "agent"
    _fake(fake)
    monkeypatch.setenv("CURSOR_FAKE", "out")
    assert client.put(
        "/api/providers/cursor/cli",
        json={"binary": str(fake), "transport": "login"},
    ).status_code == 200
    body = client.get("/api/providers/cursor/cli").json()
    assert body["logged_in"] is False
    assert body["error"] == "Cursor CLI is installed and not signed in."
    refused = client.post("/api/providers/cursor/validate")
    assert refused.status_code == 502

    linked = client.post("/api/providers/cursor/cli/link", json={})
    assert linked.status_code == 200
    started = linked.json()
    assert started["login_started"] is True
    assert started["login_url"] == "https://cursor.com/loginDeepControl?token=local"
    assert started["logged_in"] is False


def test_api_key_transport_uses_the_stored_key_and_refuses_linking(client: TestClient, tmp_path: Path, monkeypatch):
    fake = tmp_path / "agent"
    _fake(fake)
    monkeypatch.setenv("CURSOR_FAKE", "out")
    assert client.put(
        "/api/providers/cursor/cli",
        json={"binary": str(fake), "transport": "api_key"},
    ).status_code == 200
    missing = client.get("/api/providers/cursor/cli").json()
    assert missing["logged_in"] is False
    assert missing["error"] == "No API key is configured for Cursor."
    assert client.post("/api/providers/cursor/cli/link", json={}).status_code == 400

    stored = client.post("/api/providers/cursor/credentials", json={"api_key": "cursor_test_secret"})
    assert stored.status_code == 200
    body = client.get("/api/providers/cursor/cli").json()
    assert body["logged_in"] is True
    assert body["auth_method"] == "api_key"
    assert body["subscription"] == "Pro+"
    assert body["account"] == "k…@example.com"
    dumped = json.dumps(body)
    assert "cursor_test_secret" not in dumped
    assert "key@example.com" not in dumped
    assert client.post("/api/providers/cursor/validate").status_code == 200


def test_discover_lists_cli_models_without_a_key(client: TestClient, tmp_path: Path, monkeypatch):
    fake = tmp_path / "agent"
    _fake(fake)
    monkeypatch.setenv("CURSOR_FAKE", "in")
    assert client.put(
        "/api/providers/cursor/cli",
        json={"binary": str(fake), "transport": "login"},
    ).status_code == 200
    found = client.get("/api/providers/cursor/models")
    assert found.status_code == 200
    assert found.json()["discovered"] == ["composer-2.5", "auto"]


def test_new_run_records_the_cursor_transport(client: TestClient, tmp_path: Path):
    fake = tmp_path / "agent"
    _fake(fake)
    assert client.put(
        "/api/providers/cursor/cli",
        json={"binary": str(fake), "transport": "login"},
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
            "provider_id": "cursor",
            "model_id": "composer-2.5",
            "modality": "sampled_frames",
        },
        enqueue=False,
    )
    assert run.config_json["cursor_transport"] == "login"
    with pytest.raises(UnsupportedMode, match="temperature"):
        create_run(
            session,
            {
                "video_id": video.id,
                "provider_id": "cursor",
                "model_id": "composer-2.5",
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
                "provider_id": "cursor",
                "model_id": "auto",
                "modality": "native_video",
            },
            enqueue=False,
        )
    session.close()


def test_labeled_cli_model_list_keeps_the_id():
    from app.providers.cursor_cli import parse_model_list, parse_model_records

    text = "Available models\n\nauto - Auto (default)\ncomposer-2.5 - Composer 2.5\ngrok-4.7-low - Grok 4.7  Low\n"
    assert parse_model_list(text) == ["auto", "composer-2.5", "grok-4.7-low"]
    assert parse_model_records(text)[2] == ("grok-4.7-low", "Grok 4.7 Low")


def test_cursor_account_marks_gemini_and_muse_as_video():
    from app.providers.catalog import lookup, models_for
    from app.providers import model_list

    catalog = [caps.as_dict() for caps in models_for("cursor")]
    ids = [
        "gemini-3-flash",
        "muse-spark-1.3-high",
        "muse-wonder-1.0-high",
        "grok-4.7-high",
        "claude-sonnet-5-high",
        "composer-2.5",
    ]
    options = model_list.overlay_options(
        "cursor",
        catalog,
        [{"model_id": model_id, "label": model_id} for model_id in ids],
    )
    by_id = {item["model_id"]: item for item in options}
    assert by_id["gemini-3-flash"]["native_video"] is True
    assert by_id["muse-spark-1.3-high"]["native_video"] is True
    assert by_id["muse-wonder-1.0-high"]["native_video"] is True
    assert by_id["grok-4.7-high"]["native_video"] is False
    assert by_id["claude-sonnet-5-high"]["native_video"] is False
    assert by_id["composer-2.5"]["native_video"] is False
    assert lookup("cursor", "auto").native_video is False
    for model_id in ids:
        model_list._saved.pop(("cursor", model_id), None)


def test_manual_cursor_model_rejects_temperature():
    from app.providers.catalog import lookup

    added = lookup("cursor", "composer-experimental")
    assert added.supports_temperature is False
    assert added.native_video is False
    assert added.image is True


def test_login_completion_reads_frames_in_ask_mode_without_the_api_key(tmp_path: Path, monkeypatch):
    from app.providers.cursor_cli import complete
    from app.providers.types import FrameInput, GenerationRequest

    fake = tmp_path / "agent"
    _fake(fake)
    log = tmp_path / "log.json"
    monkeypatch.setenv("CURSOR_API_KEY", "cursor_test_secret")
    monkeypatch.setenv("CURSOR_FAKE_LOG", str(log))
    monkeypatch.setenv("LEARNLOOP_DATA_DIR", str(tmp_path / "data"))
    from app.config import settings

    settings.data_dir = tmp_path / "data"
    from app.providers.cursor_cli import save_cursor_cli

    save_cursor_cli(str(fake), "login")
    frame = tmp_path / "shot.jpg"
    frame.write_bytes(b"\xff\xd8\xff\xd9")
    import asyncio

    response = asyncio.run(
        complete(
            GenerationRequest(
                model_id="composer-2.5",
                system_prompt="Describe the clip.",
                user_prompt="Where is the bolt?",
                frames=[FrameInput(1.5, frame)],
                json_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}},
                metadata={"transport": "login"},
            )
        )
    )
    assert response.text == '{"ok": true}'
    assert response.raw["billing"] == "cursor_subscription"
    assert response.request_id == "req-1"
    recorded = json.loads(log.read_text())
    args = recorded["args"]
    assert args[args.index("--mode") + 1] == "ask"
    assert args[args.index("--output-format") + 1] == "stream-json"
    assert "--stream-partial-output" in args
    assert args[args.index("--sandbox") + 1] == "enabled"
    assert "--force" not in args
    assert "--yolo" not in args
    assert "--api-key" not in args
    assert "cursor_test_secret" not in json.dumps(args)
    assert recorded["key"] == ""
    assert "Where is the bolt?" in recorded["instructions"]
    assert "Describe the clip." in recorded["instructions"]
    assert '"ok"' in recorded["instructions"]
    assert recorded["frames"]
    assert Path(recorded["cwd"]).resolve() == Path(args[args.index("--workspace") + 1]).resolve()
    assert Path(recorded["cwd"]).resolve() != Path.cwd().resolve()


def test_video_instruction_requires_the_whole_recording():
    from app.pipeline.infer import _video_coverage_instruction

    text = _video_coverage_instruction(21.952, [0.0, 21.935], frames_attached=True)
    assert "21.952" in text
    assert "0.000s" in text or "0.000" in text
    assert "21.935" in text
    assert "opening" in text and "end" in text


def test_video_model_sends_the_file_and_the_timeline(tmp_path: Path, monkeypatch):
    from app.providers.cursor_cli import complete
    from app.providers.types import GenerationRequest, UnsupportedMode

    fake = tmp_path / "agent"
    _fake(fake)
    log = tmp_path / "log.json"
    monkeypatch.setenv("CURSOR_FAKE_LOG", str(log))
    monkeypatch.setenv("LEARNLOOP_DATA_DIR", str(tmp_path / "data"))
    from app.config import settings

    settings.data_dir = tmp_path / "data"
    from app.providers.cursor_cli import save_cursor_cli

    save_cursor_cli(str(fake), "login")
    clip = tmp_path / "source.mp4"
    clip.write_bytes(b"video-bytes")
    frame = tmp_path / "shot.jpg"
    frame.write_bytes(b"\xff\xd8\xff\xd9")
    import asyncio

    from app.providers.types import FrameInput

    response = asyncio.run(
        complete(
            GenerationRequest(
                model_id="gemini-3-flash",
                system_prompt="Describe the clip.",
                user_prompt="Where is the bolt?",
                frames=[FrameInput(0.0, frame), FrameInput(21.935, frame)],
                video_path=clip,
                modality="native_video",
                json_schema={"type": "object"},
                metadata={"transport": "login"},
            )
        )
    )
    assert response.text == '{"ok": true}'
    recorded = json.loads(log.read_text())
    assert len(recorded["frames"]) == 2
    assert "0.000s" in recorded["instructions"]
    assert "21.935s" in recorded["instructions"]
    assert "clip.mp4" in recorded["instructions"]
    assert "first moment through the last" in recorded["prompt"]
    assert "every frame file" in recorded["prompt"]
    with pytest.raises(UnsupportedMode, match="native video"):
        asyncio.run(
            complete(
                GenerationRequest(
                    model_id="composer-2.5",
                    system_prompt="s",
                    user_prompt="u",
                    video_path=clip,
                    modality="native_video",
                    metadata={"transport": "login"},
                )
            )
        )


def test_api_key_completion_puts_the_key_in_the_environment_only(tmp_path: Path, monkeypatch):
    from app.providers.cursor_cli import complete, save_cursor_cli
    from app.providers.types import GenerationRequest

    fake = tmp_path / "agent"
    _fake(fake)
    log = tmp_path / "log.json"
    monkeypatch.setenv("CURSOR_FAKE_LOG", str(log))
    monkeypatch.delenv("CURSOR_API_KEY", raising=False)
    monkeypatch.setenv("LEARNLOOP_DATA_DIR", str(tmp_path / "data"))
    from app.config import settings

    settings.data_dir = tmp_path / "data"
    save_cursor_cli(str(fake), "api_key")
    import asyncio

    response = asyncio.run(
        complete(
            GenerationRequest(
                model_id="auto",
                system_prompt="Merge.",
                user_prompt="Combine the windows.",
                metadata={"transport": "api_key"},
            ),
            "cursor_test_secret",
        )
    )
    assert response.raw["billing"] == "cursor_api_key"
    recorded = json.loads(log.read_text())
    assert recorded["key"] == "cursor_test_secret"
    assert "cursor_test_secret" not in json.dumps(recorded["args"])
    assert "This call has no frames." in recorded["instructions"]


def test_truncated_result_line_keeps_the_assistant_deltas():
    from app.providers.cursor_cli import stream_answer

    stdout = "\n".join(
        [
            json.dumps(
                {
                    "type": "assistant",
                    "message": {"role": "assistant", "content": [{"type": "text", "text": '{"ok": '}]},
                }
            ),
            json.dumps(
                {
                    "type": "assistant",
                    "message": {"role": "assistant", "content": [{"type": "text", "text": "true}"}]},
                }
            ),
            '{"type":"result","result":"{\\"ok\\": true',
        ]
    )
    assert stream_answer(stdout) == '{"ok": true}'


def test_repair_leaves_frames_and_other_files_unchanged(tmp_path: Path):
    from app.providers.cursor_cli import _restore_workspace, _snapshot_workspace

    frame = tmp_path / "frames" / "000.jpg"
    frame.parent.mkdir()
    frame.write_bytes(b"jpg")
    (tmp_path / "instructions.md").write_text("schema")
    saved = _snapshot_workspace(tmp_path)
    frame.write_bytes(b"edited")
    (tmp_path / "scratch.txt").write_text("nope")
    (tmp_path / "answer.json").write_text('{"ok": true}')
    _restore_workspace(tmp_path, saved)
    assert frame.read_bytes() == b"jpg"
    assert not (tmp_path / "scratch.txt").exists()
    assert (tmp_path / "answer.json").read_text() == '{"ok": true}'


def test_broken_json_is_repaired_by_editing_answer_json(tmp_path: Path, monkeypatch):
    from app.providers.cursor_cli import complete, save_cursor_cli
    from app.providers.types import FrameInput, GenerationRequest

    fake = tmp_path / "agent"
    _fake(fake)
    log = tmp_path / "log.json"
    monkeypatch.setenv("CURSOR_FAKE_BROKEN", "1")
    monkeypatch.setenv("CURSOR_FAKE_LOG", str(log))
    monkeypatch.setenv("LEARNLOOP_DATA_DIR", str(tmp_path / "data"))
    from app.config import settings

    settings.data_dir = tmp_path / "data"
    save_cursor_cli(str(fake), "login")
    frame = tmp_path / "shot.jpg"
    frame.write_bytes(b"\xff\xd8\xff\xd9")
    import asyncio

    response = asyncio.run(
        complete(
            GenerationRequest(
                model_id="grok-4.7-high-fast",
                system_prompt="Describe the clip.",
                user_prompt="Where is the bolt?",
                frames=[FrameInput(1.5, frame)],
                json_schema={"type": "object"},
                metadata={"transport": "login"},
            )
        )
    )
    assert response.text == '{"ok": true}'
    recorded = json.loads(log.read_text())
    assert recorded["prompt"].startswith("answer.json")
    assert "--mode" not in recorded["args"]
    assert frame.read_bytes() == b"\xff\xd8\xff\xd9"


def test_stream_result_is_taken_from_the_final_event():
    from app.providers.cursor_cli import parse_result

    stdout = "\n".join(
        [
            json.dumps({"type": "thinking", "subtype": "delta", "text": "looking"}),
            json.dumps({"type": "tool_call", "subtype": "completed", "call_id": "read-1"}),
            json.dumps(
                {
                    "type": "assistant",
                    "message": {"role": "assistant", "content": [{"type": "text", "text": '{"ok": true}'}]},
                }
            ),
            json.dumps(
                {
                    "type": "result",
                    "subtype": "success",
                    "is_error": False,
                    "result": '{"ok": true}',
                    "session_id": "sess-1",
                    "request_id": "req-1",
                }
            ),
        ]
    )
    payload = parse_result(stdout)
    assert payload["result"] == '{"ok": true}'
    assert payload["request_id"] == "req-1"


def test_cursor_cli_uses_the_global_provider_timeout():
    from app.providers.types import PROVIDER_TIMEOUT_S, provider_timeout

    assert PROVIDER_TIMEOUT_S == 30 * 60
    assert provider_timeout(180) == 1800
    assert provider_timeout(600) == 1800
    assert provider_timeout(2400) == 2400


def _child(path: Path, body: str) -> list[str]:
    path.write_text("import sys, time\n" + body)
    return [sys.executable, str(path)]


async def test_cursor_cli_stays_alive_while_output_continues(tmp_path: Path):
    from app.providers.cursor_cli import _execute

    argv = _child(
        tmp_path / "stream.py",
        """
sys.stdout.write("thinking\\n")
sys.stdout.flush()
time.sleep(0.5)
sys.stderr.write("tool\\n")
sys.stderr.flush()
time.sleep(0.5)
sys.stdout.write("{\\"type\\": \\"result\\", \\"result\\": \\"{}\\"}\\n")
sys.stdout.flush()
""",
    )
    code, out, err = await _execute(argv, os.environ.copy(), str(tmp_path), 8, idle_timeout=0.8)
    assert code == 0
    assert "thinking" in out
    assert '"result"' in out
    assert "tool" in err


async def test_cursor_cli_kills_a_quiet_process(tmp_path: Path):
    from app.providers.cursor_cli import _execute
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
        await _execute(argv, os.environ.copy(), str(tmp_path), 8, idle_timeout=1)


async def test_cursor_cli_kills_a_slow_trickle_at_the_hard_ceiling(tmp_path: Path):
    from app.providers.cursor_cli import _execute
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
        await _execute(argv, os.environ.copy(), str(tmp_path), 1, idle_timeout=5)
