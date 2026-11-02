"""Model calls, windowing, and local document construction."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from sqlalchemy.orm import Session

from app.demo.scenarios import SCENARIOS, build_document
from app.models.media import FrameAsset, VideoAsset
from app.models.runs import AnalysisRun, ModelRequestRow
from app.perception.evidence import evidence_text
from app.perception.ground import apply_perception
from app.perception.roster import align_to_roster, drop_split_objects, omit_workspace_and_hands
from app.perception.sam_fusion import prompt_prior, sam_evidence_block
from app.pipeline.contracts import (
    OutputContract,
    SchemaContractError,
    StructuredOutputError,
    contract_for,
    schema_hash,
    validate_output,
)
from app.pipeline.frame_boxes import (
    FRAME_BOX_ATTEMPTS,
    FRAME_BOX_WINDOW,
    answered_keys,
    frame_box_prompt,
    frame_box_rejection,
    frame_box_schema,
    mark_unmeasured,
    merge_frame_boxes,
    parse_frame_boxes,
)
from app.pipeline.merge import restore_omitted_records
from app.pipeline.parsing import ParseFailure, parse_analysis, parse_interpretation
from app.pipeline.prompts import DISCOVERY, INTERPRETATION, compose
from app.providers.meta_provider import MAX_OUTPUT_TOKENS
from app.providers.model_list import capabilities_for, resolve_model_id
from app.providers.registry import get_provider, needs_secret, run_transport
from app.providers.types import (
    DEFAULT_MAX_REQUESTS,
    FrameInput,
    GenerationRequest,
    UnsupportedMode,
    rough_input_tokens,
)
from app.schemas.analysis import (
    AnalysisDocument,
    Coverage,
    InterpretationResponse,
)
from app.services.costs import fail_provider_call, finish_provider_call, open_provider_call
from app.services.credentials import load_secret
from app.services.experiments import sends_video_file
from app.services.storage import data_path, resolve_stored
from app.video.sampling import budget_indices

logger = logging.getLogger(__name__)


class BudgetExceeded(RuntimeError):
    pass


class StageSkipped(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def model_for_call(session: Session, run: AnalysisRun) -> tuple[str, str]:
    """Return the model to send and the id the run asked for.

    An id the account does not list is replaced from that account's live list.
    The run keeps the replacement so later calls do not send the missing id again.
    """
    requested = run.model_id
    chosen = resolve_model_id(session, run.provider_id, requested, run.modality)
    if chosen != requested:
        run.model_id = chosen
        config = dict(run.config_json or {})
        config["model_id"] = chosen
        config["requested_model_id"] = requested
        run.config_json = config
    return chosen, requested


def load_working(run_id: str) -> AnalysisDocument:
    path = _working_path(run_id)
    if not path.exists():
        return AnalysisDocument()
    return AnalysisDocument.model_validate(json.loads(path.read_text()))


def save_working(run_id: str, document: AnalysisDocument) -> None:
    path = _working_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document.model_dump_json(indent=2))


def _working_path(run_id: str) -> Path:
    return data_path("runs", run_id, "working.json")


SEMANTIC_STAGES = ("scene_analysis", "action_analysis", "state_tracking", "workflow", "ambiguity", "evidence_fusion")
SEMANTIC_LISTS = ("actions", "states", "transitions", "relationships", "workflow_steps", "dependencies")


def semantic_mode(run: AnalysisRun) -> str:
    """``combined`` interprets in the scene call; ``grounded`` interprets once after evidence fusion."""
    mode = str((run.config_json or {}).get("semantic_mode") or "combined")
    return mode if mode in {"combined", "grounded"} else "combined"


def save_semantic_raw(run_id: str, document: AnalysisDocument) -> None:
    """The scene call's parsed output, before any repair. Replaced on retry."""
    path = data_path("runs", run_id, "semantic_raw.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document.model_dump_json(indent=2))


def semantic_calls(session: Session, run_id: str) -> int:
    """Semantic passes that reached a provider. A windowed or re-grounded pass is one pass.

    Every provider request, including windows, merges, and locate calls, is
    still recorded on its own row.
    """
    stages = (
        session.query(ModelRequestRow.stage)
        .filter(ModelRequestRow.run_id == run_id, ModelRequestRow.stage.in_(SEMANTIC_STAGES))
        .distinct()
        .all()
    )
    return len(stages)


def fixture_document_for(video: VideoAsset) -> AnalysisDocument | None:
    key = _fixture_key(video)
    return build_document(SCENARIOS[key]()) if key else None


def has_fixture(video: VideoAsset) -> bool:
    return _fixture_key(video) is not None


def _fixture_key(video: VideoAsset) -> str | None:
    index_path = data_path("fixture_index.json")
    if not index_path.exists():
        return None
    index = json.loads(index_path.read_text())
    key = index.get(video.checksum_sha256)
    return key if key in SCENARIOS else None


def register_fixture(checksum: str, scenario_key: str) -> None:
    path = data_path("fixture_index.json")
    current = json.loads(path.read_text()) if path.exists() else {}
    current[checksum] = scenario_key
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(current, indent=2))


async def produce_document(
    session: Session,
    run: AnalysisRun,
    video: VideoAsset,
    frames: list[FrameAsset],
    stage: str,
    coverage: Coverage,
    spatial_document: AnalysisDocument | None,
    user_extra: str = "",
    purpose: str = "full",
) -> AnalysisDocument:
    """One semantic call for a stage.

    `purpose` is ``full`` for the combined path, ``discovery`` for the grounded
    objects-and-boxes call, or ``interpretation`` for the grounded call that
    reads resolved geometry and returns actions, states, and workflow.
    """
    if run.provider_id == "fixture":
        document = fixture_document_for(video)
        if document is None:
            raise UnsupportedMode(
                "The fixture provider only returns an authored analysis when the video checksum "
                "matches a generated LearnLoop fixture. It will not invent results for other files."
            )
        document.coverage = coverage
        return document
    if run.provider_id == "deterministic":
        if spatial_document is None:
            raise UnsupportedMode("spatial tracking did not produce a document")
        spatial_document.coverage = coverage
        return spatial_document
    if run.provider_id == "manual_json":
        raise UnsupportedMode("Import JSON from the library instead of launching a model run.")

    config = run.config_json or {}
    prompt_key = {
        "scene_analysis": "scene",
        "action_analysis": "actions",
        "state_tracking": "states",
        "workflow": "workflow",
        "ambiguity": "ambiguity",
        "reinspection": "reinspect",
        "evidence_fusion": "workflow",
    }.get(stage, "scene")
    if config.get("analysis_pass") == "single" and stage == "scene_analysis":
        extra = "\n\n".join(
            [
                "Return one JSON object with objects, observations, states, transitions, relationships, "
                "actions, workflow_steps, dependencies, and ambiguities.",
            ]
        )
        prompt_key = "scene"
    else:
        extra = ""
    if stage != "scene_analysis":
        prior = load_working(run.id)
        extra += (
            "\n\nPrior structured analysis JSON. Per-frame SAM boxes are omitted; "
            "the SAM summary is the geometric evidence:\n" + prompt_prior(prior)
        )
        extra += "\n\nReturn JSON for this stage only. Keep object ids stable."
    system = config.get("prompt_snapshot") or compose(prompt_key, run.research_mode, extra)
    if config.get("analysis_pass") == "single" and stage == "scene_analysis":
        system = compose("scene", run.research_mode, extra)
        system += "\n\n" + compose("actions", run.research_mode)
        system += "\n\n" + compose("states", run.research_mode)
        system += "\n\n" + compose("workflow", run.research_mode)
        system += "\n\n" + compose("ambiguity", run.research_mode)
    elif extra and "prompt_snapshot" not in config:
        system = compose(prompt_key, run.research_mode, extra)
    elif stage != "scene_analysis":
        system = compose(prompt_key, run.research_mode, extra)
    if purpose == "discovery":
        system = config.get("prompt_snapshot") or compose("scene", run.research_mode)
        system += "\n\n" + DISCOVERY
    elif purpose == "interpretation":
        system = compose("actions", run.research_mode, extra)
        for key in ("states", "workflow", "ambiguity"):
            system += "\n\n" + compose(key, run.research_mode)
        system += "\n\n" + INTERPRETATION

    interpreting = purpose == "interpretation"
    # These two calls are the only source of actions and steps for the run.
    single_pass = purpose == "full" and config.get("analysis_pass") == "single" and stage == "scene_analysis"
    parser = parse_interpretation if interpreting or single_pass else None
    contract = contract_for(run.provider_id, stage, purpose, config.get("analysis_pass"))
    legacy_model = InterpretationResponse if interpreting else AnalysisDocument
    schema = contract.schema if contract else legacy_model.model_json_schema()
    video_file = sends_video_file(run)
    user = (
        f"Recording duration {video.duration_sec:.3f} seconds. "
        f"Research mode {run.research_mode}. Input modality {run.modality}. "
        "Use only the attached evidence."
    )
    from app.pipeline.clip_time import load_transform

    subclip = load_transform(run.id) if video_file and run.provider_id != "cursor" else None
    offset = float(subclip.get("offset") or 0) if subclip else 0.0
    if video_file:
        times = [_frame_time(frame) for frame in _budget_frames(frames, int(config.get("max_frames") or 48))]
        clip_duration = float(video.duration_sec or 0)
        if subclip:
            clip_duration = float(subclip.get("duration") or clip_duration)
            times = [round(item - offset, 3) for item in times]
        user += _video_coverage_instruction(
            clip_duration,
            # Interpretation does not report boxes, so it gets no per-time observation grid.
            [] if interpreting and run.provider_id != "cursor" else times,
            frames_attached=run.provider_id == "cursor",
        )
        if offset:
            user += (
                f" The video file is the requested interval of the recording. File time zero is recording time "
                f"{offset:.3f}s. Report every time in file time. Tracking evidence below is in recording time; "
                f"subtract {offset:.3f}s to compare."
            )
        user += _displayed_frame_instruction(video)
        _note_video_coverage(coverage, timeline_frames=len(times) if run.provider_id == "cursor" else None)
    if user_extra:
        user += "\n\n" + user_extra
    sam_block = sam_evidence_block(run.id, stage)
    if sam_block:
        user += "\n\n" + sam_block
    if spatial_document is not None:
        tracked = evidence_text(spatial_document)
        if tracked:
            user += "\n\n" + tracked
    if video_file and run.provider_id != "cursor":
        document = await _call(
            session,
            run,
            stage,
            system,
            user,
            [],
            video,
            schema,
            attach_video=True,
            parser=parser,
            contract=contract,
        )
        if offset:
            from app.pipeline.clip_time import shift_to_source

            document = shift_to_source(document, offset)
        if purpose != "interpretation":
            # A surface on the roster spends a box on every frame and can never be a part.
            document = omit_workspace_and_hands(document)
            document = await _reground_video_boxes(session, run, stage, document, frames, video, coverage)
        return _with_perception(document, spatial_document)
    selected = _budget_frames(frames, int(config.get("max_frames") or 48))
    if not video_file:
        coverage.frames_submitted = len(selected)
    coverage.frame_timestamps = [frame.timestamp for frame in selected]
    windows = _windows(selected, int(config.get("max_images_per_call") or 16))
    partials: list[AnalysisDocument] = []
    for index, window in enumerate(windows):
        note = _window_note(index, windows)
        if partials:
            note += _closed_roster(partials[0])
        text_doc = await _call(
            session,
            run,
            stage,
            system,
            user + note,
            window,
            video,
            schema,
            attach_video=bool(video_file),
            parser=parser,
            contract=contract,
        )
        if partials:
            # The previous window is the one that shares this boundary.
            # The first window often does not.
            text_doc = align_to_roster(partials[-1], text_doc)
        partials.append(text_doc)
    if len(partials) == 1:
        return _with_perception(partials[0], spatial_document)
    merged = await _call(
        session,
        run,
        stage,
        system,
        _merge_instruction(windows) + "\n\n" + "\n\n".join(item.model_dump_json() for item in partials),
        [],
        video,
        schema,
        attach_video=False,
        parser=parser,
        contract=contract,
    )
    merged = drop_split_objects(partials[0], restore_omitted_records(merged, partials))
    return _with_perception(merged, spatial_document)


def _with_perception(document: AnalysisDocument, spatial: AnalysisDocument | None) -> AnalysisDocument:
    if spatial is None or spatial.perception is None:
        return document
    return apply_perception(document, spatial)


async def _call(
    session,
    run,
    stage,
    system,
    user,
    frames,
    video,
    schema,
    *,
    attach_video: bool = False,
    modality: str | None = None,
    parser=None,
    contract: OutputContract | None = None,
) -> AnalysisDocument:
    config = run.config_json or {}
    if run.input_tokens and False:
        pass
    sent = session.query(ModelRequestRow).filter(ModelRequestRow.run_id == run.id).count()
    if sent >= int(config.get("max_requests") or DEFAULT_MAX_REQUESTS):
        raise BudgetExceeded("request budget reached; no further model calls were made")
    transport = run_transport(run.provider_id, config)
    secret = ""
    if needs_secret(run.provider_id, transport):
        secret, _source = load_secret(session, run.provider_id)
        if not secret:
            raise UnsupportedMode(
                f"No API key is configured for {run.provider_id}. "
                "Add one on the provider page or export the documented environment variable."
            )
    provider = get_provider(run.provider_id, session)
    model_id, requested = model_for_call(session, run)
    video_path = _video_file(video, run if run.provider_id != "cursor" else None) if attach_video else None
    if video_path is not None:
        _require_video_length(run, video)
    # A video API sees the file itself. The Cursor CLI only reads images it is given,
    # so those calls keep the timeline frames beside the file.
    image_frames = [] if attach_video and run.provider_id != "cursor" else frames
    sent_modality = modality or run.modality
    request = GenerationRequest(
        model_id=model_id,
        system_prompt=system,
        user_prompt=user,
        frames=[FrameInput(frame.timestamp, data_path(*Path(frame.path).parts)) for frame in image_frames],
        video_path=video_path,
        modality=sent_modality,
        json_schema=schema,
        schema_name=contract.name if contract else "learnloop_analysis",
        temperature=config.get("temperature"),
        reasoning_effort=config.get("reasoning_effort"),
        image_detail=config.get("image_detail") or "high",
        timeout_s=float(config.get("timeout_s") or 180),
        metadata={"transport": transport} if run.provider_id in {"openai", "anthropic", "cursor"} else {},
    )
    estimate = rough_input_tokens(request)
    ceiling = config.get("cost_ceiling_usd")
    if ceiling is not None and run.cost_usd >= float(ceiling):
        raise BudgetExceeded(
            f"estimated spend {run.cost_usd:.4f} USD has reached the ceiling of {ceiling} USD"
        )
    row = open_provider_call(
        session,
        run,
        stage=stage,
        provider_id=run.provider_id,
        model_id=model_id,
        modality=sent_modality,
        prompt_version_id=run.prompt_version_id,
        parameters={
            "temperature": request.temperature,
            "reasoning_effort": request.reasoning_effort,
            "image_detail": request.image_detail,
            "estimated_input_tokens": estimate,
            "sdk_note": "recorded on the response",
            "schema_name": request.schema_name,
            "schema_hash": schema_hash(schema),
            **(
                {
                    "strict_output": True,
                    "output_token_limit": MAX_OUTPUT_TOKENS,
                    "schema_version": contract.version if contract else None,
                }
                if run.provider_id == "meta"
                else {}
            ),
            **({"requested_model_id": requested} if requested != model_id else {}),
            **({"openai_transport": transport} if run.provider_id == "openai" else {}),
            **({"anthropic_transport": transport} if run.provider_id == "anthropic" else {}),
            **({"cursor_transport": transport} if run.provider_id == "cursor" else {}),
        },
        frame_timestamps=[frame.timestamp for frame in frames],
    )
    session.commit()
    try:
        response = await provider.generate(secret, request)
    except Exception as exc:
        fail_provider_call(row, exc)
        session.commit()
        raise
    saved = finish_provider_call(session, run, row, response)
    session.commit()
    if contract is not None:
        return _accept_contract_output(session, contract, response, row, saved, stage)
    if response.incomplete_reason:
        raise ParseFailure(
            f"{model_id} stopped before finishing its answer ({response.incomplete_reason}) "
            f"after {response.usage.output_tokens} output tokens.",
            response.text,
        )
    from app.perception.resolve import mark_estimates

    parsed = (parser or parse_analysis)(response.text)
    return mark_estimates(parsed) if isinstance(parsed, AnalysisDocument) else parsed


def _accept_contract_output(session, contract, response, row, saved, stage) -> AnalysisDocument:
    """Validate against the contract that was sent. The raw text stays on the response row either way."""
    from app.perception.resolve import mark_estimates

    label = f"{stage} call {row.provider_request_id or row.id}"
    try:
        document = validate_output(contract, response.text, incomplete_reason=response.incomplete_reason, label=label)
    except StructuredOutputError as exc:
        row.error = str(exc)[:2000]
        saved.raw_json = {**(saved.raw_json or {}), "output_check": {"result": exc.code, **exc.diagnostics}}
        session.commit()
        raise
    saved.raw_json = {
        **(saved.raw_json or {}),
        "output_check": {"result": "valid", "raw_output_chars": len(response.text)},
    }
    session.commit()
    return mark_estimates(document)


def _video_file(video: VideoAsset, run: AnalysisRun | None = None):
    """The requested-interval subclip when preprocess cut one, else the playback mp4 or the original."""
    if run is not None:
        from app.pipeline.clip_time import load_transform, subclip_path

        if load_transform(run.id) is not None:
            return subclip_path(run.id)
    return resolve_stored(video.proxy_path or video.stored_path)


def _require_video_length(run: AnalysisRun, video: VideoAsset) -> None:
    limit = capabilities_for(run.provider_id, run.model_id).max_video_seconds
    duration = float(video.duration_sec or 0)
    from app.pipeline.clip_time import load_transform

    transform = load_transform(run.id) if getattr(run, "id", None) else None
    if transform is not None:
        duration = float(transform.get("duration") or duration)
    if limit and duration > float(limit):
        raise UnsupportedMode(
            f"{run.model_id} accepts at most {limit:g} seconds of video. "
            f"This recording is {duration:.1f} seconds. The file was not replaced with frames."
        )


def _video_coverage_instruction(duration: float, times: list[float], *, frames_attached: bool) -> str:
    span = ""
    if times:
        span = f" The timeline runs from {times[0]:.3f}s through {times[-1]:.3f}s."
    if frames_attached:
        return (
            f" The video file is the full {duration:.3f}s recording.{span} "
            "The images are moments from that recording, from the opening through the end. "
            "On every supplied image, box every object that is even partly visible, including when it has not moved. "
            "Cover the opening of the recording and the end of the recording."
        )
    listed = ", ".join(f"{stamp:.3f}" for stamp in times)
    grid = f" Include an observation for each visible object at each of these times, in seconds: {listed}." if listed else ""
    return (
        f" The video file is the full {duration:.3f}s recording.{span} "
        "Cover the recording from the first moment through the last."
        f"{grid} "
        "Include moments when an object is still, hidden, or back in view."
    )


def _displayed_frame_instruction(video: VideoAsset) -> str:
    width = int(video.display_width or 0)
    height = int(video.display_height or 0)
    if width <= 0 or height <= 0:
        return ""
    return (
        f" The displayed frame is {width} pixels wide and {height} pixels tall. "
        f"In every box, x and w are fractions of {width}, and y and h are fractions of {height}. "
        f"A square part has w/h = {height / width:.3f}."
    )


async def _reground_video_boxes(
    session, run, stage, document: AnalysisDocument, frames, video: VideoAsset, coverage: Coverage
):
    """Measure boxes on the displayed frames.

    A video answer can report the right width and height and still place the
    corner in the model's own canvas. The still frames are the picture the
    player shows, so those coordinates are the ones that land on the parts.
    """
    if not document.observations:
        return document
    width = int(video.display_width or 0)
    height = int(video.display_height or 0)
    config = run.config_json or {}
    selected = _budget_frames(frames, int(config.get("max_frames") or 48))
    if not selected:
        return document
    from app.perception.resolve import canonical_objects, fence_repair

    measured = canonical_objects(document)
    pending = [obj for obj in document.objects if obj.local_id not in measured]
    if not pending:
        logger.info("frame box pass skipped: every object already has canonical geometry")
        return document
    roster_ids = {obj.local_id for obj in pending}
    roster = "\n".join(f"{obj.local_id}: {obj.label}" for obj in pending)
    system = frame_box_prompt(width, height)
    merged = document
    measured: set[tuple[str, float]] = set()
    failed: list[tuple[float, float, str]] = []
    windows = _windows(selected, FRAME_BOX_WINDOW, overlap=0)
    for index, window in enumerate(windows):
        stamps = ", ".join(f"{_frame_time(frame):.3f}" for frame in window)
        user = (
            f"Objects:\n{roster}\n\n"
            f"The attached images are the frames at these times, in seconds, in order: {stamps}. "
            "Use those times on the observations. Box only the named parts."
        )
        try:
            fresh, reason = await _measure_window(
                session, run, stage, system, user, window, video, roster_ids, width, height
            )
        except SchemaContractError:
            raise
        except Exception:
            logger.exception("frame box pass failed; the video boxes were kept")
            for rest in windows[index:]:
                failed.append(_note_unmeasured(coverage, rest, "the call failed"))
            break
        if reason:
            logger.warning("frame box pass skipped: %s", reason)
            failed.append(_note_unmeasured(coverage, window, reason))
            continue
        merged = fence_repair(merged, lambda doc, answer=fresh: merge_frame_boxes(doc, answer))
        measured |= answered_keys(fresh)
    if measured:
        logger.info("replaced video-file boxes with boxes measured on the displayed frames")
    return mark_unmeasured(merged, roster_ids, measured, failed)


async def _measure_window(
    session, run, stage, system, user, window, video, roster_ids: set[str], width: int, height: int
) -> tuple[AnalysisDocument, str]:
    """One window's frame answer and why it was refused, asked again once when it was."""
    prompt = user
    for _attempt in range(FRAME_BOX_ATTEMPTS):
        fresh = await _call(
            session,
            run,
            stage,
            system,
            prompt,
            window,
            video,
            frame_box_schema(sorted(roster_ids)),
            attach_video=False,
            modality="sampled_frames",
            parser=lambda text, frame_w=width, frame_h=height: parse_frame_boxes(text, frame_w, frame_h),
        )
        reason = frame_box_rejection(fresh.observations, width, height, roster_ids)
        if not reason:
            break
        logger.warning("frame box answer refused: %s", reason)
        prompt = (
            f"{user} A previous answer for these frames was refused because {reason}. "
            "Look at each image again. Every named part that is visible needs its own box."
        )
    return fresh, reason


def _note_unmeasured(coverage: Coverage, window, reason: str) -> tuple[float, float, str]:
    """Say which moments still show unmeasured video-file boxes, so a run cannot look measured when it is not."""
    start, end = _frame_time(window[0]), _frame_time(window[-1])
    note = f"Boxes from {start:.1f}s to {end:.1f}s were not measured on frames: {reason}."
    coverage.temporal_coverage_note = f"{coverage.temporal_coverage_note or ''} {note}".strip()
    return start, end, reason


def _note_video_coverage(coverage: Coverage, *, timeline_frames: int | None = None) -> None:
    coverage.modality = "native_video"
    coverage.native_video_submitted = True
    if timeline_frames is None:
        coverage.frames_submitted = 0
        extra = "The model receives the video file. Preview frames stay in the inspector."
    else:
        coverage.frames_submitted = timeline_frames
        extra = "The video file is sent with the timeline frames so the whole recording is visible."
    note = coverage.temporal_coverage_note or ""
    if "video file" not in note.lower():
        coverage.temporal_coverage_note = (note + " " + extra).strip()


def _budget_frames(frames: list[FrameAsset], limit: int) -> list[FrameAsset]:
    ordered = sorted(frames, key=lambda item: item.timestamp)
    return [ordered[index] for index in budget_indices(len(ordered), limit)]


# Adjacent calls repeat this many frames so a window boundary is a shared moment,
# not a fresh scene. Two frames is enough to show the arrangement on both sides.
WINDOW_OVERLAP = 2


def _windows(
    frames: list[FrameAsset], size: int, overlap: int = WINDOW_OVERLAP
) -> list[list[FrameAsset]]:
    size = max(size, 1)
    if not frames:
        return [[]]
    if len(frames) <= size:
        return [list(frames)]
    overlap = min(max(overlap, 0), size - 1)
    step = size - overlap
    windows: list[list[FrameAsset]] = []
    start = 0
    while start < len(frames):
        windows.append(frames[start : start + size])
        if start + size >= len(frames):
            break
        start += step
    return windows


def _frame_time(frame: FrameAsset) -> float:
    return float(frame.timestamp)


def _shared_times(left: list[FrameAsset], right: list[FrameAsset]) -> list[float]:
    right_times = {round(_frame_time(frame), 3) for frame in right}
    return [_frame_time(frame) for frame in left if round(_frame_time(frame), 3) in right_times]


def _fmt_times(times: list[float]) -> str:
    parts = [f"{value:.3f}".rstrip("0").rstrip(".") + " s" for value in times]
    if len(parts) <= 1:
        return parts[0] if parts else ""
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + ", and " + parts[-1]


def _closed_roster(roster: AnalysisDocument) -> str:
    lines = [f"{obj.local_id}: {obj.label}" for obj in roster.objects if obj.label]
    if not lines:
        return ""
    return (
        "\n\nThe objects are already named. Use only these ids. A part seen at a new angle, "
        "including a second arm of a piece that was one object, keeps that id and one box around "
        "the whole piece. Do not add an object.\n" + "\n".join(lines)
    )


def _window_note(index: int, windows: list[list[FrameAsset]]) -> str:
    total = len(windows)
    sentences = [f"This is window {index + 1} of {total} in temporal order."]
    if total > 1:
        if index > 0:
            shared = _shared_times(windows[index - 1], windows[index])
            if shared:
                sentences.append(
                    f"The opening frames at {_fmt_times(shared)} repeat the end of the previous window "
                    "and are the same moments."
                )
        if index + 1 < total:
            shared = _shared_times(windows[index], windows[index + 1])
            if shared:
                sentences.append(
                    f"The closing frames at {_fmt_times(shared)} are repeated at the start of the next window."
                )
        sentences.append(
            "Continue existing object ids across repeated frames. A later view that shows fewer pieces "
            "because of occlusion, stacking, or an end-on angle keeps those ids and records an ambiguity. "
            "Create a new object id only when appearance contradicts every existing object."
        )
    sentences.append(_boxing_rule(windows[index]))
    return "\n\n" + " ".join(sentences)


def _boxing_rule(frames: list[FrameAsset]) -> str:
    times = _fmt_times([_frame_time(frame) for frame in frames])
    where = f"The supplied frames are at {times}. " if times else ""
    return (
        where
        + "On each of those frames, give every object that is even partly visible its own observation with a "
        "tight box around that object alone. Leave an object out of a frame when it is fully hidden, and do "
        "not keep its id by boxing a different part."
    )


def _merge_instruction(windows: list[list[FrameAsset]]) -> str:
    shared: list[float] = []
    for left, right in zip(windows, windows[1:], strict=False):
        for timestamp in _shared_times(left, right):
            if timestamp not in shared:
                shared.append(timestamp)
    boundary = f" The shared timestamps are {_fmt_times(shared)}." if shared else ""
    return (
        "Merge these partial JSON analyses in time order. Adjacent windows share their boundary frames, "
        "and those shared timestamps are the same moments."
        + boundary
        + " Continue object ids across the boundary. When a later window shows fewer pieces of an arrangement "
        "already described, because the parts are occluded, stacked, or seen end-on, keep the earlier ids "
        "and record which of them remain possible as an ambiguity. Create a new object id only when appearance "
        "contradicts every earlier object. Do not duplicate actions that fall on the shared frames. "
        "Keep every observation. A shared timestamp is one moment: keep one box per object there, "
        "from a window that saw the frame. Do not retain both boxes, and do not describe the overlap "
        "as extra frames. If a window boxed an object at a timestamp, that box stays; do not drop "
        "it because the object did not move or because a larger object already has a box. "
        "Do not invent events from gaps between windows. Keep every state, transition, relationship, "
        "workflow step, dependency, and ambiguity from the windows."
    )
