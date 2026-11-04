"""hackathon_guided: one Meta call for a short fixed-camera procedure.

No local tracking and no automatic retry. A demonstration gets SAM object views
for the page and appearance references (see object_views.py); an attempt gets the
same views, checked against those references. Boxes never decide a verdict. A response
is stored and replayed only for the exact video, model, prompt, and schema it
answered. A cached answer is labeled cached. A failed answer is reported, kept
on its request row, and never cached.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sqlalchemy.orm import Session

from app.config import BACKEND_ROOT, settings
from app.models.media import VideoAsset
from app.models.runs import AnalysisRun, Job, ModelRequestRow
from app.models.util import utcnow
from app.pipeline.contracts import GUIDED_TASK, StructuredOutputError, schema_hash, validate_output
from app.pipeline.prompts import CAMERA_CONTEXT, FOUNDATIONAL
from app.providers.types import (
    GenerationRequest,
    ProviderError,
    UnsupportedMode,
    rough_input_tokens,
)
from app.schemas.analysis import AnalysisDocument, Coverage
from app.services.storage import data_path

PROFILE = "hackathon_guided"
DEFAULT_MODEL = "muse-wonder-1.0"
DEFAULT_SEGMENTATION_MODEL = "sam-3.1"
MODALITY = "native_video"
MAX_CLIP_SECONDS = 60.0
MAX_REQUESTS = 1
REPLAY_DIR = BACKEND_ROOT.parent / "demo" / "replay"

# Stages that do not run for this profile, and what each one would have done.
SKIPPED_STAGES = {
    "spatial_tracking": "hackathon_guided verifies results, not tracks. No local tracker or appearance model ran.",
    "action_analysis": "hackathon_guided asks for actions in its one call.",
    "state_tracking": "hackathon_guided asks for transitions and relationships in its one call.",
    "segmentation": "SAM is off for hackathon_guided. No geometry was generated.",
    "evidence_fusion": "hackathon_guided has no tracked geometry to fuse.",
    "workflow": "hackathon_guided asks for the procedure in its one call.",
    "ambiguity": "hackathon_guided asks for ambiguities in its one call.",
    "reinspection": "hackathon_guided makes no automatic follow-up calls.",
    "transcription": "Whisper is not part of this build.",
    "audio_context": "Whisper is not part of this build.",
}

_TASK = """
This is a short recording from a fixed camera of a person handling a few distinct objects.
Return only what matters for checking the physical result. Do not return boxes or per-frame tracks.

objects: the objects the task acts on, and a base or container they end up on. Leave out hands and the bare work surface. The label is short and names the kind of part, its color, and any shape feature that tells it apart from similar parts, such as a split, teeth, a flange, a slot, or a thread. visual_characteristics states those features. Identical parts each get their own object with the same label.
observations: for each object, one observation at the first moment and one at the last moment of the recording, with its visibility at that moment. Use visible, partial, occluded, or absent. Use absent only when the place the object would be is in view and the object is not there, and classify that DIRECTLY_OBSERVED. If the place is hidden or out of frame, use occluded and classify it UNKNOWN.
relationships: for each pair that ends in a physical arrangement, one relationship. The subject is the object that was moved. The other is the part in direct contact that holds it. When parts are stacked, do not skip to the largest body: name the nearest part it was pressed into or onto, judged from the last clear view before and the first clear view after the hand moves away. time_start is when the arrangement is first achieved. time_end is the last moment it is visible. Choose the most specific kind: inside, attached, supported_by, above, below, left_of, right_of, around, touching. Use inside only when most of the part disappears into the other. A part that stays mostly visible while held on is attached. Do not use near for a completed placement. Classify it DIRECTLY_OBSERVED only when the arrangement itself can be seen in the frames, and INFERRED or UNKNOWN when it cannot. If an object was moved and the arrangement is wrong or incomplete, report the arrangement that is actually there.
transitions: to_state and from_state are chosen from: separate, placed, attached, inserted, removed, open, closed, flipped, empty, full, other.
actions: one action per physical operation, with its outcome. An attempt that did not reach a result is attempted or failed, not completed.
workflow_steps: the ordered procedure seen in the recording, usually one step per placed or changed object. required_objects lists every object in the step, including the object it is placed on. action_local_ids lists its actions. time_start and time_end bound the work. completion_condition states the visible end state. observed_result states what was seen.
dependencies: only between different steps. Use necessary_physical only when the physics requires the order.
ambiguities: anything hidden, unclear, or not verifiable, including hand occlusion of the final state.
Keep descriptions and notes to one short sentence.
""".strip()

_REFERENCE = """
A reference list of objects from an earlier recording follows. It says what to look for. It does not say this attempt is correct, and it may be wrong or incomplete: parts may be missing, misplaced, swapped, or extra.
Rules for the list:
- Use an entry's exact label only for an object that is the same kind of part: the same shape, with the features the label and description name, such as a split, teeth, a flange, a slot, or a thread. A different color or finish alone does not make it a different part, so it keeps the entry's label.
- An object that looks similar but is a different kind of part, such as a flat ring where the entry is a split ring, gets its own label naming what it is, and visual_characteristics says how it differs from the entry. Never give it the entry's label. Look closely at the shape before deciding.
- Every entry gets observations, including one that is never seen. If it is not seen, still list it as an object, with visibility absent or occluded under the rules above.
- For each pair to check, report the arrangement you see at the end, if any. Do not report an arrangement you cannot see.
""".strip()


def model_choices(session: Session) -> dict:
    """The account's models for each job, as vision-test-bench splits them: SAM segments, the rest analyze."""
    from app.providers.catalog import models_for
    from app.providers.model_list import account_options, choose_fallback

    options = account_options(session, "meta")
    inference = [item for item in options if not _is_segmenter(item) and item.get("native_video")]
    # The account list may leave SAM out, and it is called through its own endpoint.
    segmentation = [item for item in options if _is_segmenter(item)] or [caps.as_dict() for caps in models_for("meta") if _is_segmenter(caps.as_dict())]
    if not settings.object_views:
        segmentation = []
    default_inference = DEFAULT_MODEL if _offers(inference, DEFAULT_MODEL) else choose_fallback(inference, native_video=True)
    default_segmentation = DEFAULT_SEGMENTATION_MODEL if _offers(segmentation, DEFAULT_SEGMENTATION_MODEL) else next((str(item["model_id"]) for item in segmentation), "")
    return {
        "inference": inference,
        "segmentation": segmentation,
        "default_inference": default_inference,
        "default_segmentation": default_segmentation,
    }


def _is_segmenter(item: dict) -> bool:
    return str(item.get("model_id")).startswith("sam")


def _offers(options: list[dict], model_id: str) -> bool:
    return any(item.get("model_id") == model_id for item in options)


def choose_models(session: Session, model_id: str, segmentation_model_id: str | None) -> tuple[str, str]:
    """Apply a person's picks. No inference pick or None for segmentation uses the default; "" turns segmentation off."""
    choices = model_choices(session)
    model = model_id or choices["default_inference"]
    if not _offers(choices["inference"], model):
        raise UnsupportedMode(f"{model} cannot analyze a video on this account.")
    segmentation = choices["default_segmentation"] if segmentation_model_id is None else segmentation_model_id
    if segmentation and not _offers(choices["segmentation"], segmentation):
        raise UnsupportedMode(f"{segmentation} is not available for segmentation.")
    return model, segmentation


def plan(
    video: VideoAsset, *, template: dict | None = None, model_id: str = DEFAULT_MODEL, segmentation_model_id: str = "", fresh: bool = False
) -> dict:
    """What a call would send, whether a stored answer will be replayed, and what it could cost. Fresh replays none."""
    from app.pipeline import object_views

    system, user = _prompts(video, template)
    key = cache_key(video.checksum_sha256, model_id, system, user)
    follows = follows_objects(segmentation_model_id, template)
    request = GenerationRequest(model_id=model_id, system_prompt=system, user_prompt=user, modality=MODALITY)
    return {
        "profile": PROFILE,
        "video_id": video.id,
        "video_seconds": video.duration_sec,
        "provider_id": "meta",
        "model_id": model_id,
        "modality": MODALITY,
        "max_requests": MAX_REQUESTS,
        "automatic_retries": 0,
        "max_output_tokens": settings.guided_max_output_tokens,
        "estimated_input_tokens": rough_input_tokens(request),
        # Meta lists no rate in this repository, so no figure is invented here.
        "estimated_cost_usd": None,
        "cache_key": key,
        "cached": not fresh and find_cached(key) is not None,
        "fresh": fresh,
        "live_enabled": settings.live_inference,
        "segmentation_model_id": segmentation_model_id if follows else "",
        # A second follows, forward only, only when some object is boxed after the first frame. Recovery may add more.
        "object_view_requests": 2 + (settings.recovery_max_sam_requests if settings.appearance_enabled and settings.appearance_recovery else 0) if follows else 0,
        # One still on the first frame, and a second only when some object is not on it or SAM loses one.
        "object_view_locate_requests": 2 if follows else 0,
        "object_view_frames": object_views.estimated_frames(float(video.duration_sec or 0)),
    }


def create_guided_run(
    session: Session,
    video: VideoAsset,
    *,
    template: dict | None = None,
    template_id: str = "",
    allow_live: bool = False,
    model_id: str = DEFAULT_MODEL,
    segmentation_model_id: str = "",
    fresh: bool = False,
) -> AnalysisRun:
    from app.services.experiments import create_run

    if float(video.duration_sec or 0) > MAX_CLIP_SECONDS:
        raise ValueError(
            f"This recording is {video.duration_sec:.0f} seconds. LearnLoop supports up to {MAX_CLIP_SECONDS:.0f} seconds."
        )
    system, user = _prompts(video, template)
    run = create_run(
        session,
        {
            "video_id": video.id,
            "provider_id": "meta",
            "model_id": model_id,
            "modality": MODALITY,
            "research_mode": "fixed_camera",
            "sampling_preset": "sparse",
            "analysis_pass": "single",
            "semantic_mode": "combined",
            "segmentation_mode": "disabled",
            "max_requests": MAX_REQUESTS,
            "name": f"{video.original_filename} · {model_id}",
        },
        enqueue=False,
    )
    if run.model_id != model_id:
        # The account list replaced the model. This profile does not switch silently.
        requested, offered = model_id, run.model_id
        session.delete(run)
        session.commit()
        raise UnsupportedMode(f"{requested} is not available on this account. It lists {offered}. Nothing was sent.")
    run.config_json = {
        **(run.config_json or {}),
        "analysis_profile": PROFILE,
        "guided_system": system,
        "guided_user": user,
        "cache_key": cache_key(video.checksum_sha256, model_id, system, user),
        "allow_live": bool(allow_live),
        # Stored answers are skipped and the new ones replace them.
        "fresh": bool(fresh),
        "reference_template_id": template_id,
        "max_output_tokens": settings.guided_max_output_tokens,
        "max_retries": 0,
        "inference_source": "pending",
        "object_views": follows_objects(segmentation_model_id, template),
        "segmentation_model_id": segmentation_model_id if follows_objects(segmentation_model_id, template) else "",
    }
    session.add(Job(run_id=run.id, max_attempts=1))
    session.commit()
    session.refresh(run)
    return run


def follows_objects(segmentation_model_id: str, template: dict | None) -> bool:
    """A demonstration is followed with SAM for its views and its appearance references. An attempt is followed
    only when appearance checks are on, so its boxes can be checked against the procedure's references."""
    if not segmentation_model_id:
        return False
    return not template or settings.appearance_enabled


def is_guided(run: AnalysisRun) -> bool:
    return (run.config_json or {}).get("analysis_profile") == PROFILE


async def run_guided_call(session: Session, run: AnalysisRun, video: VideoAsset) -> AnalysisDocument:
    """The one semantic call, or its stored answer. Validation is the same for both."""
    config = dict(run.config_json or {})
    key = config["cache_key"]
    stored = None if config.get("fresh") else find_cached(key)
    if stored is not None:
        document = _check(stored["text"], "", "cached", config)
        run.config_json = {
            **config,
            "inference_source": "cached",
            "cached_request_id": stored.get("request_id", ""),
            "cached_at": stored.get("created_at", ""),
        }
        session.commit()
        return _with_coverage(document, run, video)
    if not config.get("allow_live"):
        raise UnsupportedMode("No stored answer matches this recording, and a live call was not approved. Nothing was sent.")
    if not settings.live_inference:
        raise UnsupportedMode("Live Meta calls are off. Set LEARNLOOP_LIVE_INFERENCE=1 to allow them. Nothing was sent.")
    return await _live(session, run, video, config, key)


async def _live(session, run, video, config, key) -> AnalysisDocument:
    from app.pipeline.infer import _require_video_length, _video_file
    from app.providers.registry import get_provider
    from app.services.costs import fail_provider_call, finish_provider_call, open_provider_call
    from app.services.credentials import load_secret

    sent = session.query(ModelRequestRow).filter(ModelRequestRow.run_id == run.id, ModelRequestRow.stage == "scene_analysis").count()
    if sent >= MAX_REQUESTS:
        raise UnsupportedMode("This run already used its one request. Start a new analysis to try again.")
    secret, _source = load_secret(session, "meta")
    if not secret:
        raise UnsupportedMode("No Meta API key is configured. Set MODEL_API_KEY in .env. Nothing was sent.")
    _require_video_length(run, video)
    request = GenerationRequest(
        model_id=run.model_id,
        system_prompt=config["guided_system"],
        user_prompt=config["guided_user"],
        video_path=_video_file(video),
        modality=MODALITY,
        json_schema=GUIDED_TASK.schema,
        schema_name=GUIDED_TASK.name,
        timeout_s=float(config.get("timeout_s") or 180),
        max_output_tokens=int(config.get("max_output_tokens") or settings.guided_max_output_tokens),
    )
    row = open_provider_call(
        session,
        run,
        stage="scene_analysis",
        provider_id="meta",
        model_id=run.model_id,
        modality=MODALITY,
        prompt_version_id=run.prompt_version_id,
        parameters={
            "profile": PROFILE,
            "schema_name": GUIDED_TASK.name,
            "schema_hash": GUIDED_TASK.hash,
            "strict_output": True,
            "output_token_limit": request.max_output_tokens,
            "estimated_input_tokens": rough_input_tokens(request),
            "cache_key": key,
        },
    )
    session.commit()
    try:
        response = await get_provider("meta", session).generate(secret, request)
    except ProviderError as exc:
        fail_provider_call(row, exc)
        session.commit()
        raise
    except Exception as exc:
        fail_provider_call(row, exc)
        session.commit()
        raise
    saved = finish_provider_call(session, run, row, response)
    session.commit()
    try:
        document = _check(response.text, response.incomplete_reason, row.provider_request_id or row.id, config)
    except StructuredOutputError as exc:
        row.error = str(exc)[:2000]
        saved.raw_json = {**(saved.raw_json or {}), "output_check": {"result": exc.code, **exc.diagnostics}}
        session.commit()
        raise
    saved.raw_json = {**(saved.raw_json or {}), "output_check": {"result": "valid", "raw_output_chars": len(response.text)}}
    store_cached(
        key,
        {
            "text": response.text,
            "request_id": response.request_id,
            "model_id": run.model_id,
            "modality": MODALITY,
            "video_checksum": video.checksum_sha256,
            "schema_hash": GUIDED_TASK.hash,
            "usage": {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens},
            "created_at": utcnow().isoformat(),
        },
    )
    run.config_json = {**config, "inference_source": "live", "request_id": response.request_id}
    session.commit()
    return _with_coverage(document, run, video)


def _check(text: str, incomplete: str, label: str, config: dict) -> AnalysisDocument:
    document = validate_output(GUIDED_TASK, text, incomplete_reason=incomplete, label=f"guided call {label}")
    if not document.objects:
        raise StructuredOutputError(
            "empty_analysis", f"guided call {label}: the answer names no objects, so there is nothing to learn or check.", text
        )
    if not config.get("reference_template_id") and not document.workflow_steps:
        raise StructuredOutputError(
            "empty_analysis", f"guided call {label}: the answer has no procedure steps. No steps were invented.", text
        )
    return document


def _with_coverage(document: AnalysisDocument, run: AnalysisRun, video: VideoAsset) -> AnalysisDocument:
    from app.pipeline.infer import _note_video_coverage

    coverage = Coverage.model_validate(run.coverage_json or {"modality": MODALITY, "sampling_preset": "sparse"})
    _note_video_coverage(coverage)
    coverage.duration_sec = float(video.duration_sec or 0)
    coverage.semantic_calls = 1
    run.coverage_json = coverage.model_dump(mode="json")
    return document.model_copy(update={"coverage": coverage})


def _prompts(video: VideoAsset, template: dict | None) -> tuple[str, str]:
    parts = [FOUNDATIONAL, CAMERA_CONTEXT["fixed_camera"], _TASK]
    user = (
        f"Recording duration {video.duration_sec:.3f} seconds. Timestamps are seconds from the start of this file. "
        "Use only the attached video."
    )
    if template:
        parts.append(_REFERENCE)
        user += "\n\n" + _reference_block(template)
    return "\n\n".join(parts), user


def _reference_block(template: dict) -> str:
    """Objects and pairs to look at. The expected relation is deliberately not sent."""
    labels = {role["role_id"]: role.get("label") or role["role_id"] for role in template.get("roles") or []}
    entries = [
        f"- {role.get('label')}: {role.get('visual_characteristics') or role.get('description') or ''}".rstrip(": ")
        for role in template.get("roles") or []
    ]
    pairs = []
    for step in template.get("steps") or []:
        if step.get("removed"):
            continue
        for item in step.get("completion_conditions") or []:
            if item.get("target_role"):
                pair = f"- {labels.get(item['subject_role'], '')} and {labels.get(item['target_role'], '')}"
                if pair not in pairs:
                    pairs.append(pair)
    return "Reference objects:\n" + "\n".join(entries) + "\n\nPairs to check:\n" + ("\n".join(pairs) or "- none")


def cache_key(checksum: str, model_id: str, system: str, user: str) -> str:
    parts = {
        "video": checksum,
        "model": model_id,
        "modality": MODALITY,
        "system": system,
        "user": user,
        "schema": schema_hash(GUIDED_TASK.schema),
        "max_output_tokens": settings.guided_max_output_tokens,
    }
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


def _cache_file(key: str) -> Path:
    return data_path("inference_cache", f"{key}.json")


def find_cached(key: str) -> dict | None:
    for path in (_cache_file(key), REPLAY_DIR / f"{key}.json"):
        if path.exists():
            return json.loads(path.read_text())
    return None


def store_cached(key: str, record: dict) -> None:
    path = _cache_file(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2))
