"""Automatic second look at required steps a verification could not decide.

After a report is made, each required CANNOT_VERIFY step (a few at most) gets one
question to the attempt's analysis model: its unresolved conditions and a handful
of sharp frames from that attempt. The answer is checked against what was asked
and what was sent before it can change a predicate, and the engine still decides
the verdict. Nothing is sent unless the person approved live calls for that
attempt and the server allows them; a stored answer for the same question and
frames is replayed. The first machine report is kept as it was.
"""

from __future__ import annotations

import hashlib
import json
import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models.media import VideoAsset
from app.models.runs import AnalysisRevision, AnalysisRun, Job
from app.models.util import utcnow
from app.models.verification import (
    LearnedWorkflowTemplate,
    VerificationReinspection,
    VerificationRunRow,
)
from app.pipeline.contracts import REINSPECT
from app.providers.types import FrameInput, GenerationRequest
from app.schemas.generation import ReinspectOutput
from app.services.storage import data_path
from app.verification.read import as_document

log = logging.getLogger("learnloop.reinspection")

SYSTEM = """
You are checking whether specific physical conditions hold in still frames from one recording of a person assembling parts. Each image follows a line with its timestamp.
For each listed condition, return one finding with its step_id and predicate_id copied exactly.
outcome: supported only when the frames directly show the condition holding. contradicted only when the frames directly show it not holding, for example the part is visibly somewhere else, visibly absent from where it should be, or visibly the wrong part. Otherwise inconclusive.
Do not assume the expected condition is true. A part that cannot be seen is not missing: if a hand, another part, or the frame edge hides where it would be, the outcome is inconclusive and visibility is occluded or out_of_frame.
visibility describes the place the condition is about in the clearest frame: visible, partial, occluded, or out_of_frame.
evidence_timestamps lists only timestamps of supplied images that show what you report, copied exactly. Leave it empty for inconclusive.
evidence_class is DIRECTLY_OBSERVED only when the deciding pixels are in a supplied image; INFERRED when you reasoned from context; UNKNOWN otherwise.
reason is one short sentence naming what is seen. limitations lists anything that weakens the reading.
Return JSON only.
""".strip()
_OUTPUT_TOKENS = 4_000
# A moment closer than this to one already chosen adds little.
_MERGE_SEC = 0.4
# The sharpest frame within this distance of a chosen moment is sent instead of it.
_SHARP_WITHIN = 0.25


# ---------------------------------------------------------------- scheduling


def schedule(session: Session, row: VerificationRunRow, report: dict) -> dict:
    """Decide whether a second look runs, queue it, and say so in the report. Returns the report's reinspection block."""
    targets = targets_of(report)
    if not settings.auto_reinspect:
        return {"status": "off", "reason": "Automatic reinspection is turned off."}
    if not targets:
        return {"status": "not_needed", "reason": "No required step was left unresolved."}
    run = session.get(AnalysisRun, row.execution_run_id)
    approved = bool(run and (run.config_json or {}).get("allow_live")) and settings.live_inference
    cached = all(_cached(_key(session, row, target)) is not None for target in targets)
    if not approved and not cached:
        return {
            "status": "unavailable",
            "reason": "A second look needs a live Meta call, and live calls were not approved for this recording. The unresolved steps keep their verdict.",
            "steps": [target["step_id"] for target in targets],
        }
    stage = f"reinspect:{row.id}"
    if session.scalar(select(Job.id).where(Job.run_id == row.execution_run_id, Job.from_stage == stage)) is None:
        session.add(Job(run_id=row.execution_run_id, from_stage=stage, max_attempts=1))
    return {"status": "pending", "steps": [target["step_id"] for target in targets], "max_calls": settings.reinspect_max_calls}


def targets_of(report: dict) -> list[dict]:
    """Required CANNOT_VERIFY steps a reviewer has not decided, with their unresolved required predicates."""
    chosen = []
    for step in report.get("steps") or []:
        if step.get("verdict") != "CANNOT_VERIFY" or not step.get("required"):
            continue
        if step.get("basis") in {"human_override", "human_confirmed"}:
            continue
        predicates = [item for item in step.get("predicates") or [] if item.get("status") == "cannot" and not item.get("human") and not item.get("reinspected")]
        if not predicates:
            continue
        chosen.append({"step_id": step["step_id"], "step": step, "predicates": predicates})
        if len(chosen) >= settings.reinspect_max_calls:
            break
    return chosen


# ---------------------------------------------------------------- the job


async def run_reinspection_job(session: Session, job: Job, verification_id: str) -> None:
    from app.services import verification_store as store

    row = session.get(VerificationRunRow, verification_id)
    if row is None:
        _finish(session, job, "canceled")
        return
    report = dict(row.report_json or {})
    targets = targets_of(report)
    outcomes = []
    try:
        for target in targets:
            outcomes.append(await _inspect(session, row, target))
            session.commit()
        status = "completed"
        reason = ""
    except Exception as exc:
        session.rollback()
        log.exception("reinspection of %s failed", verification_id)
        status, reason = "failed", str(exc)[:400]
    row = session.get(VerificationRunRow, verification_id)
    store.apply_reinspections(session, row, {"status": status, "reason": reason, "outcomes": outcomes})
    _finish(session, job, "completed")


def _finish(session, job: Job, status: str) -> None:
    job.status = status
    job.finished_at = utcnow()
    session.commit()


async def _inspect(session: Session, row: VerificationRunRow, target: dict) -> dict:
    run = session.get(AnalysisRun, row.execution_run_id)
    video = session.get(VideoAsset, row.execution_video_id)
    revision = session.get(AnalysisRevision, row.execution_revision_id)
    template = session.get(LearnedWorkflowTemplate, row.template_id)
    if run is None or video is None or revision is None or template is None:
        return {"step_id": target["step_id"], "status": "failed", "reason": "The attempt or its procedure is missing."}
    times = choose_frames(target["step"], revision.document_json, video)
    question = _question(target, template.document_json)
    key = _key(session, row, target, times, question)
    stored = _cached(key)
    request_id, source = "", "cached"
    if stored is not None:
        text = stored["text"]
        request_id = stored.get("request_id") or ""
    else:
        if not ((run.config_json or {}).get("allow_live") and settings.live_inference):
            return _record(session, row, target, times, question, "not_sent", "Live calls were not approved for this recording.")
        text, request_id, error = await _call(session, run, video, times, question)
        source = "live"
        if text is None:
            return _record(session, row, target, times, question, "failed", error)
    try:
        answer = ReinspectOutput.model_validate_json(text, strict=True)
    except ValueError as exc:
        return _record(session, row, target, times, question, "invalid", f"The answer did not match its contract: {str(exc)[:300]}")
    if source == "live":
        _store(key, {"text": text, "request_id": request_id, "model_id": run.model_id, "created_at": utcnow().isoformat()})
    findings, problems = validate_findings(answer, target, times)
    for finding in findings:
        finding["model_id"] = run.model_id
        finding["request_id"] = request_id
        finding["source"] = source
    return _record(session, row, target, times, question, "answered", "; ".join(problems), findings, request_id, run.model_id, source)


def _record(session, row, target, times, question, status, reason, findings=None, request_id="", model_id="", source="") -> dict:
    session.add(
        VerificationReinspection(
            verification_id=row.id,
            step_id=target["step_id"],
            question=question,
            frame_timestamps=times,
            status=status,
            output_json={"findings": findings or [], "reason": reason, "request_id": request_id, "model_id": model_id, "source": source},
            previous_verdict=target["step"].get("verdict") or "",
        )
    )
    return {"step_id": target["step_id"], "status": status, "reason": reason, "findings": len(findings or []), "request_id": request_id}


def validate_findings(answer: ReinspectOutput, target: dict, times: list[float]) -> tuple[list[dict], list[str]]:
    """Keep findings that answer what was asked, citing frames that were sent. Two answers for one condition
    that disagree count as none. A supported or contradicted finding without direct visible evidence is
    kept as inconclusive."""
    asked = {item["predicate_id"] for item in target["predicates"]}
    kept: dict[str, dict] = {}
    problems: list[str] = []
    conflicted: set[str] = set()
    for item in answer.findings:
        if item.step_id != target["step_id"] or item.predicate_id not in asked:
            problems.append(f"ignored a finding for {item.step_id}/{item.predicate_id}, which was not asked")
            continue
        stamps = []
        for value in item.evidence_timestamps:
            match = next((sent for sent in times if abs(sent - value) <= 0.05), None)
            if match is None:
                problems.append(f"{item.predicate_id} cited {value:.2f}s, which was not sent")
            else:
                stamps.append(match)
        finding = item.model_dump()
        finding["evidence_timestamps"] = sorted(set(stamps))
        if finding["outcome"] != "inconclusive" and not (
            finding["evidence_class"] == "DIRECTLY_OBSERVED" and finding["visibility"] == "visible" and finding["evidence_timestamps"]
        ):
            finding["limitations"] = [*finding["limitations"], f"Reported {finding['outcome']} without a directly observed, visible frame that was sent."]
            finding["outcome"] = "inconclusive"
        previous = kept.get(item.predicate_id)
        if previous is not None and previous["outcome"] != finding["outcome"]:
            conflicted.add(item.predicate_id)
        kept[item.predicate_id] = finding
    for predicate_id in conflicted:
        kept[predicate_id] = {**kept[predicate_id], "outcome": "inconclusive", "limitations": ["The answer gave conflicting outcomes."]}
        problems.append(f"{predicate_id} had conflicting outcomes")
    return list(kept.values()), problems


# ---------------------------------------------------------------- frames and prompt


def choose_frames(step: dict, execution: dict, video: VideoAsset) -> list[float]:
    """A few sharp moments from this attempt: around the matched work, after it, and the end state.
    Demonstration timestamps are not used: the attempt has its own timing."""
    from app.verification.predicates import _final_window

    document = as_document(execution)
    duration = max(float(video.duration_sec or 0) - 0.1, 0.0)
    moments: list[float] = []
    actions = {action.local_id: action for action in document.actions}
    for action_id in step.get("matched_action_ids") or []:
        action = actions.get(action_id)
        if action is not None:
            moments += [action.start, (action.start + action.end) / 2, action.end, action.end + 1.0]
    for evidence in step.get("evidence") or []:
        if evidence.get("timestamp") is not None:
            moments.append(float(evidence["timestamp"]))
    if step.get("execution_time") is not None:
        moments.append(float(step["execution_time"]))
    start, end = _final_window(document)
    # The end state, after the hands have usually moved away.
    moments += [start, (start + duration) / 2, duration]
    kept: list[float] = []
    for moment in sorted(min(max(float(value), 0.0), duration) for value in moments):
        if all(abs(moment - other) >= _MERGE_SEC for other in kept):
            kept.append(round(moment, 3))
    limit = max(settings.reinspect_max_frames, 2)
    if len(kept) > limit:
        # Keep the last frame, and spread the rest.
        step_size = (len(kept) - 1) / (limit - 1)
        kept = [kept[round(index * step_size)] for index in range(limit - 1)] + [kept[-1]]
    return _sharpest(video, list(dict.fromkeys(kept)), duration)


def _sharpest(video: VideoAsset, moments: list[float], duration: float) -> list[float]:
    from app.services.ingest import playback_path
    from app.video.frames import extract_frame, read_bgr, sharpness_score

    path = playback_path(video)
    chosen = []
    for moment in moments:
        options = sorted({round(min(max(moment + offset, 0.0), duration), 3) for offset in (-_SHARP_WITHIN, 0.0, _SHARP_WITHIN)})
        scored = []
        for value in options:
            image = _still(video, value)
            try:
                if not image.exists():
                    extract_frame(path, value, image, accurate=True)
                frame = read_bgr(image)
                scored.append((sharpness_score(frame), -abs(value - moment), value))
            except Exception:
                continue
        if scored:
            chosen.append(max(scored)[2])
    return sorted(dict.fromkeys(chosen))


def _still(video: VideoAsset, moment: float):
    return data_path("reinspect_frames", video.id, f"{moment:.3f}.jpg")


def _question(target: dict, template: dict) -> str:
    roles = {role["role_id"]: role for role in template.get("roles") or []}
    conditions = {}
    for step in template.get("steps") or []:
        if step.get("step_id") == target["step_id"]:
            conditions = {item["id"]: item for item in step.get("completion_conditions") or []}
    lines = [f"Step {target['step_id']}: {target['step'].get('name') or ''}".strip()]
    for item in target["predicates"]:
        condition = conditions.get(item["predicate_id"]) or {}
        lines.append(f"- step_id {target['step_id']}, predicate_id {item['predicate_id']}: {_describe(condition, roles) or target['step'].get('expected') or 'the step result'}")
    return "Required conditions:\n" + "\n".join(lines) + "\n\nDecide each from the images only."


_RELATION_WORDS = {
    "attached": "attached to",
    "supported_by": "resting on",
    "inside": "inside",
    "around": "around",
    "touching": "touching",
    "left_of": "left of",
    "right_of": "right of",
}


def _describe(condition: dict, roles: dict) -> str:
    def name(role_id: str) -> str:
        role = roles.get(role_id) or {}
        detail = role.get("visual_characteristics") or ""
        return f"the {role.get('label') or role_id}" + (f" ({detail})" if detail else "")

    if condition.get("relation"):
        relation = _RELATION_WORDS.get(condition["relation"], condition["relation"].replace("_", " "))
        return f"{name(condition.get('subject_role', ''))} is {relation} {name(condition.get('target_role', ''))}"
    if condition.get("to_state") or condition.get("state_label"):
        return f"{name(condition.get('subject_role', ''))} is {condition.get('to_state') or condition.get('state_label')}"
    return ""


async def _call(session, run: AnalysisRun, video: VideoAsset, times: list[float], question: str) -> tuple[str | None, str, str]:
    from app.providers.registry import get_provider
    from app.services.costs import fail_provider_call, finish_provider_call, open_provider_call
    from app.services.credentials import load_secret

    secret, _source = load_secret(session, "meta")
    if not secret:
        return None, "", "No Meta API key is configured."
    stills = [FrameInput(moment, _still(video, moment)) for moment in times]
    request = GenerationRequest(
        model_id=run.model_id,
        system_prompt=SYSTEM,
        user_prompt=question,
        frames=stills,
        modality="sampled_frames",
        json_schema=REINSPECT.schema,
        schema_name=REINSPECT.name,
        max_output_tokens=_OUTPUT_TOKENS,
    )
    call = open_provider_call(
        session,
        run,
        stage="reinspection",
        provider_id="meta",
        model_id=run.model_id,
        modality="sampled_frames",
        parameters={"schema_name": REINSPECT.name, "strict_output": True, "output_token_limit": _OUTPUT_TOKENS},
        frame_timestamps=times,
    )
    session.commit()
    try:
        response = await get_provider("meta", session).generate(secret, request)
    except Exception as exc:
        fail_provider_call(call, exc)
        session.commit()
        return None, "", str(exc)[:400]
    finish_provider_call(session, run, call, response)
    session.commit()
    if response.incomplete_reason:
        return None, response.request_id or "", f"The answer stopped early ({response.incomplete_reason})."
    return response.text, response.request_id or call.id, ""


# ---------------------------------------------------------------- replay


def _key(session: Session, row: VerificationRunRow, target: dict, times: list[float] | None = None, question: str | None = None) -> str:
    if times is None or question is None:
        video = session.get(VideoAsset, row.execution_video_id)
        revision = session.get(AnalysisRevision, row.execution_revision_id)
        template = session.get(LearnedWorkflowTemplate, row.template_id)
        if video is None or revision is None or template is None:
            return ""
        times = choose_frames(target["step"], revision.document_json, video)
        question = _question(target, template.document_json)
    video = session.get(VideoAsset, row.execution_video_id)
    run = session.get(AnalysisRun, row.execution_run_id)
    parts = {
        "video": video.checksum_sha256 if video else "",
        "model": run.model_id if run else "",
        "system": SYSTEM,
        "user": question,
        "times": times,
        "schema": REINSPECT.hash,
    }
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


def _cached(key: str) -> dict | None:
    if not key:
        return None
    try:
        return json.loads(data_path("inference_cache", "reinspect", f"{key}.json").read_text())
    except (OSError, ValueError):
        return None


def _store(key: str, record: dict) -> None:
    path = data_path("inference_cache", "reinspect", f"{key}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))
