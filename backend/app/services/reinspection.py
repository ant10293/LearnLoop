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
from app.pipeline.guided import shares_cache
from app.providers.types import FrameInput, GenerationRequest
from app.schemas.generation import ReinspectOutput
from app.services.storage import data_path
from app.verification.read import as_document

log = logging.getLogger("learnloop.reinspection")

SYSTEM = """
You are checking whether specific physical conditions hold in still frames from one recording of a person assembling parts. Each image follows a line with its timestamp.
For each listed condition, return one finding with its step_id and predicate_id copied exactly.
outcome: supported only when the frames directly show the condition holding. contradicted only when the frames directly show it not holding, for example the part is visibly somewhere else, visibly absent from where it should be, or visibly the wrong part. Otherwise inconclusive.
A condition holds from the moment it is achieved until a frame shows it undone, such as the part taken back off or out. A hand or another part covering it later does not undo it, and neither does the condition being out of view at the end. A part pressed into or under another may be visible only while it goes in: judge it from those frames.
Do not assume the expected condition is true. Several parts can look alike: decide only about the part that fits the listed description. A part that cannot be seen is not missing: if a hand, another part, or the frame edge hides where it would be, the outcome is inconclusive and visibility is occluded or out_of_frame.
visibility describes the place the condition is about in the clearest frame: visible, partial, occluded, or out_of_frame.
evidence_timestamps lists only timestamps of supplied images that show what you report, copied exactly. Leave it empty for inconclusive.
evidence_class is DIRECTLY_OBSERVED only when the deciding pixels are in a supplied image; INFERRED when you reasoned from context; UNKNOWN otherwise.
reason is one short sentence naming what is seen. limitations lists anything that weakens the reading.
Return JSON only.
""".strip()
# The analysis model reasons before it answers. 4,000 tokens ran out on a real attempt.
_OUTPUT_TOKENS = 16_000
# Frames a contradicted finding must cite. A false FAIL costs more than an unresolved step.
MIN_CONTRADICTING_FRAMES = 2
# Relations whose subject ends up mostly hidden in its target.
HIDDEN_BY_DESIGN = frozenset({"inside"})
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
    cached = shares_cache(run.config_json if run else None) and all(_cached(_key(session, row, target)) is not None for target in targets)
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
            # Steps looked at so far, so the report can show progress while the rest run.
            current = dict(row.report_json or {})
            current["reinspection"] = {**(current.get("reinspection") or {}), "done": [item["step_id"] for item in outcomes]}
            row.report_json = current
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
    times, question = plan_question(session, row, target)
    key = _key(session, row, target, times, question)
    # An isolated run (Reliability Lab) asks again rather than replay another run's answer.
    stored = _cached(key) if shares_cache(run.config_json) else None
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
    if source == "live" and shares_cache(run.config_json):
        _store(key, {"text": text, "request_id": request_id, "model_id": run.model_id, "created_at": utcnow().isoformat()})
    findings, problems = validate_findings(answer, target, times, predicate_relations(target, template.document_json))
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


def validate_findings(
    answer: ReinspectOutput, target: dict, times: list[float], relations: dict[str, str] | None = None
) -> tuple[list[dict], list[str]]:
    """Keep findings that answer what was asked, citing frames that were sent. Two answers for one condition
    that disagree count as none. A supported or contradicted finding without direct visible evidence is
    kept as inconclusive. relations maps a predicate id to its relation, for the one exception below."""
    relations = relations or {}
    asked = {item["predicate_id"] for item in target["predicates"]}
    kept: dict[str, dict] = {}
    problems: list[str] = []
    conflicted: set[str] = set()
    for item in answer.findings:
        if item.predicate_id not in asked:
            problems.append(f"ignored a finding for {item.step_id}/{item.predicate_id}, which was not asked")
            continue
        if item.step_id != target["step_id"]:
            # One call asks about one step, so the predicate id alone says which condition this answers.
            problems.append(f"{item.predicate_id} came back under step_id {item.step_id!r}; read as step {target['step_id']}")
        stamps = []
        for value in item.evidence_timestamps:
            match = next((sent for sent in times if abs(sent - value) <= 0.05), None)
            if match is None:
                problems.append(f"{item.predicate_id} cited {value:.2f}s, which was not sent")
            else:
                stamps.append(match)
        finding = item.model_dump()
        if item.step_id != target["step_id"]:
            finding["reported_step_id"] = item.step_id
            finding["step_id"] = target["step_id"]
        finding["evidence_timestamps"] = sorted(set(stamps))
        if finding["outcome"] == "contradicted" and len(finding["evidence_timestamps"]) < MIN_CONTRADICTING_FRAMES:
            # Seen live: a model took a similar part for the subject in one frame while a hand hid the real one.
            finding["limitations"] = [*finding["limitations"], f"A contradiction needs at least {MIN_CONTRADICTING_FRAMES} frames that show it; one frame can hide a part behind a hand."]
            finding["outcome"] = "inconclusive"
        # A part inside another is only ever partly visible. For that condition alone, a partial view supports it
        # when it is directly observed in as many frames as a contradiction needs. Contradictions still need a full view.
        hidden_by_design = (
            finding["outcome"] == "supported"
            and relations.get(item.predicate_id) in HIDDEN_BY_DESIGN
            and finding["visibility"] == "partial"
            and len(finding["evidence_timestamps"]) >= MIN_CONTRADICTING_FRAMES
        )
        if hidden_by_design:
            finding["partial_accepted"] = True
        if finding["outcome"] != "inconclusive" and not (
            finding["evidence_class"] == "DIRECTLY_OBSERVED"
            and (finding["visibility"] == "visible" or hidden_by_design)
            and finding["evidence_timestamps"]
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


def predicate_relations(target: dict, template: dict) -> dict[str, str]:
    for step in template.get("steps") or []:
        if step.get("step_id") == target["step_id"]:
            return {item["id"]: item.get("relation") or "" for item in step.get("completion_conditions") or []}
    return {}


def condition_parts(target: dict, template: dict, report: dict) -> dict:
    """The attempt's objects for each condition's subject and target, from the report's role mapping."""
    by_role = {role.get("role_id"): role.get("object_local_ids") or [] for role in report.get("roles") or []}
    conditions = {}
    for step in template.get("steps") or []:
        if step.get("step_id") == target["step_id"]:
            conditions = {item["id"]: item for item in step.get("completion_conditions") or []}
    subjects: set[str] = set()
    targets: set[str] = set()
    for item in target["predicates"]:
        condition = conditions.get(item["predicate_id"]) or {}
        subjects.update(by_role.get(condition.get("subject_role"), []))
        targets.update(by_role.get(condition.get("target_role"), []))
    return {"subjects": subjects, "targets": targets}


def episodes(execution: dict, parts: dict) -> list[dict]:
    """When the analysis says this condition's parts were handled or changed, from its actions, transitions,
    and relations. Both parts together rank first. Step order is not used: an attempt can do steps in any order."""
    document = as_document(execution)
    subjects, targets = parts["subjects"], parts["targets"]
    found: list[dict] = []
    for action in document.actions:
        ids = set(action.object_local_ids)
        if ids & subjects:
            found.append({"start": action.start, "end": action.end, "both": bool(ids & targets), "kind": "handled"})
    for transition in document.transitions:
        if transition.object_local_id in subjects:
            found.append({"start": transition.timestamp, "end": transition.timestamp, "both": False, "kind": "changed"})
    for relation in document.relationships:
        pair = {relation.subject_local_id, relation.other_local_id}
        if pair & subjects and relation.time_start is not None:
            found.append({"start": relation.time_start, "end": relation.time_start, "both": bool(pair & targets), "kind": "joined"})
    return sorted(found, key=lambda item: (not item["both"], item["start"]))


def choose_frames(step: dict, execution: dict, video: VideoAsset, parts: dict | None = None) -> list[float]:
    """A few sharp moments that can show this condition: while its parts are put together, just after, a
    moment later when hands usually move, and the last frame. A condition reached mid-video and hidden later
    is judged from the middle. Without parts, the step's matched work is used."""
    document = as_document(execution)
    duration = max(float(video.duration_sec or 0) - 0.1, 0.0)
    moments: list[float] = []
    for item in episodes(execution, parts or {"subjects": set(), "targets": set()})[:3]:
        moments += [(item["start"] + item["end"]) / 2, item["end"], item["end"] + 0.6, item["end"] + 1.5]
    if not moments:
        actions = {action.local_id: action for action in document.actions}
        for action_id in step.get("matched_action_ids") or []:
            action = actions.get(action_id)
            if action is not None:
                moments += [(action.start + action.end) / 2, action.end, action.end + 1.0]
        if step.get("execution_time") is not None:
            moments.append(float(step["execution_time"]))
    if not moments:
        # Nothing ties this condition to a moment: spread the look over the recording, not only its end.
        moments = [duration * share for share in (0.35, 0.55, 0.7, 0.85)]
    limit = max(settings.reinspect_max_frames, 2)
    kept: list[float] = []
    # In ranked order, so the moments of the most relevant episode survive the limit.
    # The last frame is always one of them: it is the result if nothing undid it later.
    # Extracted like the others, so the still exists when the request is built.
    reserved = _sharpest(video, [round(duration, 3)], duration)[-1:] or [round(duration, 3)]
    for moment in (min(max(float(value), 0.0), duration) for value in moments):
        if len(kept) < limit - 1 and all(abs(moment - other) >= _MERGE_SEC for other in kept + reserved):
            kept.append(round(moment, 3))
    sharp = _sharpest(video, sorted(kept), duration)
    distinct: list[float] = []
    for moment in sharp:
        if all(abs(moment - other) >= _MERGE_SEC for other in distinct + reserved):
            distinct.append(moment)
    return sorted(distinct + reserved)


def timeline(execution: dict, parts: dict) -> str:
    lines = []
    for item in sorted(episodes(execution, parts)[:4], key=lambda value: value["start"]):
        span = f"{item['start']:g}s" if item["start"] == item["end"] else f"{item['start']:g}-{item['end']:g}s"
        lines.append(f"- {span}: the listed part was {item['kind']}{' together with the other listed part' if item['both'] else ''}")
    return "\n".join(lines)


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


def plan_question(session: Session, row: VerificationRunRow, target: dict) -> tuple[list[float], str] | None:
    """Frames and question for one step, the same every time for the same report, so a stored answer replays."""
    video = session.get(VideoAsset, row.execution_video_id)
    revision = session.get(AnalysisRevision, row.execution_revision_id)
    template = session.get(LearnedWorkflowTemplate, row.template_id)
    if video is None or revision is None or template is None:
        return None
    parts = condition_parts(target, template.document_json, row.report_json or {})
    times = choose_frames(target["step"], revision.document_json, video, parts)
    question = _question(target, template.document_json, times, timeline(revision.document_json, parts))
    return times, question


def _question(target: dict, template: dict, times: list[float], history: str = "") -> str:
    roles = {role["role_id"]: role for role in template.get("roles") or []}
    conditions = {}
    for step in template.get("steps") or []:
        if step.get("step_id") == target["step_id"]:
            conditions = {item["id"]: item for item in step.get("completion_conditions") or []}
    lines = [f"Step: {target['step'].get('name') or ''}".strip()]
    for item in target["predicates"]:
        condition = conditions.get(item["predicate_id"]) or {}
        ids = json.dumps({"step_id": target["step_id"], "predicate_id": item["predicate_id"]})
        lines.append(f"- {ids}: {_describe(condition, roles) or target['step'].get('expected') or 'the step result'}")
    stamps = ", ".join(f"{value:g}" for value in times)
    asked_roles = {conditions.get(item["predicate_id"], {}).get(key) for item in target["predicates"] for key in ("subject_role", "target_role")}
    others = [
        f"- {role.get('label')}: {role.get('visual_characteristics') or role.get('description') or ''}".rstrip(": ")
        for role_id, role in roles.items()
        if role_id not in asked_roles
    ]
    if others:
        lines.append(
            "\nOther parts of this procedure, which may look similar. Identify the listed part by its description before deciding, "
            "and do not report one of these as it:\n" + "\n".join(others)
        )
    return (
        "Required conditions, each with the exact step_id and predicate_id to copy:\n"
        + "\n".join(lines)
        + (f"\n\nAccording to the analysis of this recording:\n{history}\nThe images come from those moments, so the condition may be visible in the middle of the recording rather than at the end." if history else "")
        + f"\n\nThe images were taken at these timestamps, in seconds: {stamps}. "
        "A supported or contradicted finding must list in evidence_timestamps the ones that show it. "
        "Decide each from the images only."
    )


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
        planned = plan_question(session, row, target)
        if planned is None:
            return ""
        times, question = planned
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
