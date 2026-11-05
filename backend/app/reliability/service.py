"""Create, read, cancel, and label Reliability Lab batches."""

from __future__ import annotations

import csv
import hashlib
import io
import json

from sqlalchemy import select
from sqlalchemy.orm import Session

from app import __version__
from app.config import settings
from app.models.media import Project, VideoAsset
from app.models.product import ProcedureRecording
from app.models.reliability import ReliabilityBatch, ReliabilityRun
from app.models.runs import AnalysisRun, Job, PipelineStageRun
from app.models.util import elapsed_seconds, utcnow
from app.models.verification import LearnedWorkflowTemplate
from app.pipeline import guided
from app.reliability import learning, metrics
from app.services.storage import missing_original

MODES = ("verification", "learning")
ACTIVE = ("queued", "running", "canceling")
STEP_LABELS = {"PASS", "FAIL", "NOT_APPLICABLE"}

# Settings that change what a run sends or how its evidence is read. Copied into every batch.
SETTINGS_SNAPSHOT = (
    "live_inference",
    "guided_max_output_tokens",
    "object_views",
    "appearance_enabled",
    "auto_gallery",
    "appearance_recovery",
    "auto_reinspect",
    "appearance_max_references",
    "appearance_min_crop_px",
    "appearance_sample_fps",
    "appearance_batch_size",
    "appearance_accept",
    "appearance_margin",
    "appearance_floor",
    "recovery_max_intervals",
    "recovery_max_candidates",
    "recovery_max_sam_requests",
    "reinspect_max_calls",
    "reinspect_max_frames",
)

# Work shared between repetitions because it is deterministic and holds no model answer. Disclosed with each batch.
SHARED_DETERMINISTIC = [
    "The original video file and its playback copy are read, never copied per run.",
    "Frame presentation timestamps of the video (cached by file checksum).",
    "DINOv2 appearance embeddings of identical crops (frozen weights, cached by crop hash). The loaded model is one per process.",
    "The demonstration's appearance references, built once when the procedure was learned.",
]


class LiveCallRequired(Exception):
    def __init__(self, plan: dict):
        super().__init__("live_call_required")
        self.plan = plan


def _git() -> str:
    from app.services.experiments import _git as git_revision

    return git_revision()


def fingerprint(config: dict) -> str:
    """A short hash of everything that decides a run's behaviour, so batches made differently are easy to tell apart."""
    keys = ("model_id", "segmentation_model_id", "settings", "prompt_cache_key", "template_id", "independent_inference", "cache_policy")
    payload = {key: config.get(key) for key in keys}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]


def missing_recording_message(path) -> str:
    return f"The recording was added by reference and is no longer at {path}. Put the file back there, or add it to the procedure again."


def create_batch(session: Session, payload: dict) -> ReliabilityBatch:
    mode = payload.get("mode") or "verification"
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode}. Use verification or learning.")
    project = session.get(Project, payload.get("project_id") or "")
    if project is None:
        raise LookupError("procedure not found")
    video = session.get(VideoAsset, payload.get("video_id") or "")
    recording = session.get(ProcedureRecording, payload.get("video_id") or "")
    if video is None or recording is None or recording.project_id != project.id:
        raise LookupError("recording not found in this procedure")
    if (video.ingest_status or "ready") != "ready":
        raise ValueError("This recording is still being prepared.")
    if (gone := missing_original(video.stored_path)) is not None:
        raise ValueError(missing_recording_message(gone))
    if float(video.duration_sec or 0) > guided.MAX_CLIP_SECONDS:
        raise ValueError(f"This recording is {video.duration_sec:.0f} seconds. LearnLoop supports up to {guided.MAX_CLIP_SECONDS:.0f} seconds.")
    repetitions = int(5 if payload.get("repetitions") is None else payload["repetitions"])
    if not 1 <= repetitions <= settings.reliability_max_repetitions:
        raise ValueError(f"Repetitions must be between 1 and {settings.reliability_max_repetitions}.")
    concurrency = int(2 if payload.get("max_concurrency") is None else payload["max_concurrency"])
    if not 1 <= concurrency <= settings.reliability_max_concurrency:
        raise ValueError(f"Concurrent runs must be between 1 and {settings.reliability_max_concurrency}.")
    independent = bool(payload.get("independent_inference", True))

    template = None
    if mode == "verification":
        template = session.get(LearnedWorkflowTemplate, payload.get("template_id") or "")
        if template is None or template.project_id != project.id:
            raise LookupError("procedure version not found")
        if template.acceptance_status != "accepted":
            raise ValueError("Choose an accepted procedure version. Verification runs only against a human-approved procedure.")
    reference = None
    if mode == "learning" and payload.get("reference_template_id"):
        reference = session.get(LearnedWorkflowTemplate, payload["reference_template_id"])
        if reference is None or reference.project_id != project.id:
            raise LookupError("reference procedure not found")

    model_id, segmentation_model_id = guided.choose_models(session, payload.get("model_id") or "", payload.get("segmentation_model_id"))
    document = template.document_json if template else None
    plan = guided.plan(video, template=document, model_id=model_id, segmentation_model_id=segmentation_model_id, fresh=independent)
    allow_live = bool(payload.get("allow_live"))
    # Independent repetitions send their own requests, so they always need approval and the server gate.
    needs_live = independent or not plan["cached"]
    if needs_live and not (allow_live and settings.live_inference):
        raise LiveCallRequired({**plan, "repetitions": repetitions, "independent_inference": independent})

    ground_truth = clean_ground_truth(payload.get("ground_truth") or {}, document)
    config = {
        "mode": mode,
        "procedure_name": project.name,
        "template_id": template.id if template else "",
        "template_version": template.version if template else None,
        "template_family_id": template.family_id if template else "",
        "template_name": template.name if template else "",
        # The exact procedure every repetition is checked against. The template row itself is immutable.
        "template_document": document,
        "reference_template_id": reference.id if reference else "",
        "reference_document": reference.document_json if reference else None,
        "reference_version": reference.version if reference else None,
        "video": {
            "id": video.id,
            "filename": video.original_filename,
            "duration_sec": video.duration_sec,
            "checksum_sha256": video.checksum_sha256,
            "role": recording.role,
        },
        "model_id": model_id,
        "segmentation_model_id": plan.get("segmentation_model_id") or "",
        "modality": guided.MODALITY,
        "profile": guided.PROFILE,
        "prompt_cache_key": plan.get("cache_key"),
        "settings": {key: getattr(settings, key) for key in SETTINGS_SNAPSHOT},
        "independent_inference": independent,
        "cache_policy": guided.ISOLATED if independent else "shared",
        "inference_policy": (
            "Every repetition sends its own Meta requests: the analysis call, first-frame boxes, SAM tracking, and any "
            "second look. None reads a stored answer and none stores one, so no run can be served another run's answer "
            "and ordinary analyses keep replaying exactly what they did before."
            if independent
            else "Stored answers may be replayed. Repetitions that replay the same answer measure the deterministic verifier, not model variation."
        ),
        "shared_deterministic": SHARED_DETERMINISTIC if independent else [],
        "allow_live": allow_live,
        "plan": plan,
        "git_revision": _git(),
        "app_version": __version__,
        "created_at": utcnow().isoformat(),
    }
    config["fingerprint"] = fingerprint(config)
    batch = ReliabilityBatch(
        name=(payload.get("name") or "").strip()[:300] or f"{mode.title()} · {video.original_filename} × {repetitions}",
        mode=mode,
        status="queued",
        project_id=project.id,
        template_id=template.id if template else "",
        video_id=video.id,
        repetitions=repetitions,
        max_concurrency=concurrency,
        independent_inference=independent,
        config_json=config,
        ground_truth_json=ground_truth,
    )
    session.add(batch)
    session.flush()
    for index in range(1, repetitions + 1):
        session.add(ReliabilityRun(batch_id=batch.id, index=index, status="queued"))
    session.commit()
    return batch


def clean_ground_truth(value: dict, template_document: dict | None) -> dict:
    expected = value.get("expected_verdict") or None
    if expected not in {None, "PASS", "FAIL"}:
        raise ValueError("The expected verdict is PASS, FAIL, or unlabeled.")
    step_ids = {step["step_id"] for step in (template_document or {}).get("steps") or [] if not step.get("removed")}
    labels = {}
    for step_id, label in (value.get("step_labels") or {}).items():
        if not label:
            continue
        if label not in STEP_LABELS:
            raise ValueError(f"A step label is PASS, FAIL, or NOT_APPLICABLE, not {label}.")
        if step_id not in step_ids:
            raise ValueError(f"Step {step_id} is not in this procedure version.")
        labels[step_id] = label
    return {"expected_verdict": expected, "step_labels": labels, "note": str(value.get("note") or "")[:500], "updated_at": utcnow().isoformat()}


def set_ground_truth(session: Session, batch_id: str, value: dict) -> dict:
    batch = _batch(session, batch_id)
    if batch.mode != "verification":
        raise ValueError("Ground truth verdicts apply to verification experiments.")
    batch.ground_truth_json = clean_ground_truth(value, (batch.config_json or {}).get("template_document"))
    session.commit()
    return batch_payload(session, batch)


def cancel_batch(session: Session, batch_id: str) -> dict:
    batch = _batch(session, batch_id)
    for run in session.scalars(select(ReliabilityRun).where(ReliabilityRun.batch_id == batch.id)):
        _cancel_run(session, run)
    if batch.status in ACTIVE:
        batch.status = "canceling"
    session.commit()
    return batch_payload(session, batch)


def delete_batch(session: Session, batch_id: str) -> list[str]:
    """Remove a finished experiment, its runs, and the verification reports they made. Returns the analysis runs
    it launched, which the caller removes with their files. A running experiment must be canceled first."""
    from app.models.verification import VerificationRunRow
    from app.services.verification_store import _delete_reports

    batch = _batch(session, batch_id)
    if batch.status in ACTIVE:
        raise ValueError("Cancel the experiment and let it stop before deleting it.")
    runs = session.scalars(select(ReliabilityRun).where(ReliabilityRun.batch_id == batch.id)).all()
    analysis_ids = [run.analysis_run_id for run in runs if run.analysis_run_id]
    reports = {run.verification_id for run in runs if run.verification_id}
    if analysis_ids:
        reports |= set(session.scalars(select(VerificationRunRow.id).where(VerificationRunRow.execution_run_id.in_(analysis_ids))))
    _delete_reports(session, list(reports))
    for run in runs:
        session.delete(run)
    session.delete(batch)
    session.commit()
    return analysis_ids


def cancel_run(session: Session, batch_id: str, run_id: str) -> dict:
    batch = _batch(session, batch_id)
    run = session.get(ReliabilityRun, run_id)
    if run is None or run.batch_id != batch.id:
        raise LookupError("run not found")
    _cancel_run(session, run)
    session.commit()
    return batch_payload(session, batch)


def _cancel_run(session: Session, run: ReliabilityRun) -> None:
    """A queued run never starts. A running one is stopped through the pipeline's own cancellation: the runner
    checks its analysis run's status before each stage. A request already sent finishes first."""
    if run.status == "queued":
        run.status = "canceled"
        run.error_kind = "canceled"
        run.error = "Canceled before it started."
        run.finished_at = utcnow()
        return
    if run.status != "running":
        return
    run.cancel_requested = True
    analysis = session.get(AnalysisRun, run.analysis_run_id) if run.analysis_run_id else None
    if analysis is not None and analysis.status in {"pending", "running"}:
        analysis.status = "canceled"
        analysis.error = "Canceled from the Reliability Lab."
    if run.analysis_run_id:
        for job in session.scalars(select(Job).where(Job.run_id == run.analysis_run_id, Job.status == "pending")):
            job.status = "canceled"
            job.finished_at = utcnow()


def list_batches(session: Session) -> list[dict]:
    rows = session.scalars(select(ReliabilityBatch).order_by(ReliabilityBatch.created_at.desc())).all()
    return [batch_payload(session, row, brief=True) for row in rows]


def get_batch(session: Session, batch_id: str) -> dict:
    return batch_payload(session, _batch(session, batch_id))


def _batch(session: Session, batch_id: str) -> ReliabilityBatch:
    batch = session.get(ReliabilityBatch, batch_id)
    if batch is None:
        raise LookupError("experiment not found")
    return batch


def _iso(value) -> str | None:
    return value.isoformat() if value else None


def _progress(session: Session, run: ReliabilityRun, mode: str) -> dict | None:
    """Where a running repetition is, from its analysis run's stages and any queued second look."""
    if run.status != "running" or not run.analysis_run_id:
        return None
    stages = {item.stage: item for item in session.scalars(select(PipelineStageRun).where(PipelineStageRun.run_id == run.analysis_run_id))}
    order = [name for name in ("metadata", "preprocess", "sample_frames", "scene_analysis", "object_extraction", "segmentation", "validation", "artifacts") if name in stages]
    done = sum(stages[name].status in {"completed", "skipped", "failed", "canceled"} for name in order)
    current = next((name for name in order if stages[name].status == "running"), "")
    analysis = session.get(AnalysisRun, run.analysis_run_id)
    phase = "analysis"
    if analysis is not None and analysis.status == "pending":
        phase = "waiting for a slot"
    elif analysis is not None and analysis.status == "completed":
        phase = "second look" if mode == "verification" else "learning"
    return {"phase": phase, "stage": current, "stages_done": done, "stages_total": len(order)}


def run_payload(session: Session, run: ReliabilityRun, mode: str, *, brief: bool = False) -> dict:
    payload = {
        "id": run.id,
        "index": run.index,
        "status": run.status,
        "analysis_run_id": run.analysis_run_id or "",
        "verification_id": run.verification_id,
        "error": run.error,
        "error_kind": run.error_kind,
        "cancel_requested": run.cancel_requested,
        "duration_seconds": run.duration_seconds,
        "created_at": _iso(run.created_at),
        "started_at": _iso(run.started_at),
        "finished_at": _iso(run.finished_at),
        "progress": _progress(session, run, mode),
        "result": run.result_json or {},
    }
    if mode == "learning" and not brief:
        payload["learned"] = run.learned_json or {}
    return payload


def _wall_clock(batch: ReliabilityBatch) -> float | None:
    if not batch.started_at:
        return None
    return elapsed_seconds(batch.started_at, batch.finished_at or utcnow())


def batch_payload(session: Session, batch: ReliabilityBatch, *, brief: bool = False) -> dict:
    config = batch.config_json or {}
    runs = list(session.scalars(select(ReliabilityRun).where(ReliabilityRun.batch_id == batch.id).order_by(ReliabilityRun.index)))
    run_payloads = [run_payload(session, run, batch.mode, brief=brief) for run in runs]
    seconds = _wall_clock(batch)
    template_steps = [
        {"step_id": step["step_id"], "name": step.get("name") or step["step_id"], "required": bool(step.get("required"))}
        for step in (config.get("template_document") or {}).get("steps") or []
        if not step.get("removed")
    ]
    if batch.mode == "verification":
        figures = metrics.verification_metrics(run_payloads, template_steps, batch.ground_truth_json or {}, seconds)
    else:
        full = run_payloads if not brief else [run_payload(session, run, batch.mode) for run in runs]
        figures = learning.learning_metrics(full, config.get("reference_document"), seconds)
    project = session.get(Project, batch.project_id)
    payload = {
        "id": batch.id,
        "name": batch.name,
        "mode": batch.mode,
        "status": batch.status,
        "project_id": batch.project_id,
        "project_exists": project is not None,
        "procedure_name": config.get("procedure_name") or (project.name if project else ""),
        "template_id": batch.template_id,
        "template_version": config.get("template_version"),
        "video": config.get("video") or {"id": batch.video_id},
        "repetitions": batch.repetitions,
        "max_concurrency": batch.max_concurrency,
        "independent_inference": batch.independent_inference,
        "ground_truth": batch.ground_truth_json or {},
        "fingerprint": config.get("fingerprint") or "",
        "git_revision": config.get("git_revision") or "",
        "model_id": config.get("model_id") or "",
        "segmentation_model_id": config.get("segmentation_model_id") or "",
        "error": batch.error,
        "created_at": _iso(batch.created_at),
        "started_at": _iso(batch.started_at),
        "finished_at": _iso(batch.finished_at),
        "wall_clock_seconds": round(seconds, 2) if seconds is not None else None,
        "counts": figures["counts"],
    }
    if brief:
        payload["summary"] = _brief_summary(batch.mode, figures)
        return payload
    payload.update(
        {
            "steps": template_steps,
            "config": {key: value for key, value in config.items() if key not in {"template_document", "reference_document"}},
            "template_document": config.get("template_document"),
            "runs": run_payloads,
            "metrics": figures,
        }
    )
    return payload


def _brief_summary(mode: str, figures: dict) -> dict:
    if mode == "verification":
        return {
            "distribution": figures["consistency"]["distribution"],
            "modal_agreement": figures["consistency"]["modal_agreement"]["value"],
            "exact_match": (figures["accuracy"].get("exact_match") or {}).get("value"),
            "mean_runtime": figures["performance"]["runtime_seconds"]["mean"],
        }
    return {
        "step_counts": figures["step_count_distribution"],
        "steps_in_every_run": figures["steps_in_every_run"],
        "reviewable": figures["reviewable"]["value"],
        "mean_runtime": figures["performance"]["runtime_seconds"]["mean"],
    }


def export_csv(session: Session, batch_id: str) -> str:
    """One row per repetition, with each step's verdict in its own column."""
    batch = _batch(session, batch_id)
    data = batch_payload(session, batch)
    out = io.StringIO()
    writer = csv.writer(out)
    if batch.mode == "verification":
        step_ids = [step["step_id"] for step in data["steps"]]
        writer.writerow(["run", "status", "outcome", "verdict", "raw_overall", "error_kind", "error", "duration_seconds", "evidence_count", "fully_fresh", "analysis_run_id", "verification_id", *step_ids])
        for run in data["runs"]:
            result = run["result"] or {}
            steps = {step["step_id"]: step.get("verdict") or "" for step in result.get("steps") or []}
            fresh = ((result.get("execution_metadata") or {}).get("inference") or {}).get("fully_fresh")
            writer.writerow([run["index"], run["status"], metrics.outcome_of(run) or "", result.get("verdict") or "", result.get("raw_overall") or "", run["error_kind"], run["error"], run["duration_seconds"], result.get("evidence_count", ""), fresh, run["analysis_run_id"], run["verification_id"], *[steps.get(step_id, "") for step_id in step_ids]])
    else:
        writer.writerow(["run", "status", "error_kind", "error", "duration_seconds", "steps", "required_steps", "checkable_steps", "roles", "objects", "reviewable", "validation_errors", "analysis_run_id"])
        for run in data["runs"]:
            result = run["result"] or {}
            writer.writerow([run["index"], run["status"], run["error_kind"], run["error"], run["duration_seconds"], result.get("step_count", ""), result.get("required_steps", ""), result.get("checkable_steps", ""), result.get("role_count", ""), result.get("object_count", ""), result.get("reviewable", ""), (result.get("validation") or {}).get("errors", ""), run["analysis_run_id"]])
    return out.getvalue()


def options(session: Session) -> dict:
    """What an experiment can be made from: procedures, their recordings, accepted versions, and models."""
    procedures = []
    for project in session.scalars(select(Project).where(Project.description == "learnloop_procedure").order_by(Project.created_at.desc())):
        recordings = []
        for row in session.scalars(select(ProcedureRecording).where(ProcedureRecording.project_id == project.id).order_by(ProcedureRecording.created_at)):
            video = session.get(VideoAsset, row.video_id)
            if video is None:
                continue
            recordings.append(
                {
                    "video_id": video.id,
                    "filename": video.original_filename,
                    "role": row.role,
                    "duration_sec": video.duration_sec,
                    "ready": (video.ingest_status or "ready") == "ready",
                    "missing": missing_original(video.stored_path) is not None,
                }
            )
        templates = [
            {
                "id": row.id,
                "version": row.version,
                "name": row.name,
                "family_id": row.family_id,
                "acceptance_status": row.acceptance_status,
                "created_at": _iso(row.created_at),
                "source_filename": source.original_filename if (source := session.get(VideoAsset, row.source_video_id or "")) else "",
                "steps": [
                    {"step_id": step["step_id"], "name": step.get("name") or step["step_id"], "required": bool(step.get("required"))}
                    for step in (row.document_json or {}).get("steps") or []
                    if not step.get("removed")
                ],
            }
            for row in session.scalars(
                select(LearnedWorkflowTemplate).where(LearnedWorkflowTemplate.project_id == project.id).order_by(LearnedWorkflowTemplate.created_at.desc())
            )
        ]
        procedures.append({"id": project.id, "name": project.name, "recordings": recordings, "templates": templates})
    choices = guided.model_choices(session)
    return {
        "procedures": procedures,
        "models": {key: choices[key] for key in ("default_inference", "default_segmentation")},
        "live_enabled": settings.live_inference,
        "max_concurrency": settings.reliability_max_concurrency,
        "max_repetitions": settings.reliability_max_repetitions,
    }
