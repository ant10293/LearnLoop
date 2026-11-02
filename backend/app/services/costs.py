"""Persist one cost record per provider call and roll those records up."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.media import Project, VideoAsset
from app.models.runs import AnalysisRun, ModelRequestRow, ModelResponseRow
from app.providers.types import ProviderResponse


def open_provider_call(
    session: Session,
    run: AnalysisRun,
    *,
    stage: str,
    provider_id: str,
    model_id: str,
    modality: str,
    prompt_version_id: str | None = None,
    parameters: dict | None = None,
    frame_timestamps: list | None = None,
) -> ModelRequestRow:
    row = ModelRequestRow(
        run_id=run.id,
        stage=stage,
        provider_id=provider_id,
        model_id=model_id,
        modality=modality,
        prompt_version_id=prompt_version_id,
        parameters_json=parameters or {},
        frame_timestamps=list(frame_timestamps or []),
        status="running",
    )
    session.add(row)
    session.flush()
    return row


def fail_provider_call(row: ModelRequestRow, exc: BaseException) -> None:
    row.status = "failed"
    row.error = str(exc)[:2000]


def uses_api_key(provider_id: str, parameters: dict | None) -> bool:
    """True when the call was sent with an API key rather than a linked plan."""
    from app.providers.registry import needs_secret, run_transport

    return needs_secret(provider_id, run_transport(provider_id, parameters or {}))


def finish_provider_call(
    session: Session,
    run: AnalysisRun,
    row: ModelRequestRow,
    response: ProviderResponse,
) -> ModelResponseRow:
    usage = response.usage
    known = bool(usage.price_known)
    cost = round(float(usage.cost_usd or 0), 6) if known else 0.0
    row.status = "completed"
    row.latency_ms = int(response.latency_ms or 0)
    row.provider_request_id = response.request_id or ""
    parameters = dict(row.parameters_json or {})
    if response.sdk_version:
        parameters["sdk_version"] = response.sdk_version
    row.parameters_json = parameters
    saved = ModelResponseRow(
        request_id=row.id,
        run_id=run.id,
        raw_json=response.raw or {},
        text=response.text or "",
        input_tokens=int(usage.input_tokens or 0),
        output_tokens=int(usage.output_tokens or 0),
        cost_usd=cost,
        price_known=known,
    )
    session.add(saved)
    run.input_tokens = int(run.input_tokens or 0) + saved.input_tokens
    run.output_tokens = int(run.output_tokens or 0) + saved.output_tokens
    if known:
        run.cost_usd = round(float(run.cost_usd or 0) + cost, 6)
    return saved


def project_costs(session: Session, project_id: str) -> dict | None:
    project = session.get(Project, project_id)
    if project is None:
        return None
    runs = list(session.scalars(select(AnalysisRun).where(AnalysisRun.project_id == project_id)))
    run_by_id = {run.id: run for run in runs}
    videos = {
        video.id: video
        for video in session.scalars(select(VideoAsset).where(VideoAsset.project_id == project_id))
    }
    pairs = _paired_calls(session, list(run_by_id))
    calls = []
    unpriced = 0
    itemized = 0.0
    for request, response in pairs:
        run = run_by_id[request.run_id]
        video = videos.get(run.video_id)
        known = bool(response.price_known) if response is not None else False
        keyed = uses_api_key(request.provider_id, request.parameters_json)
        cost = float(response.cost_usd or 0) if response is not None and known else 0.0
        if request.status == "completed" and response is not None and not known and not keyed:
            unpriced += 1
        if known:
            itemized += cost
        calls.append(
            {
                "id": request.id,
                "run_id": run.id,
                "run_name": run.name,
                "video_id": run.video_id,
                "video_filename": video.original_filename if video else "",
                "stage": request.stage,
                "provider_id": request.provider_id,
                "model_id": request.model_id,
                "status": request.status,
                "latency_ms": request.latency_ms,
                "created_at": request.created_at.isoformat() if request.created_at else "",
                "input_tokens": response.input_tokens if response else 0,
                "output_tokens": response.output_tokens if response else 0,
                "cost_usd": round(cost, 6),
                "price_known": known,
                "api_key": keyed,
            }
        )
    calls.sort(key=lambda item: item["created_at"], reverse=True)
    total = round(sum(float(run.cost_usd or 0) for run in runs), 6)
    return {
        "project_id": project.id,
        "cost_usd": total,
        "input_tokens": sum(int(run.input_tokens or 0) for run in runs),
        "output_tokens": sum(int(run.output_tokens or 0) for run in runs),
        "unpriced_calls": unpriced,
        "unitemized_usd": _unitemized(total, itemized),
        "calls": calls,
    }


def aggregate_spend(session: Session, runs: list[AnalysisRun]) -> dict:
    pairs = _paired_calls(session, [run.id for run in runs])
    buckets: dict[str, dict] = {}
    unpriced = 0
    itemized = 0.0
    for request, response in pairs:
        if request.status != "completed" or response is None:
            continue
        bucket = buckets.setdefault(
            request.provider_id,
            {
                "provider_id": request.provider_id,
                "cost_usd": 0.0,
                "calls": 0,
                "unpriced_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
            },
        )
        bucket["calls"] += 1
        bucket["input_tokens"] += int(response.input_tokens or 0)
        bucket["output_tokens"] += int(response.output_tokens or 0)
        if response.price_known:
            cost = round(float(response.cost_usd or 0), 6)
            bucket["cost_usd"] = round(bucket["cost_usd"] + cost, 6)
            itemized += cost
        elif not uses_api_key(request.provider_id, request.parameters_json):
            bucket["unpriced_calls"] += 1
            unpriced += 1
    total = round(sum(float(run.cost_usd or 0) for run in runs), 6)
    providers = sorted(buckets.values(), key=lambda item: (-item["cost_usd"], item["provider_id"]))
    return {
        "cost_usd": total,
        "input_tokens": sum(int(run.input_tokens or 0) for run in runs),
        "output_tokens": sum(int(run.output_tokens or 0) for run in runs),
        "unpriced_calls": unpriced,
        "unitemized_usd": _unitemized(total, itemized),
        "cost_by_provider": providers,
    }


def _paired_calls(
    session: Session, run_ids: list[str]
) -> list[tuple[ModelRequestRow, ModelResponseRow | None]]:
    if not run_ids:
        return []
    requests = list(
        session.scalars(select(ModelRequestRow).where(ModelRequestRow.run_id.in_(run_ids)))
    )
    if not requests:
        return []
    responses = list(
        session.scalars(
            select(ModelResponseRow).where(
                ModelResponseRow.request_id.in_([row.id for row in requests])
            )
        )
    )
    latest: dict[str, ModelResponseRow] = {}
    for response in responses:
        current = latest.get(response.request_id)
        if current is None or response.created_at >= current.created_at:
            latest[response.request_id] = response
    return [(row, latest.get(row.id)) for row in requests]


def unpriced_call_count(session: Session, run_id: str) -> int:
    """Plan calls only. An API-key call with no catalog rate is not unpriced."""
    return sum(
        1
        for request, response in _paired_calls(session, [run_id])
        if _plan_usage(request, response)
    )


def _plan_usage(request: ModelRequestRow, response: ModelResponseRow | None) -> bool:
    return (
        response is not None
        and request.status == "completed"
        and not response.price_known
        and not uses_api_key(request.provider_id, request.parameters_json)
    )


def _unitemized(total: float, itemized: float) -> float:
    gap = round(total - itemized, 6)
    return gap if gap > 0 else 0.0
