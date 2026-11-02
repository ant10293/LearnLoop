# Architecture

```
Browser (React, 4 screens)
  │  /api/learnloop/*, /api/workflow-templates/*, /api/verifications/*, /api/media/*
FastAPI ── SQLite ── local media files
  │
  ├─ Worker: one job per analysis, one attempt
  │    metadata → sample frames → guided call → validation → artifacts
  │    (tracking, fusion, reinspection and audio stages are recorded as skipped;
  │     demonstrations also get SAM object views, which never feed a verdict)
  │
  ├─ guided.py: stored answer for the exact inputs, or one approved Meta call
  ├─ verification/learn.py: analysis → versioned procedure template
  └─ verification/engine.py + predicates.py: template + analysis + reviewer observations → report
```

## Analysis: `hackathon_guided`

`backend/app/pipeline/guided.py` makes one call to Meta with a strict, closed JSON schema (`GuidedTaskOutput`) that omits boxes and per-frame tracks. The SAM object views get their starting boxes from a separate call on the first frame. Its field names match the canonical `AnalysisDocument`, so there is one representation of an analysis. The strict contract validator runs on live and cached answers alike. Nothing is repaired or fabricated.

Failure handling:

- Incomplete, malformed, off-schema, or empty answers fail the run with the reason shown in the UI.
- The raw text stays on its response row. A failed answer is never stored for replay.
- There are no retries. The SDK's retries and job retries are both off.
- The model is never swapped silently. If the account does not list it, the run is refused.

Live calls pass three gates: a stored-answer check, explicit approval per request, and the server flag `LEARNLOOP_LIVE_INFERENCE`. The Meta adapter checks the flag again before sending.

## Execution prompts

An attempt's analysis is told the reference object labels and the pairs to report on. It is not told the expected relation or that the attempt is correct. Bias toward the reference list is still possible, and only real recordings can show how much.

## Learning

`learn_template(..., results_first=True)` keeps physical arrangements as the checks. It drops weak relations such as `near` and `held_by`, and treats state labels as supporting only when an arrangement exists. A step is promoted to required only when a condition was classified as directly observed. Everything else stays optional until a person decides.

## Verification

The engine maps roles functionally, aligns actions, judges each condition, and combines the results. The final state counts, so a continuous track is not required. Reports reference the template version, execution run, and execution revision. The first machine report is kept unchanged next to any later version.

Changes made for this submission to the reused verifier:

- A part mapped to another step no longer counts as a wrong component on a shared base.
- `above`, `below`, `left_of`, and `right_of` respect direction. A reversed arrangement is a wrong placement, and an inverse statement of the same arrangement counts as support.
- Removed steps are excluded and listed in the report.

## Human input

| Action | Stored as | Effect |
| --- | --- | --- |
| Edit a procedure | New template version with a `review_log` | The earlier version is unchanged. Changed content returns to draft. |
| Answer a checkpoint | Append-only `human_observations` row with reviewer, timestamp, and revision | The verifier re-reads it as evidence. The step's basis becomes `human_confirmed`. |

Original model responses and analysis revisions are never edited.

## Research code left in place

Tracking, SAM, reacquisition, and evaluation modules remain in the backend because other modules import them. Their audio, recovery, and task API routes are not mounted unless `LEARNLOOP_RESEARCH_ROUTES=1`, and the product UI does not use them. The general lab routes in `api/routes.py` stay mounted because the product uses their media endpoints. SAM refuses to run unless live calls are enabled.
