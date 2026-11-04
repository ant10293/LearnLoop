# Architecture

```
Browser (React, 4 screens)
  │  /api/learnloop/*, /api/workflow-templates/*, /api/verifications/*, /api/media/*
FastAPI ── SQLite ── local media files
  │
  ├─ Worker: one job per analysis, one attempt
  │    metadata → sample frames → guided call → object views → validation → artifacts
  │    (the other stage names are recorded as skipped, with the reason)
  │
  ├─ guided.py: stored answer for the exact inputs, or one approved Meta call
  ├─ object_views.py: first-frame locate (Muse Wonder) → SAM 3.1 tracks → DINOv2 appearance checks
  ├─ verification/learn.py: analysis → versioned procedure template
  ├─ verification/engine.py + predicates.py: template + analysis + reviewer observations → report
  └─ services/reinspection.py: one narrow Meta question per unresolved required step → revised report
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

### What predates the hackathon

The core of the verifier, `backend/app/verification/` (role mapping, action alignment, condition judging, and the template schema), was written in vision-test-bench before kickoff. Its first commit was on 2026-10-26 at 01:54 CDT. Since the import (`2babd11`), these files have changed by 578 added and 204 removed lines (`git diff --stat 2babd11 HEAD -- backend/app/verification`).

### Changed during the hackathon

| Change | Commits |
| --- | --- |
| A part mapped to another step no longer counts as a wrong component on a shared base. | `6ae49d5` |
| `above`, `below`, `left_of`, and `right_of` respect direction. A reversed arrangement is a wrong placement, and an inverse statement of the same arrangement counts as support. | `6ae49d5` |
| Removed steps are excluded and listed in the report. | `6ae49d5` |
| Reviewer observations are read as evidence, and the step's basis says so (`apply_human`). | `6ae49d5` |
| Second-look findings are read as evidence (`apply_reinspection`). | `29cc5ca` |
| Learning keeps the arrangement a step produced, assigns each relation to the step that made it, and gives similar parts distinct labels. | `6ae49d5`, `0e79cc1` |
| Part identity is checked without relying on relations, which catches swapped part types. | `ce1fe1a` |
| A missing size measurement no longer lowers a role score. | `72c6161` |
| A reviewer can require, lift, or add an order between steps, and impossible orders are refused. | `d95903b` |
| A partial view supports a part that ends up inside another. | `e9018a5` |
| Steps are matched to the action that moved their part. | `e7eaeca` |
| DINOv2 appearance matches count as sightings of a part. | `29cc5ca`, `19dd462`, `049ed06` |
| A part that never appears in the attempt, in the analysis or in any SAM box matched to its learned appearance, leaves its condition CANNOT VERIFY, marked "not seen anywhere", and goes to the second look and the reviewer checkpoint. It is not called missing without a direct observation of absence. | `596663a`, `31f82f5`, `a86697f`, `3adbb17`, `39444b8` |

## Human input

| Action | Stored as | Effect |
| --- | --- | --- |
| Edit a procedure | New template version with a `review_log` | The earlier version is unchanged. Changed content returns to draft. |
| Answer a checkpoint | Append-only `human_observations` row with reviewer, timestamp, and revision | The verifier re-reads it as evidence. The step's basis becomes `human_confirmed`. |

Original model responses and analysis revisions are never edited.

## Object views and appearance

`pipeline/object_views.py` asks Muse Wonder for a box around each object on the first frame, and on a few later stills when an object is missing or lost. SAM 3.1 follows those boxes at 12 frames per second, in one request and a second only for objects found later. The tracks become the thumbnails and per-object timelines in the UI.

On a demonstration, DINOv2 ViT-S/14 embeds crops from the SAM tracks into an appearance gallery per object. On an attempt, a SAM box that matches an object's gallery counts as a sighting of that object. Sightings can only keep a part from being called never seen. Appearance never passes or fails a step on its own.

SAM runs only when live calls are enabled and approved, and a failure here never costs the analysis.

## Second look

When a required step ends CANNOT VERIFY and no reviewer has decided it, `services/reinspection.py` sends Muse Wonder a few frames from that step with one question about its unresolved condition. The answer is stored as a finding and the verifier runs again. The first report stays unchanged next to the revision, and a step decided this way has basis `model_reinspection`. It sends at most three requests per verification, only when the attempt's analysis was approved for live calls or a stored answer exists.

## What is not here

The research code reused from vision-test-bench that the product does not run, including frame tracking, recovery experiments, evaluation, audio, and the lab API, was removed in `449e931`. See [PROVENANCE.md](PROVENANCE.md).
