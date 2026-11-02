# Provenance

## Open eligibility item

**The signed hackathon rules are not in either repository and were not reviewed when this file was written.** The rules are authoritative for whether the reuse below is permitted. The participant must check this before submitting.

## Source

| | |
| --- | --- |
| Source repository | https://github.com/ant10293/vision-test-bench |
| Source commit used | `386cba7349450cd0d1567cfef8e80c6353faab1e` (2026-11-02) |
| First source commit | 2026-10-21 |
| Commits before 2026-10-26 00:00 CDT | 36 of 116. The last is `f6b2a06` (2026-10-25 23:12) |
| LearnLoop repository created | 2026-10-29 (empty until this work) |
| Import commit in this repository | `2babd11`, an unmodified copy of `backend/`, `frontend/`, `scripts/` |
| Hackathon start, as stated by the participant | 2026-10-26 |

The source repository was not modified. Its data directory, database, credentials, caches, and stored results were not copied.

## Reused foundations

First-commit dates come from `git log` in the source repository. A module added before the start date may also have been changed after it.

| Component | Path | First commit | Last source change |
| --- | --- | --- | --- |
| Correction revisions | `services/corrections.py` | 2026-10-21 | 2026-10-21 |
| Media and ingest | `video/`, `services/ingest.py` | 2026-10-21 | 2026-11-01, 2026-10-26 |
| Database models | `models/` | 2026-10-21 | 2026-11-01 |
| Materialization | `services/materialize.py` | 2026-10-21 | 2026-11-01 |
| Jobs | `jobs/` | 2026-10-21 | 2026-10-27 |
| Pipeline runner and inference | `pipeline/runner.py`, `pipeline/infer.py` | 2026-10-21 | 2026-11-01 |
| Meta adapter | `providers/meta_provider.py` | 2026-10-21 | 2026-11-01 |
| Tracking, inactive in the product | `perception/` | 2026-10-25 | 2026-11-02 |
| Verification engine, learner, predicates | `verification/` | 2026-10-26 | 2026-11-01 |
| Template and report store | `services/verification_store.py` | 2026-10-26 | 2026-10-26 |
| Strict output contracts | `pipeline/contracts.py` | 2026-11-01 | 2026-11-01 |

The verification code was first committed on 2026-10-26 at 01:54 CDT. Whether that falls inside the hackathon window depends on the official start time and timezone.

## New for this submission

Developed 2026-11-02 and later, in this repository's history after `2babd11`:

- `hackathon_guided` analysis: `pipeline/guided.py`, the `GuidedTaskOutput` contract, prompts, replay keyed by exact inputs, and live-call gates
- Product API: `api/product.py`
- Procedure review: step and condition editing, soft removal, review log, and review states
- Human observations as audited evidence: `models/product.py`, `apply_human`, and step `basis`
- Verifier fixes described in [ARCHITECTURE.md](ARCHITECTURE.md)
- A new frontend: four screens in `frontend/src/pages`, `components`, and `lib/learnloop.ts`
- New tests: `test_learnloop_flow.py`, `learnloop.test.tsx`, `e2e/smoke.spec.ts`
- All documentation in this repository

## Removed from the imported code

Non-Meta providers (OpenAI, Anthropic, Google, Cursor, OpenAI-compatible, and CLI transports), Whisper transcription, and the research frontend. Tests covering that code were removed or skipped, and they remain in the source repository. Skips are listed in `backend/tests/conftest.py` with the reason.

## Tools

Claude Code (Anthropic) was used as a development tool. No Anthropic model runs in the submitted application.

## Demo assets

No demo recordings are included yet. Record here who made each one and that it may be shared. Files in `demo/replay/` must come from real Meta responses.
