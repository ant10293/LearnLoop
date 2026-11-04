# Provenance

LearnLoop reuses parts of the author's earlier research repository, vision-test-bench. This file records what came from there, when it was written, and what was built for this submission.

## Timeline

| | |
| --- | --- |
| vision-test-bench first commit | 2026-10-21 |
| Hackathon kickoff | 2026-10-26, 12:00 PM ET (11:00 AM CDT) |
| vision-test-bench commits before kickoff | 39 of 116. The last is `27859ac` (2026-10-26 02:29 CDT) |
| vision-test-bench commits during the hackathon | 77 of 116 |
| LearnLoop repository created | 2026-10-29, empty until 2026-11-02 |
| Import into LearnLoop | `2babd11` (2026-11-02), an unmodified copy of vision-test-bench `386cba7`: `backend/`, `frontend/`, `scripts/` |
| Unused imported code removed | `449e931` (2026-11-04), about 32,600 lines |

The import commit copied the whole research backend so the history shows exactly what was brought in. Its data directory, database, credentials, caches, and stored results were not copied.

## How much predates kickoff

Measured by comparing each source line of more than 25 characters in LearnLoop with every line in vision-test-bench at `27859ac`, the last commit before kickoff. A matching line is counted as pre-kickoff even if it was moved or the code around it changed, so these numbers overstate the reuse slightly.

| Area | Lines matching pre-kickoff code |
| --- | --- |
| Backend application | 4,799 of 9,499 (51%) |
| Backend tests | 851 of 2,862 (30%) |
| Frontend | 205 of 1,949 (11%) |
| All source | 5,855 of 14,310 (41%) |

Files with the most pre-kickoff lines:

| File | Pre-kickoff lines | What it does |
| --- | --- | --- |
| `services/verification_store.py` | 280 of 507 | Stores procedures and reports, versions edits |
| `models/domain.py`, `runs.py`, `media.py`, `verification.py`, `sessions.py` | 680 of 687 | Database tables |
| `services/ingest.py` | 257 of 268 | Video upload, probing, and proxies |
| `api/serialize.py` | 256 of 312 | JSON for runs and videos |
| `verification/predicates.py`, `engine.py`, `learn.py`, `schema.py`, `reinspect.py`, `roles.py` | 755 of 1,068 | The deterministic verifier |
| `services/materialize.py` | 157 of 198 | Writes analysis records and crops |
| `pipeline/runner.py` | 148 of 221 | Runs the pipeline stages |
| `providers/catalog.py`, `meta_provider.py` | 200 of 271 | Model list and the Meta API adapter |

The verifier was first committed on 2026-10-26 at 01:54 CDT, about nine hours before kickoff, and was changed during the hackathon.

## Built during the hackathon

Everything below was written after kickoff, in vision-test-bench between 2026-10-26 and 2026-11-02 or in this repository from 2026-11-02:

- `hackathon_guided` analysis: `pipeline/guided.py`, the strict `GuidedTaskOutput` contract, prompts, replay keyed by exact inputs, and live-call approval
- SAM 3.1 object views: `pipeline/object_views.py`, first-frame locating, tracking, and re-finding lost objects
- DINOv2 appearance references: `perception/appearance.py`, `pipeline/appearance_stage.py`, `services/appearance.py`, and the never-seen check
- The automatic second look: `services/reinspection.py`
- Product API: `api/product.py`
- Procedure review: step and condition editing, step ordering, soft removal, review log, and review states
- Human observations as audited evidence: `models/product.py`, `apply_human`, and step `basis`
- Changes to the reused verifier, listed with their commits in [ARCHITECTURE.md](ARCHITECTURE.md#changed-during-the-hackathon)
- The whole frontend: `frontend/src/pages`, `components`, and `lib/learnloop.ts`
- Tests: `test_learnloop_flow.py`, `test_reinspection.py`, `test_appearance.py`, `learnloop.test.tsx`, `e2e/smoke.spec.ts`
- All documentation in this repository

## Removed from the imported code

Non-Meta providers (OpenAI, Anthropic, Google, Cursor, OpenAI-compatible, and CLI transports), Whisper transcription, the research frontend, and, in `449e931`, the tracking, recovery, evaluation, multi-pass parsing, audio, and demo-scenario modules and the lab API routes the product does not use. Their tests were removed with them. They remain in vision-test-bench.

## Tools

Claude Code (Anthropic) was used as a development tool. No Anthropic model runs in the submitted application.

## Demo assets

The demo recordings were made by the author, who may share them.
