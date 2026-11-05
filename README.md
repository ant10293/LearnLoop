# LearnLoop

**Teach a physical workflow once. Check a new attempt against what was learned.**

**Demo video:** [youtu.be/MBAlZAZBJjg](https://youtu.be/MBAlZAZBJjg)

LearnLoop learns a short physical procedure from a recorded demonstration and checks a later recording against it. Meta models propose what happened and follow each part through the video. A person reviews the procedure. Deterministic code decides each verdict and shows the evidence.

## The problem

Checking physical work usually needs a person watching, or a custom vision system trained for one task. LearnLoop tries a third route for simple tasks: show the task once, confirm what the system understood, and check later attempts against explicit conditions. When the evidence is not there, it says so.

This is a demonstration of reusable procedure learning with human oversight. It is not an autonomous industrial inspection system.

## How it works

1. **Teach.** Upload a recording of the task done correctly.
2. **Learn.** One Muse Wonder call returns the objects, actions, final arrangements, and procedure steps as strict JSON. A second call finds each object on the first frame, and SAM 3.1 follows them through the video. These become a draft procedure with checkable completion conditions, object thumbnails, and a timeline per object.
3. **Review.** Rename steps, correct object labels and conditions, mark steps required, remove wrong ones, and accept. Each saved edit is a new version, and the model's version is kept.
4. **Repeat.** Upload another attempt. Meta is told which objects to look for, not what the correct arrangement is or whether the attempt is correct.
5. **Verify.** The verifier compares the reported evidence with the accepted conditions. No model decides the verdict.
6. **Second look.** When a required step is left unresolved, LearnLoop asks Meta one narrow question about a few frames of that step. The answer is read as evidence by the same verifier, and the first report is kept beside it.

Each step reports **PASS**, **FAIL**, **CANNOT VERIFY**, or **NOT APPLICABLE**, with a timestamp and the frame behind it. Selecting a step seeks the demonstration and the attempt side by side.

### Evidence rules

- An object that was not detected is not automatically absent. A missing part needs the place it belongs in view and a direct report that it is not there, or it must never appear anywhere in the attempt. Otherwise the step is CANNOT VERIFY.
- An attempted action is not a completed one.
- Object IDs do not need to match between recordings. Objects are matched by role, and appearance references from the demonstration help tell similar parts apart. Appearance never passes or fails a step on its own.
- A reviewer's answer is stored as a named, timestamped observation and read as evidence. It must cite the final interval of the recording. The report marks the result as human-confirmed and keeps the automatic reading beside it.

## Meta models

| Purpose | Model | Path |
| --- | --- | --- |
| Procedure analysis of the whole video | `muse-wonder-1.0` | Meta Responses API, native video file, strict JSON schema |
| Finding each object on the first frame, and again on later stills when one is lost | `muse-wonder-1.0` | Meta Responses API, still frames, strict JSON schema |
| Following each object through the recording, 12 frames per second | SAM 3.1 | Meta Model API |
| Second look at an unresolved step | `muse-wonder-1.0` | Meta Responses API, a few frames and one question |
| Appearance references for each object | DINOv2 ViT-S/14 (Meta open weights) | Runs locally with frozen weights |

No non-Meta model runs in this build. Meta API calls go through a small httpx client (`backend/app/providers/meta_client.py`); no third-party AI SDK is installed. OpenCV, NumPy, and ffmpeg handle media and use no AI models.

## Live results

Every flow below ran against the live Meta API with real recordings, made by the author on 2026-11-02 to 2026-11-04.

- 7 procedures, 20 analysis runs, 16 verifications, and 75 Meta requests across Muse Wonder and SAM 3.1.
- Fixed-camera tabletop recordings and head-mounted recordings, where the camera moves with the person.
- Lego motorcycle builds of 5 steps and a small hardware assembly.

One case, end to end. The demonstration is a 39.6-second head-mounted recording of a Lego motorcycle build. LearnLoop learned 5 steps, and SAM followed all 6 parts. The attempt is a 20.6-second recording that leaves out the small clear stud. The report is **FAIL** on "Attach clear stud", as a missing component that never appears in the attempt. The second look resolved one more step to PASS.

Results vary from run to run on the same recording. The same attempt has come back with its middle steps as PASS on one run and CANNOT VERIFY on another. When the evidence is weak, LearnLoop reports CANNOT VERIFY instead of guessing.

## Run it

Requires Python 3.12, Node 20 or newer, and ffmpeg.

```bash
python3.12 -m venv backend/.venv
```

```bash
backend/.venv/bin/pip install -e './backend[dev,perception]'
```

Load the DINOv2 weights once, so the first analysis does not download them:

```bash
backend/.venv/bin/python -m app.perception.encoder
```

```bash
npm install --prefix frontend
```

```bash
scripts/dev.sh
```

Open http://127.0.0.1:5180. The API runs on port 8010.

### Live analysis

Live calls are off by default. A live analysis needs all three of these, and cached replay needs none of them:

1. `cp .env.example .env`, then set `MODEL_API_KEY`.
2. Set `LEARNLOOP_LIVE_INFERENCE=1` in `.env`.
3. Approve the request in the app, which shows the model, the number of requests, and the cost if known.

Nothing is retried automatically. A valid answer is stored against the exact video checksum, model, prompts, schema, and output limit, then replayed and labeled as cached. An invalid answer is reported, kept for audit, and never stored for replay. Keep the Verify page open until an attempt's analysis finishes, because the page starts the check.

## Recording tips

- Up to 60 seconds, with good lighting and the workspace in view
- A high resolution camera (1080p - 2160p). A fixed camera is easiest, and head-mounted recordings also work.

These make learning easier, but aren't required:

- A few large, distinct objects
- All parts visible in the first frame
- Controlled, deliberate movements

## Known limitations

- Results vary between runs on the same recording, and a handful of recordings do not establish accuracy.
- Relations are coarse. Two positions on the same object can both read as "attached", so a misplacement may need a distinct relation, such as beside instead of on, to be caught.
- Small parts handled close to the hand are often left CANNOT VERIFY.
- Recordings longer than 60 seconds are not supported.
- Single-user local app with no authentication. The reviewer name is typed in, not verified.

## What was built during the hackathon

LearnLoop was built from 2026-11-02 in this repository. It reuses parts of the author's earlier research repository, vision-test-bench, which was started on 2026-10-21. Some of that code existed before the hackathon kickoff (2026-10-26, 12:00 PM ET): mainly the database models, media ingest, and the core of the deterministic verifier. [PROVENANCE.md](docs/PROVENANCE.md) lists what was reused, with dates and line counts.

Built during the hackathon:

- The guided Meta flow: one strict-schema Muse Wonder call per recording, replay keyed by exact inputs, and live-call approval
- SAM 3.1 object views: first-frame locating, tracking, and re-finding lost objects
- DINOv2 appearance references and the check that a part never appears in an attempt
- The automatic second look at unresolved steps
- The product API, procedure review and versioning, and human observations as audited evidence
- The whole frontend
- Verifier fixes described in [ARCHITECTURE.md](docs/ARCHITECTURE.md)
- The Reliability Lab, a developer tool at `/lab` that repeats independent runs on one recording and compares the results

## Tests

```bash
cd backend && .venv/bin/python -m pytest -q
```

```bash
cd frontend && npm test
```

```bash
cd frontend && npx playwright test
```

Automated tests use authored responses and a stubbed Meta client. They show the application logic works and say nothing about what a model infers from footage.

## Documentation

[Architecture](docs/ARCHITECTURE.md) · [Provenance](docs/PROVENANCE.md) · [Demo script](docs/DEMO.md) · [Submission](docs/SUBMISSION.md)
