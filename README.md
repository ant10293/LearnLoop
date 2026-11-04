# LearnLoop

**Teach a physical workflow once. Check a new attempt against what was learned.**

LearnLoop learns a short physical procedure from a recorded demonstration and checks a later recording against it. A Meta model proposes what happened. A person reviews it. Deterministic code decides each verdict and shows the evidence.

## The problem

Checking physical work usually needs a person watching, or a custom vision system trained for one task. LearnLoop tries a third route for simple tasks: show the task once, confirm what the system understood, and check later attempts against explicit conditions. When the evidence is not there, it says so.

This is a demonstration of reusable procedure learning with human oversight. It is not an autonomous industrial inspection system.

## How it works

1. **Teach.** Upload a recording of the task done correctly.
2. **Learn.** One Meta call returns the objects, actions, final arrangements, and procedure steps. These become a draft procedure with checkable completion conditions.
3. **Review.** Rename steps, correct object labels and conditions, mark steps required, remove wrong ones, and accept. Each saved edit is a new version, and the model's version is kept.
4. **Repeat.** Upload another attempt. Meta is told which objects to look for, not what the correct arrangement is or whether the attempt is correct.
5. **Verify.** The verifier compares the reported evidence with the accepted conditions. No model decides the verdict.

Each step reports **PASS**, **FAIL**, **CANNOT VERIFY**, or **NOT APPLICABLE**, with a timestamp and the frame behind it. Selecting a step seeks the demonstration and the attempt side by side.

### Evidence rules

- An object that was not detected is not automatically absent. A missing part needs the place it belongs in view and a direct report that it is not there. Otherwise the step is CANNOT VERIFY.
- An attempted action is not a completed one.
- Object IDs do not need to match between recordings. Objects are matched by role.
- A reviewer's answer is stored as a named, timestamped observation and read as evidence. It must cite the final interval of the recording. The report marks the result as human-confirmed and keeps the automatic reading beside it.

## Meta models

| Purpose | Model | Path |
| --- | --- | --- |
| Video analysis | `muse-wonder-1.0` | Meta Responses API (`https://api.meta.ai/v1/responses`), video file, strict JSON schema |

`muse-wonder-1.0` is the default because it returned procedure steps more consistently than `muse-spark-1.3` in earlier stored runs, on different footage from what this product supports. The `hackathon_guided` prompt and schema have not yet been run against a live model. See [Known limitations](#known-limitations).

No non-Meta model runs in this build. The `openai` Python package is only the HTTP client for Meta's compatible API. OpenCV, NumPy, and ffmpeg handle media and use no AI models. SAM 3.1 (a Meta model) makes the object thumbnails and timelines on demonstrations. It never feeds a verdict.

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
3. Approve the request in the app, which shows the model, the limits, and the cost if known.

Each analysis sends one analysis request, with no automatic retry. A demonstration also sends one Meta request that finds each object on the first frame, then one SAM request at 8 frames per second that follows them. A valid answer is stored against the exact video checksum, model, prompts, schema, and output limit, then replayed and labeled as cached. An invalid answer is reported, kept for audit, and never stored for replay.

## Recording tips

- Up to 60 seconds, with good lighting and the workspace in view
- A steady, high resolution camera (1080p - 2160p)

These make learning easier, but aren't required:

- A few large, distinct objects
- All parts visible in the first frame
- Controlled, deliberate movements

## Known limitations

- **Real-model accuracy is unproven.** The `hackathon_guided` flow has not yet been run against a live Meta model. Automated tests use authored responses, which show the application logic works and say nothing about what a model infers from footage.
- Relations are coarse. Two positions on the same object can both read as "attached", so a misplacement may need a distinct relation, such as beside instead of on, to be caught.
- Earlier stored Meta analyses of small parts assembled in the hand produced no false failures, and every required step came back unresolved.
- Recordings longer than 60 seconds are not supported.
- Single-user local app with no authentication. The reviewer name is typed in, not verified.

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

## Documentation

[Architecture](docs/ARCHITECTURE.md) · [Provenance](docs/PROVENANCE.md) · [Demo script](docs/DEMO.md) · [Submission](docs/SUBMISSION.md)
