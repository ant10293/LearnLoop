# Submission

## Summary

LearnLoop learns a short physical procedure from a recorded demonstration and checks a later recording against it. Meta models propose the procedure and follow each part through the video, a person reviews it, and deterministic code decides each verdict, with the timestamp and frame shown.

## Problem and intended user

Teams that train people on simple physical tasks, such as kit assembly or packing, often rely on someone watching to confirm the work. Custom vision inspection is costly to build per task. LearnLoop is for someone who wants to show a task once and check later attempts, with a clear signal when a recording cannot settle the question.

## Technical approach

- One Muse Wonder video call returns a strict, schema-validated task analysis: objects, actions, final arrangements, and steps.
- Muse Wonder finds each object on the first frame, SAM 3.1 follows it through the recording, and DINOv2 builds appearance references for each part.
- The analysis becomes a versioned procedure with explicit completion conditions, and a person reviews and accepts it.
- A later recording is analyzed the same way. A deterministic verifier, not a model, compares the evidence with the conditions.
- Missing evidence yields CANNOT VERIFY, not FAIL. For an unresolved required step, LearnLoop asks Meta one narrow question about a few frames and verifies again. Reviewers can add named, timestamped observations, which are labeled in the result.

## Meta model integration

- `muse-wonder-1.0` through the Meta Responses API: native video analysis with strict structured output, object locating on still frames, and the second look.
- SAM 3.1 through the Meta Model API: object tracking at 12 frames per second.
- DINOv2 ViT-S/14, Meta open weights, run locally: appearance references.

No non-Meta model runs in the application. Responses are stored against the exact video and prompt, and replay is labeled as cached.

## Demonstrated capabilities

- Tested live with real recordings: 7 procedures, 20 analysis runs, 16 verifications, and 75 Meta requests, on fixed-camera and head-mounted footage. On a head-mounted Lego build, an attempt missing one small part is reported FAIL on that step as a missing component.
- Tested with authored responses and a stubbed Meta client: the teach, review, verify, evidence, second look, human-confirmation, replay, and export flow, in 240 backend tests, 44 frontend tests, and a browser end-to-end test.

## Originality

LearnLoop builds on the author's earlier research code, vision-test-bench, started on 2026-10-21. About 4,800 lines of the backend existed before kickoff, mainly the database models, media ingest, and the core of the deterministic verifier. The guided Meta flow, SAM object views, appearance references, the second look, the product API, and the whole frontend were built during the hackathon. [PROVENANCE.md](PROVENANCE.md) has the dates and line counts.

## Limitations

Short clips, a few distinct objects, and a visible final state. Results vary between runs on the same recording, and a handful of recordings do not establish accuracy. Relations are coarse. It is not an autonomous inspection system.

## Future roadmap

Longer multi-stage procedures, finer spatial conditions, measured accuracy on a labeled set of recordings, and checking attempts while they happen.
