# Submission draft

**Fill in the real-video results before submitting. Do not submit the capability statements below as tested until they are.**

## Summary

LearnLoop learns a short physical procedure from a recorded demonstration and checks a later recording against it. A Meta model proposes the procedure, a person reviews it, and deterministic code decides each verdict, with the timestamp and frame shown.

## Problem and intended user

Teams that train people on simple physical tasks, such as kit assembly or packing, often rely on someone watching to confirm the work. Custom vision inspection is costly to build per task. LearnLoop is for someone who wants to show a task once and check later attempts, with a clear signal when a recording cannot settle the question.

## Technical approach

- One Meta video call returns a strict, schema-validated task analysis: objects, actions, final arrangements, and steps.
- The analysis becomes a versioned procedure with explicit completion conditions, and a person reviews and accepts it.
- A later recording is analyzed in the same way. A deterministic verifier, not a model, compares the evidence with the conditions.
- Missing evidence yields CANNOT VERIFY, not FAIL. Reviewers can add named, timestamped observations, which are labeled in the result.

## Meta model integration

`muse-wonder-1.0` through the Meta Responses API, with native video input and strict structured output. No non-Meta model runs in the application. Responses are stored against the exact video and prompt, and replay is labeled as cached.

## Originality

LearnLoop builds on the author's earlier research code, `vision-test-bench`. Part of that code predates the hackathon start. [PROVENANCE.md](PROVENANCE.md) lists what was reused, with dates, and what was built for this submission. **Confirm against the signed rules before submitting.**

## Demonstrated capabilities

- Tested with authored responses and a stubbed Meta client: the full teach, review, verify, evidence, human-confirmation, replay, and export flow.
- Tested with real Meta responses on real footage: **not yet. Add results here.**

## Limitations

Fixed camera, short clips, a few large distinct objects, and a visible final state. Relations are coarse. Accuracy on real footage has not been measured, and one demonstration does not establish it. It is not an autonomous inspection system.

## Future roadmap

Head-mounted footage, object identity tracking through occlusion, finer spatial conditions, and longer multi-stage procedures.
