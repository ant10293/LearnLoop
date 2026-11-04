# Demo script

**Status:** this flow has run live with real recordings and real Meta responses, on fixed-camera and head-mounted footage. Results vary between runs, so record the take you show and do not edit verdicts by hand.

The video is three minutes. Show only what was built during the hackathon: the guided Meta analysis, SAM object views, appearance references, procedure review, verification with its evidence, the second look, and human confirmation. See [PROVENANCE.md](PROVENANCE.md).

## Recordings

The tested pair is a head-mounted Lego motorcycle build:

| File | Attempt | Expected result |
| --- | --- | --- |
| A, demonstration | `lego-motorcyle-head-mounted-pov.MOV`, 39.6 seconds, all 5 steps | Procedure with 5 steps, 6 parts followed by SAM |
| B, incorrect | `lego-motorcyle-head-mounted-pov-missing-small-clear-disc.mov`, 20.6 seconds, clear stud left out | FAIL on "Attach clear stud", missing component |

A correct attempt, ideally by a different route, makes a good third clip. Keep each recording under 60 seconds and hold the final state in view for about 3 seconds.

## Script

1. Open LearnLoop. State the promise on the home screen.
2. **Create procedure** and name it.
3. Upload A, then **Learn the procedure**. A live call shows the approval panel with the models and request count. A stored answer says "Replayed from a stored Meta response".
4. On **Learned procedure**, point out the "AI-generated draft" status, the SAM boxes on the video, each object's thumbnails and timeline, and what the verifier checks for each step.
5. Make one real edit if needed, and say it is a human edit. Do not present human edits as model output. **Accept procedure**.
6. On **Verify**, upload B, then **Analyze and verify**. Keep this page open until the result appears, because the page starts the check.
7. Select the failed step. Show expected and observed, and both videos seeking to the step.
8. Point out the second-look line, such as "Reinspection resolved 1 step", and that the first report is kept.
9. If a step reports CANNOT VERIFY, leave it visible and explain why. Optionally answer the checkpoint, and show that the result becomes "Human-confirmed" with the automatic reading kept.
10. **Export report**.

## What is genuine

| Part | Source |
| --- | --- |
| Objects, actions, steps, relationships | Muse Wonder, live or labeled cached |
| Object boxes and tracks | Muse Wonder on the first frame, then SAM 3.1 |
| Appearance matches | DINOv2 on SAM crops |
| Verdicts | Deterministic verifier reading that output |
| Second-look findings | Muse Wonder, one question per unresolved step |
| Edited steps, labels, conditions | Human, marked as such |
| Checkpoint answers | Human, named and timestamped |

## Recovering from a failure

- **A failed analysis:** the failure reason is on screen. Trying again sends another request, which needs another approval. Read the reason before retrying.
- **`incomplete_output`:** the answer hit the output limit. Raise `LEARNLOOP_GUIDED_MAX_OUTPUT_TOKENS` deliberately. This changes the replay key.
- **Everything reports CANNOT VERIFY:** the framing or an obstruction probably hid the final state, or the model classified the arrangement as inferred. Re-record rather than weakening the rules. Conditions can be corrected on the procedure screen.
- **The page was closed before an attempt finished:** reopening it does not start the check. Remove the attempt and analyze it again, which needs a new approval.
- **Offline or no credits:** use cached replay. Keep the exact same video files, because a different file never replays. After a successful call, copy files from `data/app/inference_cache/` to `demo/replay/`.
