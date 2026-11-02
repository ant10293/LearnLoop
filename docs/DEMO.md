# Demo script

**Status:** this script has run only with authored responses and a stubbed Meta client. It still has to be run with real recordings and real Meta responses. Update this file with the results.

## Recordings to make

Use a fixed camera on a tripod or stack of books, looking down at a plain surface. Use the same framing for all three, keep each under 30 seconds, and move your hands fully away at the end. Hold the final state for about 3 seconds.

Suggested task: a large flat base, plus three large blocks of different colors. Place red on the base, then blue on the base, then yellow on the base.

| File | Attempt | Expected result |
| --- | --- | --- |
| A, demonstration | All three placed correctly | Procedure with three steps |
| B, correct | All three placed correctly, ideally by a different route | PASS |
| C, incorrect | Yellow set on the table beside the base, clearly visible | FAIL on the yellow step |

C uses a positively visible wrong placement on purpose. It is stronger evidence than an omission, which relies on a report that something is absent.

## Script

1. Open LearnLoop. State the promise on the home screen.
2. **Create procedure** and name it.
3. Upload A, then **Learn the procedure**. Say which it is: a live call shows the approval panel, and a stored answer says "Replayed from a stored Meta response".
4. On **Learned procedure**, point out the "AI-generated draft" status and what the verifier checks for each step.
5. Make one real edit if needed, and say it is a human edit. Do not present human edits as model output. **Accept procedure**.
6. On **Verify**, upload B, then **Analyze and verify**. Show the result.
7. Upload C, then **Analyze and verify**. Select the failed step. Show expected and observed, the frame, and both videos seeking to the step.
8. If a step reports CANNOT VERIFY, leave it visible and explain why. Optionally answer the checkpoint, and show that the result becomes "Human-confirmed" with the automatic reading kept.
9. **Export report**.

## What is genuine

| Part | Source |
| --- | --- |
| Objects, actions, steps, relationships | Meta model, live or labeled cached |
| Verdicts | Deterministic verifier reading that output |
| Edited steps, labels, conditions | Human, marked as such |
| Checkpoint answers | Human, named and timestamped |

## Recovering from a failure

- **"Analyze again" after a failed analysis:** the failure reason is on screen. Trying again sends another request, which needs another approval. Do not retry repeatedly. Read the reason first.
- **`incomplete_output`:** the answer hit the output limit. Raise `LEARNLOOP_GUIDED_MAX_OUTPUT_TOKENS` deliberately. This changes the replay key.
- **Everything reports CANNOT VERIFY:** the hands or framing probably hid the final state, or the model classified the arrangement as inferred. Re-record rather than weakening the rules. Conditions can be corrected on the procedure screen.
- **Offline or no credits:** use cached replay. Keep the exact same video files, because a different file never replays. After a successful call, copy files from `data/app/inference_cache/` to `demo/replay/`.
