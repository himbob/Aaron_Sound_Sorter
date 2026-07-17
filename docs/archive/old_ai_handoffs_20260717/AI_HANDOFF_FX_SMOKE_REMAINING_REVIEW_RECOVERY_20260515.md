# AI Handoff: FX Smoke Remaining Review Recovery

Date: 2026-05-15

## Trigger

Aaron reran the curated FX_Aaron2 smoke test after the prior recovery patch. The manifest still had 9 `_TO_REVIEW/Measured Role Conflict` rows:

- several sax/brass/woodwind pitched loops
- two drum-top/beat loops
- one vocal one-shot
- one keys/pad loop
- one Compton/sustained-pad style loop

Aaron stated the FX smoke ZIP is a known-good smoke pack and should not produce `_TO_REVIEW` rows.

## Design rule

No source filenames/folder names/path tokens may be used. The fix relies only on measured roles, measured shape vote/confidence, raw consensus, and candidate folders produced by voters.

## Change

`DecisionCoreV2.apply_eligibility()` now accepts optional `SharedAudioFacts`. `choose()` passes the facts object into the eligibility stage.

A new helper, `_measured_broad_bucket_for_smoke_stability()`, runs before the old conflict-review logic. It routes clear measured structures to broad useful buckets:

- pitched/bass/sustained musical phrase -> `Instruments/Instrument Loops/Loops`
- beat/top/drum loop -> `Drums/Drum Loops/Loops`
- vocal phrase -> `FX/Human and Voice FX`

It is intentionally broad and conservative. It does not promote exact leaves.

## Tests

New test file:

```text
tests/test_decision_core_fx_smoke_remaining_review_rows.py
```

Full local focused result:

```text
48 passed
```

## Follow-up

After install, Aaron should rerun the exact FX smoke command and run the no-review manifest audit. If review rows remain, inspect only those rows and add a failing synthetic test before patching.
