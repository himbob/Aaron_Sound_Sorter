# AI Handoff: Role/Candidate Conflict Resolver Patch

Date: 2026-05-15

## Problem fixed

Fresh random-folder and failure-review logs showed real bug classes:

- guitar/strum/chord one-shots were being forced into Drums/Toms or generic percussion
- vocal loops were being forced into Drum Loops
- whoosh/swoosh motion FX were being forced into percussion one-shots
- wind chimes / non-vocal tonal one-shots were being forced into Human/Voice FX
- pitched percussion loops were being kept as generic Instrument Loops instead of being reviewed

The common architecture failure was broad role forcing without a central candidate-conflict check.

## TDD rule followed

A synthetic regression file was added before the fix:

```text
tests/test_decision_core_conflict_resolver_tdd.py
```

It initially failed against the current code:

```text
4 failed, 1 passed
```

Then the smallest conflict-resolver seam was patched and the same tests passed:

```text
5 passed
```

This red/green workflow has now been added as a permanent project rule in:

```text
docs/AI_FIX_WORKFLOW_RULE_RED_GREEN_SYNTHETIC_20260515.md
docs/TESTING_POLICY.md
```

## Code changed

```text
src/aaron_sound_sorter/engine/decision_core_v2.py
src/aaron_sound_sorter/engine/eligibility.py
```

## Main design change

`DecisionCoreV2` now has a central role/candidate conflict review seam. It sends dangerous contradictions to:

```text
_TO_REVIEW/Measured Role Conflict
```

instead of forcing a wrong broad bucket.

This is intentionally conservative. Review is safer than a wrong top family.

## Real audio spot checks in this environment

The copied-audio review pack produced these after the patch:

```text
Guitar_D.wav
=> _TO_REVIEW/Measured Role Conflict

Strum 22_Amin.wav
=> _TO_REVIEW/Measured Role Conflict

Chord 15_E-B.wav
=> _TO_REVIEW/Measured Role Conflict

whoosh_and_swoosh_and_swish_val_389458.wav
=> _TO_REVIEW/Measured Role Conflict

wind_chime_val_95752.wav
=> _TO_REVIEW/Measured Role Conflict
```

The bongo loop case is now detected by eligibility as `pitched_percussion_conflict_loop`; the conflict role is forced to review instead of being allowed to remain a generic instrument loop.

## Focused pytest result

```text
57 passed
```

## Validation command

```bash
cd /Volumes/T9/testbed/Aaron_Sound_Sorter
./commands/quality/RUN_CONFLICT_RESOLVER_REGRESSION_TESTS.command
```

## What remains

This patch increases review behavior for conflict cases. That is intentional. It does not yet solve exact deep leaf placement for guitar, bongo, conga, triangle, or whoosh. It prevents the bad confident folders first.
