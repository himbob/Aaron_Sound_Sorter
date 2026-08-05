# AI Handoff: FX Smoke Stability Hotfix

Date: 2026-05-15

## Why this patch exists

The true-bucket conflict resolver got too aggressive. It converted stable winners from the old FX smoke test into `_TO_REVIEW/Measured Role Conflict` even when the raw top family was already well supported by candidate evidence.

The user saw 11 files in `_TO_REVIEW` in the FX smoke run. The correct architectural fix is not to use filenames and not to whitelist files. The fix is to prevent a generic measured-conflict role from overriding a stable raw top-family winner.

## Source-name rule

This patch does not inspect producer filenames, source folders, ZIP member names, or path tokens for sorting decisions.

## Code change

Changed:

```text
src/aaron_sound_sorter/engine/decision_core_v2.py
```

Added `_has_stable_raw_top_support(raw)` and changed the generic conflict-role branch:

```text
if measured conflict role wants _TO_REVIEW
and raw path is allowed
and raw top family is decisively supported by candidates
    keep raw
else
    review
```

This preserves real conflict review for unstable cases and for known high-risk drum-family conflicts.

## New tests

Added:

```text
tests/test_decision_core_fx_smoke_no_review_guard.py
```

These tests were red before the fix:

```text
3 failed, 2 passed
```

They pass after the fix:

```text
5 passed
```

## What to run

```bash
cd /path/to/Aaron_Sound_Sorter
./commands/quality/RUN_FX_SMOKE_STABILITY_HOTFIX_TESTS.command
```

Then rerun the old FX smoke sort and verify that stable files no longer move to `_TO_REVIEW/Measured Role Conflict`.
