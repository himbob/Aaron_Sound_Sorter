# Phase 4 Brain Structure Context Patch Status

Date: 2026-05-11

## Purpose

This bundle patches the Stage 4 / Phase 4 sorter so the brain respects measured structure earlier in the ranking process and reports coherent feature-group penalties.

The work was triggered by `DOJO_FBP_Female_Vocal_Shout.wav`, where the raw brain treated a short vocal shout as drum-like because the sound was short, bright, noisy, and transient-heavy.

## Architecture intent

This is not a filename patch and not a category hard-code for one sample.

The patch is intended to make the brain smarter in a general way:

1. Structure/role is evaluated before specific identity.
2. Clear short one-shot evidence removes loop candidates before the label contest.
3. Coherent groups of related feature violations are penalized, instead of treating every feature as an unrelated one-off.
4. The final guard still sends unresolved conflicts to review rather than forcing a confident wrong answer.

This matches the larger project rule: measured audio evidence first, structure/role before identity, and conflicts go to review instead of fake certainty.

## Changed files

- `src/aaron_sound_sorter/reports.py`
- `src/aaron_sound_sorter/preview.py`
- `tests/test_phase4_contextual_brain_structure_and_groups.py`
- `PHASE4_BRAIN_STRUCTURE_CONTEXT_PATCH_STATUS.md`

## Behavior on the vocal shout

Expected current behavior:

- The file is measured as a hard one-shot before ranking.
- Loop labels are removed up front.
- The raw brain may still see drum-like one-shot similarity because the current trained shout/scream/voice one-shot support is weak.
- The final placement should not go to Drums.
- It should go to review as a voiced/formant non-drum conflict unless stronger short-vocal training examples are added.

## Known remaining limitation

This patch does not magically teach the brain a strong short human shout class. The training support for short shout/scream/vocal one-shot material still appears too weak or too different from the uploaded shout sample.

Next likely architecture-safe step:

- Add 20 to 40 clean short vocal shout/scream/vocal-one-shot examples to the trusted training tree.
- Rebuild or update the brain.
- Then allow the contextual contrast logic to select the learned voice/human category when its membership support is strong enough.

## Tests run in the working session

Targeted tests passed:

```bash
pytest -q \
  tests/test_phase4_contextual_brain_structure_and_groups.py \
  tests/test_phase4_voice_guard.py \
  tests/test_phase4_dynamic_structure_gate.py \
  tests/test_phase4_dynamic_structure_gate_v0485.py
```

Additional test files were run in chunks and passed before environment time ran out. Two tests remained skipped as before because their optional real ZIP fixtures were unavailable in that test context.

The full one-shot `pytest -q` run timed out in the sandbox environment, but chunked file-level runs did not show failing assertions.

## Suggested local smoke command

From the project root on Aaron's Mac:

```bash
python3 -m pytest -q tests/test_phase4_contextual_brain_structure_and_groups.py tests/test_phase4_voice_guard.py tests/test_phase4_dynamic_structure_gate.py tests/test_phase4_dynamic_structure_gate_v0485.py
```

Then run a real-file sort against the shout or any suspicious short vocal/percussive one-shot sample and inspect the manifest columns for:

- `measured_structure`
- `measured_structure_confidence`
- `measured_structure_reason`
- `coherent_group_penalty`
- `coherent_group_reason`


Git status before packaging:
 M src/aaron_sound_sorter/preview.py
 M src/aaron_sound_sorter/reports.py
?? PHASE4_BRAIN_STRUCTURE_CONTEXT_PATCH_STATUS.md
?? tests/test_phase4_contextual_brain_structure_and_groups.py
