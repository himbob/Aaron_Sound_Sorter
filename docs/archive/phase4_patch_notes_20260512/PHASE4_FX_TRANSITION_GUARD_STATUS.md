# Phase 4 FX Transition Guard Status

Date: 2026-05-11

## Purpose

The latest giant run showed that FX transition leaves, especially risers, sweeps, whooshes, and downlifters, were being used for many sounds that did not have transition-style motion.

The fix is measured-physics based:

- no source filenames
- no source folder routing
- no one-file exceptions
- no invented categories

If a learned FX transition label is selected, the audio now needs envelope or spectral-motion evidence such as a reverse/tail shape, a slow swell, directional spectral motion, or a noisy moving sweep. If that evidence is missing, the sorter switches to an already nominated non-transition FX candidate or sends the file to review.

## Changed Files

- `src/aaron_sound_sorter/committee.py`
- `tests/test_phase4_fx_transition_guard.py`

## Spot Checks

Small real-file spot sort:

```text
Riser Short Effect.wav -> FX / Structural and Transitional FX / Risers and Builds / Short Riser, auto_place
Desire_reversedbeat2.wav -> FX / Structural and Transitional FX / Risers and Builds / Generic Riser, auto_place
BS_Fx45.wav -> _TO_REVIEW / Conflicting Evidence / Unsupported FX Transition Physics
```

The old giant manifest had 778 auto-placed FX transition-style rows. A manifest simulation with the new motion guard would block or repair 435 of them.

## Tests Run

```text
pytest -q tests/test_phase4_fx_transition_guard.py tests/test_phase4_v060_physical_family_guard_regressions.py
7 passed

pytest -q tests/test_phase4_fx_transition_guard.py tests/test_phase4_v060_physical_family_guard_regressions.py tests/test_phase4_contextual_brain_structure_and_groups.py tests/test_phase4_voice_guard.py tests/test_phase4_v066_loop_family_policy_regressions.py tests/test_phase4_pitched_percussion_guard.py
31 passed
```
