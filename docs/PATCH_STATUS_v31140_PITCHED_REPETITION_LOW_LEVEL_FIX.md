# v31.140 pitched repetition low-level fix

## Purpose

Fix the big-test failure mode where fast repeated pitched one-shots and musical loops were being treated as drum loops only because they had many attacks.

## Failure type

This was a low-level shape/physics boundary bug, not a filename problem and not a single-file rescue problem.

The bad pattern was:

- repeated events across the file
- strong pitched or tonal event evidence
- weak or moderate drum-loop source evidence
- later rhythmic-loop invariants still able to push the file into Drums

## Architecture-safe change

The patch adds a measured shape named `pitched_repetition_phrase` and a matching low-level physics score named `pitched_repetition_phrase_score`.

That shape/score separates:

- repeated pitched instrument phrases
- synth/keys/guitar/bass style repeated attacks
- vocal/rap repeated pitched phrases

from real drum-loop structure.

The drum-loop source score is now reduced when the new pitched-repetition decoy evidence is strong and drum evidence is weak.

## Files changed

- `src/aaron_sound_sorter/voters/shape_voter.py`
- `src/aaron_sound_sorter/domain/physics_subpanels.py`
- `src/aaron_sound_sorter/voters/physics_top_family_layer.py`
- `src/aaron_sound_sorter/voters/physics_instrument_layer.py`
- `src/aaron_sound_sorter/engine/consensus.py`
- `src/aaron_sound_sorter/engine/family_claim_arbiter.py`
- `tests/test_phase4_shape_voter.py`
- `tests/test_low_level_physics_subpanels.py`

## Guardrails added

- Fast pitched repetition now stays compatible with Instruments and Textures, not Drums.
- Real percussive drum loops still keep strong drum-loop source evidence.
- Strong synth repetition no longer gets narrowed to electric piano just because the tone is clean and mid-heavy.
- Real rap/vocal repetition can stay under Voice when low-level Human/Voice evidence is strong.
- High-density ambiguous pitched loops avoid false Voice overclaim and remain broad Instrument Loops.
- Clean keys/piano repeated loops can still restore to Keys when the physics top is Keys and the event/body evidence is keys-like.

## Validation run

Focused tests were run one at a time where needed:

- `tests/test_phase4_shape_voter.py` passed, 17 tests
- `tests/test_low_level_physics_subpanels.py` passed, 11 tests
- `tests/test_hierarchical_abstaining_arbitration_v31_99.py` passed, 29 tests
- `tests/test_decision_core_v2.py` passed, 12 tests
- `tests/test_two_voter_redesign.py` passed, 18 tests
- `tests/test_parent_eligibility_uploaded_audio.py` passed, 7 tests
- `tests/test_parent_eligibility_v24_drum_loop_steal_guard.py` passed, 5 tests
- `tests/test_v31138_clean_pitched_stab_role_guard.py` passed, 2 tests
- `tests/test_parent_eligibility_percussion_oneshots.py` passed, 7 tests
- `tests/test_phase4_pitched_percussion_guard.py` passed, 8 tests
- `tests/test_physics_voter_piano_struck_identity.py` passed, 3 tests
- `tests/test_v31116_synth_pad_keys_decoy_guard.py` passed, 2 tests
- `RUN_NO_SOURCE_NAME_SORTING_AUDIT.command` passed

The locked smoke acceptance panel was run case-by-case. All 32 protected cases passed after targeted fixes for the side effects exposed during the panel.

## Notes

The full one-by-one shell runner showed timeout behavior in the container after some cases even when the case log had already passed. I switched to direct one-case invocations with explicit shell timeouts and verified every protected case individually.
