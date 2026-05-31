# Aaron Sound Sorter v31.97 Raw-Voter Architecture Patch

## Problem found

The v31.96 claim-validity refactor moved final decision ownership into the arbiter, but old rule behavior still existed earlier in the pipeline.

Two remaining pre-arbiter mechanisms were still changing outcomes before the arbiter saw clean evidence:

1. `role_compatibility_adjustment()` changed BrainVoter/PhysicsVoter scores with measured-role boosts and penalties.
2. `ConsensusWinnerPreselector.choose()` replaced the raw best shared Brain/Physics candidate using measured-role logic.

That meant a file could have correct full/core/spread/outlier brain evidence and still be redirected because `percussive_one_shot` made Drums look compatible and Instruments look incompatible.

## Architecture change

Brain and physics voters now report raw identity evidence. Role/shape facts are diagnostic-only at voter ranking time.

`role_compatibility_adjustment()` still reports:

- measured parent role
- candidate role signature
- family compatibility
- recommended adjustment

But it always returns an applied adjustment of `0.0` and marks `role_adjustment_mode=diagnostic_only`.

`ConsensusWinnerPreselector.choose()` now returns the first shared candidate from the already sorted shared candidate table. It no longer uses measured role to preselect a different raw winner.

Profile kick shortcuts can no longer cross families from a concrete raw candidate.

## Real sample panel result in sandbox

After this patch:

- `AA_TSS_D_RUMBLER_808_BASS.wav` -> `Instruments/Bass/808 Bass/One Shots`
- `boom_eval_108640.wav` -> `FX/Impacts and Hits/Boom/One Shots`
- `clipped_sub_hit.wav` no longer routes to Kick
- `Clap 17.wav` is still not a clean truth case and should not be used for lane tuning

## What this patch intentionally does not solve

This does not tune kick identity, explosion identity, clap identity, or lane weights. It removes old pre-arbiter behavior first so the next failures are easier to read.

## Validation run in sandbox

Passed, one node/file at a time:

- `tests/test_v3197_no_role_score_shaping_architecture.py::test_role_compatibility_never_changes_voter_score`
- `tests/test_v3197_no_role_score_shaping_architecture.py::test_winner_preselector_returns_raw_best_shared_candidate`
- `tests/test_v3197_no_role_score_shaping_architecture.py::test_profile_kick_shortcut_cannot_cross_family_from_specific_bass_raw`
- `tests/test_v3184_deep_voter_and_baby_recall.py::test_role_evidence_is_diagnostic_only_for_incompatible_fx_candidate`
- `tests/test_v3184_deep_voter_and_baby_recall.py` full file
- `tests/test_claim_validity_refactor_v3196.py`
- `tests/test_claim_arbiter_architecture.py`
- `tests/test_family_claim_arbitration_matrix.py`
- `tests/test_two_voter_redesign.py`
- `tests/test_broad_bucket_claim_producer.py`
- `tests/test_no_source_name_sorting_invariant.py`
- `tests/test_brain_competence_policy.py`
- `tests/test_v3187_architecture_doc_and_lane_competence.py`
- `tools/audit_no_source_name_sorting.py`
- `py_compile` on changed files

