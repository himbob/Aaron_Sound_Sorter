# Aaron Sound Sorter Phase 4 v0.6.4 Status

## Focus

This build is a test-first repair pass for the failed-file scenarios found in the 2026-05-09 big real preview.

The failure class was not only dirty training data. Several files showed that the raw brain, model tournament, committee, and membership fallback can each fail in different ways.

## Main fixes

- Added failed-file diagnostic pytest coverage for:
  - `Piano_G.wav`
  - `Acoustic Guitar Rhythm_25_KeyDm_100bpm.wav`
  - synthetic `OrganChord1_Bmin`
  - synthetic `PlinkyPianoArp1_keyBbmin`
  - synthetic `Fill_4_130`
- Added unit tests that isolate raw brain ranking, tournament ranking, blocked membership behavior, same-family switching, and cross-family laundering.
- Kept broad-family cross-family switches blocked unless a future brain explicitly opts in through the new explicit override flag.
- Preserved legitimate same-family support-balance refinement, which v0.6.3 had accidentally weakened.
- Updated membership helper tests so cross-family “safe alternatives” go to review, while same-family alternatives can still switch.

## Known status

Targeted validation run in this bundle:

```text
python3 -S -m py_compile Aaron_Sound_Sorter.py
python3 -m pytest -q \
  tests/test_phase4_v064_failed_file_diagnostics_contracts.py \
  tests/test_phase4_v061_cross_family_policy_regressions.py \
  tests/test_phase4_v052_physics_atlas_validation.py::test_helper_reviews_instead_of_cross_family_switch_when_one_and_two_fail_physics \
  tests/test_phase4_v052_physics_atlas_validation.py::test_helper_can_choose_candidate_three_inside_same_family_when_one_and_two_fail_physics \
  tests/test_stage4_adaptive_label_model.py::test_support_balance_boost_helps_close_underrepresented_folder_without_names \
  tests/test_phase4_v062_tournament_calculation_guards.py \
  tests/test_phase4_v063_tournament_factor_contracts.py
```

Result:

```text
122 passed
```

Self-test result:

```text
All Stage 4 v0.6.4 self-tests passed.
```

## Important limitation

This build prevents the known bad files from becoming confident wrong-family auto-placements. It does not prove the raw brain is good. In particular, the acoustic guitar rhythm still exposes a raw-distance failure: the scratch brain ranks `FX / Human and Voice FX / Applause / One Shots` above `Instruments / Guitar / Acoustic Guitar / Loops`, although the guitar candidate is nearby.

Next repair target: per-feature distance contribution audit for why rhythmic acoustic guitar is closer to applause than acoustic guitar loop.
