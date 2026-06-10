# HANDOFF v31.172 — percussion offsets 150–185 low-level patch

## Scope

Continued the `one_shot_percussive_sounds.zip` sweep from the v31.171 checkpoint.

- Previous screened total: `594 / 10,254`
- New screened rows: `200`
- New range: offsets `150,155,160,165,170,175,180,185`
- Per-folder selection: `5`
- Source folders: `5`
- Ordinals covered: `151–190` per source folder
- Current screened total: `794 / 10,254 = 7.7433%`

## Final corrected outcome

After low-level fixes and third-party-enabled reruns of bad/borderline rows:

```text
Drums: 194
FX: 0
_TO_REVIEW/Broken Or Tiny: 6
Instruments failures: 0
Non-broken bad _TO_REVIEW: 0
```

The six review rows were rerun individually with third-party features enabled and returned `_TO_REVIEW/Broken Or Tiny`. They are counted as acceptable review, not sorter failures.

## Real failures fixed

### 1. `one_shot_percussive_sounds/1/15582.wav`

Before:

```text
_TO_REVIEW/Measured Role Conflict
status: blocked_role_shape_routing_review
shape: texture_bed / noise_texture
```

Third-party-enabled evidence showed a short high/noisy struck percussion hit:

```text
duration ~= 0.341s
parent role: protected_percussive_one_shot
drumlike_frame_ratio: 1.0
percussive_event_ratio: 1.0
cymbal/guiro/metallic panels strong
```

After:

```text
Drums/Percussion/Guiros Scrapes and Rasps/One Shots
status: final_measured_protected_percussive_parent_claim
```

### 2. `one_shot_percussive_sounds/5/439835.wav`

Before:

```text
_TO_REVIEW/Measured Role Conflict
status: role_candidate_conflict_review / blocked_role_shape_routing_review
shape: texture_bed / noise_texture
```

Third-party-enabled evidence showed strong drum-material panels with voice/formant false positives:

```text
duration ~= 0.882s
role_one_shot_score ~= 0.960
compact_struck_tonal_percussion_score ~= 0.816
drum_guiro_scrape ~= 0.871
drum_metallic_percussion ~= 0.824
onset_percussive ~= 0.848
voice/formant panels high but treated as decoys because the struck drum-material body is stronger
```

After:

```text
Drums/Percussion/Guiros Scrapes and Rasps/One Shots
status: final_measured_protected_percussive_parent_claim
```

## Architecture notes

The fixes were intentionally pushed down below the arbiter:

- `eligibility.py`
  - Adds a source-name-blind drum-material voice-decoy guard before processed-vocal eligibility can steal compact metallic/guiro/cymbal hits.
  - Keeps the decision at broad parent-family level: protected percussion vs vocal/FX/instrument.

- `measured_drum_structures.py`
  - Extends measured protected-percussion parent claim support for:
    - bright noisy tail hits shaped as `texture_bed`
    - bright metallic/guiro/cymbal one-shots with false voice/formant pressure
  - Emits `final_measured_protected_percussive_parent_claim` before the arbiter needs to rescue review.

- `family_claim_arbiter.py`
  - No new broad arbiter rescue was added in this patch.

No filename/source-folder production sorting was added. No brain rebuild. No JSON ledger mutation.

## Ledgers included

```text
coverage_ledgers/percussion_v31172_offsets150_185_corrected.csv
coverage_ledgers/percussion_v31172_summary.json
coverage_ledgers/percussion_v31172_summary.txt
```

## Tests run

Passed:

```bash
NUMBA_DISABLE_JIT=1 python3 -m pytest tests/test_measured_percussion_zip_regression_guards.py -q
# 19 passed

NUMBA_DISABLE_JIT=1 python3 -m pytest tests/test_claim_arbiter_real_panel_surrogates.py -q
# 25 passed

NUMBA_DISABLE_JIT=1 python3 -m pytest \
  tests/test_consensus_concrete_fx_gate_regressions.py::test_stable_pitched_music_phrase_without_instrument_candidate_goes_to_review \
  tests/test_decision_core_fx_smoke_remaining_review_rows.py::test_sax_like_pitched_loop_raw_fx_becomes_broad_instrument_loop_not_review \
  tests/test_family_claim_stability_followup_matrix.py::test_stable_pitched_instrument_claim_reviews_siren_false_positive_without_leaf_support -q
# 3 passed

NUMBA_DISABLE_JIT=1 python3 -m pytest tests/test_measured_bass_loop_drum_authority_guard.py -q
# 2 passed

NUMBA_DISABLE_JIT=1 python3 -m pytest tests/test_fx_zip_concrete_fx_bass_steal_guard.py -q
# 4 passed

python3 -m compileall -q src tests tools Aaron_Sound_Sorter.py
python3 tools/audit_no_source_name_sorting.py --project-root .
# PASS
```

Not claimed:

- Full pytest was not run to completion in the sandbox.
- `tests/test_parent_eligibility_v24_drum_loop_steal_guard.py` hung in this sandbox after printing a couple dots, so it is not claimed green here. Aaron should run it locally.

## Next sweep

Continue at:

```text
offset 190
ordinals 191+
```

Current target progress:

```text
794 / 10,254 = 7.7433%
```

The next 250–300 new rows should push the percussion ZIP over 10% coverage.
