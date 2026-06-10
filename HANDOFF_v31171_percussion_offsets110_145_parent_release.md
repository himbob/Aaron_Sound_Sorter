# HANDOFF v31.171 — Percussion offsets 110–145 parent-release patch

## Base

Built on top of v31.170 percussion low-level parent-claim patch.

No brain rebuild. No JSON ledger mutation. No production source-name sorting.

## What was tested

Source ZIP:

```text
one_shot_percussive_sounds.zip
```

New-only range tested after the prior trusted range:

```text
offsets: 110, 115, 120, 125, 130, 135, 140, 145
per-folder count: 5
source folders: 5
ordinals covered per folder: 111–150
new rows screened: 200
```

Bulk sweep was run with third-party features disabled to avoid the known sandbox librosa/numba hang. Every non-broken bad/borderline row was rerun individually with third-party features enabled and bounded before diagnosis or patching.

## Coverage summary

```text
previous_trusted_screened_before_v31171: 394
new_screened_offsets110_145: 200
new_drums_after_fixes: 199
new_fx_after_fixes: 0
new_instruments_after_fixes: 0
new_review_after_fixes: 1
new_broken_or_tiny_review_accepted: 1
new_bad_after_fixes_excluding_broken_tiny: 0
cumulative_screened: 594
zip_total_audio: 10254
cumulative_screened_percent: 5.7929
```

The single remaining review is accepted by policy because it is `_TO_REVIEW/Broken Or Tiny`:

```text
one_shot_percussive_sounds/2/104845.wav
```

## Real failures fixed with third-party-enabled reruns

```text
one_shot_percussive_sounds/1/14491.wav
before: Instruments/Instrument Loops/Loops
reason: compact tonal struck percussion treated as short pitched instrument
final:  Drums/Percussion/Bells and Metallic Percussion/One Shots
source: final_protected_percussive_parent_release

one_shot_percussive_sounds/2/105434.wav
before: Instruments/Instrument Loops/Loops, then Review during narrowing
reason: short low percussive kick/sub hit stolen by instrument/bass-loop logic
final:  Drums/Kick Drums/Generic Kick/One Shots
source: final_parent_low_kick_like_release

one_shot_percussive_sounds/4/205973.wav
before: Instruments/Woodwinds/Flute/One Shots
reason: parent eligibility had protected percussion, but clean tonal/woodwind release overrode it
final:  Drums/Rims and Sticks/Generic Rim or Stick/One Shots
source: final_protected_percussive_parent_release

one_shot_percussive_sounds/4/207170.wav
before: Instruments/Instrument Loops/Loops
reason: raw Drums/protected parent overwritten by broad Instrument Loops
final:  Drums/Rims and Sticks/Rimshot/One Shots
source: final_weak_review_measured_drums_release

one_shot_percussive_sounds/1/14820.wav
before: _TO_REVIEW/Measured Role Conflict
reason: very short voice-like struck material blocked despite strong compact/hand/wood evidence
final:  Drums/Toms/Generic Tom/One Shots
source: final_protected_percussive_parent_release

one_shot_percussive_sounds/5/439816.wav
before: _TO_REVIEW/Measured Role Conflict
reason: tiny high cymbal/metallic texture-bed hit blocked by texture/voice-like shape
final:  Drums/Percussion/Guiros Scrapes and Rasps/One Shots
source: final_protected_percussive_parent_release

one_shot_percussive_sounds/1/15312.wav
before: _TO_REVIEW/Measured Role Conflict
reason: short hit-with-tail low/mid struck event had protected parent but thresholds were too strict
final:  Drums/Percussion/Generic Percussion/One Shots
source: final_protected_percussive_parent_release

one_shot_percussive_sounds/1/15314.wav
before: Instruments/Keys/Rhodes/One Shots
reason: low pitched repeated struck drum hit looked like Rhodes
final:  Drums/Toms/Generic Tom/One Shots
source: final_protected_percussive_parent_release

one_shot_percussive_sounds/5/439825.wav
before: _TO_REVIEW/Measured Role Conflict
reason: protected short tonal percussion parent blocked by role conflict
final:  Drums/Toms/Generic Tom/One Shots
source: final_protected_percussive_parent_release
```

## Architecture changes

### `src/aaron_sound_sorter/engine/eligibility.py`

Added lower-level measured parent eligibility for:

- compact struck tonal percussion hits
- short low percussive one-shots / kick-like hits

These block unsafe Instrument/Bass/Instrument Loop steals before winner selection.

### `src/aaron_sound_sorter/engine/claim_producers/measured_drum_structures.py`

Extended measured kick one-shot claim production so the new low-kick-like parent emits a real measured Drum claim.

### `src/aaron_sound_sorter/engine/placement_depth.py`

Added measured short-low-percussive guard to prevent broad Instrument placement when the lower-level facts support a short low drum parent.

### `src/aaron_sound_sorter/engine/family_claim_arbiter.py`

Kept changes narrow and supportive:

- accepts the new low-kick-like parent release
- prevents Instrument claims from competing when parent eligibility already blocks Instruments for a protected drum hit
- makes clean-tonal/FX-broad-instrument release stand down when protected percussion parent evidence exists
- broadens protected parent release only for measured short/repeated/material-supported percussion cases

This is not a new filename/source rescue layer.

### `tests/test_measured_percussion_zip_regression_guards.py`

Added regression coverage for the new patterns. This file now contains 16 tests and passed as a focused set.

## Tests run

Passed:

```bash
NUMBA_DISABLE_JIT=1 python3 -m pytest tests/test_measured_percussion_zip_regression_guards.py -q
# 16 passed

NUMBA_DISABLE_JIT=1 python3 -m pytest tests/test_claim_arbiter_real_panel_surrogates.py -q
# 25 passed

NUMBA_DISABLE_JIT=1 python3 -m pytest \
  tests/test_consensus_concrete_fx_gate_regressions.py::test_stable_pitched_music_phrase_without_instrument_candidate_goes_to_review \
  tests/test_decision_core_fx_smoke_remaining_review_rows.py::test_sax_like_pitched_loop_raw_fx_becomes_broad_instrument_loop_not_review \
  tests/test_family_claim_stability_followup_matrix.py::test_stable_pitched_instrument_claim_reviews_siren_false_positive_without_leaf_support -q
# 3 passed

NUMBA_DISABLE_JIT=1 python3 -m pytest tests/test_measured_bass_loop_drum_authority_guard.py tests/test_fx_zip_concrete_fx_bass_steal_guard.py -q
# 6 passed

python3 -m compileall -q src tests tools Aaron_Sound_Sorter.py

python3 tools/audit_no_source_name_sorting.py --project-root .
# PASS: production sorting/voting/decision code contains no banned source-name evidence markers.
```

Not claimed:

- Full pytest was not run to completion in the sandbox.
- A larger parent-eligibility command hit the outer sandbox timeout after starting, so do not claim it as passed.

## Next recommended sweep

Continue new-only percussion testing from:

```text
offset 150
ordinals 151+
```

Use the same method:

1. Bulk screen small chunks.
2. Do not count timed-out/unwritten rows.
3. Rerun every non-broken failure with third-party features enabled and bounded.
4. Fix shape/voter/claim/eligibility first; arbiter last and only to honor lower-level facts.

## Install command

```bash
cd /Volumes/T9/testbed/Aaron_Sound_Sorter && rm -rf /tmp/ass_v31171 && unzip -q ~/Downloads/Aaron_Sound_Sorter_v31171_percussion_offsets110_145_parent_release_patch.zip -d /tmp/ass_v31171 && rsync -av /tmp/ass_v31171/Aaron_Sound_Sorter_v31171_percussion_offsets110_145_parent_release_patch/ ./
```
