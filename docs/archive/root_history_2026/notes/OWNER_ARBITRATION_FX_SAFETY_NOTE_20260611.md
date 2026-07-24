# Archived: Owner Arbitration FX Safety Patch — 2026-06-11

## Problem

The previous shape-owner FX patch made FX recover better, but it made the FX rescue too broad. Real repeated drum/music loops could be routed to `FX/Hybrid Designed FX` when ShapeVoter found secondary designed-FX pressure.

Confirmed breakage:

- `Full Drum Loop 03 - 78BPM.wav` -> `FX/Hybrid Designed FX`
- `US_JF_Drum_140_Dapocket_FULL.wav` -> `FX/Hybrid Designed FX`
- `MKS_98_Beat1.wav` -> `FX/Hybrid Designed FX`
- `GangstaFunk_Cmaj_96bpm.wav` -> `FX/Hybrid Designed FX`
- source-safe voiced phrase fixture -> `bass_phrase`
- wet sax loop -> `_TO_REVIEW/Measured Role Conflict`

## Design correction

ShapeVoter can be a structural authority, but it must not directly become a broad FX override. This patch adds an owner-arbitration guard:

```text
high-confidence repeated loop/phrase owner
+ strong repetition/span evidence
+ real pitched-loop or drum-groove anchors
+ no clear directional FX motion owner
= block broad FX rescue
```

Clear directional FX is still allowed through:

```text
large spectral slope
+ FX motion support
+ weak/absent pitched musical body
= FX motion owner may beat drum/instrument decoys
```

## Code changes

- `shape_voter.py`
  - Bass phrase now needs a real bass-band owner, so voiced/formant phrases no longer become `bass_phrase` just because they are pitched and sustained.
  - Source-safe voiced/formant phrase evidence can explicitly keep `pitched_phrase_shape` as the primary shape.
  - Designed motion FX loop shape is narrowed so stable pitched/groove loops do not become `designed_motion_fx_loop` unless there is clear directional/unpitched FX morphology.

- `measured_transition_fx.py`
  - Added protected repeated loop/phrase owner gate before broad FX rescue.
  - The gate protects real drum loops, bass/full music loops, and stable pitched loops.
  - Strong down-sweeps/risers still pass when directional FX motion is clear.

- `family_claim_arbiter.py`
  - Added the same protected owner check at the arbiter invariant level so older/final transition paths cannot bypass the producer gate.

- `eligibility.py`
  - Added the same owner concept to parent eligibility so broad `designed_fx_loop_shape` cannot steal real repeated music/drum owners.

## Verified anchors

Still FX:

- `5.FX down sweep 128bpm.wav` -> `FX/Structural and Transitional FX/Drops and Downlifters/Generic Drop or Downlifter/Long FX`
- `scraping_string_drop_long.aif` -> `FX/Structural and Transitional FX/Risers and Builds/Generic Riser/Long FX`
- `MajorLaser.wav` -> `FX/Digital Mechanical Industrial Transport/Glitches and Stutters/Generic Glitch/Long FX`

Protected from FX:

- `Full Drum Loop 03 - 78BPM.wav` passes drum-loop guard
- `US_JF_Drum_140_Dapocket_FULL.wav` passes drum-loop guard
- `MKS_98_Beat1.wav` passes drum-loop guard
- `GangstaFunk_Cmaj_96bpm.wav` no longer goes to FX
- wet sax loop stays Instruments
- source-safe voiced phrase emits `pitched_phrase_shape`

## Test summary

Passed focused checks:

```text
tests/test_designed_fx_shape_claims.py                         11 passed
tests/test_phase4_shape_voter.py                               17 passed
tests/test_permanent_librosa_feature_adapter.py                  6 passed
tests/test_parent_eligibility_fx_instrument_steal_guard.py
 tests/test_fx_zip_concrete_fx_bass_steal_guard.py
 tests/test_consensus_concrete_fx_gate_regressions.py           12 passed
selected pasted failures / anchors                              passed individually
no-source-name audit                                             PASS
```

One long combined parameter run timed out in this container after already printing passing dots. The failing pasted cases were rerun individually instead.
