# v31.174 mixed-instrument + brass/woodwind loop voter patch

## Purpose
Fix two FX_Aaron2 real-audio failures at the lower measured-claim/voter layer:

1. `SC_[[Challenger]]_Melody_Multi_Sample_2_130BPM_Amin.wav`
   - Before: `_TO_REVIEW/Shape Conflict`
   - After: `Instruments/Instrument Loops/Loops`
   - Reason: low/bass body plus strong pitched-loop evidence and strong non-bass tonal panels; raw Bass One Shot leaf was too specific.

2. `RHSH_Brass_Saxophones_01_keyEm_103bpm.wav`
   - Before: `_TO_REVIEW/No Strong Voter Consensus`
   - After: `Instruments/Brass and Woodwinds/Loops`
   - Reason: pitched reed/instrument loop with mixed-instrument branch and credible woodwind/reed evidence.

A third guard case was kept intact:

3. `DH Vice city sixx vibes Loop 3 F minor 87 Bpm BAss synth.wav`
   - Remains: `Instruments/Bass/Bass Loops`
   - Reason: clean bass-loop authority remains valid; the mixed-loop patch does not steal pure bass-loop material.

## Architecture
Most logic is in:

- `src/aaron_sound_sorter/engine/claim_producers/measured_instrument_branches.py`

It adds lower measured claims:

- `mixed_instrument_loop_role_claim`
- measured brass/woodwind branch-loop emission using existing branch-loop source contract

Small arbiter acceptance change:

- `src/aaron_sound_sorter/engine/family_claim_arbiter.py`

The arbiter change only allows a review row to yield when these new lower-level claims already exist and are allowed to compete. It is not filename sorting and not a broad top-level rescue.

## Tests run

```bash
NUMBA_DISABLE_JIT=1 python3 -m pytest \
  tests/test_measured_instrument_loop_voter_claims.py \
  tests/test_measured_bass_loop_drum_authority_guard.py \
  tests/test_claim_arbiter_real_panel_surrogates.py -q
# 35 passed

NUMBA_DISABLE_JIT=1 python3 -m pytest \
  tests/test_consensus_concrete_fx_gate_regressions.py::test_stable_pitched_music_phrase_without_instrument_candidate_goes_to_review \
  tests/test_decision_core_fx_smoke_remaining_review_rows.py::test_sax_like_pitched_loop_raw_fx_becomes_broad_instrument_loop_not_review \
  tests/test_family_claim_stability_followup_matrix.py::test_stable_pitched_instrument_claim_reviews_siren_false_positive_without_leaf_support \
  tests/test_fx_zip_concrete_fx_bass_steal_guard.py \
  tests/test_measured_percussion_zip_regression_guards.py -q
# 26 passed

python3 -m compileall -q src tests tools Aaron_Sound_Sorter.py
python3 tools/audit_no_source_name_sorting.py --project-root .
# PASS
```

## Targeted real-audio verification

Sorted three extracted real files:

```text
DH Vice city sixx vibes Loop 3 F minor 87 Bpm BAss synth.wav
=> Instruments/Bass/Bass Loops

RHSH_Brass_Saxophones_01_keyEm_103bpm.wav
=> Instruments/Brass and Woodwinds/Loops

SC_[[Challenger]]_Melody_Multi_Sample_2_130BPM_Amin.wav
=> Instruments/Instrument Loops/Loops
```

## Source-name policy
No production filename/folder-name evidence was added. Source-name audit passed.
