# Archived: HANDOFF v31.173 — Bass-loop vs drum-loop authority cleanup

## Purpose
Fix the `FX_Aaron2.zip` questionable case where a clean synth-bass loop was being stolen by broad Drum Loops logic.

Primary audited file:

```text
FX_Aaron2/Premium West Coast Bounce/Loop/Bass/DH Vice city sixx vibes Loop 3 F minor 87 Bpm BAss synth.wav
```

Correct/current route after patch:

```text
Instruments/Bass/Bass Loops
```

## Important caution
This is not a filename rule. The code does not route based on `Bass` in the file or folder name.

The fix requires measured audio evidence:

- very low event/body ratio
- strong pitched/sustained tonal evidence
- strong bass/synth/sub/low-end panel evidence
- weak drum-loop/rhythmic-break evidence
- weak percussive/drumlike event evidence

If measured drum-loop evidence is strong, the bass-loop guard stands down and Drum Loops remains valid.

## Architecture layer
Most of the real work is below the arbiter:

```text
src/aaron_sound_sorter/engine/claim_producers/final_drum_loop.py
src/aaron_sound_sorter/engine/claim_producers/measured_instrument_branches.py
```

There is still a narrow final invariant in:

```text
src/aaron_sound_sorter/engine/family_claim_arbiter.py
```

That final invariant only accepts Bass Loops when lower-level measured bass-loop proof already exists. It does not identify by filename and it does not steal real drum loops.

## New/protected tests

```text
tests/test_measured_bass_loop_drum_authority_guard.py
```

Now includes both sides:

1. Clean low tonal beat-loop bass does not emit a drum-loop claim.
2. Clean low tonal beat-loop bass emits a bass-loop claim.
3. A true low drum loop with bass energy still emits a drum-loop claim.
4. A true low drum loop blocks the bass-loop claim.

## Probe result

```text
094/130 PASS_INSTRUMENT_SOURCE   FX_Aaron2/Premium West Coast Bounce/Loop/Bass/DH Vice city sixx vibes Loop 3 F minor 87 Bpm BAss synth.wav => Instruments/Bass/Bass Loops
```

Coverage ledger included:

```text
coverage_ledgers/fx_v31173_bass_synth_loop_case.csv
coverage_ledgers/fx_v31173_bass_synth_loop_case.summary.txt
```

## Tests run in sandbox

Passed:

```bash
NUMBA_DISABLE_JIT=1 python3 -m pytest tests/test_measured_bass_loop_drum_authority_guard.py tests/test_measured_percussion_zip_regression_guards.py -q
NUMBA_DISABLE_JIT=1 python3 -m pytest tests/test_claim_arbiter_real_panel_surrogates.py -q
NUMBA_DISABLE_JIT=1 python3 -m pytest \
  tests/test_consensus_concrete_fx_gate_regressions.py::test_stable_pitched_music_phrase_without_instrument_candidate_goes_to_review \
  tests/test_decision_core_fx_smoke_remaining_review_rows.py::test_sax_like_pitched_loop_raw_fx_becomes_broad_instrument_loop_not_review \
  tests/test_family_claim_stability_followup_matrix.py::test_stable_pitched_instrument_claim_reviews_siren_false_positive_without_leaf_support \
  tests/test_fx_zip_concrete_fx_bass_steal_guard.py -q
python3 -m compileall -q src tests tools Aaron_Sound_Sorter.py
python3 tools/audit_no_source_name_sorting.py --project-root .
```

Results:

```text
25 passed
25 passed
7 passed
source-name audit PASS
```

Known sandbox limitation:

```text
NUMBA_DISABLE_JIT=1 python3 -m pytest tests/test_parent_eligibility_v24_drum_loop_steal_guard.py -q
```

printed two dots and then hit the outer sandbox timeout, matching prior local-container behavior. Run it on Aaron's Mac after install.

## Install command

```bash
cd /Volumes/T9/testbed/Aaron_Sound_Sorter && rm -rf /tmp/ass_v31173 && unzip -q ~/Downloads/Aaron_Sound_Sorter_v31173_bass_loop_drum_authority_patch.zip -d /tmp/ass_v31173 && rsync -av /tmp/ass_v31173/Aaron_Sound_Sorter_v31173_bass_loop_drum_authority_patch/ ./
```
