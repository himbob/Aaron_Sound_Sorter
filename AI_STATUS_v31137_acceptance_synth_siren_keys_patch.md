# AI Status v31.137 acceptance synth/siren/keys patch

## Problem class

Resolver/arbitration failure, not install failure.

The late arbitration layer was allowing broad measured invariants to overrule better source identity boundaries:

1. Low tonal kick loops could be measured too much like bass because high autocorrelation pitch confidence was treated as a sustained voiced bass line.
2. Clean pitched-loop restoration could demote a real siren/alarm FX case into generic Instruments.
3. Synth Lead restoration was too broad and could steal strings, pads, and keys/electric piano.
4. Acoustic piano needed to route to Piano, while Rhodes/keys needed to stay Keys and not become Synth Lead.

## Files changed

- `src/aaron_sound_sorter/domain/roles.py`
- `src/aaron_sound_sorter/engine/family_claim_arbiter.py`

## Architecture-safe fix summary

- Added a voiced-tonal-line gate to the low-pulse drum-loop role calculation so repeated tonal kick loops are not mistaken for sustained bass lines.
- Added a final measured tonal-alert/siren invariant, but tightened it so secondary siren-shape evidence alone cannot steal wet sax or other pitched instruments into FX.
- Centralized synth-loop depth selection so Pads, Synth Lead, and generic Synth Loops are chosen by measured panel/shape evidence instead of a blunt `pitched_phrase == Synth Lead` rule.
- Tightened Synth Lead restoration so weak synth-lead panels cannot steal strings, pads, or electric keys.
- Added a measured keys-body guard so Rhodes/keys loops stay under Keys when the physics voter and keys panels support keys and synth-lead evidence is not decisive.
- Preserved acoustic piano routing for the protected obvious piano case.

## Tests run in this container

### Direct pasted failures

All passed:

```bash
PYTHONPATH=src python3 -m pytest -q tests/test_direct_body_voice_and_low_drum_loop_policy.py::test_low_tonal_kick_loop_measures_as_drum_loop_not_bass_loop
PYTHONPATH=src python3 -m pytest -q tests/test_hierarchical_abstaining_arbitration_v31_99.py::test_bass_phrase_with_internal_synth_candidate_refines_to_synth_loop
PYTHONPATH=src python3 -m pytest -q tests/test_hierarchical_abstaining_arbitration_v31_99.py::test_measured_synth_lead_specialist_can_beat_fx_siren_false_positive
PYTHONPATH=src python3 -m pytest -q tests/test_hierarchical_abstaining_arbitration_v31_99.py::test_clean_synth_loop_with_incidental_sax_candidate_routes_synth_not_piano_or_sax
PYTHONPATH=src python3 -m pytest -q tests/test_parent_eligibility_v28_fx_zip_matrix.py::test_fx_zip_tonal_police_fx_stays_fx_not_generic_instrument
```

### Related pytest files

Passed:

```bash
PYTHONPATH=src python3 -m pytest -q \
  tests/test_direct_body_voice_and_low_drum_loop_policy.py \
  tests/test_hierarchical_abstaining_arbitration_v31_99.py \
  tests/test_v31110_synth_pad_sax_overreach.py \
  tests/test_v3198_fx_pitched_false_positive_and_baby_recall.py \
  tests/test_physics_voter_piano_struck_identity.py \
  tests/test_v31116_synth_pad_keys_decoy_guard.py \
  tests/test_v31101_instrument_subpanels.py \
  tests/test_v31108_non_voice_tonal_loop_guard.py \
  tests/test_v31118_calibrated_panel_authority_firewall.py \
  tests/test_v31119_panel_source_specific_authority_split.py \
  tests/test_consensus_concrete_fx_gate_regressions.py \
  tests/test_claim_arbiter_architecture.py \
  tests/test_claim_arbiter_real_panel_surrogates.py
```

### No-source-name audit

Passed:

```bash
PROJECT_ROOT=$PWD PYTHON_BIN=$(command -v python3) ./commands/quality/RUN_NO_SOURCE_NAME_SORTING_AUDIT.command
```

Result:

```text
PASS: production sorting/voting/decision code contains no banned source-name evidence markers.
Scanned Python files: 94
```

### Locked smoke acceptance

The direct full acceptance command timed out in this container before finishing as one monolithic run. The single-case harness was used instead, which avoids the container/report-opening problem and lets failures be isolated.

The original failed acceptance cases now pass:

```text
piano_loop_key_c -> Instruments/Keys/Piano/Loops
synth_lead_k_dot_fluteish -> Instruments/Synths/Synth Lead/Loops
police_fx_siren_not_piano_or_sax -> FX/Designed Noise FX/Siren/Long FX
```

Additional regression cases found during the fix were also corrected and rerun:

```text
strings_loop_77_ebm -> Instruments/Instrument Loops/Loops
wet_sax_hiphoptapes_no_voice -> Instruments/Woodwinds/Saxophone/Loops
electric_keys_ews_not_sax_or_guitar -> Instruments/Keys/Electric Piano/Loops
synth_loop_05_emn -> Instruments/Synths/Synth Loops
piano_loop_dhb_vintage -> Instruments/Keys/Electric Piano/Loops
electric_piano_not_sax_ws2_101_fm -> Instruments/Keys/Piano/Loops
```

## Not claimed

I did not get a clean one-command full `pytest` completion in this container. It timed out partway through the full suite. The targeted and related tests above passed, and the protected acceptance regressions were run as isolated cases.
