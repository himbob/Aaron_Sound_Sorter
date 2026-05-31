# AI Handoff v31110: Synth Pad vs Sax Overreach

## Purpose

This changed-files-only patch fixes a final-invariant overreach where a sustained synth pad loop was stolen by the measured sax/reed loop invariant.

Regression sample:

```text
CS_NJ2_135bpm_Pad_Aster_Am.wav
```

The test copies it to a neutral filename before sorting, so the fix does not rely on filename text.

## Failure before patch

Neutralized sample result:

```text
sample_0001.wav -> Instruments/Woodwinds/Saxophone/Loops
consensus_status = final_measured_sax_loop_invariant
```

Raw voter evidence was conflicted:

```text
Brain ensemble: Instruments/Guitar/Nylon Guitar/One Shots
Physics voter:  Instruments/Woodwinds/Flute/One Shots
Physics panels: strong synth-pad evidence was present
```

Important measured evidence:

```text
synth_pad_score                 0.784578
synth_tonal_source_score        0.714126
woodwind_sax_score              0.691305
reed_wind_score                 0.619390
shape_vote                      vocal_phrase
shape_confidence                1.0
pitched_event_ratio             1.0
sustained_tonal_frame_ratio     1.0
f0_voiced_ratio                 1.0
low_event_ratio                 0.554302
high_event_ratio                0.002381
spectral_flatness_mean          0.051226
percussive_event_ratio          0.0
drumlike_frame_ratio            0.0
```

## Failure class

```text
final invariant / sax identity overreach
```

The bug was not a filename issue and not a pure missing-candidate issue. The sax invariant had too much authority over clean low-air synth-pad bodies.

## Code changes

Changed file:

```text
src/aaron_sound_sorter/engine/family_claim_arbiter.py
```

Changes:

1. Added `_facts_support_strong_synth_pad_loop()`.
2. Let strong synth-pad loop evidence block sax/reed final invariant overreach.
3. Let strong synth-pad loop evidence block clean-keys loop overreach.
4. Allowed synth-loop protection to run again after broad instrument-loop parent broadening.
5. Routes strong synth-pad loop bodies to:

```text
Instruments/Synths/Pads/Loops
```

## New regression

```text
tests/test_v31110_synth_pad_sax_overreach.py
```

Expected:

```text
CS_NJ2_135bpm_Pad_Aster_Am.wav -> Instruments/Synths/Pads/Loops
```

## Tests run one at a time

Pytest nodes:

```text
PASS tests/test_v31110_synth_pad_sax_overreach.py::test_uploaded_pad_loop_is_not_stolen_by_sax_invariant
PASS tests/test_v31109_strong_drum_consensus_firewall.py::test_uploaded_snare_clap_is_not_released_to_blip_fx
PASS tests/test_v31108_uploaded_audio_routing.py::test_uploaded_synth_bells_loop_is_not_voice
PASS tests/test_v31108_uploaded_audio_routing.py::test_uploaded_belize_loop_is_not_voice
PASS tests/test_v31108_uploaded_audio_routing.py::test_uploaded_av5_hit_escapes_measured_role_conflict_review
PASS tests/test_v31108_non_voice_tonal_loop_guard.py::test_bell_like_tonal_loop_does_not_promote_to_voice_branch
PASS tests/test_no_source_name_sorting_invariant.py::test_production_sorting_code_has_no_source_name_evidence_markers
PASS tests/test_physics_voter_reed_sax_identity.py -k clean_synth_pad_loop_decoy
PASS tests/test_physics_voter_reed_sax_identity.py -k can_rank_sax_identity_before_generic_loop
```

Locked smoke acceptance cases run one at a time using `RUN_LOCKED_SMOKE_ACCEPTANCE_AI_SAFE.command --case-id ...`:

```text
PASS synth_pad_e_pad_097 -> Instruments/Synths/Pads/Loops
PASS synth_bells_no_safety -> Instruments/Instrument Loops/Loops
PASS sax_loop_aajbl_11 -> Instruments/Woodwinds/Saxophone/Loops
PASS wet_sax_jazzhiphop_15 -> Instruments/Woodwinds/Saxophone/Loops
PASS electric_piano_not_sax_ws2_101_fm -> Instruments/Keys/Electric Piano/Loops
PASS synth_loop_05_emn -> Instruments/Synths/Synth Loops
```

## Install note

This is a changed-files-only patch. It does not replace the whole project.
