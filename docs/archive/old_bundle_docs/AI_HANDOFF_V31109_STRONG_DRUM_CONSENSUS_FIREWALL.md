# AI Handoff v31109: Strong Drum Consensus Firewall

## Purpose

Fix false FX/Blip release for short snare/clap drum hits that measure as clean, pitched, and tonal.

Regression sample:

- `ES_TEDR2_Claps&Snares_76.wav`

The file was neutralized to `sample_0001.wav` before sorting, so the test does not use filename evidence.

## Failure

Before the patch:

- Final label: `FX/Designed Noise FX/Blip/One Shots`
- Final status: `final_clean_tonal_non_drum_hit_fx_invariant`

The committee evidence was drum-family:

- Brain ensemble: `Drums/Snares/Acoustic Snare/One Shots`
- Full brain: `Drums/Snares/Acoustic Snare/One Shots`
- Core baby: `Drums/Snares/Acoustic Snare/One Shots`
- Spread baby: `Drums/Snares/Acoustic Snare/One Shots`
- Outlier baby: `Drums/Hi Hats/Closed Hat/One Shots`
- Physics: `Drums/Hi Hats/Closed Hat/One Shots`
- Shape: `single_hit`

The bug was not that voters rejected Drums. The late clean-tonal FX invariant overruled the drum-family committee.

## Code change

Changed:

- `src/aaron_sound_sorter/engine/family_claim_arbiter.py`

Added:

- `_strong_drum_one_shot_committee_blocks_clean_tonal_fx_release(...)`

The guard blocks `_release_clean_tonal_non_drum_hit_to_fx(...)` when the winning claim is a strong drum-family one-shot committee result. It reads only voter and measured-fact evidence. It does not inspect filenames or source paths.

## Test change

Added:

- `tests/test_v31109_strong_drum_consensus_firewall.py`
- `tests/regression_audio/ES_TEDR2_Claps&Snares_76.wav`

The test neutralizes the filename before sorting and verifies the sample does not release to FX/Blip.

## Test results run one at a time

Passed:

```bash
python3 -m py_compile src/aaron_sound_sorter/engine/family_claim_arbiter.py
python3 -m pytest -q tests/test_v31109_strong_drum_consensus_firewall.py::test_uploaded_snare_clap_is_not_released_to_blip_fx
python3 -m pytest -q tests/test_v31108_uploaded_audio_routing.py::test_uploaded_av5_hit_escapes_measured_role_conflict_review
python3 -m pytest -q tests/test_v31108_uploaded_audio_routing.py::test_uploaded_synth_bells_loop_is_not_voice
python3 -m pytest -q tests/test_v31108_uploaded_audio_routing.py::test_uploaded_belize_loop_is_not_voice
python3 -m pytest -q tests/test_v31108_non_voice_tonal_loop_guard.py::test_bell_like_tonal_loop_does_not_promote_to_voice_branch
python3 -m pytest -q tests/test_no_source_name_sorting_invariant.py
```

Selected locked-smoke acceptance cases run individually with `RUN_LOCKED_SMOKE_ACCEPTANCE_AI_SAFE.command --case-id ...`:

```text
clap_clean_clap2 PASS
snare_clean_jackbaby PASS
synth_bells_no_safety PASS
vocal_money_female_rap PASS
vocal_or_brass_hit_av5 PASS
```

Note: the all-case one-by-one wrapper started with `kick_clean_cs_ne_monroe`, which passed internally, but the wrapper process did not return cleanly in the container before the tool timeout. The direct one-case command works and was used for the selected cases above.

## Known caveat

The uploaded snare/clap sample now stays in Drums but lands under `Drums/Hi Hats/Closed Hat/One Shots` because PhysicsVoter still prefers hat. This v31109 patch fixes the cross-family Blip theft. It does not yet refine snare-vs-hat leaf selection.
