# AI Handoff: Synthetic Instrument Regression Patch 2026-05-14

## Context

Aaron reported three closely related failures:

1. A string-like WAV was placed into Voice or FX even though it should be treated as an instrument.
2. A sax file was placed into Voice, while many other sax files still went to the correct instrument area.
3. Some drum or percussion loop material can still be mistaken for bass or other non-drum loop families.

The code is close, so this patch intentionally avoids broad rewrites and avoids changing voter scores. The change is limited to the parent-eligibility safety seam and synthetic pytest coverage.

## Architecture-safe change made

File changed:

```text
src/aaron_sound_sorter/engine/eligibility.py
```

Added a new broad parent role:

```text
clean_sustained_tonal_instrument_phrase
```

This role catches clean, stable, low-flatness, sustained pitched instrument phrases before the vocal-shape rescue can route them into `FX/Human and Voice FX`.

The role requires all of these broad physics traits:

- sustained tonal behavior
- strong voiced/pitched evidence
- low percussive and drumlike loop ratios
- low spectral flatness
- weak formant-light voice evidence
- not tonal alert/siren FX
- not low-body rhythmic drum loop
- not repeated drum loop

The fallback is deliberately broad:

```text
Instruments/Instrument Loops/Loops
```

It does not claim exact strings, sax, or guitar identity. It only prevents the catastrophic parent error of clean tonal instruments becoming Voice, FX, Drums, animal, machine, or ambience folders.

## Synthetic tests added

File added:

```text
tests/test_parent_eligibility_synthetic_instrument_regressions.py
```

Tests included:

1. `test_clean_sustained_string_like_phrase_blocks_voice_and_fx_bucket`
   - synthetic string-like phrase
   - raw decision is Voice/FX
   - expected final parent is Instruments

2. `test_sax_like_sustained_phrase_stays_reed_branch_not_voice`
   - synthetic sax/reed-like phrase
   - raw decision is Voice/FX
   - expected final parent is Instruments

3. `test_real_vocal_stab_still_goes_to_voice_bucket`
   - synthetic vocal stab
   - protects existing voice behavior from being broken by the new instrument guard

4. `test_clean_tonal_bass_phrase_is_not_forced_to_drum_loop`
   - synthetic clean bass-like tonal loop
   - protects against over-routing tonal bass/instrument phrases into Drum Loops

## Validation run in this sandbox

Passed:

```bash
pytest -q \
  tests/test_parent_eligibility_synthetic_instrument_regressions.py \
  tests/test_parent_eligibility_fact_roles_v27.py \
  tests/test_decision_core_v2.py \
  tests/test_phase4_v060_physical_family_guard_regressions.py \
  tests/test_phase4_voice_guard.py \
  tests/test_phase4_shape_voter.py \
  tests/test_phase4_committee_physics_gap_locks.py \
  tests/test_two_voter_redesign.py
```

Result:

```text
53 passed
```

Also compiled:

```bash
python3 -m py_compile src/aaron_sound_sorter/engine/eligibility.py
```

## Tests that could not be fully validated here

The full pytest suite is not self-contained in this uploaded code-only bundle. Some real-audio fixture folders are missing or empty, causing fixture errors such as:

```text
No audio files found under folder input: tests/regression_audio_drum_loop_steal_guard
Expected audio file, ZIP, or folder input: tests/regression_audio_v28_fx_zip_matrix
missing fixture Piano_G.wav
```

These are packaging/test-fixture problems, not proof that the patch failed.

## Real sample spot check from FX_Aaron2 drum/percussion folders

I selected 3 random files per drum/percussion folder from `FX_Aaron2.zip`, where available. Four completed in this sandbox before long WAV feature extraction started hanging:

```text
2.Drum Loop_1_100bpm.wav => Drums/Drum Loops/Loops
AAJV_Kick_Happiness.wav => Drums/Kick Drums/Generic Kick/One Shots
CS_NE_Kick_OneShot_Monroe.wav => Drums/Kick Drums/Generic Kick/One Shots
Clap2.wav => Drums/Claps Snaps Slaps/Generic Clap/One Shots
```

The sandbox repeatedly hung on longer 24-bit stereo WAV drum loops such as:

```text
EWS_Drumloop_Boombap_HipHop_87BPM.wav
Full Drum Loop 03 - 78BPM.wav
```

Do not interpret that as a category failure. It is a runtime/audio-analysis timeout issue in this environment and should be debugged separately from parent eligibility.

## Next AI should do first

1. Run the synthetic tests above first. They are fast and do not require WAV fixtures.
2. On Aaron's Mac, run the real audio fixture tests that were missing here.
3. Inspect the specific failed strings WAV and sax WAV one at a time with `brain-lab` and the manifest fields:
   - `parent_eligibility_v2`
   - `raw_consensus_before_eligibility_v2`
   - `shape_vote`
   - `measured_roles`
   - `feature_values_by_name`
4. If the sax file still goes to Voice, do not globally weaken voice. Compare it to sax files that already work and identify which measured feature differs.
5. Add a new synthetic fact vector for that exact sax failure pattern before changing code again.
6. Separately investigate long-WAV hangs in the audio reader or feature extractor. Do not mix that with classifier placement logic.

## Synthetic clone idea

Yes, a real sample can be turned into a synthetic regression sound, but not by copying the audio. The safer approach is:

1. Analyze the real sample and save a feature snapshot.
2. Generate a synthetic approximation with matching broad traits:
   - envelope length
   - attack and decay shape
   - rough pitch range
   - harmonic/noise ratio
   - transient count
   - stereo width
   - loop event spacing
3. Use that generated WAV or the extracted synthetic fact vector in pytest.

For this patch, I used fact-vector synthesis because it is faster and more stable than generated audio. A future helper could generate actual synthetic WAVs from a real sample fingerprint for deeper integration testing.
