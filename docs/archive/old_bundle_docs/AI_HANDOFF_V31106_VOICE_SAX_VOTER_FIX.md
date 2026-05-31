# AI Handoff: v31.106 Voice/Sax Voter Fix

## Purpose
Fix the uploaded vocal regression where a real vocal phrase fell to `_TO_REVIEW/Measured Role Conflict` instead of staying in `Instruments/Voice`.

This is an architecture-level voter/arbiter repair. It does not use filenames, folder names, ZIP member names, or sample-pack labels as classifier evidence.

## Files changed

- `src/aaron_sound_sorter/voters/physics_instrument_layer.py`
- `src/aaron_sound_sorter/engine/family_claim_arbiter.py`
- `src/aaron_sound_sorter/engine/claim_producers/profile_candidate_instruments.py`
- `tests/test_v31106_voice_voter_invariant.py`
- `commands/quality/RUN_V31106_FX_VOICE_SAX_SELECTOR_AUDIT.command`

## Root cause

The failing vocal sample already had human/voice evidence in the raw candidates, and the subpanels measured a strong human/spoken voice texture. The problem was that `PhysicsInstrumentLayer` required the broad `vocal_role` score before it would promote the `Voice` branch. For this phrase, `vocal_role` was zero, so the system fell through to broad instrument-loop arbitration and then review.

## Fix summary

### 1. PhysicsInstrumentLayer voice texture signal

Added a measured `human_voice_texture` signal from:

- `voice_score`
- `human_spoken_voice_score`
- `human_breath_mouth_score`
- `fx_formant_score`

The branch voter can now promote `Voice` when the measured human/spoken/formant evidence is strong, even if the old broad role signal is missing.

### 2. Final arbiter voice invariant

The final voice invariant now recognizes the new `instrument_human_voice_phrase_signal` and does not incorrectly treat this case as a measured role conflict.

### 3. Sax protection repair

The sax path was also tightened while testing the FX pack:

- sax candidates from `physics_vote_result` and `dry_core_physics_vote_result` are now visible to the profile sax leaf probe
- close PhysicsVoter sax candidates can protect sax loops from being flattened to broad Instrument Loops
- strong dark/low-mid sax subpanel authority can protect low sax bodies from clean-tone FX/blip routing
- sax false-positive decoys are guarded so strong measured reed/sax evidence is not blocked by broad synth/key/bass similarity too early

## Direct checks run in the container

### Passing direct sample checks

- `Vocal Phrase We Up 140bpm.wav` -> `Instruments/Voice/Phrase/One Shots`
- `AMV_VRNB1_121_brass_saxophone_loop_shot_Gm.wav` -> `Instruments/Woodwinds/Saxophone/Loops`
- `AMV_VRNB1_154_brass_saxophone_loop_deepthoughts_D#m.wav` -> `Instruments/Woodwinds/Saxophone/Loops`
- `GrimyHipHop_Saxophone_26_Fm_Melody_Dark_Dusty_Warm_Loop_84bpm.wav` -> `Instruments/Woodwinds/Saxophone/Loops`
- `Lady_Saxophone_Riff_Emin_160bpm.wav` -> `Instruments/Woodwinds/Saxophone/Loops`
- `DOJO_FBP_Female_Vocal_Shout.wav` -> `Instruments/Voice/Phrase/One Shots` in direct CLI final placement

### Honest exceptions seen during filename-selected FX checks

The selector found files by filename only. The classifier was not given filename evidence.

- `Halfway_Vocal_keyC#m_82BPM.wav` still classifies as Sax by measured audio. Physics ranks sax first and the instrument layer chooses Woodwinds. Forcing this to Voice would be a filename-based override.
- `Chopart3_GodlikeLoops_90_Piano_Sax_Songstarter_Abm_Loop_Fine.wav` stays broad `Instruments/Instrument Loops/Loops`. It appears mixed/piano/sax by filename, but measured evidence was not strong enough for a sax leaf without source-name cheating.

## Tests run

```bash
PYTHONPATH=src:. python3 -m pytest -q \
  tests/test_uploaded_regression_audio.py \
  tests/test_hierarchical_abstaining_arbitration_v31_99.py \
  tests/test_v31106_voice_voter_invariant.py \
  tests/test_physics_voter_reed_sax_identity.py \
  tests/test_no_source_name_sorting_invariant.py
```

Result:

```text
sssssssssss....................................... [100%]
```

```bash
python3 -m py_compile Aaron_Sound_Sorter.py $(find src -name '*.py' -type f | sort) $(find tests -name '*.py' -type f | sort)
```

Result: passed.

```bash
./commands/quality/RUN_NO_SOURCE_NAME_SORTING_AUDIT.command
```

Result:

```text
PASS: production sorting/voting/decision code contains no banned source-name evidence markers.
Scanned Python files: 91
```

## Local full FX voice/sax audit command

Run this after installing the bundle:

```bash
cd /Volumes/T9/testbed/Aaron_Sound_Sorter
./commands/quality/RUN_V31106_FX_VOICE_SAX_SELECTOR_AUDIT.command
```

Optional override:

```bash
FX_ZIP="/path/to/FX_Aaron2.zip" ./commands/quality/RUN_V31106_FX_VOICE_SAX_SELECTOR_AUDIT.command
```

The command selects vocal/vox/sax files by filename only for testing, then runs the real sorter on each extracted file one at a time. It writes its output under `_reports/` and opens the report folder on macOS.
