# AI Handoff: True-Bucket Conflict Resolver Patch

Date: 2026-05-15

## Rule locked in

Every sorter logic fix must follow red/green TDD:

1. Add a synthetic regression test that fails before the fix.
2. Prove it fails.
3. Patch the smallest architecture-safe seam.
4. Prove the same test passes.
5. Run the focused regression panel.

This patch upgrades the prior conflict resolver from "review everything questionable" to "choose a real broad/near bucket when candidate evidence is strong enough."

## Confirmed user ground truth encoded

- Kick/drum loops should land in `Drums/Drum Loops`, not Bass Loops or generic Instrument Loops.
- Distorted kick loops are still kick loops.
- Shroom LANDR Break10_100bpm is a drum loop/break, not Human/Voice FX.
- `epchord02_CminAdd9` is a one-shot keys/Rhodes-like instrument, not Hybrid FX.
- `wt_fd_fx_seaguls` is a crazy FX sound, not Human/Voice Crowd.
- `AANO2_Techno_Melody_Loop_124bpm_Dmin_Cluster Bells` is growing bell FX, acceptable in FX.
- `AADD2_Dancehall_Melody_Loop_96bpm_G#min_Flux` is near-human/flute-like synth loop, not real voice, acceptable as generic Instrument Loop.
- `AASOY_R&B_One_Night_Pluck_Dmin_130bpm` is reverb-heavy keys/pluck loop, acceptable as Instrument Loop.
- `OMT_WestAfricanDounV1_BassControlPercussion_80bpm` is a real percussion loop.
- `088 African Percussion Loop 100.wav` is a metallic percussion loop.

## Code changes

### `src/aaron_sound_sorter/engine/decision_core_v2.py`

Adds a candidate true-bucket rescue seam before review fallback:

- confirmed drum/kick/percussion loop evidence -> `Drums/Drum Loops/Loops`
- strong instrument one-shot candidate evidence beating raw FX -> best instrument candidate path
- non-voice FX candidate beating Human/Voice FX -> best non-voice FX candidate path
- specific tonal/evolving FX candidate beating generic Instrument Loops -> best FX candidate path

The rescue uses only voter candidates and measured eligibility. It does not read source filenames.

### `src/aaron_sound_sorter/engine/eligibility.py`

Adds `pitched_percussion_loop` eligibility for dense, repeated, tonal/metallic percussion loops that were previously mistaken for generic Instrument Loops or FX risers.

Controls prevent slow, clean keys/pluck loops from being pulled into percussion.

## New/updated tests

- `tests/test_decision_core_ground_truth_true_bucket_tdd.py`
- `tests/test_decision_core_ground_truth_samples_tdd.py`
- `tests/test_parent_eligibility_synthetic_bass_alert_overreach_surrogates.py`

The important change is that several confirmed cases now expect real buckets, not review.

## Local validation completed in the sandbox

Focused regression panel:

```text
94 passed
```

Real uploaded spot-check files after patch:

```text
088 African Percussion Loop 100.wav => Drums/Drum Loops/Loops
AADD2_Dancehall_Melody_Loop_96bpm_G#min_Flux.wav => Instruments/Instrument Loops/Loops
AANO2_Techno_Melody_Loop_124bpm_Dmin_Cluster Bells.wav => FX/Structural and Transitional FX/Risers and Builds/Synth Riser/Long FX
AASOY_R&B_One_Night_Pluck_Dmin_130bpm.wav => Instruments/Instrument Loops/Loops
OMT_WestAfricanDounV1_BassControlPercussion_80bpm.wav => Drums/Drum Loops/Loops
```

## Known caveat

The exact FX leaf for growing bell FX may not be perfect yet. It is now in FX, which is much closer than generic Instrument Loops. Do not overfit that leaf without more examples.
