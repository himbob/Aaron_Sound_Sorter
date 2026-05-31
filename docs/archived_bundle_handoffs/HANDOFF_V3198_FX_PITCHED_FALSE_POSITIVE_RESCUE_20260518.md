# Aaron Sound Sorter v31.98 FX Pitched False Positive Rescue

## Purpose

This patch continues the redesign work after v31.97. It does not restore the old rule tree. It keeps raw voter scores intact and lets the arbiter compare evidence streams.

The full FX run showed a major remaining problem: clearly labeled pitched music loops inside the FX pack, especially sax/reed loops, were still landing in concrete FX leaves such as Short Impact, Siren, Riser, Bird, or Hybrid Designed FX. That was not a small leaf mistake. It meant a concrete FX candidate could still beat strong measured pitched-loop evidence.

## What changed

### FamilyClaimArbiter

Added an architecture-safe escape hatch for false concrete FX leaves:

- If raw winner is FX and looks like a suspicious concrete FX leaf,
- and measured evidence strongly supports pitched music loop/phrase,
- and concrete FX is not confirmed by multiple brain lanes,
- then a broad Instrument Loops claim may compete and win.

This changes false FX leaf outcomes to a safer broad instrument folder instead of pretending a sax loop is a short impact or siren.

The same arbiter now protects true concrete FX. Siren/riser/impact/boom/glitch style FX are not flattened to Instrument Loops when enough lanes confirm the same concrete FX key.

### BabyRecallClaimProducer

Added a conservative clap recall:

- A short hit with baby-lane clap/snap/slap evidence may rescue a false Bird result.
- This fixes the `Clap 17.wav` panel failure without using filenames.

Tightened brass/woodwind baby recall:

- Baby recall now only deep-rescues exact sax/saxophone-style evidence.
- A single trumpet/flute/horn-style baby row is not enough to force Brass and Woodwinds.
- This prevents synth/piano loops from being falsely pushed into Brass and Woodwinds.

## What was tested locally

Focused pytest files were run one at a time:

- `tests/test_v3197_no_role_score_shaping_architecture.py`
- `tests/test_claim_validity_refactor_v3196.py`
- `tests/test_v3184_deep_voter_and_baby_recall.py`
- `tests/test_v3198_fx_pitched_false_positive_and_baby_recall.py`
- `tests/test_claim_arbiter_architecture.py`
- `tests/test_broad_bucket_claim_producer.py`
- `tests/test_no_source_name_sorting_invariant.py`
- `tests/test_brain_competence_policy.py`
- `tests/test_v3187_architecture_doc_and_lane_competence.py`

Also run:

- `python3 -m py_compile` on changed Python files
- `tools/audit_no_source_name_sorting.py`

Focused real-audio checks were run one file at a time where the container allowed it. Important observed results:

- `AA_TSS_D_RUMBLER_808_BASS.wav` -> `Instruments/Bass/808 Bass/One Shots`
- `boom_eval_108640.wav` -> `FX/Impacts and Hits/Boom/One Shots`
- `Clap 17.wav` -> `Drums/Claps Snaps Slaps/Generic Clap/One Shots`
- `Brass_Saxophone_RnB_Multi_Instrument_F_Minor_80BPM.wav` no longer lands in `FX/Impacts and Hits/Short Impact/Long FX`; it lands in broad `Instruments/Instrument Loops/Loops`
- `Riser Short Effect.wav` stays in `FX/Structural and Transitional FX/Risers and Builds/Short Riser/Long FX`
- `MS_O_01_Outlaw_Fx Police_D#minor_91bpm_Wet.wav` stays in `FX/Designed Noise FX/Siren/Long FX`

## Known limitation

This patch intentionally prefers safer broad Instrument Loops over fake concrete FX for some sax/reed failures. It does not claim perfect sax identity yet. Exact sax/reed routing should be improved after this false-FX rescue is validated on Aaron's full FX run.

## Architecture note

This is not filename logic. Filenames and folder paths are only used by diagnostics/test commands to pick known samples and evaluate outcomes. Production decision code remains filename-blind.
