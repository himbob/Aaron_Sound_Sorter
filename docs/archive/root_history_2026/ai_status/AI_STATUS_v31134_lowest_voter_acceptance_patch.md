# Archived: v31.134 Lowest Voter Acceptance Patch

## Purpose

This patch addresses the current acceptance and pytest failures at the lowest practical voting layers rather than by filename rescue or broad category hiding.

## Changed files

- `src/aaron_sound_sorter/voters/shape_voter.py`
- `src/aaron_sound_sorter/voters/physics_top_family_layer.py`
- `src/aaron_sound_sorter/voters/physics_instrument_layer.py`
- `tests/acceptance/locked_smoke_v1/samples/HipHopTapes_28_Saxophone_D#m_90bpm.wav`

## Fix intent

1. **Missing protected sax sample**
   - Restores `HipHopTapes_28_Saxophone_D#m_90bpm.wav` into the locked smoke acceptance sample folder.
   - This fixes the anchor/stability failure where the protected sample ID existed but the real WAV was missing.

2. **ShapeVoter drum-loop overreach**
   - Tightens the `rhythmic_break_loop` promotion so ambiguous musical phrases do not become `beat_loop` unless there is real drum-body evidence.
   - Keeps repeated pitched/tonal phrases in `repeated_phrase_loop` instead of forcing them into Drums.

3. **Bass and pitched musical loop top-family protection**
   - Adds measured-parent top-family guards so non-percussive pitched/bass loops stay in `Instruments` when their measured role and branch evidence agree.
   - Prevents bass and mixed pitched loops from being stolen by `Drums/Drum Loops` when the loop body is tonal rather than percussive.

4. **Voice-loop protection from drum-loop theft**
   - Prevents measured voice loops with stronger Instrument evidence from being treated as drum loops merely because they have loop structure.

5. **Keys/electric-piano branch protection**
   - Strengthens the electric-keys source path when the subpanel confidently sees Electric Piano/Rhodes-like struck-key behavior.
   - Goal: keep clean keys/electric piano material from flattening to broad `Instrument Loops` or being stolen by sax/guitar.

6. **Mallet/bell subpanel correction**
   - Gives metallic pitched hits enough local authority to report as `MalletBell` rather than collapsing to `MixedInstrument` when the signal is a clear pitched metallic hit.

## What this deliberately does not do

- Does not use filenames, folder names, sample-pack words, or source path text as classification evidence.
- Does not disable sax, voice, drums, FX, synth, keys, or any full category library branch.
- Does not edit the brains.
- Does not rewrite the acceptance policy.

## Quick validation command

After installing, run:

```bash
./commands/quality/RUN_V31134_TARGETED_REPROS_ONE_BY_ONE.command
```

That command runs only the current failure nodes one at a time, matching the project rule for timeout-safe testing.
