# AI Handoff: Parent Eligibility V2.8 FX Zip Matrix

## State of the work

This bundle is the end of the current crisis stabilization thread. It is not a perfect classifier. It is a safer parent-family layer with hard regression tests for the WAVs and FX zip examples that kept breaking.

The design goal remains the v31 goal: obvious source families should be correct, uncertain sounds should go to review, and catastrophic wrong folders should become rare. Do not return to a giant first-match rule tree. Do not use filenames for final routing. The filename clues in the audit are only for selecting human review fixtures.

## What V2.8 adds

1. A forced full-thread WAV regression matrix.
2. A forced FX_Aaron2 selected-file regression matrix.
3. A one-by-one FX_Aaron2 audit command.
4. Handoff notes and status docs so the next AI does not restart the architecture debate.

## Critical protected classes

### Sax and reed loops

Must not go to:

- Drums
- Drum Loops
- FX Human Voice
- animals
- water/ocean/coins/machine

Safe current destination:

- Instruments / Brass and Woodwinds / Loops
- Instruments / Instrument Loops / Loops if branch certainty is weak
- _TO_REVIEW if source evidence conflicts

### Vocal one-shots and vocal phrases

Must not go to:

- Drums
- Percussion
- Machines
- Motor
- animals
- coins

Safe current destination:

- FX / Human and Voice FX
- Instruments / Voice
- _TO_REVIEW if ambiguous

### Drum loops

Must not go to:

- Instruments / Instrument Loops
- Bass Loops
- FX risers or downlifters
- Human Voice

Safe current destination:

- Drums / Drum Loops / Loops

### Synth, keys, electric piano, piano

Must not be overcorrected into:

- Brass and Woodwinds
- Drums
- Human Voice
- generic FX alarm unless there is strong FX evidence

Safe current destination:

- Instruments / Instrument Loops / Loops
- Instruments / Synths or Keys when available
- _TO_REVIEW if ambiguous

## Remaining known weaknesses

1. Sax leaf recovery is still broad. It usually lands in Brass and Woodwinds Loops, not exact Sax.
2. Some wet synth leads can still be hard because their modulation looks like FX or bass.
3. Some vocal-layer loops are ambiguous between Human Voice and generic Instrument Loops.
4. The V2.8 selected FX zip matrix can be slow because it sorts a directory of real WAVs through the actual command path.

## Do not do next

- Do not add a new voter.
- Do not hide categories.
- Do not hardcode filenames into sorter logic.
- Do not make role gates secretly rewrite voter scores.
- Do not make another tiny patch without adding the failing WAV to pytest first.

## Best next step

Run the QA commands on Aaron's Mac. If a file fails, first add it to the forced matrix, then fix the generalized parent-family logic.

