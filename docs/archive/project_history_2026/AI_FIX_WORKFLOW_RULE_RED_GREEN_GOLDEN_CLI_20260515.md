# Archived: AI Fix Workflow Rule — Red-Green Synthetic + Real Golden Audit

Date: 2026-05-15

## Non-negotiable rule

No future sorter-logic change is allowed unless it follows this sequence:

1. Add a failing synthetic regression test for the bug class.
2. Prove that test fails before the fix.
3. Patch the smallest production seam.
4. Prove the same synthetic test passes.
5. Run a real-data golden audit through the actual sorter output manifest.
6. Do not ship if the golden audit fails.

## Why this rule exists

Function-level green tests were not enough. The sorter failed through the full
pipeline:

```text
WAV -> feature extraction -> voters -> consensus -> decision core -> placement -> manifest
```

A unit test around one changed function can pass while the real command still
puts bass loops in risers, piano loops in FX, or curated smoke files in review.

## Golden FX smoke rule

The curated `FX_Aaron2.zip` smoke run is a golden regression set. It must pass
broad expected buckets:

```text
Loop/Bass             -> Instruments/Bass
Loop/Drums            -> Drums/Drum Loops or Drums/Percussion
Loop/Brass_Woodwind   -> Instruments
Loop/Vocals           -> Voice/Human Voice or broad Instrument Loops
One_Shot/Vocals       -> Voice/Human Voice
One_Shot/Fx           -> FX or Review, never Drums/Instruments
```

This is a test oracle only. Runtime sorting must never use filenames/folders as
evidence.

## Source-name blindness remains mandatory

The sorter may never use producer file names, folder names, ZIP member names, or
path tokens as sorting evidence. Names are allowed only in report/audit tools
after a decision has already been made.
