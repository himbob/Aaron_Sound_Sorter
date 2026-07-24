# Neural Migration Vocal Reproduction

Initial diagnosis: 2026-07-23
Full pinned-CLAP verification: 2026-07-24 UTC

## Legacy failure

The initial source-blind reproduction used one member of
`Premium Deep House Vocals.zip`. Its display name was retained only to identify
the evidence row:

```text
38.HLMVC Vocal Chop 122bpm Key B minor Dry.wav
```

Legacy result:

```text
Final folder: FX/Human and Voice FX/Altered Voice/Long FX
Final source: strong_consensus
Review: no
Measured parent role: pitched_music_phrase
```

Both old learned memory lanes repeated the Altered Voice label at near-zero
stored-fingerprint distance. This established a corpus-contamination problem,
not a missing vocal-specific rescue rule.

## Clean experiment contract

The entire 39-file pack was held out. Prototypes used 45 approved trainers,
including five broad musical-vocal examples. The evaluation also used 28
protected real-audio negatives. Byte hashes and canonical decoded-audio hashes
were disjoint across preview and held-out roots.

Names and source paths were used only for opening files and report display.
CLAP received decoded audio samples only.

## Real CLAP result

Pinned model:

```text
laion/larger_clap_music_and_speech
revision 195c3a3e68faebb3e2088b9a79e79b43ddbda76b
weights d4e5cf6317c7521ca62c11b524f5646565310e040129fe35719cad661696d745
```

Measured:

```text
held-out vocal recall: 30/39 (76.9%)
loops: 21/29 (72.4%)
one-shots: 9/10 (90.0%)
protected-negative raw voice top-1 false positives: 2/28
protected-negative in-distribution voice false positives: 0/28
vocal OOD/Review: 38/39
legacy Instruments/Voice recall: 21/39 (53.8%)
CLAP-only broad-family wins: 12
legacy-only broad-family wins: 3
```

The representation generalized beyond the five trainers, but the learned
neighborhood did not. Therefore the correct outcome is continued shadow mode,
not a wider radius or a production override.

## Contamination evidence

The focused 74-trainer audit found 26 stronger cross-label conflicts and 6
exact duplicate-byte label conflicts. Every Altered Voice trainer was closer
to another category than to Altered Voice in the CLAP leave-one-out diagnostic.
That does not automatically decide the true label, but it makes silent reuse
unsafe.

## Remaining coverage gaps

The separate disputed-label confuser panel covered sax, synth/pad, textures,
mixed loops, spoken voice, and one non-overlapping altered-voice example. It
had zero in-distribution musical-vocal claims. All eight spoken-voice examples
were predicted musical vocal but were OOD.

No breath/mouth example survived primary-experiment overlap exclusion, and the
current taxonomy has no explicit Formant FX folder. Those cases remain untested
clean truth and need new human-approved material.

See `NEURAL_AUDIO_VERIFICATION_HANDOFF_20260724.md` for the per-file reports,
cleanup plan, commands, and next gates.
