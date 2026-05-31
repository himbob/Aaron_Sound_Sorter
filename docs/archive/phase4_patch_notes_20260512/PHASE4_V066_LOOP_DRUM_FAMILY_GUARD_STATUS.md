# Phase 4 v0.6.6 Loop Drum Family Guard Patch

## Purpose

This patch addresses two failures found in the latest review logs:

1. Non-percussive rhythmic material could survive as a Drums loop because repeated events alone were treated as enough drum evidence.
2. Mixed or top-only drum loops could land in an over-specific loop leaf such as claps, shakers, or ambiguous clap/percussion loops instead of the broad learned Drum Loops folder.
3. Sub-heavy kick-pulse loops could be laundered into Bass or generic Instrument Loops because they are tonal, low-end-heavy, and sparse in the high end.

The change uses measured physics only. It does not read filenames or source folder names for routing.

## Changed files

```text
src/aaron_sound_sorter/committee.py
tests/test_phase4_v066_loop_family_policy_regressions.py
```

## Main logic changes

### Unsupported drum-loop physics guard

A Drums loop candidate now needs distributed repeated events plus drum-like body/top/noise evidence. Repeated syllables, tonal instrument events, or FX texture motion alone are not enough to keep a file in Drums.

If the candidate lacks drum-loop physics, the physical family guard switches to an already-nominated non-Drums candidate that passes membership, or sends it to review.

### Broad learned Drum Loops placement for risky drum-loop leaf contests

If the file is already Drums, measured as a loop, and the exact drum sub-leaf contest is risky or ambiguous, the sorter collapses the placement to the broad learned Drums / Drum Loops / Loops label instead of burying it under clap, shaker, snare, or other over-specific loop leaves.

### Sub-heavy kick-pulse loop repair

The drum-loop physics helper now recognizes sparse low-pulse loop evidence: distributed repeated events, dominant sub/bass energy, low high-end energy, fast attack, and loop-scale event rate. If the raw broad family already says Drums but later ranking tries to place that material as Bass or Instrument Loops, the guard switches to the learned broad Drum Loops label.

## Regression tests added

```text
tests/test_phase4_v066_loop_family_policy_regressions.py
```

Covers:

- High-bright hat/top drum loop collapses from ambiguous clap/percussion leaf to Drums / Drum Loops / Loops.
- Non-percussive rhythmic FX-like loop cannot remain a drum loop.
- Mixed drum beat physics remains a drum loop.
- Voiced rap/vocal loop still reviews instead of becoming a drum loop.
- Sub-heavy kick-pulse loop raw-ranked as Drums does not become a Bass loop.
- A similar sub-heavy loop stays Instruments when the raw broad family did not say Drums.

## Tests run in container

```text
python3 -m pytest -q tests/test_phase4_v060_physical_family_guard_regressions.py tests/test_phase4_dynamic_structure_gate.py tests/test_phase4_v066_loop_family_policy_regressions.py
14 passed

python3 -m pytest -q tests/test_phase4_synthetic_quality_matrix.py
21 passed
```

Additional longer pytest groups were started, but one command timed out in the limited container before failure output. No assertion failure was seen before timeout.

## Real-sample spot checks run with uploaded latest brain

Using `stage4_folder_brain.json` from the uploaded brain ZIP:

```text
MKS_98_Beat1.wav -> Drums / Drum Loops / Loops, auto_place
Drumloop_hats_Dark_Rap_140BPM.wav -> Drums / Drum Loops / Loops, auto_place
Vocal Phrase We Up 140bpm.wav -> _TO_REVIEW / Conflicting Evidence / Voiced Formant Non Drum
CD4_Forbidden Romance_saxophone & keys_Part_1_Gm_78.wav -> Instruments / _Ambiguous Leaf / Brass vs Woodwinds / Loops
AMV_VRNB1_122_brass_saxophone_loop_forus_Fm.wav -> Instruments / _Ambiguous Leaf / Synths vs Woodwinds / Loops
RHSH_Saxophone_Ensemble_05_keyC_89bpm.wav -> Instruments / _Ambiguous Leaf / Guitar vs Strings Bowed / Loops
22784.wav -> _TO_REVIEW due severe coherent group outlier
KickLoop_150bpm.wav -> Drums / Drum Loops / Loops, auto_place
Kick_85bpm.wav -> Drums / Drum Loops / Loops, auto_place
```

## Remaining known limitation

Sax/brass examples are now kept out of FX, but the exact saxophone leaf is still not reliable enough with the current brain. That needs stronger clean sax/brass/woodwind loop training or better dry/core identity features in a later pass.
