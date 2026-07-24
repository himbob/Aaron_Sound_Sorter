# Aaron Sound Sorter — Current Status

Updated: 2026-07-24 PDT

## Current authority boundary

The production sorter still uses the legacy committee and `FamilyClaimArbiter`.
An exact dual-memory GUI correction can now own the precise Drums, Instruments,
or FX label the human selected. Frozen CLAP/MERT prototypes remain shadow-only
and have zero placement authority. No new vocal, sax, instrument, drum, or FX
rescue branch was added to the arbiter.

## GUI correction failure fixed

The GUI had learned all six vocal corrections, but the learned-owner producer
rewrote supervised `FX/Human and Voice FX/...` labels into
`Instruments/Voice/...`. Human-trained raw protection then blocked that
cross-family rewrite and older role/shape logic returned Review.

The owner now preserves the exact supervised taxonomy label for every supported
top family. Voter/physics memory also removes an older same-audio label even
when both labels share one broad role/physics bucket. Sparse GUI corrections
now update only user, voter, physics, and shape memory brains; full/core/spread/
outlier brains are rebuild-only.

End-to-end evidence on the 39-file pack:

- 6/6 explicit GUI corrections landed in the exact selected folder;
- 6/6 were owned by `exact_human_teacher_owner_claim`;
- Review fell from 3 to 0;
- 33/39 landed in a voice-related Instruments or Human Voice FX folder.

The remaining six non-voice-family placements are not silently declared wrong;
they require explicit listening verdicts.

## Verified neural infrastructure

- Pinned local CLAP revision:
  `laion/larger_clap_music_and_speech@195c3a3e68faebb3e2088b9a79e79b43ddbda76b`.
- Model weights SHA-256:
  `d4e5cf6317c7521ca62c11b524f5646565310e040129fe35719cad661696d745`.
- Real CPU inference produced finite normalized 512-dimensional embeddings.
- Cache identity includes audio bytes, upstream model ID, revision, local
  snapshot hash, preprocessing version, and embedding schema.
- Embeddings are committed as one atomic vector/metadata file.
- Explicit `review_preview` and `heldout_eval` roots reject both byte-level and
  canonical decoded-audio overlap.
- Prototype indexes preserve the previous index if installation of a rebuilt
  index fails.
- Shadow reports include top two labels, similarity, margin, prototype support,
  radius/spread, OOD evidence, structure/trainer seams, and a human-verdict column.
- The legacy-brain inventory covers all nine current folder/dedicated memory
  brain JSON files and does not mutate any of them.

## Real held-out vocal result

The experiment used 45 approved preview trainers and 67 held-out files. The
entire 39-file `Premium Deep House Vocals.zip` remained held out. There were 28
protected negative fixtures and zero byte or decoded-audio overlaps.

| Measure | Result |
|---|---:|
| Held-out musical-vocal recall | 30/39 (76.9%) |
| Vocal loops | 21/29 (72.4%) |
| Vocal one-shots | 9/10 (90.0%) |
| Protected-negative raw musical-vocal top-1 false positives | 2/28 (7.1%) |
| In-distribution musical-vocal false positives | 0/28 (0.0%) |
| Neural vocal Review/OOD rate | 38/39 (97.4%) |
| Legacy `Instruments/Voice` recall on the pack | 21/39 (53.8%) |
| Broad-family comparison | CLAP-only 12; legacy-only 3; both 18; neither 6 |

This is evidence of better broad vocal representation, not permission to route.
The five musical-vocal trainers do not establish a reliable neighborhood. The
two negative voice guesses and 38/39 vocal guesses are OOD, so the current gate
keeps them in Review.

The separate 35-file disputed-label confuser panel produced no in-distribution
musical-vocal claims. All eight spoken-voice examples were predicted as musical
vocal but were OOD. Breath/mouth examples were all excluded for overlap with the
primary experiment, and no explicit Formant FX taxonomy folder exists.

## Real music-focused comparison

Pinned `m-a-p/MERT-v1-95M` revision
`7d1bb4c6894b70c0f958a550c20dc861c83b25c3` ran on the same 45-train/
67-held-out split and produced real 768-dimensional layer-6 embeddings.

| Held-out group | CLAP | MERT |
|---|---:|---:|
| Drums and percussion | 1/8 | 5/8 |
| Other instruments | 2/4 | 4/4 |
| Sax and woodwind | 8/12 | 10/12 |
| Synth pad and keys | 3/4 | 2/4 |
| Musical vocal | 30/39 | 10/39 |

MERT is a useful research comparison but its checkpoint is CC-BY-NC-4.0, so it
cannot be the intended distributable production default. The registry records
both the best evaluated lane and the license-compatible deployment candidate.
Fusion remains disabled because no fused lane beat a single encoder on the same
held-out panel.

## Provenance corpus and readiness

The persistent artifact inventories 366 candidate occurrences:

- 51 explicit human-approved occurrences;
- 315 legacy curated candidates kept excluded pending provenance review;
- byte, canonical decoded, and normalized/silence-trimmed duplicate groups;
- separate prototype-training, validation, calibration, and final-heldout uses;
- GUI corrections forced to training-only;
- zero duplicate-group reuse allowed across partitions.

All 274 production labels are inventoried: 0 Tier A, 2 Tier B, 7 Tier C, 265
Tier D, and 0 automatically declared Tier E. There is not enough trusted,
source-diverse evidence to promote a detailed production leaf.

## Trainer contamination facts

The 315-file CLAP leave-one-out audit reported:

- 11 identical-byte label conflicts;
- 71 stronger cross-label conflicts;
- 9 possible same-label outliers;
- 28 singletons;
- 21 limited-support trainers;
- 170 supported trainers.

The 74-row vocal/texture/instrument-loop focus contains 26 stronger cross-label
conflicts and 6 exact duplicate-label conflicts. All six current Altered Voice
trainers are closer to another category than to their assigned category under
this diagnostic.

The all-brain static inventory found 10,949 trainer occurrences across nine
brains, 6,929 stored identities, 52 conflicting identities, and 8,561 stale or
missing source paths. That inventory is evidence for a reversible cleanup
project; it is not a deletion list and not held-out model accuracy.

## Patch verification

The original handoff bundle was found under `_reports/bundles`. The claimed
`v31.177` ZIP was not present in the supplied locations. The installed root
`INSTALL_CHANGED_FILES.command` was byte-identical to the original handoff
installer and omitted the neural package and its v31.175/v31.176 cumulative
changes. This patch repairs bundle selection so root status documents and the
complete changed-file set are included by the Makefile bundle workflow.

## Not yet complete

- Neural production ownership is not approved.
- Confidence calibration has not been trained: only one nonleaking general
  calibration example exists and there are not 20 accepted/rejected outcomes.
- MERT is not distribution-compatible; another music-focused production
  candidate is still needed.
- The actual all-baby on/off replay is complete, but independent
  core/spread/outlier ablations and unique safety-veto measurement are missing.
- Breath/mouth and explicit formant-FX non-overlapping held-out coverage is missing.
- Lossy/time-stretched derived-copy detection is missing.
- Full-suite and final quality results are listed in the new handoff rather than
  inferred from focused evidence.

See `docs/NEURAL_TRAINING_CURATION_HANDOFF_20260724.md` and
`neural_artifacts/20260724_real_curation/` for commands, persistent evidence,
rollback, and the exact next gates.
