# Brain Lane Reduction Plan

## Decision

Do not delete older brains in the v31.176 patch.

The current files are large and partly overlapping:

| Brain | Approximate size | Labels | Detailed examples |
|---|---:|---:|---:|
| Full folder brain | 64 MB | 274 | 6,603 |
| Baby brain | 16 MB | 160 | 441 |
| Core baby | 28 MB | 274 | 929 |
| Spread baby | 28 MB | 273 | 921 |
| Outlier baby | 25 MB | 234 | 810 |
| User memory | 15 MB | 72 | 290 |
| Voter memory | 1.3 MB | 35 | compact role memories |
| Physics memory | 1.4 MB | 29 | compact physics memories |
| Shape memory | 1.3 MB | 17 | compact shape memories |

File size and age are not enough to prove redundancy. Some lanes may look duplicative
but still provide a unique correct win or a unique safety veto on a rare sound.

## Required ablation report

For every lane, replay the same locked datasets with the lane enabled and disabled.
Record:

1. Correct placements lost when the lane is removed.
2. Wrong placements fixed when the lane is removed.
3. Review count increase or decrease.
4. Unique top-1 wins supplied only by that lane.
5. Unique catastrophic-family vetoes supplied only by that lane.
6. Duplicate votes that never affect the final claim.
7. Runtime and memory cost.
8. Effect on fresh GUI corrections and adaptive category recall.

## Minimum replay sets

- 32-case locked smoke panel.
- Real trained-example corpus.
- FX_Aaron2 coverage sample.
- Percussion one-shot sample.
- Known historical catastrophic failures.
- A held-out, source-name-blind random sample not used for thresholds.

## Retirement criteria

A lane may be retired or folded into another lane only when all are true:

- no protected smoke regressions;
- no unique correct wins worth preserving;
- no unique safety vetoes worth preserving;
- trained correction recall does not fall;
- Review does not increase materially;
- replacement behavior is explainable in the manifest;
- rollback brain files are retained for at least one release.

## Likely migration order

1. Measure overlap among full/core/spread/outlier folder brains.
2. Identify whether the plain baby brain contributes anything not supplied by the
   three specialized baby lanes.
3. Keep user memory separate; it represents human ownership, not generic training.
4. Keep voter/physics/shape roles separate until an ablation proves one is redundant.
5. Consolidate only the demonstrated duplicate folder-brain lanes first.
6. Rebuild a smaller unified reference brain from retained examples after replay,
   rather than deleting JSON sections in place.

## Architectural target

The long-term target should be:

- one compact general reference brain;
- one explicit human user-memory store;
- compact orthogonal role/physics/shape evidence only where each lane proves unique
  value;
- learned calibration and confidence reports;
- no policy encoded by maintaining several mystery brains solely to average mistakes.

## Verification inventory — 2026-07-24 UTC

The new static inventory scanned all nine current folder and dedicated memory
brain files. It recorded 10,949 trainer occurrences, 6,929 stored identities,
52 conflicting identities, and 8,561 occurrences whose original source path is
now missing. This proves that trainer provenance cleanup is needed, but it does
not prove that any lane is redundant.

No ablation was run and no brain was deleted. The inventory and reversible
cleanup plan are under
`_reports/neural_audio/vocal_shadow_20260724_001/trainer_contamination/`.
