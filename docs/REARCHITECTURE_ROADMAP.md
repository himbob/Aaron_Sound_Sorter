# Aaron Sound Sorter Re-Architecture Roadmap

## Purpose

This roadmap exists so future AI helpers do not lose the project goal or patch one failure while damaging the whole sorter.

The sorter should be a practical music-production sound librarian. It should place obvious files into sensible producer-friendly folders, send uncertain/conflicting files to review, and write enough diagnostics to understand mistakes.

The current refactor target is not perfect leaf naming. The target is to stop catastrophic parent-family failures.

## Non-Negotiable Architecture Rules

1. Measured audio evidence first. Do not use filenames or folder names for final category decisions.
2. Structure and role before source identity.
3. Role/shape/eligibility may block, broaden, or review. They may not promote a specific leaf identity.
4. Raw voters stay raw. BrainVoter and PhysicsVoter must not receive hidden role bonuses.
5. Broad safe folders are better than wrong leaves.
6. Review is a valid result when evidence conflicts.
7. Do not patch individual filenames.
8. Do not narrow or hide the full category library.
9. Every bundle must include current status and handoff notes.
10. Do not write outputs to Aaron's Desktop unless explicitly requested.

## Current Failure Pattern

Recent bad files show the same architectural failure:

- sax/music loop -> tom drums or spoken voice
- drum loop -> FX, drops, breath, or spoken voice
- vocal stab -> dogs or birds
- choppy music loop -> coins
- single kick -> drum loops
- clap/snare -> glitches/stutters

This means incompatible labels are being allowed to compete too late in the pipeline.

## Target Architecture

```text
Audio file
  -> bounded audio loader
  -> feature extraction / shared facts
  -> measured role classifier
  -> eligibility policy
  -> raw voter candidate generation
  -> safe decision core
       - block incompatible branches
       - broaden weak leaves
       - review conflicts
  -> manifest and audit report
```

## Phase Plan

### Phase 0: Project Control

- [x] Add this roadmap.
- [x] Add current status/handoff document.
- [x] Add decision-core contract.
- [x] Add testing policy.
- [x] Warn future helpers that project ZIPs and sample ZIPs are mixed in the project folder.

### Phase 1: Baseline and Regression Fixtures

- [x] Preserve uploaded failure WAVs as explicit regression fixtures.
- [x] Add a focused pytest that fails on the current unsafe parent-family placements.
- [x] Record baseline failure examples in `CURRENT_STATUS.md`.

### Phase 2: Parent Eligibility Layer

- [x] Add `engine/eligibility.py`.
- [x] Model eligibility as allowed tops, blocked fragments, broad fallback, and explanation.
- [x] Keep eligibility broad and physical. It does not identify sax, dog, coins, or toms.

### Phase 3: DecisionCoreV2 Seam

- [x] Add `engine/decision_core_v2.py`.
- [x] Route final decisions through V2 after raw consensus.
- [x] V2 may block, broaden, or review only.
- [x] Keep legacy consensus available internally as the raw candidate picker.

### Phase 4: Parent Safety Regression

- [x] Uploaded sax loop no longer lands in Drums or FX voice/animal/foley.
- [x] Uploaded drum loops no longer land in FX Long FX/Drops/Breath/Voice.
- [x] Uploaded vocal stab no longer lands in Dog/Bird.
- [x] Uploaded choppy music loop no longer lands in Coins.

### Phase 5: Remaining Work

- [ ] Run Aaron's full local real-sort packs under `_real_sort_tests`.
- [ ] Add a report-only comparison script for before/after manifests.
- [ ] Add branch-aware coverage gate when pytest-cov is available.
- [ ] Add exemplar/nearest-neighbor diagnostics later, after parent safety is stable.
- [ ] Consider pretrained audio embeddings only as audit evidence, not final truth.

## Definition of Done for Future Patches

Every patch must state:

- Failure class addressed
- Why the fix generalizes
- Files changed
- Tests added/updated
- Tests run
- Tests not run
- Whether raw voter scores changed
- Whether role/shape/eligibility changed
- Whether filename/path semantics were used
- Remaining known failures

---

## Absolute blind-sorting rule

Production sorting logic must never use producer filenames, source folder names,
ZIP member names, path tokens, sample-pack labels, or any other source-name text
as classification evidence. Names are allowed only for I/O, manifest display,
post-decision diagnostics, test fixture selection, and human review. Voters,
roles, eligibility, consensus, conflict resolution, and final placement must use
audio measurements, learned brain candidates, physics candidates, shape/role
facts, and explicit manual corrections only.

Every sorter-logic change must preserve this invariant and must pass:

```bash
./commands/quality/RUN_NO_SOURCE_NAME_SORTING_AUDIT.command
```

