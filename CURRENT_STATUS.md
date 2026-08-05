# Current status

Updated: 2026-08-05

## Working

- CLI sorting with local trained brains.
- Local GUI review and supervised corrections.
- Human-sized selected-file summaries, stable disclosures, 200-row queue
  windows, incremental polling, and Stop & Keep Results for large runs.
- Durable, source-blind GUI corrections feeding versioned CLAP prototypes.
- Exact content-hash ownership for human-approved audio.
- Cross-family neural/legacy conflicts routed to Review.
- Exact human-approved memory ownership across Drums, Instruments, and FX.
- Pretrained PANNs broad-event evidence, visible in both GUIs.
- Canonical 395-category registry; availability is separate from readiness.
- Read-only detailed CLAP prompt suggestions for every canonical category.
- Cluster-first review packs that retain every member under hash-only names.
- Learning Center with representative playback, safe-core approval, separate
  import, explicit background prototype rebuilding, and all-category coverage.
- Accepted and corrected GUI outcomes feeding source-blind confidence learning.
- Gain/silence-normalized duplicate protection across training and evaluation.
- A locked-seed relabel requires two identical human approvals.
- Approved embeddings survive rebuilds when their original audio is missing.
- Source-name blindness audit.
- CLAP and MERT embedding, cache, prototype, and evaluation tooling.
- Provenance, contamination, readiness, active-learning, and ablation reports.

## Production authority

- Measured evidence and trained memory: active.
- CLAP generalized ownership: gated off until a small category group passes
  calibration and independent held-out requirements.
- Exact approved hashes: authoritative; CLAP, PANNs, and measured witnesses
  remain visible but cannot veto the user's explicit choice.
- PANNs: broad support/contradiction guardrail only; no detailed ownership.
- Detailed CLAP prompts: visible advice only.
- Unknown or conflicting neural evidence: Review or legacy fallback.
- Legacy arbiter: temporary compatibility and safety infrastructure.

## Neural evidence

- Trusted hash-only runtime replay: 45/45 exact; 0 Review (2026-07-25).
- Detailed CLAP prompt benchmark: 8/45 exact top-1 and 19/45 exact top-3;
  therefore advisory only.
- Active local prototype memory: 215 unique examples across 79 labels; zero
  missing hashes after the rebuild-durability repair.
- Real altered-vocal replay: exact user label owned the final folder; combined
  PANNs Singing/Yodeling/Speech/A-cappella support was 0.359 (2026-07-26).
- Category coverage: 395 selectable; 7 Tier B, 30 Tier C, 358 Tier D.
- Independent 67-sound CLAP panel: 65.7% top-1, 82.1% top-3, 82.1%
  Review/OOD, and 4 incorrect known-distribution placements.
- Voice promotion gate: failed and remained disabled.
- Seven-file real synth-loop provenance audit: 1 exact neural owner, 2 neural
  conflict reviews, 4 transitional/legacy decisions, and 0 automatic Voice
  placements. Passing final folders are not being counted as neural wins.
- Real confidence calibration: 19 nonleaking reviewed outcomes (2 accepted,
  17 corrected); one more distinct outcome is required to fit, and substantially
  more balanced evidence is required to promote authority.
- MERT is research-only because its checkpoint is non-commercial.
- Full pytest, source-name audit, public-assets audit, and changed-file quality
  gate passed on 2026-08-05. Seven unavailable private-audio cases skipped.

## Public repository

- Source, configuration, tests, and download/build commands are tracked.
- Models, trained brains, indexes, caches, audio, private manifests, reports,
  runtime pointers, and machine paths are local-only.
- CLAP and PANNs have pinned, checksum-aware local download commands.
- New personal training requires lawfully licensed user audio.

## Remaining promotion work

- Collect at least 21 more balanced, nonleaking accepted/corrected reviews.
- Add independent held-out audio per authority group; do not lower the gate.
- Run group promotion again, starting with voice, then drums and FX.
- Audit decision-owner provenance before removing any legacy lane.
- Refactor `preview_service.py`; split the legacy arbiter only behind ablation
  tests and rollback coverage.

See `docs/NEURAL_AUDIO_ARCHITECTURE_V1.md`.
