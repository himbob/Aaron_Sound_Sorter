# Neural audio implementation status — 2026-07-24

This table reflects code inspection, focused tests, and real CPU inference. It
does not infer completeness from older handoff claims.

| File | Status | Verification and limitation |
|---|---|---|
| `neural_audio/__init__.py` | Complete and tested | Public contracts import in the maintained neural tests. |
| `neural_audio/cache.py` | Complete and tested | Atomic vector/metadata records, full model identity, hash reuse, and corruption handling are covered by `test_neural_embedding_cache.py`. Real CLAP and MERT records were written. |
| `neural_audio/calibration.py` | Implementation complete; real fit incomplete | Append-only review feedback and logistic fitting pass `test_neural_calibration_feedback.py`. Only one nonleaking general-corpus calibration example and no 20-outcome accepted/rejected panel exist, so no real calibrator was fitted. |
| `neural_audio/contracts.py` | Complete and tested | Typed embeddings, prototypes, splits, lane evaluations, feedback evidence, and validation invariants are exercised across the neural test set. |
| `neural_audio/curation.py` | Complete for current corpus policy; tested | Four-way group-safe use assignment, GUI-training-only feedback policy, conflict exclusion, and readiness tiers pass `test_neural_curation.py`. |
| `neural_audio/dataset.py` | Complete and tested | Curated discovery, preview/held-out splitting, explicit roots, byte leakage, and decoded-audio leakage are tested. The new four-way corpus policy lives in `curation.py`. |
| `neural_audio/evaluation.py` | Complete and tested | Rejects prototype/evaluation hash reuse and writes per-label held-out metrics. Both real encoders used it on the same 67-file panel. |
| `neural_audio/hashing.py` | Complete for lossless/gain/silence variants; tested | Byte, canonical decoded, and normalized/silence-trimmed digests are tested. Lossy, time-stretched, or heavily trimmed copies still need a perceptual matcher. |
| `neural_audio/lane_selection.py` | Leaderboard and fusion gate complete; real fusion missing | Per-label selection and evidence-gated fusion policy are tested. No fused encoder was evaluated, so fusion is disabled. |
| `neural_audio/legacy_brain_audit.py` | Complete and tested | Static inventory covers folder and dedicated memory brains without mutation. |
| `neural_audio/legacy_manifest.py` | Complete and tested | Legacy output is joined to neural evidence by content hash, not display name. |
| `neural_audio/prototype_index.py` | Complete and tested | Deterministic bounded multi-prototype building, cosine search, category spread, OOD evidence, atomic replacement, and rollback pass focused tests and both real builds. |
| `neural_audio/shadow.py` | Complete and tested | Disagreement-first rows include runner-up, margin, prototype, OOD, provenance seams, and verdict columns. A current 39-file comparison was generated. |
| `neural_audio/trainer_audit.py` | Complete and tested | Real CLAP leave-one-out contamination evidence covers all 315 locked trainers. It is diagnostic, not held-out accuracy. |
| `providers/__init__.py` | Complete and tested | Exposes the reviewed providers used by the lab. |
| `providers/base.py` | Complete and tested | Source-blind provider protocol is exercised by cache, index, and source-name tests. |
| `providers/huggingface_clap.py` | Complete and real-tested on CPU | Pinned local CLAP produced finite normalized 512-dimensional embeddings. MPS was unavailable in this runtime, so Apple GPU inference was unable to be tested. |
| `providers/huggingface_encoder.py` | Complete for reviewed MERT comparison; real-tested on CPU | Pinned MERT produced normalized 768-dimensional layer-6 embeddings. The model requires reviewed remote code and a compatibility default for newer Transformers. The checkpoint license is non-commercial, so it is research-only. |
| `providers/identity.py` | Complete and tested | Records upstream ID, revision, local snapshot digest, preprocessing, and embedding schema. This is model provenance, not a fake embedding provider. |

## Missing production pieces

- A production loader for the model registry and calibrated neural owner claims.
- Twenty or more nonleaking accepted/rejected review outcomes for logistic calibration.
- A legally distributable music-focused encoder that matches MERT's sax/woodwind result.
- Lossy/time-stretch perceptual duplicate detection.
- Automatic GUI feedback updates to the neural embedding/prototype index with rollback.
- Independent core/spread/outlier disable switches and complete multi-panel ablations.
- A validated category group that passes readiness, calibration, negative, OOD,
  and rollback gates. No category currently has frozen-neural placement authority.

No component is currently classified as a placeholder. The frozen neural route
is incomplete by evidence, not hidden behind fake vectors.
