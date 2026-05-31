# Aaron Sound Sorter v31.94C Import Compatibility Fix

## Problem

Aaron's local validation failed while collecting `tests/test_broad_bucket_claim_producer.py`:

```text
ImportError: cannot import name 'BroadBucketClaimProducer' from 'aaron_sound_sorter.engine.claim_producers'
```

## Cause

v31.94 renamed the broad-bucket coordinator internals to `MeasuredBucketClaimProducer`, but an older public regression test still imported `BroadBucketClaimProducer`. That public seam should not have been removed during a refactor.

## Fix

This patch adds a backward-compatible wrapper:

```text
src/aaron_sound_sorter/engine/claim_producers/broad_bucket_claims.py
```

and re-exports it from:

```text
src/aaron_sound_sorter/engine/claim_producers/__init__.py
```

`DecisionCoreV2` now instantiates the compatibility seam:

```python
BroadBucketClaimProducer(self)
```

The wrapper delegates to the current `MeasuredBucketClaimProducer`, so behavior stays the same.

## Validation run before packaging

```text
compileall: passed
no-source-name audit: passed
/tmp/test_broad_bucket_claim_producer.py: 2 passed
tests/test_claim_arbiter_architecture.py: 3 passed
tests/test_v3189_real_audio_architecture_regressions.py: 8 passed
self-test: passed
```

## Architecture note

This is an API compatibility patch, not a classifier-policy change. No scoring thresholds or routing logic were retuned.
