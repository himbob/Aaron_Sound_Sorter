# Decision Core Contract — Compatibility Status

The existing `DecisionCoreV2` and `FamilyClaimArbiter` remain production compatibility
components while the neural migration is in shadow mode. They are frozen against new
category-policy growth.

New semantic category knowledge belongs in:

```text
src/aaron_sound_sorter/neural_audio/
```

Only objective safety facts and typed compatibility adapters may be added to the legacy
decision path. The target contract is documented in `NEURAL_AUDIO_ARCHITECTURE_V1.md`.
The previous contract is archived at:

```text
docs/archive/legacy_rule_engine/DECISION_CORE_CONTRACT_legacy_20260723.md
```
