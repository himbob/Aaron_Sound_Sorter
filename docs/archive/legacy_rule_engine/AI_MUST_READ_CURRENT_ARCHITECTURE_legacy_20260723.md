# AI Must Read Current Architecture

Final placement belongs to the FamilyClaimArbiter.

Decision code should produce evidence claims, not final folder answers. Folder path mapping belongs in PlacementResolver or narrow claim construction helpers. Claim producers may create typed evidence claims, but they should not become hidden final arbiters.

Current cleanup direction:

- Keep `DecisionCoreV2` as a coordinator.
- Keep claim producers small and documented.
- Preserve source-name blindness in production sorting modules.
- Do not retune thresholds during structure-only refactors.
- Run pytest one file at a time.
- Keep brain-lane competence policy isolated from final placement.
