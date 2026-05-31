# AI Handoff: Claim-Arbiter Refactor v31.78 Work In Progress

## Goal

Replace distributed final-folder decisions with one Claim-Arbiter-Resolver workflow:

1. `ConsensusRunner` gathers raw voter agreement and consensus claims.
2. `DecisionCoreV2` gathers parent-eligibility claims.
3. `FamilyClaimArbiter` is the only production engine module that constructs `ConsensusDecision`.
4. `PlacementResolver` owns broad synthetic folder-path mapping.

This directly targets the v31.77 failure where `Instruments/Instrument Loops/Loops` swallowed too many Sax/reed loops and Human/Voice broad rescues could steal pitched instruments.

## Implemented files

- `src/aaron_sound_sorter/engine/family_claims.py`
  - Added `ConsensusClaim`.
  - Added helpers for candidate, folder, and review claims.
- `src/aaron_sound_sorter/engine/family_claim_arbiter.py`
  - New final arbiter.
  - Only production engine module containing `ConsensusDecision(`.
  - Blocks ghost cross-family broad claims unless they have safe backing.
- `src/aaron_sound_sorter/engine/placement_resolver.py`
  - New centralized broad folder map.
- `src/aaron_sound_sorter/engine/consensus.py`
  - `choose()` now returns `(raw_claim, consensus_claims)`.
  - Consensus rescue paths now emit claims instead of final decisions.
- `src/aaron_sound_sorter/engine/decision_core_v2.py`
  - `choose()` now runs raw claims + eligibility claims through `FamilyClaimArbiter`.
  - `gather_eligibility_claims()` replaces old direct-return final placement flow.
  - `apply_eligibility()` remains only as a direct unit-test seam and routes through the new arbiter path, not a second legacy decision engine.
- `src/aaron_sound_sorter/engine/placement_depth.py`
  - Placement-depth broadening now emits claims.
- `src/aaron_sound_sorter/engine/sorter.py`, `src/aaron_sound_sorter/app.py`, `Aaron_Sound_Sorter.py`
  - Composition root wiring now injects `PlacementResolver` and `FamilyClaimArbiter`.
- `tests/test_claim_arbiter_architecture.py`
  - New architecture lock tests.
- `tests/test_two_voter_redesign.py`
  - Converted to run consensus evidence through the arbiter.
- `tests/test_consensus_concrete_fx_gate_regressions.py`
  - Partially converted to run consensus evidence through the arbiter.

## Validation completed in this environment

Passed:

```bash
./commands/quality/RUN_NO_SOURCE_NAME_SORTING_AUDIT.command
python Aaron_Sound_Sorter.py self-test
python -m pytest tests/test_claim_arbiter_architecture.py tests/test_two_voter_redesign.py tests/test_no_source_name_sorting_invariant.py -q
python -m pytest tests/test_family_claim_arbitration_matrix.py -q
```

Compile check passed:

```bash
find src -name '*.py' -print0 | xargs -0 python -m py_compile
```

## Known unfinished work

The old direct `ConsensusRunner().choose(...) -> ConsensusDecision` tests are not all converted yet.  The new contract is:

```python
raw_claim, consensus_claims = ConsensusRunner().choose(brain, physics, facts)
decision = FamilyClaimArbiter().adjudicate(
    raw_claim=raw_claim,
    consensus_claims=consensus_claims,
    eligibility_claims=[],
    facts=facts,
)
```

`tests/test_decision_core_v2.py` still contains old identity assertions such as `final is raw`. Those assertions are obsolete because the arbiter now creates a new final `ConsensusDecision` from a raw claim. Convert those tests to compare decision values and claim behavior, not object identity.

The second test in `tests/test_consensus_concrete_fx_gate_regressions.py` now returns review rather than generic `Instruments/Instrument Loops/Loops` when there is no real Instrument candidate. That is intentional under the new policy: a ghost generic Instrument Loop broad bucket should not auto-place across family without candidate backing. Decide whether the test should expect review or whether an explicit real Brass/Woodwind candidate should be added.

## Important policy decisions made

- Ghost Human/Voice claims cannot steal Sax/reed/Instrument raw winners without real candidate backing.
- Ghost generic Instrument Loop claims cannot steal cross-family winners without real candidate backing.
- Strong inferred Drums/Drum Loops and Instruments/Brass Woodwinds broad claims may cross families only at high strength.
- Source-name blindness was preserved. The audit passes.

## Next recommended steps

1. Convert remaining direct-consensus tests to the new claim-arbiter contract.
2. Convert old `DecisionCoreV2.apply_eligibility` tests to assert claim behavior and final decision values.
3. Run the FX/Sax manifest again and specifically compare:
   - Sax into `Instruments/Instrument Loops/Loops`
   - Sax into `FX/Human and Voice FX`
   - total review count
4. Only after tests are converted, remove or rename the direct `apply_eligibility()` test seam if desired.
