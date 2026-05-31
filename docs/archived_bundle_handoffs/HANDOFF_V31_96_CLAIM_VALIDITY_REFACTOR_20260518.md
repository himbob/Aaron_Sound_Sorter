# V31.96 Claim Validity Refactor

## Purpose

This patch removes the old behavior where measured role/shape shortcuts could still act like routing rules after the claim-arbiter refactor.

The prior refactor moved final `ConsensusDecision` creation into `FamilyClaimArbiter`, but some claim producers still emitted high-strength routing claims based on role/shape compatibility. That meant old behavior survived under a cleaner type name.

## Architectural rule now enforced

- Brain/full/baby/physics candidate evidence remains usable.
- Role and shape observations are diagnostic evidence, not direct routing authority.
- Cross-family movement by role/shape requires real voter-candidate backing.
- A profile/baby kick shortcut cannot beat a specific non-drum raw candidate unless the kick candidate decisively beats the raw candidate by rank.
- Strong blocked role/shape conflict goes to `_TO_REVIEW/Measured Role Conflict` instead of forcing a bad broad folder.
- Review is no longer erased by old-style shortcut claims such as `candidate_true_bucket_rescue`, `parent_eligibility_broad_bucket`, `profile_candidate_kick_claim`, or `baby_recall_kick_claim`.

## Brain lane policy

This patch does not retrain or remove any brain lane.

Baby brains are preserved. Useful baby-lane claims such as `baby_recall_brass_woodwind_claim` can still beat a generic raw bucket when they are candidate-backed. What changed is that baby/profile kick shortcuts cannot use role/shape compatibility alone to steamroll a stronger concrete non-drum raw candidate.

## Files changed

- `src/aaron_sound_sorter/engine/family_claim_arbiter.py`
- `tests/test_claim_validity_refactor_v3196.py`
- `tests/test_family_claim_arbitration_matrix.py`
- `tests/test_two_voter_redesign.py`

## Local validation run in sandbox

Passed one at a time or small focused groups:

- `tests/test_claim_validity_refactor_v3196.py`
- `tests/test_claim_arbiter_architecture.py`
- `tests/test_family_claim_arbitration_matrix.py`
- `tests/test_two_voter_redesign.py`
- `tests/test_broad_bucket_claim_producer.py`
- `tests/test_brain_competence_policy.py`
- `tests/test_v3187_architecture_doc_and_lane_competence.py`
- `tests/test_no_source_name_sorting_invariant.py`
- `tests/test_profile_candidate_claim_producer.py::test_profile_candidate_producer_returns_kick_claim_from_measured_kick_context`
- `tests/test_profile_candidate_claim_producer.py::test_profile_candidate_producer_does_not_promote_reed_without_specific_reed_role`
- `python3 -S -m py_compile` on changed files
- `python3 tools/audit_no_source_name_sorting.py`

## Real sample smoke checked in sandbox

Single-file runtime tests completed without crashing:

- `clipped_sub_hit.wav` no longer routed to Generic Kick. It went to `_TO_REVIEW/No Strong Voter Consensus` because old kick/tom shortcut claims were blocked by review.
- `AA_TSS_D_RUMBLER_808_BASS.wav` still needs later identity work. It no longer selected Generic Kick in this sandbox run, but the raw product voter path still favored Drums/Toms because physics and raw shared-candidate evidence are not fixed by this claim-validity refactor.
- `Clap 17.wav` still needs identity work.
- `boom_eval_108640.wav` still needs identity work.

## What this patch does not fix

This is not the kick/bass/boom identity fix. It removes old routing shortcuts first. The next fix should study why raw product candidate evidence still turns some sub/bass/boom material into Toms/Percussion after the old shortcut claims are blocked.

