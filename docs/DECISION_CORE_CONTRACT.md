# Decision Core Contract

## Core Rule

Role, shape, and eligibility are not leaf classifiers.

They may:

- block an incompatible family or branch
- broaden an unsafe leaf to a safe parent bucket
- send the file to review

They may not:

- promote a specific source identity such as sax, dog, tom, coin, or clap
- mutate BrainVoter scores
- mutate PhysicsVoter scores
- use filenames or folder paths as decision evidence

## Data Flow

```python
raw_decision = legacy_consensus.choose(brain_votes, physics_votes, facts)
eligibility = infer_parent_eligibility(facts)
final_decision = DecisionCoreV2.apply(raw_decision, eligibility, facts)
```

## Allowed V2 Actions

1. Keep the raw decision if it is eligible.
2. Broaden to a safe parent bucket if the raw decision is incompatible.
3. Send to review if the measured role is uncertain and the raw decision is unsafe.

## Forbidden V2 Actions

- No filename rescue.
- No category-specific one-file patch.
- No hidden bonus to both voters.
- No disabling categories globally.

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

