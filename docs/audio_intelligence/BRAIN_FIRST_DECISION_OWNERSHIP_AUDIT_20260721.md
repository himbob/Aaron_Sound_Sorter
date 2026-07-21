# Brain-First Decision Ownership Audit - 2026-07-21

## Why This Exists

The sorter is moving away from static Python rescue code and toward trained
memory brains. To make that real, every run needs to answer a simple question:

```text
Who owned this final folder?
```

Possible owners are:

- learned memory brains
- older full/core/spread/outlier trained brains
- static measured contracts and guardrails
- review
- legacy or unknown routing

If a result is correct only because static code rescued it, that behavior should
eventually become training evidence. If a learned memory lane matched but did
not own the final result, that row is the first place to inspect when GUI
training feels ignored.

## New Command

```bash
cd /Volumes/T9/testbed/Aaron_Sound_Sorter
./commands/quality/RUN_DECISION_OWNERSHIP_AUDIT.command path/to/Aaron_Sorted_Sounds_manifest.csv
```

The report is written to:

```text
_reports/decision_ownership_audit/run_YYYYMMDD_HHMMSS/
```

Files:

- `decision_ownership_audit.csv`
- `decision_ownership_summary.txt`

## How To Read It

Important columns:

- `decision_owner_type`
- `learned_memory_matched`
- `learned_memory_agrees`
- `learned_memory_blocked`
- `static_over_brain`
- `broad_fallback`

Good signs:

- `decision_owner_type=learned_memory`
- `learned_memory_matched=True` and `learned_memory_agrees=True`
- static contracts are rare and mostly review or family-safety rails

Warning signs:

- `learned_memory_blocked=True`
- `static_over_brain=True`
- many correct rows owned by `static_measured_contract`
- many known source-family samples landing in broad fallback buckets

## Current Architecture Finding

The new shape, physics, and voter memory brains are real, but they are still
not the whole classifier. They store numeric fingerprints, match without source
filenames, and can emit learned owner claims. The remaining static code still
does broad body authorization:

- Drums need measured drum/percussive body.
- Instruments need measured pitched/instrument body.
- FX need measured FX motion/body.
- Voice has additional non-voice conflict checks.

That is acceptable as a temporary safety shell, but it means the system is not
yet pure brain-owned. The next redesign step is to use this audit to distill
static successes and static failures into trusted memory training panels, then
retire static leaf-level contracts after memory-backed tests cover them.

## No Source-Name Evidence

This audit reads source paths only as row identifiers from manifests. It does
not classify audio using filenames, folders, ZIP member names, or sample-pack
paths.
