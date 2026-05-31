# Category Stability Gate

This folder defines category isolation contracts for real-audio smoke testing.
The sorter must still be source-name blind; filenames here are only stable test
identifiers used by the QA harness.

Each `anchors_*.json` file is a category trap pack with four scenario types:

- `category_positive`: examples that should land in the target category.
- `category_near_miss`: nearby sounds that may share traits but should keep their own role/family.
- `category_must_not_steal`: neighbor-category examples the target category must not steal.
- `category_should_review`: uncertain examples that are safer in `_TO_REVIEW` than in a false confident folder.

Run the gate:

```bash
cd /Volumes/T9/testbed/Aaron_Sound_Sorter
./commands/quality/RUN_CATEGORY_STABILITY_GATE.command
```

Before a behavior-changing patch, capture a baseline:

```bash
./commands/quality/RUN_PRE_CHANGE_BASELINE_CATEGORY_STABILITY.command
```

After the patch, run the gate and compare:

```bash
./commands/quality/RUN_CATEGORY_STABILITY_GATE.command
./commands/quality/COMPARE_CATEGORY_STABILITY_TO_BASELINE.command
```

If a patch is intentionally allowed to move specific protected outputs, list the
case ids, anchor packs, target categories, or scenarios in `change_budget.json`
before running the comparison. The default budget approves no protected output
drift.
