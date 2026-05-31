# `make bundle` workflow

`make bundle` is the standard AI handoff command for Aaron Sound Sorter patch bundles.

## Purpose

The target makes bundle creation repeatable instead of letting every AI agent invent a new ZIP command.

It enforces this order:

1. Remove generated junk with `make clean-for-bundle`.
2. Run the AI changed-files quality gate with `make ai-check`.
3. Call `commands/build/BUILD_CLEAN_PATCH_BUNDLE.command`.
4. Stage only safe changed project files under `files/`.
5. Generate `INSTALL_NO_BACKUP.command`.
6. Write the ZIP under `_reports/bundles/`, not the project root.

## Normal command

```bash
make bundle BUNDLE_NAME=Aaron_Sound_Sorter_descriptive_patch_name
```

## Dry run

```bash
make bundle-dry-run BUNDLE_NAME=Aaron_Sound_Sorter_descriptive_patch_name
```

## Changed-file source

The builder uses this priority:

1. `AI_CHANGED_FILES`, when supplied.
2. Git changed and untracked files relative to `AI_BASE`, defaulting to `HEAD`.

For non-Git or temporary AI workspaces, set `AI_CHANGED_FILES` explicitly:

```bash
AI_CHANGED_FILES='Makefile tools/build_clean_patch_bundle.py commands/build/BUILD_CLEAN_PATCH_BUNDLE.command' \
make bundle BUNDLE_NAME=Aaron_Sound_Sorter_patch
```

## Bundle hygiene

The builder refuses to bundle:

- caches
- compiled Python files
- macOS AppleDouble/resource-fork files
- `.DS_Store`
- `_reports`
- virtual environments
- audio files
- ZIP archives
- oversized files

This keeps patch bundles small, inspectable, and safe to install.
