# Repo hygiene policy

The root folder should contain only product files, active project state, and the single AI handoff document.

## Keep in root

- `Aaron_Sound_Sorter.py`
- `AI_READ_THIS_FIRST.md`
- `README.md`
- `Makefile`
- `pyproject.toml`
- `requirements-dev.txt`
- active root brain JSONs such as `stage4_folder_brain*.json`
- `src/`, `tests/`, `tools/`, `commands/`, `docs/`

## Move out of root

- Root `RUN_*.command` files go to `commands/legacy_root_commands/`.
- Old bundle-specific Markdown files go to `docs/archive/old_bundle_docs/`.
- Generated reports and old run outputs are removed by `make clean`.
- Bundles are written to `_reports/bundles/`, not root.

## Commands

Preview cleanup:

```bash
make clean-dry-run
```

Apply cleanup:

```bash
make clean
```

Move root command files:

```bash
make organize-root-commands
```

Archive old bundle handoff docs:

```bash
./commands/docs/ARCHIVE_OLD_BUNDLE_HANDOFF_DOCS.command
```

Create AI handoff bundle with brains:

```bash
./commands/bundle/bundle.sh
```
