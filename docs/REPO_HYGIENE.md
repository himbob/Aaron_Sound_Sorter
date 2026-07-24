# Repository hygiene

Root keeps only:

- public entry documents;
- package/build files;
- the thin CLI runner;
- active runtime pointers under `config/runtime/`.

Use:

- `src/` for product code;
- `tests/` for maintained tests;
- `tools/` for developer utilities;
- `commands/` for shell entry points;
- `docs/` for active documentation;
- `docs/archive/` for history;
- `_reports/` for generated output.

Never commit audio, brains, weights, caches, private paths, or local evidence.

```bash
make clean-dry-run
make clean
make ai-check
```
