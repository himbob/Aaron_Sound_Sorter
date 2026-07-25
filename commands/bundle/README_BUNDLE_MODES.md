# Aaron Sound Sorter bundle modes

## Normal command

From the repository root, either command uses the same script:

```bash
make bundle
```

```bash
./commands/bundle/bundle.sh
```

The default build produces:

1. `Aaron_Sound_Sorter_Code_Only_<timestamp>.zip`
   - small code-review archive;
   - application source, commands, tools, config, current docs, and Python tests;
   - no brains, model weights, reports, or audio.

2. `Aaron_Sound_Sorter_End_User_Pack_<timestamp>_Part_XX_of_YY.zip`
   - a balanced multipart end-user pack;
   - all parts together contain the code archive, active trained brains, selected CLAP/MERT snapshots, and prototype indexes;
   - every finished part stays below `BUNDLE_UPLOAD_PART_MAX_MIB`, which defaults to 490 MiB.

The multipart pack names make it obvious that every file belongs to one set.
Extract every part ZIP into the same folder, then run the included
`ASSEMBLE_AND_INSTALL.command`. It verifies checksums, reconstructs the AI
runtime archive, and merges code and runtime into one `Aaron_Sound_Sorter/`
folder.

## Compression and upload size

The code and AI-runtime archives both use ZIP DEFLATE level 9. GZIP uses the
same DEFLATE compression family, so changing from ZIP to `.tar.gz` does not
meaningfully shrink dense CLAP/MERT model weights. The reliable fix is balanced
multipart packaging.

Default maximum part size:

```bash
BUNDLE_UPLOAD_PART_MAX_MIB=490 make bundle
```

Use a smaller limit when a service has a lower upload ceiling:

```bash
BUNDLE_UPLOAD_PART_MAX_MIB=400 make bundle
```

The original unsplit AI-runtime ZIP is deleted after the multipart pack is
created. Keep it only when explicitly needed:

```bash
BUNDLE_KEEP_UNSPLIT_AI_ARCHIVE=1 make bundle
```

## Preview without writing archives

```bash
make bundle-dry-run
```

The exact number of balanced parts is determined only after the real AI-runtime
archive is compressed.

## Build only one archive

Code only:

```bash
BUNDLE_OUTPUT_MODE=code ./commands/bundle/bundle.sh
```

Unsplit AI runtime only:

```bash
BUNDLE_OUTPUT_MODE=ai-runtime ./commands/bundle/bundle.sh
```

Combined legacy product ZIP:

```bash
BUNDLE_OUTPUT_MODE=product ./commands/bundle/bundle.sh
```

## Smaller runtime without downloaded model snapshots

```bash
BUNDLE_INCLUDE_NEURAL_MODELS=0 make bundle
```

This still includes all active brains and compact prototype indexes.

## Excluded from every product bundle

- `.git`
- `_reports`, previews, old bundles, outputs, and generated runs
- virtual environments and Python caches
- training folders and sample libraries
- regression and acceptance audio
- neural embedding caches and curation runs
- GUI training imports and inboxes
- backups, archives, and old ZIP files

The script does not delete or clean anything from the working repository.

## Runtime asset list

The exact live brains, model directories, and neural runtime artifact directories
are declared in:

```text
config/product_bundle_assets.txt
```

## Legacy AI patch bundle

```bash
make patch-bundle
```
