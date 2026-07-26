# Aaron Sound Sorter

Source-blind audio sample sorting for music producers.

Give it a ZIP, folder, or audio file. It builds a producer-friendly library,
keeps uncertain sounds in `_TO_REVIEW`, and records why each decision happened.
`definitely_a_kick_FINAL_9.wav` gets no special treatment.

This started as Aaron's attempt to make DSP do the tedious work of sorting
music-production samples. It turned out to be useful for almost any sound
collection, so the project grew—but the producer-friendly workflow is still
the heart of it.

## What it does

- Sorts Drums, Instruments, Voice, Textures, and FX.
- Separates loops, one-shots, and long effects.
- Uses decoded audio—not filenames or paths—as evidence.
- Uses a downloaded, pretrained LAION CLAP encoder; it is not trained from scratch.
- Uses pretrained PANNs as an independent broad sound-event witness.
- Lets approved GUI corrections teach small, versioned Aaron prototypes on top.
- Remembers exact human-approved audio immediately.
- Promotes broader neural ownership one tested category group at a time.
- Sends uncertain or conflicting evidence to Review.

## Install

Requirements: Python 3.9+, macOS or Linux, and `libsndfile`.

```bash
git clone https://github.com/himbob/Aaron_Sound_Sorter.git
cd Aaron_Sound_Sorter
git lfs install
git lfs pull
make bootstrap
.venv_phase4/bin/python Aaron_Sound_Sorter.py self-test
```

Optional local CLAP support:

```bash
./commands/neural/INSTALL_NEURAL_LAB.command
./commands/neural/PREFETCH_CLAP_MODEL.command
```

Git LFS carries the pinned model snapshots and compact neural indexes. Active
brain JSONs are included. Sample audio, training trees, and private review
reports are not.

## Sort audio

```bash
python3 Aaron_Sound_Sorter.py "/path/to/samples.zip" "/path/to/output"
```

Use a local trained brain when required:

```bash
python3 Aaron_Sound_Sorter.py sort \
  "/path/to/samples" "/path/to/output" \
  --brain "/path/to/stage4_folder_brain.json"
```

With local CLAP assets installed, CLI sorting also uses neural evidence.
Use `--no-neural` only for diagnostics.

## Train with the GUI

```bash
./commands/gui/RUN_SORTER_GUI.command
```

1. Preview a folder or ZIP.
2. Listen to Review items.
3. Choose the narrowest correct category.
4. Apply training.
5. Re-run Preview.

Large folders stay usable: the queue shows 200 sounds at a time and receives
only new results. **Stop & Keep Results** ends new work but keeps completed
sounds reviewable; **Cancel** abandons the run.

Approved corrections enter a source-blind training inbox. CLAP prototypes
rebuild first; temporary legacy memories refresh second.

The selected-file panel gives one plain-English decision. CLAP, PANNs, voter
traces, and alternatives stay collapsed unless you want them. A relabel that
conflicts with a locked seed must be chosen twice.

Open **Learning Center** for batch training:

1. Load a generated cluster review pack.
2. Listen to its center, typical, boundary, and outlier examples.
3. Approve the safe core.
4. Add that approval to the training inbox.
5. Press **Build Updated Neural Brain** once after the review session.

The coverage table keeps all 395 categories selectable while showing which
ones need more varied examples. Confidence learning uses accepted and corrected
reviews, but it cannot enable automatic ownership until independent held-out
gates pass.

CLAP and PANNs weights are downloaded separately and remain frozen. Your
corrections teach Aaron's local memory; they do not retrain the giant model.

## Categorization advice

- Decide structure first: loop, one-shot, phrase, or long effect.
- Decide family next: drum, instrument, voice, texture, or FX.
- Use Review when two families remain plausible.
- Train varied examples, not duplicates of one sound.
- Keep validation audio out of training.
- Ignore names and pack folders. Listen to the bytes.

See the concise [audio sorting guide](docs/AUDIO_SORTING_GUIDE.md).

## Development

```bash
make ai-preflight
make ai-check
make test
```

Use `make test`, not a random system `pytest`; the Makefile selects the project
environment and checks required audio dependencies. Read [AGENTS.md](AGENTS.md)
before AI-assisted changes.

## Status

- CLI and GUI: working with local trained assets.
- CLAP: pretrained encoder plus local prototypes; generalized authority is data-gated.
- PANNs: pretrained broad-event support/contradiction; never a detailed-folder owner.
- Cross-family or structural conflict: Review.
- Legacy classifier: temporary fallback while held-out coverage grows.

See [current status](CURRENT_STATUS.md) and [documentation](docs/README.md).

## License

Personal, non-commercial evaluation only.

- No commercial use.
- No redistribution.
- No proprietary training assets or derived model data may be distributed.

See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md).
