# Aaron Sound Sorter

Source-blind audio sample sorting for music producers.

Give it a ZIP, folder, or audio file. It builds a producer-friendly library,
keeps uncertain sounds in `_TO_REVIEW`, and records why each decision happened.
`definitely_a_kick_FINAL_9.wav` gets no special treatment.

## What it does

- Sorts Drums, Instruments, Voice, Textures, and FX.
- Separates loops, one-shots, and long effects.
- Uses decoded audio—not filenames or paths—as evidence.
- Lets approved GUI corrections teach versioned CLAP prototypes.
- Uses neural ownership only inside learned neighborhoods.
- Sends uncertain or conflicting evidence to Review.

## Install

Requirements: Python 3.9+, macOS or Linux, and `libsndfile`.

```bash
git clone https://github.com/himbob/Aaron_Sound_Sorter.git
cd Aaron_Sound_Sorter
make bootstrap
.venv_phase4/bin/python Aaron_Sound_Sorter.py self-test
```

Optional local CLAP support:

```bash
./commands/neural/INSTALL_NEURAL_LAB.command
./commands/neural/PREFETCH_CLAP_MODEL.command
```

Model weights, trained brains, training audio, and private evidence are not
included.

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

## Train with the GUI

```bash
./commands/gui/RUN_SORTER_GUI.command
```

1. Preview a folder or ZIP.
2. Listen to Review items.
3. Choose the narrowest correct category.
4. Apply training.
5. Re-run Preview.

Approved corrections enter a source-blind training inbox. CLAP prototypes
rebuild first; temporary legacy memories refresh second.

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
- CLAP: limited GUI ownership for known learned neighborhoods.
- Cross-family or structural conflict: Review.
- Legacy classifier: temporary fallback while held-out coverage grows.

See [current status](CURRENT_STATUS.md) and [documentation](docs/README.md).

## License

Personal, non-commercial evaluation only.

- No commercial use.
- No redistribution.
- No proprietary training assets or derived model data may be distributed.

See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md).
