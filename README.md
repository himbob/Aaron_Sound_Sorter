# Aaron Sound Sorter

Source-blind audio sample sorting for music producers.

Give it a ZIP, folder, or audio file. It builds a cleaner library, keeps
uncertain sounds in `_TO_REVIEW`, and writes evidence for every decision.
Because a folder named `final_samples_USE_THIS_ONE_7` is not a filing system.

## What it does

- Sorts drums, instruments, textures, and FX.
- Separates loops, one-shots, and long FX.
- Learns from approved GUI corrections.
- Uses measured audio—not filenames—as evidence.
- Produces manifests, summaries, and an optional ZIP.

## Project status

- CLI and GUI: working with local trained brains.
- Neural CLAP/MERT pipeline: real, but shadow-only.
- Production neural ownership: not enabled yet.
- Trained brains, sample libraries, model weights, and private evidence: not
  distributed.

See [current status](CURRENT_STATUS.md) and [documentation](docs/README.md).

## Quick start

Requirements:

- Python 3.9+
- macOS or Linux
- `libsndfile` where required by `soundfile`

```bash
git clone https://github.com/himbob/Aaron_Sound_Sorter.git
cd Aaron_Sound_Sorter
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python Aaron_Sound_Sorter.py self-test
```

## Sort samples

A trained brain is required for production sorting. It is intentionally not
included in the public repository.

```bash
python Aaron_Sound_Sorter.py sort \
  "/path/to/samples.zip" \
  "/path/to/output" \
  --brain "/path/to/stage4_folder_brain.json"
```

Expected output:

```text
Aaron_Sorted_Sounds/
Aaron_Sorted_Sounds_manifest.csv
Aaron_Sorted_Sounds_summary.txt
Aaron_Sorted_Sounds.zip
```

## GUI

```bash
./commands/gui/RUN_SORTER_GUI.command
```

GUI corrections teach dedicated user, role, physics, and shape memories.

## Development

```bash
python -m pip install -e ".[dev]"
make ai-preflight
make ai-check
make test
```

Read [AGENTS.md](AGENTS.md) before AI-assisted changes.

## Data and model policy

- Do not commit sample audio, trained brains, model weights, or private paths.
- Optional neural models are downloaded separately.
- Third-party assets keep their original licenses.
- Filename and folder text must never influence runtime classification.

## License

Source-available for personal, non-commercial evaluation only.

- No commercial use.
- No redistribution.
- No proprietary training assets or derived model data may be distributed.

See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md).
