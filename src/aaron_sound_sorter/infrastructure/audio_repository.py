"""Audio input preparation for the product sorter."""

from __future__ import annotations

import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

from aaron_sound_sorter.core import AUDIO_EXTS


@dataclass(frozen=True)
class PreparedAudioInput:
    """Prepared audio files for one sort request."""

    source_root: Path
    audio_files: list[Path]
    cleanup_root: Path | None = None


class AudioInputRepository:
    """Prepare folder, single audio file, or ZIP input for sorting."""

    def prepare(self, input_path: Path, output_dir: Path) -> PreparedAudioInput:
        """Return a safe list of audio files to sort."""
        resolved = Path(input_path).expanduser().resolve()
        if resolved.is_dir():
            files = sorted(path for path in resolved.rglob("*") if is_audio_file(path))
            if not files:
                raise ValueError(f"No audio files found under folder input: {resolved}")
            return PreparedAudioInput(source_root=resolved, audio_files=files)
        if resolved.is_file() and is_audio_file(resolved):
            return PreparedAudioInput(source_root=resolved.parent, audio_files=[resolved])
        if resolved.is_file() and resolved.suffix.lower() == ".zip":
            extract_root = self.extract_zip(resolved, output_dir)
            files = sorted(path for path in extract_root.rglob("*") if is_audio_file(path))
            if not files:
                raise ValueError(f"No audio files found inside ZIP input: {resolved}")
            return PreparedAudioInput(source_root=extract_root, audio_files=files, cleanup_root=None)
        raise ValueError(f"Expected audio file, ZIP, or folder input: {resolved}")

    def extract_zip(self, zip_path: Path, output_dir: Path) -> Path:
        """Extract safe audio members from ZIP into output-local cache."""
        cache_root = Path(output_dir).expanduser().resolve() / "_source_cache" / zip_path.stem
        if cache_root.exists():
            shutil.rmtree(cache_root)
        cache_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                parts = safe_audio_parts(info.filename)
                if not parts:
                    continue
                target = cache_root.joinpath(*parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info, "r") as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination, length=1024 * 1024)
        return cache_root


def is_audio_file(path: Path) -> bool:
    """Return true for supported audio files only."""
    if any(part == "__MACOSX" for part in path.parts):
        return False
    if path.name.startswith("._") or path.name.startswith("."):
        return False
    return path.suffix.lower() in AUDIO_EXTS


def safe_audio_parts(raw_name: str) -> list[str]:
    """Return safe ZIP path parts for an audio member."""
    name = str(raw_name).replace("\\", "/")
    parts = [part for part in name.split("/") if part]
    if not parts or any(part in {"..", ""} for part in parts):
        return []
    if any(part == "__MACOSX" or part.startswith("._") or part.startswith(".") for part in parts):
        return []
    if Path(parts[-1]).suffix.lower() not in AUDIO_EXTS:
        return []
    return parts
