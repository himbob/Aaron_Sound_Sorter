#!/usr/bin/env python3
"""Split the complete Aaron Sound Sorter payload into balanced upload-safe ZIP parts.

The input code and AI-runtime archives are already compressed. This tool does
not recompress their contents; it stores balanced byte chunks inside clearly
named ZIP wrappers. After all wrapper ZIPs are extracted into one directory,
``ASSEMBLE_AND_INSTALL.command`` verifies, reconstructs, and installs the code
and runtime archives into the same ``Aaron_Sound_Sorter`` folder.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

MIB = 1024 * 1024
WRAPPER_OVERHEAD_RESERVE = 1 * MIB
COPY_BLOCK_SIZE = 4 * MIB


@dataclass(frozen=True)
class UploadPackResult:
    """Result of building one balanced multipart upload pack."""

    part_paths: tuple[Path, ...]
    part_count: int
    balanced_payload_bytes: int


def human_size(byte_count: int) -> str:
    """Return a compact binary file-size string."""
    value = float(byte_count)
    for suffix in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or suffix == "TiB":
            return f"{value:.1f} {suffix}"
        value /= 1024.0
    return f"{byte_count} B"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while block := file_handle.read(COPY_BLOCK_SIZE):
            digest.update(block)
    return digest.hexdigest()


def safe_stem(raw_name: str) -> str:
    """Return a filesystem-safe archive stem."""
    cleaned = "".join(character if character.isalnum() or character in "-_." else "_" for character in raw_name)
    cleaned = cleaned.strip("._-")
    if not cleaned:
        raise ValueError("pack name cannot be empty")
    return cleaned


def calculate_balanced_layout(total_payload_bytes: int, max_part_bytes: int) -> tuple[int, int]:
    """Return part count and near-equal payload bytes per wrapper.

    A small reserve is kept for ZIP metadata, the README, checksum file, and
    installer script so each finished wrapper remains below the requested limit.
    """
    usable_part_bytes = max_part_bytes - WRAPPER_OVERHEAD_RESERVE
    if usable_part_bytes <= 0:
        raise ValueError("max part size must exceed 1 MiB")
    part_count = max(1, math.ceil(total_payload_bytes / usable_part_bytes))
    balanced_payload_bytes = math.ceil(total_payload_bytes / part_count)
    return part_count, balanced_payload_bytes


def installer_script(
    *,
    code_archive_name: str,
    runtime_archive_name: str,
    runtime_sha256: str,
    code_sha256: str,
    part_count: int,
) -> str:
    """Return a portable macOS/Linux assembly and installation script."""
    return f"""#!/usr/bin/env bash
set -euo pipefail

PACK_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
PAYLOAD_DIR="$PACK_DIR/payload"
DESTINATION="${{1:-$(cd "$PACK_DIR/.." && pwd)}}"
CODE_ARCHIVE="$PACK_DIR/{code_archive_name}"
RUNTIME_ARCHIVE="$PACK_DIR/{runtime_archive_name}"
EXPECTED_PARTS={part_count}

fail() {{
  echo "ERROR: $*" >&2
  exit 1
}}

[[ -f "$CODE_ARCHIVE" ]] || fail "Missing code archive: $CODE_ARCHIVE"

part_files=("$PAYLOAD_DIR/{runtime_archive_name}.part-"*)
[[ ${{#part_files[@]}} -eq $EXPECTED_PARTS ]] || fail "Expected $EXPECTED_PARTS runtime chunks, found ${{#part_files[@]}}. Extract every pack part into the same folder first."

cat "${{part_files[@]}}" > "$RUNTIME_ARCHIVE"

sha256_value() {{
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{{print $1}}'
  elif command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{{print $1}}'
  else
    fail "Neither shasum nor sha256sum is installed."
  fi
}}

[[ "$(sha256_value "$CODE_ARCHIVE")" == "{code_sha256}" ]] || fail "Code archive checksum failed."
[[ "$(sha256_value "$RUNTIME_ARCHIVE")" == "{runtime_sha256}" ]] || fail "AI runtime checksum failed. One or more pack parts are missing or damaged."

mkdir -p "$DESTINATION"
unzip -q -o "$CODE_ARCHIVE" -d "$DESTINATION"
unzip -q -o "$RUNTIME_ARCHIVE" -d "$DESTINATION"

echo
echo "Aaron Sound Sorter installed into:"
echo "  $DESTINATION/Aaron_Sound_Sorter"
echo
echo "Next:"
echo "  cd \"$DESTINATION/Aaron_Sound_Sorter\""
echo "  python3 Aaron_Sound_Sorter.py self-test"
"""


def readme_text(*, pack_name: str, part_count: int, max_part_mib: int) -> str:
    """Return concise end-user assembly instructions."""
    return f"""Aaron Sound Sorter end-user pack
=================================

Pack: {pack_name}
Parts: {part_count}
Maximum wrapper size: {max_part_mib} MiB

1. Put every ZIP named `{pack_name}_Part_XX_of_{part_count:02d}.zip` in the same folder.
2. Extract every part ZIP into that same folder.
3. Open the merged `{pack_name}` folder.
4. Double-click `ASSEMBLE_AND_INSTALL.command`.

All parts belong to one pack. No single part is a complete product by itself.
The installer verifies SHA-256 checksums before extracting the code and AI runtime.
"""


def write_stored_member(
    archive: zipfile.ZipFile, source_path: Path, archive_name: str, *, executable: bool = False
) -> None:
    """Store one file without trying to recompress already-compressed payloads."""
    info = zipfile.ZipInfo(archive_name)
    info.compress_type = zipfile.ZIP_STORED
    if executable:
        info.external_attr = (0o100755 & 0xFFFF) << 16
    else:
        info.external_attr = (0o100644 & 0xFFFF) << 16
    with source_path.open("rb") as source_handle, archive.open(info, "w") as target_handle:
        while block := source_handle.read(COPY_BLOCK_SIZE):
            target_handle.write(block)


def write_text_member(archive: zipfile.ZipFile, archive_name: str, text: str, *, executable: bool = False) -> None:
    """Store one UTF-8 text member."""
    info = zipfile.ZipInfo(archive_name)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = ((0o100755 if executable else 0o100644) & 0xFFFF) << 16
    archive.writestr(info, text.encode("utf-8"), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def build_balanced_upload_pack(
    *,
    code_archive: Path,
    runtime_archive: Path,
    output_dir: Path,
    pack_name: str,
    max_part_mib: int,
    dry_run: bool = False,
) -> UploadPackResult:
    """Build balanced, independently uploadable ZIP wrappers for one product pack."""
    code_archive = code_archive.expanduser().resolve()
    runtime_archive = runtime_archive.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    pack_name = safe_stem(pack_name)

    if not code_archive.is_file():
        raise SystemExit(f"Code archive is missing: {code_archive}")
    if not runtime_archive.is_file():
        raise SystemExit(f"AI runtime archive is missing: {runtime_archive}")
    if max_part_mib < 2:
        raise SystemExit("Maximum part size must be at least 2 MiB.")

    code_size = code_archive.stat().st_size
    runtime_size = runtime_archive.stat().st_size
    total_payload = code_size + runtime_size
    max_part_bytes = max_part_mib * MIB
    part_count, balanced_payload_bytes = calculate_balanced_layout(total_payload, max_part_bytes)

    print("Balanced end-user upload pack:")
    print(f"  code archive: {human_size(code_size)}")
    print(f"  AI runtime archive: {human_size(runtime_size)}")
    print(f"  parts: {part_count}")
    print(f"  target payload per part: {human_size(balanced_payload_bytes)}")
    print(f"  maximum finished part: {max_part_mib} MiB")

    part_paths = tuple(
        output_dir / f"{pack_name}_Part_{part_number:02d}_of_{part_count:02d}.zip"
        for part_number in range(1, part_count + 1)
    )
    if dry_run:
        for part_path in part_paths:
            print(f"Dry run: would write {part_path}")
        return UploadPackResult(part_paths, part_count, balanced_payload_bytes)

    output_dir.mkdir(parents=True, exist_ok=True)
    code_sha256 = sha256_file(code_archive)
    runtime_sha256 = sha256_file(runtime_archive)
    readme = readme_text(pack_name=pack_name, part_count=part_count, max_part_mib=max_part_mib)
    installer = installer_script(
        code_archive_name=code_archive.name,
        runtime_archive_name=runtime_archive.name,
        runtime_sha256=runtime_sha256,
        code_sha256=code_sha256,
        part_count=part_count,
    )
    checksums = f"{code_sha256}  {code_archive.name}\n{runtime_sha256}  {runtime_archive.name}\n"

    with tempfile.TemporaryDirectory(prefix="aaron_balanced_pack_") as temporary_directory:
        temporary_root = Path(temporary_directory)
        remaining_runtime = runtime_size
        with runtime_archive.open("rb") as runtime_handle:
            for part_number, part_path in enumerate(part_paths, start=1):
                desired_payload = balanced_payload_bytes
                if part_number == 1:
                    desired_payload = max(0, desired_payload - code_size)
                runtime_chunk_bytes = min(remaining_runtime, desired_payload)
                if part_number == part_count:
                    runtime_chunk_bytes = remaining_runtime

                chunk_name = f"{runtime_archive.name}.part-{part_number:02d}-of-{part_count:02d}"
                chunk_path = temporary_root / chunk_name
                with chunk_path.open("wb") as chunk_handle:
                    bytes_to_copy = runtime_chunk_bytes
                    while bytes_to_copy > 0:
                        block = runtime_handle.read(min(COPY_BLOCK_SIZE, bytes_to_copy))
                        if not block:
                            raise SystemExit("Unexpected end of AI runtime archive while splitting pack.")
                        chunk_handle.write(block)
                        bytes_to_copy -= len(block)
                remaining_runtime -= runtime_chunk_bytes

                if part_path.exists():
                    part_path.unlink()
                with zipfile.ZipFile(part_path, "w", allowZip64=True) as wrapper:
                    root = f"{pack_name}/"
                    write_text_member(wrapper, f"{root}README_FIRST.txt", readme)
                    write_text_member(wrapper, f"{root}SHA256SUMS.txt", checksums)
                    write_text_member(
                        wrapper,
                        f"{root}ASSEMBLE_AND_INSTALL.command",
                        installer,
                        executable=True,
                    )
                    if part_number == 1:
                        write_stored_member(wrapper, code_archive, f"{root}{code_archive.name}")
                    write_stored_member(wrapper, chunk_path, f"{root}payload/{chunk_name}")

                finished_size = part_path.stat().st_size
                if finished_size > max_part_bytes:
                    part_path.unlink(missing_ok=True)
                    raise SystemExit(
                        f"Finished pack part exceeded {max_part_mib} MiB: {part_path.name} ({human_size(finished_size)})"
                    )
                print(f"Wrote: {part_path.name} ({human_size(finished_size)})")
                chunk_path.unlink(missing_ok=True)

    if remaining_runtime != 0:
        raise SystemExit(f"AI runtime split ended with {remaining_runtime} bytes remaining.")
    return UploadPackResult(part_paths, part_count, balanced_payload_bytes)


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-archive", required=True)
    parser.add_argument("--runtime-archive", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pack-name", required=True)
    parser.add_argument("--max-part-mib", type=int, default=int(os.environ.get("BUNDLE_UPLOAD_PART_MAX_MIB", "490")))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    """Run the balanced upload-pack builder."""
    arguments = build_parser().parse_args()
    build_balanced_upload_pack(
        code_archive=Path(arguments.code_archive),
        runtime_archive=Path(arguments.runtime_archive),
        output_dir=Path(arguments.output_dir),
        pack_name=str(arguments.pack_name),
        max_part_mib=int(arguments.max_part_mib),
        dry_run=bool(arguments.dry_run),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
