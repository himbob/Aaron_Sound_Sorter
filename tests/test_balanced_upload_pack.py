from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

from tools.build_balanced_upload_pack import build_balanced_upload_pack


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def make_zip(path: Path, member_name: str, payload: bytes) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(member_name, payload)


def test_balanced_pack_parts_are_named_bounded_and_reconstruct_runtime(tmp_path: Path) -> None:
    code_archive = tmp_path / "Aaron_Sound_Sorter_Code_Only_test.zip"
    runtime_archive = tmp_path / "Aaron_Sound_Sorter_AI_Runtime_test.zip"
    make_zip(code_archive, "Aaron_Sound_Sorter/src/app.py", b"code" * 200_000)
    make_zip(runtime_archive, "Aaron_Sound_Sorter/_models/model.bin", bytes(range(256)) * 32_000)

    output_dir = tmp_path / "output"
    result = build_balanced_upload_pack(
        code_archive=code_archive,
        runtime_archive=runtime_archive,
        output_dir=output_dir,
        pack_name="Aaron_Sound_Sorter_End_User_Pack_test",
        max_part_mib=4,
    )

    assert result.part_count == 3
    assert [path.name for path in result.part_paths] == [
        "Aaron_Sound_Sorter_End_User_Pack_test_Part_01_of_03.zip",
        "Aaron_Sound_Sorter_End_User_Pack_test_Part_02_of_03.zip",
        "Aaron_Sound_Sorter_End_User_Pack_test_Part_03_of_03.zip",
    ]
    assert all(path.stat().st_size < 4 * 1024 * 1024 for path in result.part_paths)
    assert max(path.stat().st_size for path in result.part_paths) - min(
        path.stat().st_size for path in result.part_paths
    ) < 256 * 1024

    extracted = tmp_path / "extracted"
    for part_path in result.part_paths:
        with zipfile.ZipFile(part_path) as archive:
            archive.extractall(extracted)

    pack_root = extracted / "Aaron_Sound_Sorter_End_User_Pack_test"
    assert (pack_root / code_archive.name).is_file()
    assert (pack_root / "ASSEMBLE_AND_INSTALL.command").is_file()
    assert (pack_root / "README_FIRST.txt").is_file()

    chunks = sorted((pack_root / "payload").glob(f"{runtime_archive.name}.part-*"))
    reconstructed = tmp_path / "reconstructed.zip"
    with reconstructed.open("wb") as target:
        for chunk in chunks:
            target.write(chunk.read_bytes())
    assert sha256(reconstructed) == sha256(runtime_archive)
