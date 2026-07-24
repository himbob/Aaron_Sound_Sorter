"""Write Aaron Audio Intelligence sidecars from sorter results."""

from __future__ import annotations

import json
from pathlib import Path

from aaron_audio_intelligence.sidecar_schema import build_sidecar_from_sort_result
from aaron_sound_sorter.domain.models import SortFileResult

AUDIO_INTELLIGENCE_SIDECAR_NAME = "Aaron_Audio_Intelligence_Sidecars.jsonl"


def write_audio_intelligence_sidecars(
    path: Path,
    file_results: list[SortFileResult],
    *,
    debug_packet_name: str = "Aaron_Sorted_Sounds_debug_packets.jsonl",
) -> None:
    """Write one versioned audio-intelligence sidecar row per sorted file.

    Args:
        path: Destination JSONL path.
        file_results: Sorter results to serialize.
        debug_packet_name: Optional debug packet sidecar filename. Empty means
            the full debug packet was intentionally not written.

    Side Effects:
        Creates parent directories and writes UTF-8 JSON Lines. This does not
        change final folders, train brains, or mutate sorter decisions.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index, result in enumerate(file_results, start=1):
            debug_packet_ref = f"{debug_packet_name}:{index}" if debug_packet_name else ""
            sidecar = build_sidecar_from_sort_result(
                result,
                analysis_version="sorter_read_only_sidecar_v1",
                debug_packet_ref=debug_packet_ref,
            )
            handle.write(json.dumps(sidecar.as_dict(), sort_keys=True, default=str) + "\n")
