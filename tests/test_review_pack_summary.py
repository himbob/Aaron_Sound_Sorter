from __future__ import annotations

from tools.summarize_cluster_review_packs import prioritize_rows


def test_review_priority_favors_sparse_unknown_boundary_audio() -> None:
    rows = [
        {
            "cluster_id": "Cluster_0001",
            "review_file": "members/audio_hash.wav",
            "file_sha256": "a" * 64,
            "membership": "boundary",
            "structure_bucket": "Loops",
            "broad_measured_family": "Tonal Instrument",
            "suggested_category": "Instruments/Synths/Synth Pad/Loops",
            "known_distribution": "0",
            "margin": "0.02",
        },
        {
            "cluster_id": "Cluster_0002",
            "review_file": "members/audio_hash_2.wav",
            "file_sha256": "b" * 64,
            "membership": "safe_core",
            "structure_bucket": "Loops",
            "broad_measured_family": "Tonal Instrument",
            "suggested_category": "Instruments/Keys/Piano/Loops",
            "known_distribution": "1",
            "margin": "0.30",
        },
    ]

    prioritized = prioritize_rows(
        "pack",
        rows,
        {"Instruments/Synths/Synth Pad/Loops": 0, "Instruments/Keys/Piano/Loops": 8},
    )

    assert float(prioritized[0]["priority_score"]) > float(prioritized[1]["priority_score"])
    assert prioritized[0]["examples_needed_for_generalization"] == "3"
    assert prioritized[1]["examples_needed_for_generalization"] == "0"
    assert "locator" not in prioritized[0]
