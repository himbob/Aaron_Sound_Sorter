from tools.neural_vocal_experiment_report import build_report_rows, summarize_rows, summarize_supplement


def test_vocal_experiment_summary_separates_recall_false_positives_and_ood() -> None:
    vocal_hash = "1" * 64
    negative_hash = "2" * 64
    manifest = {
        vocal_hash: {"label": "Voice/Musical Vocal", "subgroup": "Loop", "provenance": "pack"},
        negative_hash: {"label": "NonVoice/Sax", "subgroup": "Sax", "provenance": "protected"},
    }
    predictions = {
        vocal_hash: {
            "expected_label": "Voice/Musical Vocal",
            "predicted_label": "Voice/Musical Vocal",
            "second_label": "NonVoice/Sax",
            "top_similarity": "0.8",
            "second_similarity": "0.6",
            "margin": "0.2",
            "known_distribution": "0",
            "radius_ratio": "1.3",
        },
        negative_hash: {
            "expected_label": "NonVoice/Sax",
            "predicted_label": "Voice/Musical Vocal",
            "second_label": "NonVoice/Sax",
            "top_similarity": "0.7",
            "second_similarity": "0.6",
            "margin": "0.1",
            "known_distribution": "1",
            "radius_ratio": "0.8",
        },
    }
    legacy = {vocal_hash: "FX/Human and Voice FX/Altered Voice/Long FX"}

    rows = build_report_rows(
        manifest,
        predictions,
        legacy,
        voice_label="Voice/Musical Vocal",
    )
    summary = summarize_rows(rows, voice_label="Voice/Musical Vocal")

    assert summary["heldout_vocal_recall"] == 1.0
    assert summary["heldout_vocal_review_rate"] == 1.0
    assert summary["confusing_negative_false_positive_rate"] == 1.0
    assert summary["confusing_negative_known_false_positive_rate"] == 1.0
    assert summary["legacy_voice_family_recall"] == 0.0
    assert summary["legacy_vs_clap_broad_family"] == {"clap_only": 1}


def test_supplement_reports_known_voice_claims_separately() -> None:
    digest = "3" * 64
    manifest = {digest: {"group": "Spoken Voice", "legacy_label": "FX/Human/Spoken"}}
    shadow = {
        digest: {
            "neural_label": "Voice/Musical Vocal",
            "neural_second_label": "NonVoice/Other",
            "neural_similarity": "0.7",
            "neural_margin": "0.2",
            "neural_radius_ratio": "0.7",
            "neural_known_distribution": "1",
        }
    }

    _, summary = summarize_supplement(
        manifest,
        shadow,
        voice_label="Voice/Musical Vocal",
    )

    assert summary["voice_claim_rate"] == 1.0
    assert summary["known_voice_claim_rate"] == 1.0
