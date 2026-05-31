from __future__ import annotations

from aaron_sound_sorter.domain.physics_category_panels import (
    CategoryCalibrationCurve,
    build_category_panel_scores,
    calibrate_category_panel_score,
)


def test_calibrate_category_panel_score_uses_percentile_curve() -> None:
    curve = CategoryCalibrationCurve(raw_p10=0.2, raw_p50=0.5, raw_p90=0.8)
    calibration = {"example_score": curve}

    assert calibrate_category_panel_score("example_score", 0.2, calibration) == 0.1
    assert calibrate_category_panel_score("example_score", 0.5, calibration) == 0.5
    assert calibrate_category_panel_score("example_score", 0.8, calibration) == 0.9


def test_build_category_panel_scores_reports_calibration_status() -> None:
    output = build_category_panel_scores(
        feature_values={
            "tail_energy_ratio": 0.2,
            "log_transient_count": 0.0,
            "spectral_flatness_mean": 0.1,
            "spectral_entropy_mean": 0.2,
        },
        source_scores={"drum_kick_source_score": 0.8, "role_one_shot_score": 0.9},
        calibration_profile={
            "drums_kick_drums_generic_kick_one_shots_score": CategoryCalibrationCurve(
                raw_p10=0.2,
                raw_p50=0.5,
                raw_p90=0.8,
            )
        },
    )

    coverage = output["coverage"]
    assert coverage["calibration_status"] == "loaded"
    assert coverage["calibration_curve_count"] == 1
    assert coverage["missing_category_panels"] == []
