from __future__ import annotations

from pathlib import Path

from aaron_sound_sorter import features


def test_third_party_feature_profile_times_out_without_crashing(monkeypatch, tmp_path):
    sample = tmp_path / "sample.wav"
    sample.write_bytes(b"not used")

    def slow_profile(path: Path):
        raise features.ThirdPartyFeatureTimeout("forced")

    monkeypatch.setenv("AARON_THIRD_PARTY_TIMEOUT_SECONDS", "1")
    monkeypatch.setattr(features, "_third_party_feature_profile_unbounded", slow_profile)

    profile = features.third_party_feature_profile(sample)

    assert profile["status"] == "third_party_feature_timeout:1s"
    assert profile["flat"] == {}
    assert profile["adapters"]["librosa"]["status"] == "timeout:1s"


def test_third_party_feature_profile_can_be_unbounded_for_worker_threads(monkeypatch, tmp_path):
    sample = tmp_path / "sample.wav"
    sample.write_bytes(b"not used")

    monkeypatch.setenv("AARON_THIRD_PARTY_TIMEOUT_SECONDS", "0")
    monkeypatch.setattr(
        features,
        "_third_party_feature_profile_unbounded",
        lambda path: {"status": "ok", "adapters": {}, "flat": {"x": 1.0}},
    )

    assert features.third_party_feature_profile(sample)["flat"]["x"] == 1.0


def test_third_party_feature_profile_can_be_disabled_by_environment(monkeypatch, tmp_path):
    sample = tmp_path / "sample.wav"
    sample.write_bytes(b"not used")

    monkeypatch.setenv("AARON_DISABLE_THIRD_PARTY_FEATURES", "1")

    profile = features.third_party_feature_profile(sample)

    assert profile["status"] == "third_party_feature_disabled_by_environment"
    assert profile["adapters"]["librosa"]["status"] == "disabled_by_environment"
