from __future__ import annotations

import csv
import os
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import aaron_training_reseed_manager as mgr


def make_wav(path: Path, seconds: float = 0.5, sr: int = 8000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = max(1, int(seconds * sr))
    t = np.arange(n, dtype=np.float32) / sr
    y = 0.25 * np.sin(2 * np.pi * 440 * t)
    ints = np.clip(y * 32767, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes(ints.tobytes())


def test_discover_training_slots_counts_long_fx(tmp_path: Path) -> None:
    root = tmp_path / "training"
    (root / "FX" / "Risers" / "Generic Riser" / "_LONG_FX").mkdir(parents=True)
    (root / "FX" / "Risers" / "Generic Riser" / "_ONE_SHOTS").mkdir(parents=True)
    slots = mgr.discover_training_slots(root)
    assert {slot.structure for slot in slots} == {"_LONG_FX", "_ONE_SHOTS"}


def test_electric_piano_is_not_clean_piano() -> None:
    text = mgr.text_tokens("Dmin140BPMElectricFryPiano.wav")
    entries = mgr.all_taxonomy_entries(Path("/no/project"))
    piano = next(e for e in entries if str(e.label_rel) == "Instruments/Keys/Piano")
    electric = next(e for e in entries if str(e.label_rel) == "Instruments/Keys/Electric Piano")
    assert mgr.entry_matches_text(electric, text)[0] is True
    assert mgr.entry_matches_text(piano, text)[0] is False


def test_read_wav_stats_handles_tiny_wav_without_transient_reshape_crash(tmp_path: Path) -> None:
    tiny = tmp_path / "tiny.wav"
    make_wav(tiny, seconds=0.026875, sr=8000)
    stats = mgr.read_wav_stats(tiny)
    assert stats.readable is True
    assert stats.reason == "ok"
    assert stats.duration_sec > 0.0
    assert stats.transient_ratio == 0.0


def test_structure_policy_covers_every_non_fx_category_with_loops_and_one_shots() -> None:
    expected = ("_LOOPS", "_ONE_SHOTS")
    assert mgr.structures_for_label(Path("Drums/Kick Drums/Short Kick")) == expected
    assert mgr.structures_for_label(Path("Drums/Cymbals/Generic Cymbal")) == expected
    assert mgr.structures_for_label(Path("Drums/Cymbals/Splash Cymbal")) == expected
    assert mgr.structures_for_label(Path("Drums/Claps Snaps Slaps/Snap")) == expected
    assert mgr.structures_for_label(Path("Drums/Drum Loops/Full Drum Loops")) == expected
    assert mgr.structures_for_label(Path("Instruments/Guitar/Guitar Loops")) == expected
    assert mgr.structures_for_label(Path("Instruments/Keys/Keys Loops")) == expected
    assert mgr.structures_for_label(Path("Instruments/Mixed Musical Loops/Multi Instrument")) == expected
    assert mgr.structures_for_label(Path("FX/Crashes and Breaks/Glass and Metal")) == ("_ONE_SHOTS", "_LONG_FX")
    assert "_LOOPS" not in mgr.structures_for_label(Path("FX/Crashes and Breaks/Glass and Metal"))


def test_phrase_guards_block_exact_bad_matches() -> None:
    entries = mgr.all_taxonomy_entries(Path("/no/project"))

    drum_roll = next(e for e in entries if str(e.label_rel) == "Drums/Drum Fills and Rolls/Drum Roll")
    splash = next(e for e in entries if str(e.label_rel) == "Drums/Cymbals/Splash Cymbal")
    glass_metal = next(e for e in entries if str(e.label_rel) == "FX/Crashes and Breaks/Glass and Metal")
    keys_loops = next(e for e in entries if str(e.label_rel) == "Instruments/Keys/Keys Loops")
    string_loops = next(e for e in entries if str(e.label_rel) == "Instruments/Strings/String Loops")
    vibraphone = next(e for e in entries if str(e.label_rel) == "Instruments/Mallets and Bells/Vibraphone")

    assert mgr.entry_matches_text(drum_roll, mgr.text_tokens("OrganRoll.wav"))[0] is False
    assert mgr.entry_matches_text(splash, mgr.text_tokens("splash_and_splatter_train_112152.wav"))[0] is False
    assert mgr.entry_matches_text(glass_metal, mgr.text_tokens("Dive Break_88bpm.wav"))[0] is False
    assert mgr.entry_matches_text(keys_loops, mgr.text_tokens("Acoustic Loop 3_keyAbmin_125bpm.wav"))[0] is False
    assert (
        mgr.entry_matches_text(string_loops, mgr.text_tokens("Nylon String Guitar Rhythm_53_KeyAm_85bpm.wav"))[0]
        is False
    )
    assert mgr.entry_matches_text(vibraphone, mgr.text_tokens("JUST_VIBE.wav"))[0] is False

    assert mgr.entry_matches_text(drum_roll, mgr.text_tokens("Snare Roll 90bpm.wav"))[0] is True
    assert mgr.entry_matches_text(splash, mgr.text_tokens("Splash Cymbal.wav"))[0] is True
    assert mgr.entry_matches_text(glass_metal, mgr.text_tokens("Glass Break.wav"))[0] is True
    assert mgr.entry_matches_text(keys_loops, mgr.text_tokens("Keys Loop 90bpm.wav"))[0] is True
    assert mgr.entry_matches_text(string_loops, mgr.text_tokens("String Section Loop 90bpm.wav"))[0] is True
    assert mgr.entry_matches_text(vibraphone, mgr.text_tokens("Vibraphone.wav"))[0] is True


def test_safe8_specific_names_do_not_match_generic_siblings() -> None:
    entries = mgr.all_taxonomy_entries(Path("/no/project"))

    generic_hat = next(e for e in entries if str(e.label_rel) == "Drums/Hi Hats/Generic Hat")
    open_hat = next(e for e in entries if str(e.label_rel) == "Drums/Hi Hats/Open Hat")
    generic_tom = next(e for e in entries if str(e.label_rel) == "Drums/Toms/Generic Tom")
    high_tom = next(e for e in entries if str(e.label_rel) == "Drums/Toms/High Tom")
    mid_tom = next(e for e in entries if str(e.label_rel) == "Drums/Toms/Mid Tom")
    generic_cymbal = next(e for e in entries if str(e.label_rel) == "Drums/Cymbals/Generic Cymbal")
    crash_cymbal = next(e for e in entries if str(e.label_rel) == "Drums/Cymbals/Crash Cymbal")
    generic_bass = next(e for e in entries if str(e.label_rel) == "Instruments/Bass/Generic Bass")
    bass_808 = next(e for e in entries if str(e.label_rel) == "Instruments/Bass/808 Bass")

    assert mgr.entry_matches_text(open_hat, mgr.text_tokens("Open Hat 3.wav"))[0] is True
    assert mgr.entry_matches_text(generic_hat, mgr.text_tokens("Open Hat 3.wav"))[0] is False
    assert mgr.entry_matches_text(high_tom, mgr.text_tokens("High Tom 2.wav"))[0] is True
    assert mgr.entry_matches_text(mid_tom, mgr.text_tokens("Mid Tom 2.wav"))[0] is True
    assert mgr.entry_matches_text(generic_tom, mgr.text_tokens("High Tom 2.wav"))[0] is False
    assert mgr.entry_matches_text(generic_tom, mgr.text_tokens("Mid Tom 2.wav"))[0] is False
    assert mgr.entry_matches_text(crash_cymbal, mgr.text_tokens("Crash Cymbal 1.wav"))[0] is True
    assert mgr.entry_matches_text(generic_cymbal, mgr.text_tokens("Crash Cymbal 1.wav"))[0] is False
    assert mgr.entry_matches_text(bass_808, mgr.text_tokens("808 Bass.wav"))[0] is True
    assert mgr.entry_matches_text(generic_bass, mgr.text_tokens("808 Bass.wav"))[0] is False


def test_sync_taxonomy_creates_fx_long_fx_not_loops(tmp_path: Path) -> None:
    training = tmp_path / "training"
    samples = tmp_path / "samples"
    project = tmp_path / "project"
    make_wav(samples / "FX Pack" / "Huge Riser 128BPM.wav", seconds=3.0)
    args = type(
        "Args",
        (),
        {
            "training_root": str(training),
            "samples_root": str(samples),
            "project_dir": str(project),
            "ledger_json": "",
            "apply": True,
            "create_all": False,
        },
    )()
    assert mgr.sync_taxonomy(args) == 0
    assert (
        training / "FX" / "Structural and Transitional FX" / "Risers and Builds" / "Generic Riser" / "_LONG_FX"
    ).is_dir()
    assert not (
        training / "FX" / "Structural and Transitional FX" / "Risers and Builds" / "Generic Riser" / "_LOOPS"
    ).exists()


def test_sync_taxonomy_creates_complete_structure_rows(tmp_path: Path) -> None:
    training = tmp_path / "training"
    samples = tmp_path / "samples"
    project = tmp_path / "project"
    for name in [
        "wt_adm_55b_cymbal_crash.wav",
        "BanginSlowStorm Drum loop_124bpm.wav",
        "Guitar Loop 120bpm.wav",
        "Keys Loop 90bpm.wav",
        "Acoustic Loop 3_keyAbmin_125bpm.wav",
        "Huge Riser 128BPM.wav",
        "OrganRoll.wav",
        "splash_and_splatter_train_112152.wav",
        "Dive Break_88bpm.wav",
        "Dmin140BPMElectricFryPiano.wav",
    ]:
        make_wav(samples / name, seconds=3.0)
    args = type(
        "Args",
        (),
        {
            "training_root": str(training),
            "samples_root": str(samples),
            "project_dir": str(project),
            "ledger_json": "",
            "apply": False,
            "create_all": False,
        },
    )()
    assert mgr.sync_taxonomy(args) == 0
    manifest = sorted(project.rglob("taxonomy_sync_manifest.csv"))[-1]
    rows = list(csv.DictReader(manifest.open()))

    def rows_for(label: str, structure=None):
        return [
            row for row in rows if row["label_rel"] == label and (structure is None or row["structure"] == structure)
        ]

    assert rows_for("Drums/Cymbals/Generic Cymbal", "_LOOPS")
    assert rows_for("Drums/Cymbals/Splash Cymbal", "_LOOPS")
    assert rows_for("Drums/Drum Loops/Full Drum Loops", "_ONE_SHOTS")
    assert rows_for("Instruments/Guitar/Guitar Loops", "_ONE_SHOTS")
    assert rows_for("Instruments/Keys/Keys Loops", "_ONE_SHOTS")
    assert not any(row["family"] == "FX" and row["structure"] == "_LOOPS" for row in rows)

    assert rows_for("Drums/Drum Loops/Full Drum Loops", "_LOOPS")[0]["matching_sample_count"] == "1"
    assert rows_for("Drums/Drum Fills and Rolls/Drum Roll", "_ONE_SHOTS")[0]["matching_sample_count"] == "0"
    assert rows_for("Drums/Cymbals/Splash Cymbal", "_ONE_SHOTS")[0]["matching_sample_count"] == "0"
    assert rows_for("FX/Crashes and Breaks/Glass and Metal", "_LONG_FX")[0]["matching_sample_count"] == "0"
    assert rows_for("Instruments/Keys/Keys Loops", "_LOOPS")[0]["matching_sample_count"] == "1"
    assert rows_for("Instruments/Keys/Electric Piano", "_LOOPS")[0]["matching_sample_count"] == "1"
    assert rows_for("Instruments/Keys/Piano", "_LOOPS")[0]["matching_sample_count"] == "0"
    assert rows_for("Instruments/Keys/Piano", "_ONE_SHOTS")


def test_kick_stem_with_bpm_is_loop_not_one_shot_structure(tmp_path: Path) -> None:
    path = tmp_path / "100 BPM" / "Audio Files" / "1 Kick 100.wav"
    make_wav(path, seconds=4.8)
    stats = mgr.read_wav_stats(path)
    text = mgr.text_tokens(path)
    assert mgr.structure_guess(text, stats, "Drums") == {"_LOOPS"}
    assert mgr.structure_score(text, "_LOOPS", stats, "Drums")[0] > 0
    assert mgr.structure_score(text, "_ONE_SHOTS", stats, "Drums")[0] < 0


def test_candidate_source_path_excludes_generated_sorted_outputs(tmp_path: Path) -> None:
    samples = tmp_path / "samples"
    generated = samples / "Sorted samples" / "Drums" / "Kick Drums" / "Generic Kick" / "_ONE_SHOTS" / "1 Kick 100.wav"
    real_pack = samples / "Midi Pack" / "Loops" / "100 BPM" / "1 Kick 100.wav"
    make_wav(generated, seconds=4.8)
    make_wav(real_pack, seconds=4.8)
    assert mgr.is_candidate_source_path(generated, samples) is False
    assert mgr.is_candidate_source_path(real_pack, samples) is True
    records = mgr.sample_records(samples)
    assert [path for path, _ in records] == [real_pack]


def test_opaque_numeric_generic_percussive_file_has_no_specific_category_evidence(tmp_path: Path) -> None:
    samples = tmp_path / "samples"
    generic_numeric = samples / "one_shot_percussive_sounds" / "2" / "100005.wav"
    bongo_numeric = samples / "Bongo" / "100005.wav"
    bass_808 = samples / "one_shot_percussive_sounds" / "1" / "808.wav"
    make_wav(generic_numeric, seconds=0.2)
    make_wav(bongo_numeric, seconds=0.2)
    make_wav(bass_808, seconds=0.2)
    entries = mgr.all_taxonomy_entries(Path("/no/project"))
    generic_perc = next(e for e in entries if str(e.label_rel) == "Drums/Percussion/Generic Percussion")
    guiro = next(e for e in entries if str(e.label_rel) == "Drums/Percussion/Guiros Scrapes and Rasps")
    bongo = next(e for e in entries if str(e.label_rel) == "Drums/World Percussion/Latin Percussion/Bongo")
    bass_808_entry = next(e for e in entries if str(e.label_rel) == "Instruments/Bass/808 Bass")

    assert mgr.entry_matches_text(generic_perc, mgr.text_tokens(generic_numeric.relative_to(samples)))[0] is True
    assert mgr.source_path_has_specific_evidence(generic_perc, generic_numeric, samples) is False
    assert mgr.entry_matches_text(guiro, mgr.text_tokens(generic_numeric.relative_to(samples)))[0] is False
    assert mgr.source_path_has_specific_evidence(bongo, bongo_numeric, samples) is True
    assert mgr.source_path_has_specific_evidence(bass_808_entry, bass_808, samples) is True


def test_real_ledger_entries_obey_structure_policy_when_present() -> None:
    from tests.synthetic_audio_fixtures import ensure_synthetic_ledger_file

    ledger = ROOT / "Aaron_Master_Attribute_Classification_Ledger_v1_0.json"
    if not ledger.exists():
        ledger = ensure_synthetic_ledger_file(
            ROOT / "tests" / "fixtures" / "Aaron_Master_Attribute_Classification_Ledger_v1_0.json"
        )
    entries = mgr.load_ledger_taxonomy(ROOT, str(ledger))
    assert entries
    for entry in entries:
        assert entry.structures == mgr.structures_for_label(entry.label_rel, entry.family)


def test_stage_candidates_excludes_electric_piano_from_clean_piano(tmp_path: Path) -> None:
    training = tmp_path / "training"
    samples = tmp_path / "samples"
    project = tmp_path / "project"
    (training / "Instruments" / "Keys" / "Piano" / "_LOOPS").mkdir(parents=True)
    (training / "Instruments" / "Keys" / "Electric Piano" / "_LOOPS").mkdir(parents=True)
    make_wav(samples / "Keys" / "Dmin140BPMElectricFryPiano.wav", seconds=4.0)
    make_wav(samples / "Keys" / "Grand Piano Loop 120BPM.wav", seconds=4.0)
    args = type(
        "Args",
        (),
        {
            "training_root": str(training),
            "samples_root": str(samples),
            "project_dir": str(project),
            "ledger_json": "",
            "per_structure": 10,
            "max_candidates_per_slot": 25,
            "min_score": 8.0,
        },
    )()
    assert mgr.stage_candidates(args) == 0
    manifests = sorted(project.rglob("candidate_manifest_all_ranked.csv"))
    assert manifests
    rows = list(csv.DictReader(manifests[-1].open()))
    piano_sources = [r["source_path"] for r in rows if r["label_rel"] == "Instruments/Keys/Piano"]
    ep_sources = [r["source_path"] for r in rows if r["label_rel"] == "Instruments/Keys/Electric Piano"]
    assert any("Grand Piano" in p for p in piano_sources)
    assert not any("ElectricFryPiano" in p for p in piano_sources)
    assert any("ElectricFryPiano" in p for p in ep_sources)


def test_stage_candidates_skips_eval_files_when_folder_has_non_eval_option(tmp_path: Path) -> None:
    training = tmp_path / "training"
    samples = tmp_path / "samples"
    project = tmp_path / "project"
    (training / "FX" / "Impacts and Hits" / "Boom" / "_ONE_SHOTS").mkdir(parents=True)
    make_wav(samples / "Booms" / "boom_eval_108640.wav", seconds=0.5)
    make_wav(samples / "Booms" / "boom_108641.wav", seconds=0.5)
    args = type(
        "Args",
        (),
        {
            "training_root": str(training),
            "samples_root": str(samples),
            "project_dir": str(project),
            "ledger_json": "",
            "per_structure": 10,
            "max_candidates_per_slot": 25,
            "min_score": 8.0,
        },
    )()
    assert mgr.stage_candidates(args) == 0
    manifest = sorted(project.rglob("candidate_manifest_all_ranked.csv"))[-1]
    rows = list(csv.DictReader(manifest.open()))
    boom_sources = [r["source_path"] for r in rows if r["label_rel"] == "FX/Impacts and Hits/Boom"]
    assert any("boom_108641" in p for p in boom_sources)
    assert not any("boom_eval" in p for p in boom_sources)


def test_stage_candidates_skips_eval_when_non_eval_sibling_is_already_training(tmp_path: Path) -> None:
    training = tmp_path / "training"
    samples = tmp_path / "samples"
    project = tmp_path / "project"
    slot = training / "FX" / "Impacts and Hits" / "Boom" / "_ONE_SHOTS"
    slot.mkdir(parents=True)
    eval_source = samples / "Booms" / "boom_eval_108640.wav"
    clean_source = samples / "Booms" / "boom_108641.wav"
    make_wav(eval_source, seconds=0.5)
    make_wav(clean_source, seconds=0.5)
    os.symlink(str(clean_source), str(slot / clean_source.name))
    args = type(
        "Args",
        (),
        {
            "training_root": str(training),
            "samples_root": str(samples),
            "project_dir": str(project),
            "ledger_json": "",
            "per_structure": 10,
            "max_candidates_per_slot": 25,
            "min_score": 8.0,
        },
    )()
    assert mgr.stage_candidates(args) == 0
    manifest = sorted(project.rglob("candidate_manifest_all_ranked.csv"))[-1]
    rows = list(csv.DictReader(manifest.open()))
    assert not rows


def test_stage_candidates_skips_non_taxonomy_legacy_slots(tmp_path: Path) -> None:
    training = tmp_path / "training"
    samples = tmp_path / "samples"
    project = tmp_path / "project"
    (training / "Drums" / "Claps Snaps Slaps" / "Generic Clap" / "_ONE_SHOTS").mkdir(parents=True)
    make_wav(samples / "Clap 1.wav", seconds=0.5)
    args = type(
        "Args",
        (),
        {
            "training_root": str(training),
            "samples_root": str(samples),
            "project_dir": str(project),
            "ledger_json": "",
            "per_structure": 10,
            "max_candidates_per_slot": 25,
            "min_score": 8.0,
        },
    )()
    assert mgr.stage_candidates(args) == 0
    manifest = sorted(project.rglob("candidate_manifest_all_ranked.csv"))[-1]
    rows = list(csv.DictReader(manifest.open()))
    assert not rows


def test_apply_reviewed_candidates_defaults_to_top_ten_per_structure(tmp_path: Path) -> None:
    review = tmp_path / "review"
    training = tmp_path / "training"
    project = tmp_path / "project"
    slot = review / "Drums" / "Kick Drums" / "Generic Kick" / "_ONE_SHOTS"
    slot.mkdir(parents=True)
    sources = []
    for index in range(12):
        source = tmp_path / "samples" / f"Kick {index:02d}.wav"
        make_wav(source, seconds=0.5)
        sources.append(source)
        os.symlink(str(source), str(slot / f"{index + 1:02d}__score_10.00__{source.name}"))
    args = type(
        "Args",
        (),
        {
            "review_root": str(review),
            "training_root": str(training),
            "project_dir": str(project),
            "apply": True,
            "max_per_structure": 10,
        },
    )()
    assert mgr.apply_reviewed_candidates(args) == 0
    manifest = sorted(project.rglob("apply_reviewed_candidates_manifest.csv"))[-1]
    rows = list(csv.DictReader(manifest.open()))
    assert sum(1 for row in rows if row["action"] == "add_reviewed_training_symlink") == 10
    assert sum(1 for row in rows if row["action"] == "skip_not_selected_for_spread_limit") == 2
    assert len(list((training / "Drums" / "Kick Drums" / "Generic Kick" / "_ONE_SHOTS").iterdir())) == 10


def test_apply_reviewed_candidates_spreads_selection_across_source_folders(tmp_path: Path) -> None:
    review = tmp_path / "review"
    training = tmp_path / "training"
    project = tmp_path / "project"
    slot = review / "Drums" / "Kick Drums" / "Generic Kick" / "_ONE_SHOTS"
    slot.mkdir(parents=True)
    rank = 1
    for folder_name, count in [("pack_a", 10), ("pack_b", 1), ("pack_c", 1)]:
        for _index in range(count):
            source = tmp_path / "samples" / folder_name / f"Kick {rank:02d}.wav"
            make_wav(source, seconds=0.5)
            os.symlink(str(source), str(slot / f"{rank:02d}__score_10.00__{source.name}"))
            rank += 1
    args = type(
        "Args",
        (),
        {
            "review_root": str(review),
            "training_root": str(training),
            "project_dir": str(project),
            "apply": True,
            "max_per_structure": 3,
        },
    )()
    assert mgr.apply_reviewed_candidates(args) == 0
    manifest = sorted(project.rglob("apply_reviewed_candidates_manifest.csv"))[-1]
    rows = list(csv.DictReader(manifest.open()))
    added_sources = [Path(row["source_target"]) for row in rows if row["action"] == "add_reviewed_training_symlink"]
    assert len(added_sources) == 3
    assert {source.parent.name for source in added_sources} == {"pack_a", "pack_b", "pack_c"}


def test_apply_reviewed_candidates_fills_existing_slot_only_to_target_count(tmp_path: Path) -> None:
    review = tmp_path / "review"
    training = tmp_path / "training"
    project = tmp_path / "project"
    review_slot = review / "Drums" / "Kick Drums" / "Generic Kick" / "_ONE_SHOTS"
    training_slot = training / "Drums" / "Kick Drums" / "Generic Kick" / "_ONE_SHOTS"
    review_slot.mkdir(parents=True)
    training_slot.mkdir(parents=True)
    for index in range(8):
        source = tmp_path / "existing" / f"Existing Kick {index:02d}.wav"
        make_wav(source, seconds=0.5)
        os.symlink(str(source), str(training_slot / source.name))
    for index in range(5):
        source = tmp_path / "samples" / f"New Kick {index:02d}.wav"
        make_wav(source, seconds=0.5)
        os.symlink(str(source), str(review_slot / f"{index + 1:02d}__score_10.00__{source.name}"))
    args = type(
        "Args",
        (),
        {
            "review_root": str(review),
            "training_root": str(training),
            "project_dir": str(project),
            "apply": True,
            "max_per_structure": 10,
        },
    )()
    assert mgr.apply_reviewed_candidates(args) == 0
    manifest = sorted(project.rglob("apply_reviewed_candidates_manifest.csv"))[-1]
    rows = list(csv.DictReader(manifest.open()))
    assert sum(1 for row in rows if row["action"] == "add_reviewed_training_symlink") == 2
    assert sum(1 for row in rows if row["action"] == "skip_not_selected_for_spread_limit") == 3
    assert len(list(training_slot.iterdir())) == 10


def test_apply_reviewed_candidates_skips_slot_already_at_target_count(tmp_path: Path) -> None:
    review = tmp_path / "review"
    training = tmp_path / "training"
    project = tmp_path / "project"
    review_slot = review / "Drums" / "Kick Drums" / "Generic Kick" / "_ONE_SHOTS"
    training_slot = training / "Drums" / "Kick Drums" / "Generic Kick" / "_ONE_SHOTS"
    review_slot.mkdir(parents=True)
    training_slot.mkdir(parents=True)
    for index in range(10):
        source = tmp_path / "existing" / f"Existing Kick {index:02d}.wav"
        make_wav(source, seconds=0.5)
        os.symlink(str(source), str(training_slot / source.name))
    for index in range(2):
        source = tmp_path / "samples" / f"New Kick {index:02d}.wav"
        make_wav(source, seconds=0.5)
        os.symlink(str(source), str(review_slot / f"{index + 1:02d}__score_10.00__{source.name}"))
    args = type(
        "Args",
        (),
        {
            "review_root": str(review),
            "training_root": str(training),
            "project_dir": str(project),
            "apply": True,
            "max_per_structure": 10,
        },
    )()
    assert mgr.apply_reviewed_candidates(args) == 0
    manifest = sorted(project.rglob("apply_reviewed_candidates_manifest.csv"))[-1]
    rows = list(csv.DictReader(manifest.open()))
    assert sum(1 for row in rows if row["action"] == "add_reviewed_training_symlink") == 0
    assert sum(1 for row in rows if row["action"] == "skip_existing_slot_at_or_above_limit") == 2
    assert len(list(training_slot.iterdir())) == 10


def test_apply_reviewed_candidates_skips_invalid_legacy_slots(tmp_path: Path) -> None:
    review = tmp_path / "review"
    training = tmp_path / "training"
    project = tmp_path / "project"
    source = tmp_path / "samples" / "Riser Loop.wav"
    make_wav(source, seconds=0.5)
    bad_slot = review / "FX" / "Structural and Transitional FX" / "Risers and Builds" / "Generic Riser" / "_LOOPS"
    bad_slot.mkdir(parents=True)
    os.symlink(str(source), str(bad_slot / source.name))

    args = type(
        "Args",
        (),
        {
            "review_root": str(review),
            "training_root": str(training),
            "project_dir": str(project),
            "apply": True,
            "max_per_structure": 10,
        },
    )()
    assert mgr.apply_reviewed_candidates(args) == 0
    manifest = sorted(project.rglob("apply_reviewed_candidates_manifest.csv"))[-1]
    rows = list(csv.DictReader(manifest.open()))
    assert rows[0]["action"] == "skip_invalid_structure_policy"
    assert not (
        training / "FX" / "Structural and Transitional FX" / "Risers and Builds" / "Generic Riser" / "_LOOPS"
    ).exists()
