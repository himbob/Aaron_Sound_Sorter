# Auto-split from Aaron_Sound_Sorter.py.
# This is a component module, not a legacy wrapper.
from __future__ import annotations

from .core import *


def _write_test_wav(path: Path, samples: np.ndarray, sr: int = TARGET_SR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y = np.asarray(samples, dtype=np.float32)
    y = np.clip(y, -1.0, 1.0)
    pcm = (y * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def _synthetic_hit(sr: int = TARGET_SR) -> np.ndarray:
    n = int(sr * 0.50)
    t = np.arange(n, dtype=np.float32) / sr
    return (np.exp(-t * 35.0) * np.sin(2 * np.pi * 90.0 * t)).astype(np.float32)


def _synthetic_loop(sr: int = TARGET_SR) -> np.ndarray:
    n = int(sr * 4.0)
    y = np.zeros(n, dtype=np.float32)
    hit = _synthetic_hit(sr)[: int(sr * 0.12)]
    for start in range(0, n - hit.size, int(sr * 0.50)):
        y[start : start + hit.size] += hit
    return np.clip(y, -1.0, 1.0)


def _synthetic_long_tone(sr: int = TARGET_SR) -> np.ndarray:
    n = int(sr * 6.0)
    t = np.arange(n, dtype=np.float32) / sr
    env = np.exp(-t * 0.15)
    return (0.35 * env * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


def run_self_tests(tmp_root: Optional[Path] = None) -> int:
    """Targeted regression tests for the Phase 3 pure brain lab.

    These tests intentionally use tiny synthetic WAVs and fake paths. They catch
    the exact failures Aaron saw without running a giant library scan.
    """
    import tempfile

    if tmp_root is None:
        tmp_root = Path(tempfile.mkdtemp(prefix="phase3_v0417_selftest_"))
    tmp_root.mkdir(parents=True, exist_ok=True)
    failures: List[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        if cond:
            print(f"PASS {name}")
        else:
            print(f"FAIL {name}: {detail}")
            failures.append(f"{name}: {detail}")

    # 1. AppleDouble, hidden files, __MACOSX, AKWF are skipped before decode.
    root = tmp_root / "preview_scan"
    _write_test_wav(root / "Good" / "real.wav", _synthetic_hit())
    _write_test_wav(root / "Good" / "._real.wav", _synthetic_hit())
    _write_test_wav(root / "Good" / ".hidden.wav", _synthetic_hit())
    _write_test_wav(root / "__MACOSX" / "junk.wav", _synthetic_hit())
    _write_test_wav(root / "AKWF" / "akwf_0001.wav", _synthetic_hit()[:1000])
    files = iter_real_preview_audio_files(root, max_total=100, per_folder=20, random_seed=1)
    names = sorted(p.name for p in files)
    check("real preview skips Apple/hidden/AKWF junk", names == ["real.wav"], f"selected={names}")

    sorted_training_root = tmp_root / "samples" / "Sorted samples"
    outside_root = tmp_root / "samples" / "Loose Sounds"
    _write_test_wav(sorted_training_root / "FX" / "Animals" / "Cat" / "meow.wav", _synthetic_hit())
    _write_test_wav(outside_root / "random.wav", _synthetic_hit())
    mixed_files = iter_real_preview_audio_files(
        tmp_root / "samples", max_total=100, per_folder=20, random_seed=1, extra_exclude_roots=[sorted_training_root]
    )
    mixed_names = sorted(p.name for p in mixed_files)
    check(
        "real preview excludes Sorted samples training pool", mixed_names == ["random.wav"], f"selected={mixed_names}"
    )
    sorted_only_files = iter_real_preview_audio_files(
        sorted_training_root, max_total=100, per_folder=20, random_seed=1, extra_exclude_roots=[sorted_training_root]
    )
    check(
        "real preview returns nothing when root is Sorted samples",
        sorted_only_files == [],
        f"selected={[str(p) for p in sorted_only_files]}",
    )

    # 2. Onset regularity separates a repeated loop from a one-shot.
    hit_path = tmp_root / "audio" / "hit.wav"
    loop_path = tmp_root / "audio" / "loop.wav"
    tone_path = tmp_root / "audio" / "long_tone.wav"
    _write_test_wav(hit_path, _synthetic_hit())
    _write_test_wav(loop_path, _synthetic_loop())
    _write_test_wav(tone_path, _synthetic_long_tone())
    hit_fp, hit_dur, hit_status = make_fingerprint(hit_path)
    loop_fp, loop_dur, loop_status = make_fingerprint(loop_path)
    tone_fp, tone_dur, tone_status = make_fingerprint(tone_path)
    check(
        "fingerprint size matches FP_SIZE",
        hit_fp.size == FP_SIZE and loop_fp.size == FP_SIZE,
        f"sizes={hit_fp.size},{loop_fp.size}",
    )
    check(
        "synthetic files read",
        hit_status == "ok" and loop_status == "ok" and tone_status == "ok",
        f"statuses={hit_status},{loop_status},{tone_status}",
    )
    check(
        "loop has more transients than hit",
        float(loop_fp[35]) > float(hit_fp[35]),
        f"hit={hit_fp[35]} loop={loop_fp[35]}",
    )
    check(
        "loop regularity beats hit",
        float(loop_fp[43]) < float(hit_fp[43]),
        f"hit_reg={hit_fp[43]} loop_reg={loop_fp[43]}",
    )
    pseudo_cat_row = {
        "duration_sec": "1.25",
        "onset_count": "5",
        "loop_periodicity": "0.30",
        "group_key": "FX/Animals and Creatures/Cat/_ONE_SHOTS",
    }
    real_loop_row = {
        "duration_sec": "3.20",
        "onset_count": "8",
        "loop_periodicity": "0.50",
        "group_key": "Drums/Drum Loops/_ONE_SHOTS",
    }
    check(
        "long animal one-shot is not repaired to loop",
        not strong_audio_loop_evidence(pseudo_cat_row),
        f"row={pseudo_cat_row}",
    )
    check(
        "clear repeated-event row can repair to loop", strong_audio_loop_evidence(real_loop_row), f"row={real_loop_row}"
    )
    repaired_tone, tone_reason = repair_structure_lane_with_fingerprint("one_shot", tone_fp, tone_dur)
    repaired_loop, loop_reason = repair_structure_lane_with_fingerprint("one_shot", loop_fp, loop_dur)
    check(
        "long single tone stays one_shot", repaired_tone == "one_shot", f"repaired={repaired_tone} reason={tone_reason}"
    )
    check(
        "synthetic repeated pulse reports loop evidence without changing locked one-shot folder",
        repaired_loop == "one_shot" and "REPORT_ONLY" in loop_reason,
        f"repaired={repaired_loop} reason={loop_reason} transients={np.expm1(loop_fp[35])} regularity={loop_fp[43]}",
    )
    tiny_path = tmp_root / "audio" / "tiny.wav"
    _write_test_wav(tiny_path, _synthetic_hit()[: int(TARGET_SR * 0.02)])
    tiny_fp, tiny_dur, tiny_status = make_fingerprint(tiny_path)
    check(
        "tiny normal sample is skipped after read",
        tiny_status == "ok" and should_skip_tiny_after_read(tiny_dur, tiny_path),
        f"tiny_status={tiny_status} tiny_dur={tiny_dur}",
    )
    check(
        "long tone is not tiny",
        tone_dur > 4.0 and not should_skip_tiny_after_read(tone_dur, tone_path),
        f"tone_dur={tone_dur}",
    )

    # 3. Learned structure head should prefer loop for repeated pulse and one_shot for single hit.
    train_rows = [
        FeatureRow(
            str(hit_path),
            "Test/Hit",
            "Drums/Test Hit/One Shots",
            "Drums",
            "one_shot",
            hit_dur,
            hit_fp.astype(float).tolist(),
            "ok",
        ),
        FeatureRow(
            str(tone_path),
            "Test/Tone",
            "Instruments/Test Tone/One Shots",
            "Instruments",
            "one_shot",
            tone_dur,
            tone_fp.astype(float).tolist(),
            "ok",
        ),
        FeatureRow(
            str(loop_path),
            "Test/Loop",
            "Drums/Test Loop/Loops",
            "Drums",
            "loop",
            loop_dur,
            loop_fp.astype(float).tolist(),
            "ok",
        ),
    ]
    brain = build_brain(train_rows, max_centroids=3)
    hs = structure_distance_scores(brain, hit_fp.tolist())
    ls = structure_distance_scores(brain, loop_fp.tolist())
    hit_best = hs[0][0] if hs else ""
    loop_best = ls[0][0] if ls else ""
    check("structure head picks one_shot for hit", hit_best == "one_shot", f"scores={hs}")
    check("structure head picks loop for repeated pulse", loop_best == "loop", f"scores={ls}")
    pred_loop, pred_top, sim, margin, top5 = predict(brain, loop_fp.tolist(), "")
    check(
        "pure predict can choose loop label",
        label_default_structure(pred_loop) == "loop",
        f"pred={pred_loop} top5={top5}",
    )

    # 4. Conservative gate: a huge margin does not rescue a weak similarity
    # unless the legacy override is explicitly requested.
    gated = gate_prediction("Drums/Test Hit/One Shots", "Drums", 0.40, 9.0, "ok", 0.5, 0.55, 0.60)
    check(
        "weak similarity auto-places with weak_pick audit under decisive product policy",
        gated[2] == "auto_place" and "weak_pick" in str(gated[3]),
        f"gate={gated}",
    )
    gated_legacy = gate_prediction(
        "Drums/Test Hit/One Shots", "Drums", 0.40, 9.0, "ok", 0.5, 0.55, 0.60, allow_low_similarity_clear_margin=True
    )
    check("legacy low-similarity margin override is explicit", gated_legacy[2] == "auto_place", f"gate={gated_legacy}")

    # 5. Filename shortening stays below macOS limits.
    long_name = "Very_Long_Source_Name_" + ("x" * 160) + ".wav"
    long_path = tmp_root / "audio" / long_name
    _write_test_wav(long_path, _synthetic_hit())
    dest = copy_real_preview_file(
        str(long_path), tmp_root / "sorted", "Drums", "Drums/Test Hit/One Shots", 1, "Drums/Test Hit/One Shots"
    )
    check("copy filename stays under 255 chars", bool(dest) and len(Path(dest).name) < 180, f"dest={dest}")

    # v0.4.35: no public labels are disabled by code in locked training mode.
    check(
        "Generic Percussion is not disabled by hidden code",
        not bool(public_label_is_disabled_for_training("Drums/World Percussion/Generic Percussion/One Shots")),
        "Aaron owns training cleanup manually",
    )

    # v0.4.35: pure loop labels are not rerouted by filename/source-name logic.
    dummy_loop = FeatureRow(
        "/tmp/BLV43_125_Percussion_Mixed_07.wav",
        "Drums/Kick Drums/Generic Kick/Loops",
        "Drums/Kick Drums/Generic Kick/Loops",
        "Drums",
        "loop",
        4.0,
        [0.0] * FP_SIZE,
        "ok",
    )
    action, reason = loop_purity_training_decision(dummy_loop)
    check(
        "mixed beat source is not rerouted by filename/source-name logic",
        action == "keep",
        f"action={action} reason={reason}",
    )

    # 6. Review-only physical role recommendations catch impossible one-shot roles.
    # Stage 4 folder-brain mode must not fabricate named category recommendations.
    # Older self-tests expected hand-written kick/clap/dog rescue labels, but Aaron
    # explicitly moved this phase to learned folders only. Physics still blocks unsafe
    # auto-placement through learned conflict gates; it just no longer invents alternate
    # terminal category paths outside the trained tree.
    fake_brain = {"structure_by_label": {"Instruments/Bass/Synth Bass/One Shots": "one_shot"}}
    role = physical_role_recommendation(
        fake_brain,
        loop_fp.tolist(),
        loop_dur,
        "Instruments/Bass/Synth Bass/One Shots",
        "Instruments",
        [("Instruments/Bass/Synth Bass/One Shots", 0.1)],
    )
    check(
        "physical role recommender does not fabricate unlearned category names",
        role.get("physical_role_action") == "disabled_no_named_category_recommendations"
        and role.get("physical_role_candidate_label", "") == "",
        f"role={role}",
    )

    train_only = [
        FeatureRow(
            "/tmp/train.wav", "Drums/Kick", "Drums/Kick/One Shots", "Drums", "one_shot", 0.2, [0.0] * FP_SIZE, "ok"
        )
    ]
    eval_only = [
        FeatureRow(
            "/tmp/clap.wav", "Drums/Clap", "Drums/Clap/One Shots", "Drums", "one_shot", 0.4, [0.1] * FP_SIZE, "ok"
        )
    ]
    fallback_train = ensure_eval_labels_have_training_support(train_only, eval_only, tmp_root)
    check(
        "eval-only label is reported but never injected into training",
        not any(r.label == "Drums/Clap/One Shots" for r in fallback_train),
        f"labels={[r.label for r in fallback_train]}",
    )
    overlap_train = [
        FeatureRow(
            "/tmp/shared.wav", "Drums/Kick", "Drums/Kick/One Shots", "Drums", "one_shot", 0.2, [0.0] * FP_SIZE, "ok"
        )
    ]
    overlap_eval = [
        FeatureRow(
            "/tmp/shared.wav", "Drums/Kick", "Drums/Kick/One Shots", "Drums", "one_shot", 0.2, [0.0] * FP_SIZE, "ok"
        )
    ]
    overlap_kept = remove_eval_training_overlaps(overlap_train, overlap_eval, tmp_root)
    check(
        "eval overlap with training is excluded before scoring",
        len(overlap_kept) == 0,
        f"kept={len(overlap_kept)}",
    )

    if failures:
        print("\nSELF TEST FAILURES")
        for item in failures:
            print(" - " + item)
        return 1
    print("\nAll Stage 4 v0.6.4 self-tests passed.")
    return 0
