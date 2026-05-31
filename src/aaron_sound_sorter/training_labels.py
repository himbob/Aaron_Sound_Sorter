# Auto-split from Aaron_Sound_Sorter.py.
# This is a component module, not a legacy wrapper.
from __future__ import annotations

import struct

from .core import *
from .core import _MEL_CACHE, _sf


def is_active_training_status(status: str) -> bool:
    return str(status or "").startswith("ACTIVE_")


def validate_v052_brain_or_die(brain: dict, brain_path: Path | str = "") -> None:
    """Reject stale brains. v0.5.5 is a clean 100-feature brain format only."""
    if not isinstance(brain, dict):
        raise SystemExit("Brain JSON is not an object")
    feature_size = int(brain.get("feature_size", brain.get("fingerprint_size", 0)) or 0)
    feature_names = list(brain.get("feature_names", []))
    if feature_size != FP_SIZE or len(feature_names) != FP_SIZE:
        where = f" {brain_path}" if brain_path else ""
        raise SystemExit(
            f"Incompatible brain{where}: expected v0.5.5 {FP_SIZE}-feature brain, "
            f"got feature_size={feature_size}, feature_names={len(feature_names)}. "
            "Rebuild with: python3 Aaron_Sound_Sorter.py train-brain TRAINING_FOLDER --save-brain stage4_folder_brain.json"
        )
    required = {"category_fact_profiles", "feature_weights", "scaler_mean", "scaler_std", "labels"}
    missing = sorted(k for k in required if k not in brain)
    if missing:
        raise SystemExit(f"Incompatible brain{(' ' + str(brain_path)) if brain_path else ''}: missing {missing}")


def normalize_locked_training_group_key(group_key: str) -> str:
    """Normalize Aaron-owned training folder paths without reading filenames.

    This is folder-structure normalization only. It keeps the locked training
    tree read-only, strips no identity category except structure marker folders
    handled later, and folds legacy top-level World Instruments into the final
    producer-friendly Instruments tree so those labels are not silently skipped
    by allowed-top filtering.
    """
    parts = [p.strip() for p in str(group_key or "").replace("\\", "/").split("/") if p.strip()]
    if not parts:
        return ""
    first = parts[0].lower().replace("_", " ").replace("-", " ").strip()
    if first == "world instruments":
        return "/".join(["Instruments", "World and Special Instruments", *parts[1:]])
    return "/".join(parts)


def row_duration_sec(row: Dict[str, str]) -> float:
    for key in ("duration_sec", "duration", "length_sec"):
        try:
            value = str(row.get(key, "")).strip()
            if value:
                return float(value)
        except Exception:
            pass
    return 0.0


def structure_bucket_from_group(group_key: str) -> str:
    """Read explicit structure from Aaron's locked training folder path.

    Version: v20260512_LONG_FX_LABEL_PRESERVE

    The locked tree is human-owned. The code may read these marker folders,
    but it must not reinterpret or rewrite the training folder.

    Supported explicit markers:
      _ONE_SHOTS / One Shots -> one_shot
      _LONG_FX / Long FX     -> long_fx
      _LONG_RUNNING          -> long_running
      _LOOPS / Loops         -> loop

    Important: _LONG_FX is not a one-shot. Older code collapsed _LONG_FX to
    one_shot and later appended /One Shots, which erased Aaron's true FX
    structure from stage4_folder_brain.json.
    """
    parts = [
        p.strip().lower().replace("-", "_").replace(" ", "_")
        for p in str(group_key or "").replace("\\", "/").split("/")
        if p.strip()
    ]
    if any(p in {"_loops", "loops", "loop", "_loop"} for p in parts):
        return "loop"
    if any(p in {"_long_fx", "long_fx", "long_fxs"} for p in parts):
        return "long_fx"
    if any(p in {"_long_running", "long_running"} for p in parts):
        return "long_running"
    if any(p in {"_one_shots", "one_shots", "one_shot", "_one_shot"} for p in parts):
        return "one_shot"
    # Category names that explicitly name loop roles still get loop structure.
    body = " ".join(parts)
    if re.search(
        r"\b(drum_loop|drum_loops|bass_loop|guitar_loop|synth_loop|keys_loop|strings_loop|vocal_loop|mixed_musical_loops|mixed_instrumental_loop|loop|loops|fill|fills|roll|rolls|pattern|patterns|groove|grooves|beat|beats|arp|arps)\b",
        body,
    ):
        return "loop"
    return ""


def structure_lane_for_row(row: Dict[str, str], group_key: str = "") -> str:
    """Two-lane structure-first decision for the pure brain lab.

    The split is not short-vs-long. It is:
      one_shot = one non-repeating sample event, even if long/sustained/evolving
      loop     = repeated musical/rhythmic/multi-hit material

    Duration, sustain, tail, drone, riser, ambience, and texture remain traits
    or identity categories, not a separate structure lane.

    v0.4.65: structure is determined only from explicit folder markers,
    generic loop-role folder wording, and measured audio evidence. No fixed
    terminal category vocabulary is used to name the learned folder label.
    group_has_named_loop_role() already covers the same cases correctly.
    """
    group_key = group_key or str(row.get("group_key", "")).replace("\\", "/").strip()
    explicit = structure_bucket_from_group(group_key)
    if explicit:
        return explicit
    if group_has_named_loop_role(group_key):
        return "loop"
    return "one_shot"


def structure_terminal_name(structure: str) -> str:
    """Producer-facing terminal folder for structure.

    Version: v20260512_LONG_FX_LABEL_PRESERVE

    v0.4.5 kept One Shots / Loops at the bottom of the label path. This
    version adds Long FX / Long Running so explicit training markers survive
    into the brain and can later be used for hard structure gating.
    """
    normalized = str(structure or "").lower().replace(" ", "_").replace("-", "_")
    if normalized == "loop":
        return "Loops"
    if normalized == "long_fx":
        return "Long FX"
    if normalized == "long_running":
        return "Long Running"
    return "One Shots"


def label_with_terminal_structure(public: str, structure: str) -> str:
    base = str(public or "Unknown").strip().strip("/")
    low = base.lower().replace("\\", "/")
    terminal_lows = (
        "/one shots",
        "/one shot",
        "/loops",
        "/loop",
        "/long fx",
        "/long fxs",
        "/long running",
    )
    if low.endswith(terminal_lows):
        return base
    return f"{base}/{structure_terminal_name(structure)}"


def structure_folder_name(structure: str) -> str:
    # Kept only for old reports; folder output should use structure_terminal_name().
    return structure_terminal_name(structure)


def public_label(label: str) -> str:
    text = str(label or "")
    return text.split("::", 1)[1] if "::" in text else text


def label_default_structure(label: str) -> str:
    """Return normalized structure from a terminal label.

    Version: v20260512_LONG_FX_LABEL_PRESERVE

    Long FX and Long Running are real learned structure lanes now. They must
    not default back to one_shot.
    """
    text = str(label or "")
    prefix = text.split("::", 1)[0].strip().lower().replace(" ", "_").replace("-", "_")
    if prefix in {"_loops", "loops", "loop"}:
        return "loop"
    if prefix in {"_long_fx", "long_fx", "long_fxs"}:
        return "long_fx"
    if prefix in {"_long_running", "long_running", "long"}:
        return "long_running"
    if prefix in {"_one_shots", "one_shots", "one_shot"}:
        return "one_shot"
    base = public_label(text).lower().replace("\\", "/").strip("/")
    if base.endswith("/loops") or base.endswith("/loop"):
        return "loop"
    if base.endswith("/long fx") or base.endswith("/long fxs"):
        return "long_fx"
    if base.endswith("/long running"):
        return "long_running"
    return "one_shot"


def scoped_label(structure: str, label: str) -> str:
    # New labels are plain producer folder paths with One Shots / Loops at the leaf.
    return label_with_terminal_structure(label, structure)


def top_for_public_label(label: str) -> str:
    """Return top folder only from a folder-derived label path.

    Legacy shorthand prefixes are deliberately not mapped here. If a label is
    not a real folder path, it is not a valid learned folder destination.
    """
    base = public_label(label)
    if "/" in base:
        return base.split("/", 1)[0]
    return "_TO_REVIEW"


def source_text_for_row(row: Dict[str, str]) -> str:
    """Return folder hierarchy text only.

    This helper intentionally ignores source filenames.  It is allowed for
    reports and folder-structure interpretation, not for filename hacks.
    """
    text = " ".join(
        [
            str(row.get("group_key", "")),
            str(row.get("folder", "")),
        ]
    )
    text = text.lower().replace("\\", "/").replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", text)


def group_has_named_loop_role(group_key: str) -> bool:
    """True when the hierarchy names a real loop/pattern role, not only a terminal /Loops leaf.

    v0.4.18 fixes a training-data bug where longer one-shot FX such as cat
    meows were placed in a Loops terminal and then trained as true loops.
    A terminal /Loops folder by itself is no longer enough.  The category must
    either name a musical/rhythmic loop role, or the measured audio must later
    prove repeated primary events.
    """
    g = str(group_key or "").replace("\\", "/").lower()
    parts = [p.strip() for p in g.split("/") if p.strip()]
    # Remove pure structure leaves so "Cat/Loops" does not count as named-loop intent.
    non_terminal = [
        p
        for p in parts
        if p
        not in {
            "_loops",
            "loops",
            "loop",
            "_one_shots",
            "one shots",
            "one shot",
            "_long_fx",
            "long fx",
            "_long_running",
            "long running",
        }
    ]
    body = "/".join(non_terminal)
    if not body:
        return False
    # Generic producer loop-role words only. No terminal source category names.
    return bool(
        re.search(
            r"\b(loop|loops|looping|pattern|patterns|groove|grooves|beat|beats|fill|fills|roll|rolls|arp|arps)\b",
            body.replace("_", " "),
        )
    )


def explicit_loop_requires_audio_proof(group_key: str) -> bool:
    """Return True for terminal /Loops folders that may just be long one-shots.

    This is intentionally broad for FX, animal, foley, ambience, transition,
    impact, and generic long-running material. Those categories should train as
    Loops only when the audio has repeated primary events.
    """
    explicit = structure_bucket_from_group(group_key)
    if explicit != "loop":
        return False
    return not group_has_named_loop_role(group_key)


def public_label_is_disabled_for_training(label: str) -> str:
    """Return reason when a public label must not become an active teacher."""
    pub = public_label(label).replace("\\", "/")
    for pattern in DISABLED_TRAINING_PUBLIC_LABEL_PATTERNS:
        if re.search(pattern, pub, flags=re.IGNORECASE):
            return f"disabled_training_label_pattern:{pattern}"
    return ""


def feature_row_source_text(row: FeatureRow) -> str:
    return source_text_for_row({"source_path": row.path, "group_key": row.group_key})


def pure_loop_family_from_label(label: str) -> str:
    """Deprecated compatibility wrapper.

    No fixed pure-loop family categories are inferred from label text anymore.
    The folder path itself is the learned label.
    """
    return ""


def mixed_loop_public_from_context(row_or_feature, group_key: str = "") -> str:
    """Return no invented mixed-loop label.

    Folder truth is dynamic.  If a mixed-loop category exists, it must come from
    Aaron's folder path.  This function intentionally does not manufacture
    labels from broad legacy folder names.
    """
    return ""


def loop_purity_training_decision(row: FeatureRow) -> Tuple[str, str]:
    """Locked training does not reroute/delete loop teachers.

    Folder structure supplies training truth.  Loop purity problems are reported
    through physics fields and review reports, not silently fixed by filename or
    source-text rules.
    """
    return "keep", ""


def fingerprint_primary_event_count(fingerprint: Sequence[float]) -> float:
    try:
        fp = np.asarray(fingerprint, dtype=np.float32)
        return float(np.expm1(max(0.0, float(fp[35])))) if fp.size > 35 else 0.0
    except Exception:
        return 0.0


def fingerprint_front_loaded_tail_evidence(fingerprint: Sequence[float], duration_sec: float) -> bool:
    """True for one dominant hit with a long tail/reverb/body.

    Long cymbals, kicks with giant reverb, impacts, gongs, and conga tails are
    one-shot teachers, not loop teachers. Transient chatter in the tail should
    not make them loops.
    """
    try:
        fp = np.asarray(fingerprint, dtype=np.float32)
        if fp.size < FP_SIZE:
            return False
        transients = fingerprint_primary_event_count(fp)
        temporal_center = float(fp[42])
        regularity = float(fp[43])
        attack = float(fp[44])
        onset_span = float(fp[45]) if fp.size > 45 else 0.0
        tail_ratio = float(fp[47]) if fp.size > 47 else 0.0
        decay = float(np.expm1(max(0.0, float(fp[31]))))
    except Exception:
        return False
    if duration_sec >= 1.0 and temporal_center <= 0.16 and attack <= 0.08 and onset_span <= 0.35:
        return True
    if (
        duration_sec >= 1.5
        and temporal_center <= 0.24
        and onset_span <= 0.35
        and (regularity >= 0.75 or tail_ratio >= 0.35 or decay >= 0.18)
    ):
        return True
    return bool(duration_sec >= 3.0 and temporal_center <= 0.3 and onset_span <= 0.45 and transients <= 4)


def fingerprint_loop_evidence_detail(fingerprint: Sequence[float], duration_sec: float) -> Tuple[bool, str]:
    """Strict loop proof using measured audio only.

    Low onset_interval_regularity is loop-like. High values are irregular bursts,
    single hits with tail chatter, or performance/FX material. Older v0.4.16 code
    accidentally used this backwards in some repair thresholds.
    """
    try:
        fp = np.asarray(fingerprint, dtype=np.float32)
        if fp.size < FP_SIZE:
            return False, "fingerprint_too_small"
        transients = fingerprint_primary_event_count(fp)
        regularity = float(fp[43])
        temporal_center = float(fp[42])
        attack = float(fp[44])
        onset_span = float(fp[45]) if fp.size > 45 else 0.0
        event_rate = float(fp[46]) if fp.size > 46 else 0.0
    except Exception as exc:
        return False, f"fingerprint_error:{str(exc)[:80]}"
    if fingerprint_front_loaded_tail_evidence(fp, duration_sec):
        return False, (
            f"front_loaded_single_event_tail_veto duration={duration_sec:.3f} "
            f"transients={transients:.1f} temporal_centroid={temporal_center:.3f} "
            f"regularity={regularity:.3f} span={onset_span:.3f} attack={attack:.3f}"
        )
    if (
        duration_sec >= 1.5
        and transients >= 4
        and regularity <= 0.60
        and temporal_center >= 0.22
        and onset_span >= 0.50
        and event_rate >= 0.75
    ):
        return (
            True,
            f"loop_by_repeated_primary_events transients={transients:.1f} regularity={regularity:.3f} temporal={temporal_center:.3f} span={onset_span:.3f} rate={event_rate:.3f}",
        )
    if (
        duration_sec >= 3.0
        and transients >= 4
        and regularity <= 0.55
        and temporal_center >= 0.22
        and onset_span >= 0.50
        and event_rate >= 0.55
    ):
        return (
            True,
            f"loop_by_long_regular_events transients={transients:.1f} regularity={regularity:.3f} temporal={temporal_center:.3f} span={onset_span:.3f} rate={event_rate:.3f}",
        )
    if (
        duration_sec >= 5.0
        and transients >= 6
        and regularity <= 0.50
        and temporal_center >= 0.28
        and onset_span >= 0.65
        and event_rate >= 0.60
        and attack >= 0.03
    ):
        return (
            True,
            f"loop_by_long_distributed_events transients={transients:.1f} regularity={regularity:.3f} temporal={temporal_center:.3f} span={onset_span:.3f} rate={event_rate:.3f}",
        )

    # v0.6.5: real musical/drum loops are often syncopated, reverberant, or
    # played by humans. They should not need machine-perfect onset regularity.
    # The reliable structure evidence is that events are distributed across most
    # of a multi-second file with the energy centroid in the body, not only at
    # the attack. This fixes sax/brass/reverby instrumental loops and mixed drum
    # beats being treated as structure-unknown. The front-loaded-tail veto above
    # still protects crashes, gongs, impacts, and riser tails from being called
    # loops just because the decay chatters.
    if (
        duration_sec >= 2.0
        and transients >= 8
        and onset_span >= 0.65
        and 0.24 <= temporal_center <= 0.78
        and event_rate >= 1.10
    ):
        return True, (
            f"loop_by_distributed_multi_event_pattern transients={transients:.1f} "
            f"regularity={regularity:.3f} temporal={temporal_center:.3f} "
            f"span={onset_span:.3f} rate={event_rate:.3f} attack={attack:.3f}"
        )
    if (
        duration_sec >= 6.0
        and transients >= 5
        and onset_span >= 0.70
        and 0.25 <= temporal_center <= 0.75
        and event_rate >= 0.75
    ):
        return True, (
            f"loop_by_long_loose_distributed_pattern transients={transients:.1f} "
            f"regularity={regularity:.3f} temporal={temporal_center:.3f} "
            f"span={onset_span:.3f} rate={event_rate:.3f} attack={attack:.3f}"
        )
    return (
        False,
        f"no_strong_loop_proof transients={transients:.1f} regularity={regularity:.3f} temporal={temporal_center:.3f} span={onset_span:.3f} rate={event_rate:.3f} attack={attack:.3f}",
    )


def is_mixed_loop_source(row_or_path, group_key: str = "") -> str:
    """Return mixed-loop role when a training candidate should not pollute pure labels.

    This uses folder/source text only for training-data hygiene. It does not route
    unknown user audio. Mixed instrumental loops and mixed beat/drum loops are
    their own trainer labels because they corrupt pure instrument/drum prototypes.
    """
    return mixed_loop_public_from_context(row_or_path, group_key)


def is_basic_drum_identity_group(group_key: str) -> bool:
    """Deprecated compatibility wrapper. No identity category groups are inferred."""
    return False


def likely_repeated_hit_loop(row: Dict[str, str]) -> bool:
    """Multiple strong independent hits means loop/pattern/fill.

    v0.4.5 is stricter. A long file or a folder/name hint alone is not enough.
    A single long note, riser, drone, impact tail, cymbal tail, sax sustain,
    or guitar ring-out stays in one_shot.
    """
    duration = row_float(row, "duration_sec", "duration", default=0.0)
    onset_count = row_float(
        row,
        "primary_event_count_est",
        "onset_count",
        "significant_transients",
        "transient_count",
        "transients",
        default=0.0,
    )
    regularity = row_float(row, "onset_interval_regularity", default=1.0)
    onset_span = row_float(row, "onset_span_ratio", default=0.0)
    event_rate = row_float(row, "event_rate_hz", default=0.0)
    temporal_center = row_float(row, "temporal_centroid_ratio", default=0.0)
    legacy_periodicity = row_float(row, "loop_periodicity", "tempo_confidence", "pulse_clarity", default=-1.0)
    # Strong current fingerprint evidence wins. Lower regularity is more loop-like.
    if duration >= 1.5 and onset_count >= 4 and regularity <= 0.55 and onset_span >= 0.45 and event_rate >= 0.75:
        return True
    if (
        duration >= 3.0
        and onset_count >= 4
        and regularity <= 0.60
        and temporal_center >= 0.22
        and onset_span >= 0.50
        and event_rate >= 0.55
    ):
        return True

    # Legacy pre-v0.4.3 reports used pulse confidence fields where higher was loop-like.
    if legacy_periodicity >= 0 and duration >= 1.2 and onset_count >= 5 and legacy_periodicity >= 0.25:
        return True
    if legacy_periodicity >= 0 and duration >= 1.8 and onset_count >= 4 and legacy_periodicity >= 0.32:
        return True
    return bool(legacy_periodicity >= 0 and duration >= 2.5 and onset_count >= 3 and legacy_periodicity >= 0.45)


def strong_audio_loop_evidence(row: Dict[str, str]) -> bool:
    """Strict repeated-event evidence used to repair polluted structure labels.

    v0.4.18 deliberately raises the bar.  A longer animal call, cymbal tail,
    riser, drone, crowd swell, water movement, or single foley action can produce
    several detected onsets without being a loop.  Repair to Loops only when the
    audio has both multiple events and pulse/regularity evidence.
    """
    duration = row_float(row, "duration_sec", "duration", default=0.0)
    onset_count = row_float(
        row,
        "primary_event_count_est",
        "onset_count",
        "significant_transients",
        "transient_count",
        "transients",
        default=0.0,
    )
    regularity = row_float(row, "onset_interval_regularity", default=1.0)
    temporal_center = row_float(row, "temporal_centroid_ratio", default=0.0)
    onset_span = row_float(row, "onset_span_ratio", default=0.0)
    event_rate = row_float(row, "event_rate_hz", default=0.0)
    legacy_periodicity = row_float(row, "loop_periodicity", "tempo_confidence", "pulse_clarity", default=-1.0)

    # Current fingerprint fields: lower regularity is more loop-like.
    if duration >= 1.8 and onset_count >= 8 and regularity <= 0.60 and onset_span >= 0.45 and event_rate >= 0.75:
        return True
    if (
        duration >= 3.0
        and onset_count >= 6
        and regularity <= 0.60
        and temporal_center >= 0.22
        and onset_span >= 0.50
        and event_rate >= 0.55
    ):
        return True
    if (
        duration >= 5.0
        and onset_count >= 6
        and regularity <= 0.65
        and temporal_center >= 0.28
        and onset_span >= 0.65
        and event_rate >= 0.55
    ):
        return True

    # Legacy report fields: higher pulse confidence was more loop-like.
    if legacy_periodicity >= 0 and duration >= 1.8 and onset_count >= 8 and legacy_periodicity >= 0.50:
        return True
    if legacy_periodicity >= 0 and duration >= 3.0 and onset_count >= 6 and legacy_periodicity >= 0.45:
        return True
    return bool(legacy_periodicity >= 0 and duration >= 5.0 and onset_count >= 6 and legacy_periodicity >= 0.6)


def likely_long_single_sound(row: Dict[str, str]) -> bool:
    """Report-only trait helper. Long single sounds are still one_shot."""
    duration = row_float(row, "duration_sec", "duration", default=0.0)
    onset_count = row_int(row, "onset_count", "significant_transients", "transient_count", "transients", default=0)
    if likely_repeated_hit_loop(row):
        return False
    if duration >= 1.5 and onset_count <= 1:
        return True
    return bool(duration >= 2.0 and onset_count <= 2)


def strong_fingerprint_loop_evidence(fingerprint: Sequence[float], duration_sec: float) -> bool:
    """Strict audio-only loop evidence from the extracted 45-feature fingerprint."""
    ok, _reason = fingerprint_loop_evidence_detail(fingerprint, duration_sec)
    return bool(ok)


def repair_structure_lane_with_fingerprint(
    base_structure: str, fingerprint: Sequence[float], duration_sec: float, group_key: str = ""
) -> Tuple[str, str]:
    """Trust locked folder structure; return physics note only.

    Earlier code changed loop/one-shot labels during training. That made the
    locked folder tree untrustworthy. Now the folder structure wins and physics
    evidence is only logged for Aaron to review later.
    """
    base = "loop" if str(base_structure or "").lower() == "loop" else "one_shot"
    has_loop_evidence, loop_reason = fingerprint_loop_evidence_detail(fingerprint, duration_sec)
    if base == "loop" and not has_loop_evidence:
        return base, "folder_loop_audio_loop_proof_missing_REPORT_ONLY|" + loop_reason
    if base == "one_shot" and has_loop_evidence:
        return base, "folder_one_shot_audio_looks_loop_like_REPORT_ONLY|" + loop_reason
    return base, loop_reason if base == "loop" else ""


def repair_structure_lane_for_row(row: Dict[str, str], group_key: str, base_structure: str) -> str:
    """Return the locked folder structure without repairing it.

    Physics disagreements are reported by structure_conflict_warning(), not used
    to relabel training examples.
    """
    explicit = structure_bucket_from_group(group_key)
    if explicit in {"one_shot", "loop", "long_fx", "long_running"}:
        return explicit
    return "loop" if str(base_structure or "").lower() == "loop" else "one_shot"


def structure_conflict_warning(row: Dict[str, str], group_key: str, structure: str) -> str:
    """Report possible label dirt without changing training labels."""
    explicit = structure_bucket_from_group(group_key)
    if explicit == "one_shot" and strong_audio_loop_evidence(row):
        return "REPORT_ONLY_explicit_one_shot_folder_audio_looks_loop_like"
    if explicit == "one_shot" and likely_repeated_hit_loop(row):
        return "REPORT_ONLY_explicit_one_shot_folder_audio_looks_loop_like"
    if explicit == "loop" and not strong_audio_loop_evidence(row):
        return "REPORT_ONLY_explicit_loop_folder_audio_lacks_strong_loop_evidence"
    return ""


def is_dirty_non_fx_training_row(row: Dict[str, str]) -> bool:
    """Filename/path dirt filters are disabled for locked training.

    Aaron owns the training folder now.  The code can report physics conflicts,
    but it must not silently reject a teacher because its filename or folder text
    contains words like fx, glitch, alien, soft, etc.
    """
    return False


def is_reviewed_bad_training_row(row: Dict[str, str]) -> bool:
    """No hard-coded filename blacklist is allowed in locked training mode."""
    return False


def clean_hierarchy_part(text: str) -> str:
    text = str(text or "").strip()
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text or "Unknown"


def hierarchy_parts_from_group(group_key: str) -> List[str]:
    parts = []
    for raw in str(group_key or "").replace("\\", "/").split("/"):
        part = raw.strip()
        if not part:
            continue
        low = part.lower().replace("_", " ").strip()
        if low in STRUCTURE_MARKERS:
            continue
        if part.startswith("_") and low in STRUCTURE_MARKERS:
            continue
        parts.append(clean_hierarchy_part(part))
    return parts


def hierarchy_public_label(group_key: str) -> str:
    parts = hierarchy_parts_from_group(group_key)
    return "/".join(parts) if parts else "Unknown"


def training_public_label_from_group(group_key: str, row: Dict[str, str]) -> str:
    """Return the dynamic public training label for a folder row.

    This is the single place where Stage 4 turns a trusted training folder into
    a brain label.  It does not contain category-specific folder repairs.

    Generic grammar:
      1. Split the folder path into parts.
      2. Remove only structure marker folders, such as _ONE_SHOTS, _LOOPS, or _LONG_FX.
      3. Use the remaining folder path exactly as Aaron's public label.
      4. scoped_label() appends the producer-facing structure terminal, preserving
         _LONG_FX as /Long FX instead of collapsing it to /One Shots.

    Examples:
      Drums/Drum Loops/file.wav
        -> public label Drums/Drum Loops
        -> structure inferred as loop from the folder text
        -> internal label Drums/Drum Loops/Loops

      Instruments/My New Folder/_ONE_SHOTS/file.wav
        -> public label Instruments/My New Folder
        -> structure one_shot from the marker
        -> internal label Instruments/My New Folder/One Shots

      FX/Whatever Aaron Adds/_LOOPS/file.wav
        -> public label FX/Whatever Aaron Adds
        -> structure loop from the marker
        -> internal label FX/Whatever Aaron Adds/Loops

      FX/Animals and Creatures/Bird/_LONG_FX/file.wav
        -> public label FX/Animals and Creatures/Bird
        -> structure long_fx from the marker
        -> internal label FX/Animals and Creatures/Bird/Long FX

    Filenames are not used here.  If Aaron wants Bass Loop, Synth Loop, or any
    other exact label, that name must appear in the training folder path.
    """
    return hierarchy_public_label(group_key)


def bird_behavior_public_label(group_key: str, row: Dict[str, str]) -> str:
    """Compatibility wrapper: no behavior/species categories are invented in code.

    The learned folder path is the label. If Aaron wants separate bird-call,
    wing-flutter, or mixed-bird folders, those folders must exist in the
    training tree. This function remains only so older imports do not break.
    """
    return hierarchy_public_label(group_key)


def hierarchy_scoped_label(structure: str, group_key: str) -> str:
    return scoped_label(structure, hierarchy_public_label(group_key))


def folder_parts_for_public_label(label: str) -> List[str]:
    base = public_label(label)
    if "/" in base:
        return [safe_folder_name(p) for p in base.split("/") if p.strip()]
    return [safe_folder_name(base)]


def simplified_hint_for_group(group_key: str, duration_sec: float = 0.0) -> str:
    """Report-only folder hint. It never maps to fixed music categories."""
    return hierarchy_public_label(group_key)


def expected_label_for_group(group_key: str, duration_sec: float = 0.0) -> str:
    """Deprecated compatibility wrapper.

    Older Stage 3/early Stage 4 code mapped folder text into fixed terminal
    category labels. That category vocabulary is no longer allowed in the active
    brain path. The only expected label is the folder-derived label itself.
    """
    return hierarchy_public_label(group_key)


def dct2(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.moveaxis(x, axis, -1)
    n = x.shape[-1]
    if n <= 0:
        return np.moveaxis(x, -1, axis)
    x = np.clip(x, -1.0e6, 1.0e6)
    idx = np.arange(n, dtype=np.float32)
    k = np.arange(n, dtype=np.float32)[:, None]
    basis = np.cos(np.pi / n * (idx + 0.5)[None, :] * k).astype(np.float32)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        y = np.matmul(x.astype(np.float64), basis.T.astype(np.float64)).astype(np.float32)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y[..., 0] *= np.sqrt(1.0 / n)
    if n > 1:
        y[..., 1:] *= np.sqrt(2.0 / n)
    return np.moveaxis(y, -1, axis)


def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def mel_filterbank(sr: int) -> np.ndarray:
    if sr in _MEL_CACHE:
        return _MEL_CACHE[sr]
    mel_pts = np.linspace(float(hz_to_mel(0.0)), float(hz_to_mel(sr / 2.0)), N_MELS + 2)
    hz_pts = mel_to_hz(mel_pts)
    bins = np.floor((N_FFT + 1) * hz_pts / sr).astype(int)
    bins = np.clip(bins, 0, N_FFT // 2)
    fb = np.zeros((N_MELS, N_FFT // 2 + 1), dtype=np.float32)
    for i in range(N_MELS):
        left, center, right = int(bins[i]), int(bins[i + 1]), int(bins[i + 2])
        if center <= left:
            center = min(left + 1, N_FFT // 2)
        if right <= center:
            right = min(center + 1, N_FFT // 2)
        if center > left:
            fb[i, left:center] = np.linspace(0, 1, center - left, endpoint=False)
        if right > center:
            fb[i, center:right] = np.linspace(1, 0, right - center, endpoint=False)
    fb /= np.maximum(fb.sum(axis=1, keepdims=True), 1e-9)
    _MEL_CACHE[sr] = fb
    return fb


def ensure_2d(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if y.shape[1] > 2:
        y = y[:, :2]
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)


def resample_linear(y: np.ndarray, sr: int, target: int = TARGET_SR) -> Tuple[np.ndarray, int]:
    if sr == target or y.size == 0:
        return y.astype(np.float32), sr
    old_n = y.shape[0]
    new_n = max(1, int(round(old_n * target / float(sr))))
    old_x = np.linspace(0.0, 1.0, old_n, endpoint=False, dtype=np.float32)
    new_x = np.linspace(0.0, 1.0, new_n, endpoint=False, dtype=np.float32)
    chans = []
    for ch in range(y.shape[1]):
        chans.append(np.interp(new_x, old_x, y[:, ch]).astype(np.float32))
    return np.vstack(chans).T, target


def read_wav_direct(path: Path) -> Tuple[np.ndarray, int]:
    try:
        return _read_wav_with_python_wave(path)
    except Exception:
        return _read_wav_riff_direct(path)


def _read_wav_with_python_wave(path: Path) -> Tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sr = wf.getframerate()
        sampw = wf.getsampwidth()
        total = wf.getnframes()
        read_frames = min(total, max(N_FFT, int(MAX_ANALYSIS_SECONDS * sr)))
        starts = [0]
        if total > read_frames:
            starts = [0, max(0, (total - read_frames) // 2), max(0, total - read_frames)]
        best = None
        best_rms = -1.0
        for start in sorted(set(starts)):
            wf.setpos(start)
            raw = wf.readframes(read_frames)
            if sampw == 1:
                arr = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            elif sampw == 2:
                arr = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            elif sampw == 3:
                b = np.frombuffer(raw, dtype=np.uint8)
                n = (len(b) // 3) * 3
                b = b[:n].reshape(-1, 3)
                vals = b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16)
                vals = np.where(vals >= (1 << 23), vals - (1 << 24), vals)
                arr = vals.astype(np.float32) / float(1 << 23)
            elif sampw == 4:
                arr = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
            else:
                raise ValueError(f"unsupported WAV sample width {sampw}")
            usable = (arr.size // channels) * channels
            if usable <= 0:
                continue
            y = arr[:usable].reshape(-1, channels)
            y = ensure_2d(y)
            mono = np.mean(y, axis=1)
            rms = float(np.sqrt(np.mean(mono**2))) if mono.size else 0.0
            if rms > best_rms:
                best = y
                best_rms = rms
        if best is None:
            raise ValueError("empty WAV")
        return best, sr


def _read_wav_riff_direct(path: Path) -> Tuple[np.ndarray, int]:
    """Read PCM or IEEE-float WAV when Python ``wave`` cannot open it.

    The project venv intentionally has few dependencies, so some smoke commands
    run without ``soundfile``.  Python 3.9's ``wave`` rejects IEEE-float WAVs,
    which made valid FX fixtures look broken.  This parser handles the small
    subset we need without changing the feature pipeline.
    """
    payload = Path(path).read_bytes()
    fmt, data = _wav_chunks(payload)
    audio_format = int(fmt["audio_format"])
    channels = int(fmt["channels"])
    sr = int(fmt["sample_rate"])
    bits = int(fmt["bits_per_sample"])
    block_align = int(fmt["block_align"])
    if channels <= 0 or sr <= 0 or block_align <= 0:
        raise ValueError("invalid WAV header")
    total = len(data) // block_align
    read_frames = min(total, max(N_FFT, int(MAX_ANALYSIS_SECONDS * sr)))
    starts = [0]
    if total > read_frames:
        starts = [0, max(0, (total - read_frames) // 2), max(0, total - read_frames)]

    best = None
    best_rms = -1.0
    for start in sorted(set(starts)):
        raw = data[start * block_align : (start + read_frames) * block_align]
        arr = _decode_wav_samples(raw, audio_format=audio_format, bits_per_sample=bits)
        usable = (arr.size // channels) * channels
        if usable <= 0:
            continue
        y = ensure_2d(arr[:usable].reshape(-1, channels))
        mono = np.mean(y, axis=1)
        rms = float(np.sqrt(np.mean(mono**2))) if mono.size else 0.0
        if rms > best_rms:
            best = y
            best_rms = rms
    if best is None:
        raise ValueError("empty WAV")
    return best, sr


def _wav_chunks(payload: bytes) -> Tuple[Dict[str, int], bytes]:
    if len(payload) < 12 or payload[:4] != b"RIFF" or payload[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    pos = 12
    fmt: Dict[str, int] | None = None
    audio_data: bytes | None = None
    while pos + 8 <= len(payload):
        chunk_id = payload[pos : pos + 4]
        chunk_size = struct.unpack_from("<I", payload, pos + 4)[0]
        start = pos + 8
        end = min(len(payload), start + chunk_size)
        chunk = payload[start:end]
        if chunk_id == b"fmt " and len(chunk) >= 16:
            audio_format, channels, sample_rate, _byte_rate, block_align, bits = struct.unpack_from("<HHIIHH", chunk, 0)
            if audio_format == 0xFFFE and len(chunk) >= 40:
                # WAVE_FORMAT_EXTENSIBLE stores the real format tag in the
                # first two bytes of the sub-format GUID.
                audio_format = struct.unpack_from("<H", chunk, 24)[0]
            fmt = {
                "audio_format": int(audio_format),
                "channels": int(channels),
                "sample_rate": int(sample_rate),
                "block_align": int(block_align),
                "bits_per_sample": int(bits),
            }
        elif chunk_id == b"data":
            audio_data = chunk
        pos = end + (chunk_size & 1)
    if fmt is None or audio_data is None:
        raise ValueError("missing WAV fmt or data chunk")
    return fmt, audio_data


def _decode_wav_samples(raw: bytes, *, audio_format: int, bits_per_sample: int) -> np.ndarray:
    if audio_format == 3:
        if bits_per_sample == 32:
            return np.frombuffer(raw, dtype="<f4").astype(np.float32)
        if bits_per_sample == 64:
            return np.frombuffer(raw, dtype="<f8").astype(np.float32)
        raise ValueError(f"unsupported IEEE-float WAV depth {bits_per_sample}")
    if audio_format != 1:
        raise ValueError(f"unsupported WAV format {audio_format}")
    if bits_per_sample == 8:
        return (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    if bits_per_sample == 16:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if bits_per_sample == 24:
        b = np.frombuffer(raw, dtype=np.uint8)
        n = (len(b) // 3) * 3
        b = b[:n].reshape(-1, 3)
        vals = b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16)
        vals = np.where(vals >= (1 << 23), vals - (1 << 24), vals)
        return vals.astype(np.float32) / float(1 << 23)
    if bits_per_sample == 32:
        return np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    raise ValueError(f"unsupported PCM WAV depth {bits_per_sample}")


def read_audio(path: Path) -> Tuple[np.ndarray, int]:
    suffix = path.suffix.lower()
    if suffix == ".wav":
        try:
            return read_wav_direct(path)
        except Exception:
            pass
    if _sf is not None:
        info = _sf.info(str(path))
        sr = int(info.samplerate)
        total = int(info.frames or 0)
        read_frames = (
            min(total, max(N_FFT, int(MAX_ANALYSIS_SECONDS * sr)))
            if total
            else max(N_FFT, int(MAX_ANALYSIS_SECONDS * sr))
        )
        starts = [0]
        if total > read_frames:
            starts = [0, max(0, (total - read_frames) // 2), max(0, total - read_frames)]
        best = None
        best_rms = -1.0
        with _sf.SoundFile(str(path), "r") as snd:
            for start in sorted(set(starts)):
                try:
                    snd.seek(start)
                    y = snd.read(frames=read_frames, always_2d=True, dtype="float32")
                    y = ensure_2d(y)
                    mono = np.mean(y, axis=1)
                    rms = float(np.sqrt(np.mean(mono**2))) if mono.size else 0.0
                    if rms > best_rms:
                        best = y
                        best_rms = rms
                except Exception:
                    continue
        if best is not None:
            return best, sr
    raise ValueError("no audio reader for this file")


def trim_and_normalize(y: np.ndarray) -> Tuple[np.ndarray, str]:
    y = ensure_2d(y)
    if y.size == 0:
        return np.zeros((N_FFT, 1), dtype=np.float32), "empty"
    y = y - np.mean(y, axis=0, keepdims=True)
    mono = np.mean(y, axis=1)
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    if peak <= 1e-9:
        return np.zeros((N_FFT, y.shape[1]), dtype=np.float32), "silent"
    thr = peak * (10 ** (-55.0 / 20.0))
    idx = np.where(np.abs(mono) > thr)[0]
    if idx.size:
        y = y[int(idx[0]) : int(idx[-1]) + 1]
    peak2 = float(np.max(np.abs(y))) if y.size else 0.0
    if peak2 > 1e-9:
        y = y / peak2
    return y.astype(np.float32), "ok"


def frame_audio(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.size < N_FFT:
        y = np.pad(y, (0, N_FFT - y.size), mode="constant")
    starts = list(range(0, max(1, y.size - N_FFT + 1), HOP)) or [0]
    win = np.hanning(N_FFT).astype(np.float32)
    frames = []
    for s in starts:
        f = y[s : s + N_FFT]
        if f.size < N_FFT:
            f = np.pad(f, (0, N_FFT - f.size), mode="constant")
        frames.append(f * win)
    return np.vstack(frames)


def spectral_entropy(power: np.ndarray) -> np.ndarray:
    p = power / np.maximum(np.sum(power, axis=1, keepdims=True), 1e-12)
    ent = -np.sum(p * np.log2(p + 1e-12), axis=1)
    return ent / max(1e-9, math.log2(power.shape[1]))


def clean_onset_peaks(frames: np.ndarray, sr: int) -> List[int]:
    """Return cleaned onset peak frame indices from an RMS-envelope diff.

    This keeps transient count and interval-regularity using the same detector,
    so a loop-like score is not derived from a different onset policy.
    """
    env = np.sqrt(np.mean(frames**2, axis=1))
    if env.size <= 1:
        return []
    diff = np.maximum(0.0, np.diff(env, prepend=env[0]))
    med = float(np.median(diff))
    mad = float(np.median(np.abs(diff - med))) + 1e-12
    thr = max(med + 3.0 * mad, float(np.max(diff)) * 0.18)
    peaks = np.where(diff >= thr)[0]
    # Very short one-shots can begin at full energy, so an envelope-diff
    # detector sees no "rise" and incorrectly reports zero events.  If the
    # file has real energy but no rise peaks, keep the strongest energy frame
    # as one primary event.  This repairs guiro/hat/rim/clap training recall
    # without using filenames or folder labels.
    if peaks.size == 0 and float(np.max(env)) > 1e-6:
        return [int(np.argmax(env))]
    min_gap = max(1, int(0.055 / (HOP / sr)))
    clean: List[int] = []
    last = -999
    for idx in peaks:
        if int(idx) - last >= min_gap:
            clean.append(int(idx))
            last = int(idx)
    return clean


def onset_count(frames: np.ndarray, sr: int) -> int:
    return len(clean_onset_peaks(frames, sr))


def onset_interval_regularity(frames: np.ndarray, sr: int) -> float:
    """Coefficient of variation of inter-onset intervals.

    Low values mean regularly spaced onsets, which is loop-like.  High values
    mean a single hit, clustered burst, or irregular FX/performance material.
    Fewer than three detected peaks returns 1.0 by convention. Two peaks only
    gives one interval, whose standard deviation is always 0.0, and would make
    a hit-plus-echo look like a perfectly clocked loop.
    """
    peaks = clean_onset_peaks(frames, sr)
    if len(peaks) < 3:
        return 1.0
    intervals = np.diff(np.asarray(peaks, dtype=np.float32))
    mean_ioi = float(np.mean(intervals))
    std_ioi = float(np.std(intervals))
    return float(min(4.0, std_ioi / (mean_ioi + 1e-9)))


def onset_span_ratio(frames: np.ndarray, sr: int) -> float:
    """How much of the file the detected primary events occupy.

    This fixes the old blind spot where three fast flams inside one hit looked
    like a loop. A true loop has repeated primary events spread across the file;
    a one-shot with tail chatter has clustered events near the front.
    """
    peaks = clean_onset_peaks(frames, sr)
    if len(peaks) < 2 or frames.shape[0] <= 1:
        return 0.0
    return float(max(0.0, min(1.0, (max(peaks) - min(peaks)) / float(max(1, frames.shape[0] - 1)))))


def event_rate_hz(frames: np.ndarray, sr: int) -> float:
    peaks = clean_onset_peaks(frames, sr)
    dur = float((frames.shape[0] * HOP) / max(1, sr))
    if dur <= 0.0:
        return 0.0
    return float(min(30.0, len(peaks) / dur))


def tail_energy_ratio(mono: np.ndarray) -> float:
    """Energy after the first third divided by total energy.

    High values mean the sound keeps living after the attack. This is useful
    diagnostic evidence but not loop proof by itself.
    """
    y = np.asarray(mono, dtype=np.float32)
    if y.size <= 4:
        return 0.0
    power = y**2
    total = float(np.sum(power)) + 1e-12
    start = int(y.size * 0.33)
    return float(max(0.0, min(1.0, np.sum(power[start:]) / total)))


def attack_rise_time_norm(mono: np.ndarray) -> float:
    """Normalized time from 10% to 90% of peak absolute amplitude.

    Fast hits approach 0.0.  Slow swells, reverse tails, pads, and risers move
    upward.  This is intentionally separate from temporal centroid because
    attack speed and energy center describe different envelope behavior.
    """
    y = np.abs(np.asarray(mono, dtype=np.float32))
    if y.size <= 1:
        return 0.0
    peak = float(np.max(y))
    if peak < 1e-9:
        return 0.0
    lo = np.where(y >= peak * 0.10)[0]
    hi = np.where(y >= peak * 0.90)[0]
    if lo.size == 0 or hi.size == 0:
        return 0.0
    rise = max(0, int(hi[0]) - int(lo[0]))
    return float(min(1.0, rise / float(max(1, y.size))))


def fingerprint_sanity_warnings(predicted_label: str, fingerprint: Sequence[float]) -> str:
    """Report-only physical tension warnings without named-category routing.

    These warnings describe measured conflicts only. They do not mention or
    manufacture terminal sound categories, and they never change placement.
    """
    fp = np.asarray(fingerprint, dtype=np.float32)
    warnings_out: List[str] = []
    sub_bass = float(fp[36]) if fp.size > 36 else 0.0
    air = float(fp[40]) if fp.size > 40 else 0.0
    zcr = float(fp[41]) if fp.size > 41 else 0.0
    temporal = float(fp[42]) if fp.size > 42 else 0.0
    regularity = float(fp[43]) if fp.size > 43 else 1.0
    attack = float(fp[44]) if fp.size > 44 else 0.0
    tail = float(fp[47]) if fp.size > 47 else 0.0
    if sub_bass >= 0.55 and air <= 0.03 and temporal <= 0.25:
        warnings_out.append("low_front_loaded_sub_heavy_audio")
    if air >= 0.35 and sub_bass <= 0.02:
        warnings_out.append("very_bright_low_sub_audio")
    if regularity > 1.20:
        warnings_out.append("irregular_onset_pattern")
    if temporal < 0.25 and attack < 0.02 and tail >= 0.25:
        warnings_out.append("front_loaded_audio_with_long_tail")
    if zcr < 0.02 and air >= 0.20:
        warnings_out.append("bright_audio_with_low_zero_crossing")
    return ";".join(warnings_out)


def safe_feature_value(fingerprint: Sequence[float], index: int, default: float = 0.0) -> float:
    try:
        fp = np.asarray(fingerprint, dtype=np.float32)
        if int(index) < 0 or int(index) >= fp.size:
            return float(default)
        value = float(fp[int(index)])
        if not math.isfinite(value):
            return float(default)
        return value
    except Exception:
        return float(default)


def fingerprint_physics_tags(fingerprint: Sequence[float], duration_sec: float) -> str:
    """Compact human-readable physics labels lifted from the older sorter style.

    These tags are evidence labels only. They do not override Phase 3 prediction.
    """
    transients = float(np.expm1(max(0.0, safe_feature_value(fingerprint, 35))))
    crest = float(np.expm1(max(0.0, safe_feature_value(fingerprint, 27))))
    decay = float(np.expm1(max(0.0, safe_feature_value(fingerprint, 31))))
    sub = safe_feature_value(fingerprint, 36)
    bass = safe_feature_value(fingerprint, 37)
    safe_feature_value(fingerprint, 38)
    presence = safe_feature_value(fingerprint, 39)
    air = safe_feature_value(fingerprint, 40)
    zcr = safe_feature_value(fingerprint, 41)
    temporal = safe_feature_value(fingerprint, 42)
    regularity = safe_feature_value(fingerprint, 43, 1.0)
    attack = safe_feature_value(fingerprint, 44)
    onset_span = safe_feature_value(fingerprint, 45)
    safe_feature_value(fingerprint, 46)
    tail_ratio = safe_feature_value(fingerprint, 47)
    flux_variance = safe_feature_value(fingerprint, 48)
    pitch_conf = safe_feature_value(fingerprint, 49)
    flatness = safe_feature_value(fingerprint, 29)
    entropy = safe_feature_value(fingerprint, 30)
    slope = safe_feature_value(fingerprint, 34)
    width = safe_feature_value(fingerprint, 32)
    tags: List[str] = []
    if duration_sec <= 0.35:
        tags.append("tiny_short")
    elif duration_sec <= 1.3:
        tags.append("short")
    elif duration_sec >= 4.0:
        tags.append("long")
    if transients <= 1.5:
        tags.append("single_event_or_sustain")
    elif transients >= 4:
        tags.append("multi_event")
    if temporal <= 0.18:
        tags.append("front_loaded")
    elif temporal >= 0.45:
        tags.append("energy_spread_late")
    if regularity <= 0.55 and transients >= 3:
        tags.append("regular_repetition")
    elif regularity >= 0.90:
        tags.append("irregular_or_tail_chatter")
    if transients >= 3 and onset_span >= 0.45 and duration_sec >= 1.2:
        tags.append("events_spread_across_file")
    elif transients >= 2 and onset_span <= 0.25 and temporal <= 0.22:
        tags.append("clustered_front_events")
    if duration_sec >= 1.5 and transients >= 4 and onset_span >= 0.45 and regularity <= 0.60:
        tags.append("loop_pulse_candidate")
    if tail_ratio >= 0.55 and temporal <= 0.28:
        tags.append("long_tail_or_sustain")
    if attack <= 0.04:
        tags.append("fast_attack")
    elif attack >= 0.20:
        tags.append("slow_attack")
    if decay >= 0.45:
        tags.append("long_tail")
    if crest >= 8.0:
        tags.append("punchy")
    if width >= 0.35:
        tags.append("wide")
    if sub + bass >= 0.55:
        tags.append("low_heavy")
    if sub >= 0.45 and presence + air <= 0.10:
        tags.append("sub_click_or_low_blip_risk")
    if presence + air >= 0.45:
        tags.append("bright_high_energy")
    if entropy >= 0.62 or flatness >= 0.18 or zcr >= 0.15:
        tags.append("noisy")
    if entropy <= 0.42 and flatness <= 0.10:
        tags.append("tonal_or_resonant")
    if pitch_conf >= 0.55:
        tags.append("periodic_pitch_confident")
    elif pitch_conf <= 0.15 and (flatness >= 0.12 or zcr >= 0.12):
        tags.append("aperiodic_noise_like")
    if flux_variance >= 0.012 and transients >= 3:
        tags.append("high_flux_variation")
    if abs(slope) >= 0.10:
        tags.append("moving_up" if slope > 0 else "moving_down")
    return ";".join(tags) or "plain"


def fingerprint_physics_dict(fingerprint: Sequence[float], duration_sec: float) -> Dict[str, str]:
    transients = float(np.expm1(max(0.0, safe_feature_value(fingerprint, 35))))
    crest = float(np.expm1(max(0.0, safe_feature_value(fingerprint, 27))))
    decay = float(np.expm1(max(0.0, safe_feature_value(fingerprint, 31))))
    sub = safe_feature_value(fingerprint, 36)
    bass = safe_feature_value(fingerprint, 37)
    mid = safe_feature_value(fingerprint, 38)
    presence = safe_feature_value(fingerprint, 39)
    air = safe_feature_value(fingerprint, 40)
    zcr = safe_feature_value(fingerprint, 41)
    temporal = safe_feature_value(fingerprint, 42)
    regularity = safe_feature_value(fingerprint, 43, 1.0)
    attack = safe_feature_value(fingerprint, 44)
    onset_span = safe_feature_value(fingerprint, 45)
    event_rate = safe_feature_value(fingerprint, 46)
    tail_ratio = safe_feature_value(fingerprint, 47)
    flux_variance = safe_feature_value(fingerprint, 48)
    pitch_conf = safe_feature_value(fingerprint, 49)
    flatness = safe_feature_value(fingerprint, 29)
    entropy = safe_feature_value(fingerprint, 30)
    slope = safe_feature_value(fingerprint, 34)
    width = safe_feature_value(fingerprint, 32)
    ms = safe_feature_value(fingerprint, 33)
    return {
        "duration_sec": f"{float(duration_sec):.6f}",
        "primary_event_count_est": f"{transients:.6f}",
        "crest_est": f"{crest:.6f}",
        "decay_ratio_est": f"{decay:.6f}",
        "sub_bass_ratio_lt_150hz": f"{sub:.6f}",
        "bass_ratio_150_500hz": f"{bass:.6f}",
        "mid_ratio_500_2000hz": f"{mid:.6f}",
        "presence_ratio_2000_8000hz": f"{presence:.6f}",
        "air_ratio_gt_8000hz": f"{air:.6f}",
        "low_total_ratio_lt_500hz": f"{(sub + bass):.6f}",
        "high_total_ratio_gt_2000hz": f"{(presence + air):.6f}",
        "zcr_mean": f"{zcr:.6f}",
        "spectral_flatness_mean": f"{flatness:.6f}",
        "spectral_entropy_mean": f"{entropy:.6f}",
        "centroid_slope_norm": f"{slope:.6f}",
        "stereo_width": f"{width:.6f}",
        "mid_side_ratio": f"{ms:.6f}",
        "temporal_centroid_ratio": f"{temporal:.6f}",
        "onset_interval_regularity": f"{regularity:.6f}",
        "attack_rise_time_norm": f"{attack:.6f}",
        "onset_span_ratio": f"{onset_span:.6f}",
        "event_rate_hz": f"{event_rate:.6f}",
        "tail_energy_ratio": f"{tail_ratio:.6f}",
        "spectral_flux_variance": f"{flux_variance:.6f}",
        "pitch_confidence": f"{pitch_conf:.6f}",
    }


def fingerprint_physics_summary(fingerprint: Sequence[float], duration_sec: float) -> str:
    d = fingerprint_physics_dict(fingerprint, duration_sec)
    return (
        f"dur={d['duration_sec']} events={d['primary_event_count_est']} "
        f"sub={d['sub_bass_ratio_lt_150hz']} bass={d['bass_ratio_150_500hz']} "
        f"mid={d['mid_ratio_500_2000hz']} hi={d['high_total_ratio_gt_2000hz']} "
        f"temporal={d['temporal_centroid_ratio']} regularity={d['onset_interval_regularity']} "
        f"span={d['onset_span_ratio']} rate={d['event_rate_hz']} tail={d['tail_energy_ratio']} "
        f"flux_var={d['spectral_flux_variance']} pitch_conf={d['pitch_confidence']} "
        f"attack={d['attack_rise_time_norm']} tags={fingerprint_physics_tags(fingerprint, duration_sec)}"
    )


def nearest_training_examples_for_label(
    brain: dict, label: str, fingerprint: Sequence[float], limit: int = 3
) -> List[Dict[str, str]]:
    """Return nearest kept training teachers for a predicted/expected label.

    This is Phase 3 diagnostic evidence only. It helps explain why the brain
    chose a label by showing the actual teachers closest to the tested file.
    It does not change prediction or placement.
    """
    examples_by_label = brain.get("training_examples_detailed_by_label", {})
    examples = examples_by_label.get(label, []) if isinstance(examples_by_label, dict) else []
    if not examples:
        return []
    try:
        raw_x = np.asarray(fingerprint, dtype=np.float32)
        mean, std = scaler_for_label(brain, label)
        weights = np.asarray(brain.get("feature_weights", FEATURE_WEIGHTS), dtype=np.float32)
        xw = ((raw_x - mean) / std) * weights
    except Exception:
        return []
    scored: List[Tuple[float, Dict[str, str]]] = []
    for ex in examples:
        try:
            ex_fp = np.asarray(ex.get("fingerprint", []), dtype=np.float32)
            if ex_fp.shape != raw_x.shape:
                continue
            ew = ((ex_fp - mean) / std) * weights
            dist = float(np.sqrt(np.mean((xw - ew) ** 2)))
            scored.append(
                (
                    dist,
                    {
                        "distance": f"{dist:.4f}",
                        "source_path": str(ex.get("source_path", "")),
                        "source_pack": str(ex.get("source_pack", "")),
                        "duration_sec": f"{float(ex.get('duration_sec', 0.0) or 0.0):.3f}",
                        "physics_tags": str(ex.get("physics_tags", "")),
                        "physics_summary": str(ex.get("physics_summary", "")),
                    },
                )
            )
        except Exception:
            continue
    scored.sort(key=lambda item: item[0])
    return [item for _dist, item in scored[: max(0, int(limit))]]


def format_nearest_training_examples(examples: List[Dict[str, str]]) -> str:
    """Compact nearest-teacher evidence for CSV review."""
    parts = []
    for ex in examples:
        name = Path(str(ex.get("source_path", ""))).name
        pack = str(ex.get("source_pack", ""))
        dist = str(ex.get("distance", ""))
        tags = str(ex.get("physics_tags", ""))
        parts.append(f"d={dist} pack={pack} file={name} tags={tags}")
    return " || ".join(parts)


def write_training_physics_summary(train_rows: List[FeatureRow], reports_dir: Path) -> None:
    """Summarize kept teacher physics by label so dirty prototypes are visible."""
    by_label: Dict[str, List[FeatureRow]] = defaultdict(list)
    for row in train_rows:
        by_label[row.label].append(row)
    out_rows: List[Dict[str, str]] = []
    for label in sorted(by_label):
        rows = by_label[label]
        fps = (
            np.vstack([np.asarray(r.fingerprint, dtype=np.float32) for r in rows])
            if rows
            else np.zeros((0, FP_SIZE), dtype=np.float32)
        )
        durations = [float(r.duration_sec) for r in rows]
        tag_counter: Counter = Counter()
        source_counter: Counter = Counter()
        for r in rows:
            for tag in fingerprint_physics_tags(r.fingerprint, r.duration_sec).split(";"):
                if tag:
                    tag_counter[tag] += 1
            source_counter[r.source_pack or source_pack_key_for_path(r.path)] += 1

        def feature_mean(index: int, default: float = 0.0, label_fingerprints: np.ndarray = fps) -> float:
            if label_fingerprints.size == 0:
                return default
            return float(np.nanmean(label_fingerprints[:, index]))

        out_rows.append(
            {
                "internal_label": label,
                "public_label": public_label(label),
                "top": top_for_public_label(label),
                "structure": label_default_structure(label),
                "kept_count": str(len(rows)),
                "source_group_count": str(len(source_counter)),
                "top_source_groups": ";".join(f"{k}:{v}" for k, v in source_counter.most_common(8)),
                "duration_mean": f"{float(np.mean(durations)) if durations else 0.0:.6f}",
                "duration_median": f"{float(np.median(durations)) if durations else 0.0:.6f}",
                "primary_event_count_mean": f"{float(np.mean([np.expm1(max(0.0, safe_feature_value(r.fingerprint, 35))) for r in rows])) if rows else 0.0:.6f}",
                "temporal_centroid_mean": f"{feature_mean(42):.6f}",
                "onset_regularity_mean": f"{feature_mean(43):.6f}",
                "onset_span_mean": f"{feature_mean(45):.6f}",
                "event_rate_mean": f"{feature_mean(46):.6f}",
                "tail_energy_ratio_mean": f"{feature_mean(47):.6f}",
                "spectral_flux_variance_mean": f"{feature_mean(48):.6f}",
                "pitch_confidence_mean": f"{feature_mean(49):.6f}",
                "sub_bass_ratio_mean": f"{feature_mean(36):.6f}",
                "bass_ratio_mean": f"{feature_mean(37):.6f}",
                "mid_ratio_mean": f"{feature_mean(38):.6f}",
                "high_total_ratio_mean": f"{(feature_mean(39) + feature_mean(40)):.6f}",
                "zcr_mean": f"{feature_mean(41):.6f}",
                "physics_tag_counts": ";".join(f"{k}:{v}" for k, v in tag_counter.most_common(20)),
                "review_hint": "Look for dirty labels: fake loops with high irregularity, tom/kick labels with sub_click_or_low_blip_risk, or pure labels with mixed-loop-looking tags.",
            }
        )
    fields = [
        "internal_label",
        "public_label",
        "top",
        "structure",
        "kept_count",
        "source_group_count",
        "top_source_groups",
        "duration_mean",
        "duration_median",
        "primary_event_count_mean",
        "temporal_centroid_mean",
        "onset_regularity_mean",
        "onset_span_mean",
        "event_rate_mean",
        "tail_energy_ratio_mean",
        "spectral_flux_variance_mean",
        "pitch_confidence_mean",
        "sub_bass_ratio_mean",
        "bass_ratio_mean",
        "mid_ratio_mean",
        "high_total_ratio_mean",
        "zcr_mean",
        "physics_tag_counts",
        "review_hint",
    ]
    write_csv(reports_dir / "training_physics_summary_by_label.csv", out_rows, fields)


def band_energy_ratio(power: np.ndarray, freqs: np.ndarray, low_hz: float, high_hz: Optional[float]) -> float:
    """Return file-level energy ratio inside a frequency band.

    This is intentionally direct, not inferred through MFCCs, because kick, bass,
    hat, cymbal, chirp, and flutter separation needs explicit low/high balance.
    """
    if power.size == 0 or freqs.size == 0:
        return 0.0
    mask = freqs >= float(low_hz) if high_hz is None else (freqs >= float(low_hz)) & (freqs < float(high_hz))
    total_energy = float(np.sum(power)) + 1e-12
    if not np.any(mask):
        return 0.0
    return float(np.sum(power[:, mask]) / total_energy)


def zero_crossing_rate_mean(frames: np.ndarray) -> float:
    """Mean zero-crossing rate per analysis frame, normalized to 0..1."""
    if frames.size == 0:
        return 0.0
    signs = np.signbit(frames)
    zc = np.mean(signs[:, 1:] != signs[:, :-1], axis=1)
    return float(np.mean(zc)) if zc.size else 0.0


def temporal_centroid_ratio(mono: np.ndarray) -> float:
    """Energy center of mass in time, 0=front-loaded and 1=end-loaded."""
    y = np.asarray(mono, dtype=np.float32)
    if y.size <= 1:
        return 0.0
    energy = y * y
    total = float(np.sum(energy)) + 1e-12
    idx = np.arange(y.size, dtype=np.float32) / float(max(1, y.size - 1))
    return float(np.sum(idx * energy) / total)


def stereo_metrics(stereo: np.ndarray) -> Tuple[float, float]:
    stereo = ensure_2d(stereo)
    if stereo.shape[1] < 2:
        return 0.0, 1.0
    L = stereo[:, 0].astype(np.float32)
    R = stereo[:, 1].astype(np.float32)
    mid = (L + R) * 0.5
    side = (L - R) * 0.5
    mid_rms = float(np.sqrt(np.mean(mid**2)))
    side_rms = float(np.sqrt(np.mean(side**2)))
    width = side_rms / (mid_rms + side_rms + 1e-9)
    ms_ratio = mid_rms / (mid_rms + side_rms + 1e-9)
    return float(width), float(ms_ratio)


def pitch_confidence(mono: np.ndarray, sr: int) -> float:
    """Normalized autocorrelation peak in a musical pitch range.

    This is a cheap periodicity feature, not a pitch detector. High values mean
    stable cycle repetition (bass, synth, pad, tonal instrument); low values mean
    aperiodic noise/transient material (hat, snare noise, many FX).
    """
    try:
        y = np.asarray(mono, dtype=np.float32)
        analysis = y[: min(y.size, int(0.15 * sr))]
        if analysis.size < 64:
            return 0.0
        analysis = analysis - float(np.mean(analysis))
        rms = float(np.sqrt(np.mean(analysis**2)))
        if rms <= 1e-7:
            return 0.0
        n = int(analysis.size)
        fft = np.fft.rfft(analysis, n=2 * n)
        acf = np.fft.irfft(fft * np.conj(fft))[:n]
        acf = acf / (float(acf[0]) + 1e-9)
        min_lag = max(1, int(sr / 2000.0))
        max_lag = min(n - 1, int(sr / 50.0))
        if min_lag >= max_lag:
            return 0.0
        return float(np.clip(np.max(acf[min_lag:max_lag]), 0.0, 1.0))
    except Exception:
        return 0.0


def _safe_log_ratio(num: float, den: float) -> float:
    try:
        return float(math.log1p(max(0.0, float(num)) / max(1e-12, float(den))))
    except Exception:
        return 0.0


def _estimate_f0_confidence_and_hz(segment: np.ndarray, sr: int) -> Tuple[float, float]:
    """Cheap autocorrelation F0 estimator used for expanded physics.

    Returns (confidence, frequency_hz). This is not a tuner; it is a measured
    periodicity descriptor so the brain can learn tonal-vs-noisy folder physics.
    """
    try:
        y = np.asarray(segment, dtype=np.float32).reshape(-1)
        if y.size < max(96, int(0.012 * sr)):
            return 0.0, 0.0
        y = y[: min(y.size, int(0.25 * sr))]
        y = y - float(np.mean(y))
        rms = float(np.sqrt(np.mean(y**2)))
        if rms <= 1e-7:
            return 0.0, 0.0
        y = y * np.hanning(y.size).astype(np.float32)
        n = int(y.size)
        fft = np.fft.rfft(y, n=2 * n)
        acf = np.fft.irfft(fft * np.conj(fft))[:n]
        if float(acf[0]) <= 1e-12:
            return 0.0, 0.0
        acf = acf / (float(acf[0]) + 1e-12)
        min_lag = max(1, int(sr / 2000.0))
        max_lag = min(n - 1, int(sr / 45.0))
        if min_lag >= max_lag:
            return 0.0, 0.0
        region = acf[min_lag:max_lag]
        if region.size <= 0:
            return 0.0, 0.0
        offset = int(np.argmax(region))
        lag = min_lag + offset
        conf = float(np.clip(region[offset], 0.0, 1.0))
        hz = float(sr / max(1, lag))
        return conf, hz
    except Exception:
        return 0.0, 0.0


def _frame_level_f0_stats(mono: np.ndarray, sr: int) -> Dict[str, float]:
    """Frame-level F0 contour stats for v0.5 expanded physics."""
    y = np.asarray(mono, dtype=np.float32).reshape(-1)
    if y.size < int(0.05 * sr):
        conf, hz = _estimate_f0_confidence_and_hz(y, sr)
        return {
            "f0_median_hz": hz if conf >= 0.25 else 0.0,
            "f0_voiced_ratio": 1.0 if conf >= 0.25 else 0.0,
            "f0_stability_cents": 0.0,
            "f0_slope_cents_per_sec": 0.0,
        }
    win = max(256, int(0.046 * sr))
    hop = max(128, int(0.023 * sr))
    hz_vals: List[float] = []
    times: List[float] = []
    total = 0
    for start in range(0, max(1, y.size - win + 1), hop):
        seg = y[start : start + win]
        if seg.size < win:
            continue
        total += 1
        conf, hz = _estimate_f0_confidence_and_hz(seg, sr)
        if conf >= 0.25 and 45.0 <= hz <= 2000.0:
            hz_vals.append(hz)
            times.append((start + win * 0.5) / float(sr))
    if not hz_vals or total <= 0:
        return {"f0_median_hz": 0.0, "f0_voiced_ratio": 0.0, "f0_stability_cents": 0.0, "f0_slope_cents_per_sec": 0.0}
    hz_arr = np.asarray(hz_vals, dtype=np.float32)
    med = float(np.median(hz_arr))
    cents = 1200.0 * np.log2(np.maximum(hz_arr, 1e-6) / max(med, 1e-6))
    stability = float(np.std(cents)) if cents.size > 1 else 0.0
    slope = 0.0
    if len(hz_vals) >= 3 and len(times) == len(hz_vals):
        t = np.asarray(times, dtype=np.float32)
        if float(np.max(t) - np.min(t)) > 1e-6:
            slope = float(np.polyfit(t, cents.astype(np.float32), 1)[0])
    return {
        "f0_median_hz": med,
        "f0_voiced_ratio": float(len(hz_vals) / max(1, total)),
        "f0_stability_cents": min(2400.0, abs(stability)),
        "f0_slope_cents_per_sec": float(np.clip(slope, -4800.0, 4800.0)),
    }


def _segment_descriptor(segment: np.ndarray, sr: int) -> Dict[str, float]:
    """Attack/body/tail descriptor block for expanded physics."""
    y = np.asarray(segment, dtype=np.float32).reshape(-1)
    if y.size < 64 or float(np.max(np.abs(y))) <= 1e-9:
        return {
            "pitch": 0.0,
            "flatness": 0.0,
            "entropy": 0.0,
            "zcr": 0.0,
            "high_ratio": 0.0,
            "low_ratio": 0.0,
            "noise_ratio": 0.0,
        }
    if y.size < N_FFT:
        y = np.pad(y, (0, N_FFT - y.size), mode="constant")
    frames = frame_audio(y)
    mag = np.abs(np.fft.rfft(frames, axis=1)).astype(np.float32) + 1e-12
    power = mag * mag
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / sr).astype(np.float32)
    flat = float(np.mean(np.exp(np.mean(np.log(mag), axis=1)) / (np.mean(mag, axis=1) + 1e-12)))
    ent = float(np.mean(spectral_entropy(power)))
    zcr = zero_crossing_rate_mean(frames)
    low = band_energy_ratio(power, freqs, 0.0, 500.0)
    high = band_energy_ratio(power, freqs, 2000.0, None)
    conf, _hz = _estimate_f0_confidence_and_hz(y, sr)
    # Noise ratio is a measured proxy: noisy/aperiodic material tends toward high
    # flatness/entropy/ZCR and low pitch confidence.
    noise = float(np.clip((flat + ent + min(1.0, zcr * 2.0) + (1.0 - conf)) / 4.0, 0.0, 1.0))
    return {
        "pitch": conf,
        "flatness": flat,
        "entropy": ent,
        "zcr": zcr,
        "high_ratio": high,
        "low_ratio": low,
        "noise_ratio": noise,
    }


def _spectral_peak_descriptors(power: np.ndarray, freqs: np.ndarray, f0_hz: float) -> Dict[str, float]:
    """Average-spectrum peak/formant-like descriptors."""
    try:
        spec = np.mean(np.asarray(power, dtype=np.float64), axis=0)
        if spec.size < 5 or float(np.sum(spec)) <= 1e-12:
            return {
                "spectral_peak_count": 0.0,
                "top1_peak_frequency_hz": 0.0,
                "top2_peak_frequency_hz": 0.0,
                "top3_peak_frequency_hz": 0.0,
                "peak_bandwidth_mean_hz": 0.0,
                "spectral_peak_stability": 0.0,
                "formant_like_peak_spacing": 0.0,
                "spectral_envelope_slope": 0.0,
            }
        # Ignore DC and extreme top bin noise.
        np.arange(spec.size)
        local = (spec[1:-1] > spec[:-2]) & (spec[1:-1] >= spec[2:])
        peak_idx = np.where(local)[0] + 1
        threshold = max(float(np.median(spec) * 6.0), float(np.max(spec) * 0.08))
        peak_idx = peak_idx[spec[peak_idx] >= threshold]
        if peak_idx.size == 0:
            peak_idx = np.asarray([int(np.argmax(spec))])
        order = peak_idx[np.argsort(spec[peak_idx])[::-1]]
        top = order[:3]
        top_freqs = [float(freqs[int(i)]) for i in top]
        while len(top_freqs) < 3:
            top_freqs.append(0.0)
        bws = []
        for i in top[:8]:
            i = int(i)
            half = float(spec[i] * 0.5)
            left = i
            while left > 1 and spec[left] >= half:
                left -= 1
            right = i
            while right < spec.size - 2 and spec[right] >= half:
                right += 1
            bws.append(float(freqs[right] - freqs[left]))
        peak_count = float(min(64, peak_idx.size))
        stability = float(np.sum(spec[top]) / (np.sum(spec) + 1e-12)) if top.size else 0.0
        spacing = 0.0
        if len(top_freqs) >= 3 and top_freqs[1] > 0 and top_freqs[2] > 0:
            diffs = np.diff(np.asarray([f for f in top_freqs if f > 0.0], dtype=np.float32))
            if diffs.size:
                spacing = float(np.std(diffs) / (np.mean(diffs) + 1e-9))
        # Log spectral envelope slope over nonzero bands.
        mask = (freqs >= 80.0) & (freqs <= min(10000.0, float(freqs[-1]))) & (spec > 0)
        slope = 0.0
        if int(np.sum(mask)) > 8:
            x = np.log1p(freqs[mask].astype(np.float64))
            y = np.log1p(spec[mask].astype(np.float64))
            slope = float(np.polyfit(x, y, 1)[0])
        return {
            "spectral_peak_count": peak_count,
            "top1_peak_frequency_hz": top_freqs[0],
            "top2_peak_frequency_hz": top_freqs[1],
            "top3_peak_frequency_hz": top_freqs[2],
            "peak_bandwidth_mean_hz": float(np.mean(bws)) if bws else 0.0,
            "spectral_peak_stability": float(np.clip(stability, 0.0, 1.0)),
            "formant_like_peak_spacing": float(np.clip(spacing, 0.0, 4.0)),
            "spectral_envelope_slope": float(np.clip(slope, -20.0, 20.0)),
        }
    except Exception:
        return {
            k: 0.0
            for k in [
                "spectral_peak_count",
                "top1_peak_frequency_hz",
                "top2_peak_frequency_hz",
                "top3_peak_frequency_hz",
                "peak_bandwidth_mean_hz",
                "spectral_peak_stability",
                "formant_like_peak_spacing",
                "spectral_envelope_slope",
            ]
        }


def _harmonic_descriptors(power: np.ndarray, freqs: np.ndarray, f0_hz: float) -> Dict[str, float]:
    """Harmonic stack descriptors from the average spectrum."""
    try:
        spec = np.mean(np.asarray(power, dtype=np.float64), axis=0)
        total_energy = float(np.sum(spec)) + 1e-12
        if f0_hz <= 45.0 or spec.size < 5:
            return {
                "harmonic_to_noise_ratio": 0.0,
                "harmonic_peak_count": 0.0,
                "harmonic_energy_ratio": 0.0,
                "inharmonicity": 1.0,
                "fundamental_dominance_ratio": 0.0,
                "overtone_slope": 0.0,
            }
        bin_hz = float(freqs[1] - freqs[0]) if freqs.size > 1 else 1.0
        harmonic_energy = 0.0
        harmonic_amps: List[float] = []
        harmonic_nums: List[float] = []
        for h in range(1, 33):
            target = h * float(f0_hz)
            if target > float(freqs[-1]):
                break
            idx = int(np.argmin(np.abs(freqs - target)))
            width = max(1, int(round(max(bin_hz, target * 0.015) / max(bin_hz, 1e-9))))
            lo = max(0, idx - width)
            hi = min(spec.size, idx + width + 1)
            e = float(np.sum(spec[lo:hi]))
            harmonic_energy += e
            harmonic_amps.append(e)
            harmonic_nums.append(float(h))
        harmonic_ratio = float(np.clip(harmonic_energy / total_energy, 0.0, 1.0))
        hnr = _safe_log_ratio(harmonic_energy, max(0.0, total_energy - harmonic_energy))
        peak_count = float(sum(1 for e in harmonic_amps if e >= max(harmonic_amps or [0.0]) * 0.05))
        fundamental = harmonic_amps[0] if harmonic_amps else 0.0
        fundamental_dominance = float(fundamental / (sum(harmonic_amps) + 1e-12)) if harmonic_amps else 0.0
        overtone_slope = 0.0
        if len(harmonic_amps) >= 3:
            amps = np.log1p(np.asarray(harmonic_amps, dtype=np.float64))
            nums = np.log1p(np.asarray(harmonic_nums, dtype=np.float64))
            overtone_slope = float(np.polyfit(nums, amps, 1)[0])
        # Distance of strong spectral peaks to the nearest harmonic.
        local = (spec[1:-1] > spec[:-2]) & (spec[1:-1] >= spec[2:])
        peaks = np.where(local)[0] + 1
        thresh = max(float(np.median(spec) * 6.0), float(np.max(spec) * 0.08))
        peaks = peaks[spec[peaks] >= thresh]
        distances = []
        for idx in peaks[:30]:
            hz = float(freqs[int(idx)])
            nearest_h = max(1.0, round(hz / float(f0_hz)))
            nearest = nearest_h * float(f0_hz)
            distances.append(abs(hz - nearest) / max(float(f0_hz), 1e-9))
        inharm = float(np.clip(np.mean(distances), 0.0, 4.0)) if distances else 1.0
        return {
            "harmonic_to_noise_ratio": float(hnr),
            "harmonic_peak_count": peak_count,
            "harmonic_energy_ratio": harmonic_ratio,
            "inharmonicity": inharm,
            "fundamental_dominance_ratio": float(np.clip(fundamental_dominance, 0.0, 1.0)),
            "overtone_slope": float(np.clip(overtone_slope, -20.0, 20.0)),
        }
    except Exception:
        return {
            "harmonic_to_noise_ratio": 0.0,
            "harmonic_peak_count": 0.0,
            "harmonic_energy_ratio": 0.0,
            "inharmonicity": 1.0,
            "fundamental_dominance_ratio": 0.0,
            "overtone_slope": 0.0,
        }


def _expanded_decay_and_low_source_descriptors(
    mono: np.ndarray,
    frames: np.ndarray,
    power: np.ndarray,
    freqs: np.ndarray,
    sr: int,
    centroid_frames: np.ndarray,
    f0_stats: Dict[str, float],
) -> Dict[str, float]:
    """Noise-burst, low-end, and decay descriptors for v0.5 expanded physics."""
    try:
        env = np.sqrt(np.mean(frames**2, axis=1)) if frames.size else np.zeros(1, dtype=np.float32)
        np.arange(env.size, dtype=np.float32) * (HOP / max(1, sr)) * 1000.0
        max_env = float(np.max(env)) if env.size else 0.0
        active = np.where(env >= max_env * 0.20)[0] if max_env > 1e-9 else np.asarray([], dtype=int)
        noise_burst_ms = float((active[-1] - active[0] + 1) * HOP / max(1, sr) * 1000.0) if active.size else 0.0
        freqs_arr = np.asarray(freqs, dtype=np.float32)
        high_mask = freqs_arr >= 2000.0
        sub_mask = freqs_arr < 150.0
        low_mask = freqs_arr < 500.0
        total = np.sum(power, axis=1) + 1e-12
        high_e = np.sum(power[:, high_mask], axis=1) / total if np.any(high_mask) else np.zeros(power.shape[0])
        sub_e_abs = np.sum(power[:, sub_mask], axis=1) if np.any(sub_mask) else np.zeros(power.shape[0])
        sub_e_abs / (np.sum(power, axis=1) + 1e-12)

        def slope_of(vals):
            vals = np.asarray(vals, dtype=np.float32)
            if vals.size < 3:
                return 0.0
            x = np.arange(vals.size, dtype=np.float32) * (HOP / max(1, sr))
            return float(np.clip(np.polyfit(x, vals, 1)[0], -1000.0, 1000.0))

        high_decay = slope_of(high_e)
        cent_slope = slope_of(centroid_frames / max(float(sr) / 2.0, 1.0))
        noise_tail = 0.0
        if env.size >= 3:
            tail_start = max(0, int(env.size * 0.33))
            noise_tail = slope_of(env[tail_start:])
        low_peak_hz = 0.0
        low_bw = 0.0
        if np.any(low_mask):
            avg = np.mean(power[:, low_mask], axis=0)
            lf = freqs_arr[low_mask]
            if avg.size and float(np.max(avg)) > 0:
                idx = int(np.argmax(avg))
                low_peak_hz = float(lf[idx])
                half = float(avg[idx]) * 0.5
                left = idx
                while left > 0 and avg[left] >= half:
                    left -= 1
                right = idx
                while right < avg.size - 1 and avg[right] >= half:
                    right += 1
                low_bw = float(lf[right] - lf[left]) if right > left else 0.0
        sub_attack_ms = 0.0
        sub_decay_ms = 0.0
        sub_to_click_ms = 0.0
        sub_sustain_ratio = 0.0
        if sub_e_abs.size and float(np.max(sub_e_abs)) > 1e-12:
            smax = float(np.max(sub_e_abs))
            peak_i = int(np.argmax(sub_e_abs))
            above = np.where(sub_e_abs >= smax * 0.90)[0]
            if above.size:
                first90 = int(above[0])
                sub_attack_ms = float(first90 * HOP / max(1, sr) * 1000.0)
            after = sub_e_abs[peak_i:]
            below = np.where(after <= smax * 0.30)[0]
            if below.size:
                sub_decay_ms = float(int(below[0]) * HOP / max(1, sr) * 1000.0)
            half = max(1, sub_e_abs.size // 2)
            sub_sustain_ratio = float(np.sum(sub_e_abs[half:]) / (np.sum(sub_e_abs) + 1e-12))
            high_abs = np.sum(power[:, high_mask], axis=1) if np.any(high_mask) else np.zeros(power.shape[0])
            if high_abs.size and float(np.max(high_abs)) > 1e-12:
                sub_to_click_ms = float(
                    (int(np.argmax(sub_e_abs)) - int(np.argmax(high_abs))) * HOP / max(1, sr) * 1000.0
                )
        pitch_drop = 0.0
        y = np.asarray(mono, dtype=np.float32)
        if y.size >= int(0.08 * sr):
            early = y[: max(64, int(y.size * 0.33))]
            late = y[max(0, int(y.size * 0.50)) :]
            c1, h1 = _estimate_f0_confidence_and_hz(early, sr)
            c2, h2 = _estimate_f0_confidence_and_hz(late, sr)
            if c1 >= 0.25 and c2 >= 0.25 and h1 > 0 and h2 > 0:
                pitch_drop = float(np.clip(1200.0 * math.log2(h1 / h2), -4800.0, 4800.0))
        return {
            "noise_burst_duration_ms": noise_burst_ms,
            "high_band_decay_slope": float(high_decay),
            "spectral_centroid_decay_slope": float(cent_slope),
            "noise_tail_decay_slope": float(noise_tail),
            "low_peak_frequency_hz": low_peak_hz,
            "low_peak_bandwidth_hz": low_bw,
            "sub_attack_time_ms": sub_attack_ms,
            "sub_decay_time_ms": sub_decay_ms,
            "sub_to_click_offset_ms": sub_to_click_ms,
            "kick_pitch_drop_cents": pitch_drop,
            "sub_sustain_ratio": float(np.clip(sub_sustain_ratio, 0.0, 1.0)),
        }
    except Exception:
        return {
            "noise_burst_duration_ms": 0.0,
            "high_band_decay_slope": 0.0,
            "spectral_centroid_decay_slope": 0.0,
            "noise_tail_decay_slope": 0.0,
            "low_peak_frequency_hz": 0.0,
            "low_peak_bandwidth_hz": 0.0,
            "sub_attack_time_ms": 0.0,
            "sub_decay_time_ms": 0.0,
            "sub_to_click_offset_ms": 0.0,
            "kick_pitch_drop_cents": 0.0,
            "sub_sustain_ratio": 0.0,
        }
