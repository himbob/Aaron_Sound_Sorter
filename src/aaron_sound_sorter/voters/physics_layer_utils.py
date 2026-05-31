from __future__ import annotations

from typing import Any

from aaron_sound_sorter.domain.models import SharedAudioFacts


def shape_vote(facts: SharedAudioFacts) -> dict[str, Any]:
    if not isinstance(getattr(facts, "evidence", None), dict):
        return {}
    shape = facts.evidence.get("shape_vote", {})
    if isinstance(shape, dict) and shape:
        return shape
    try:
        from aaron_sound_sorter.voters.shape_voter import classify_shape

        values = facts.feature_values_by_name or {}
        computed = classify_shape(values, facts=facts).to_dict()
        if isinstance(computed, dict):
            facts.evidence["shape_vote"] = computed
            return computed
    except Exception:
        return {}
    return {}


def fx_shape_vote(facts: SharedAudioFacts) -> dict[str, Any]:
    """Return local FX motion-shape evidence without changing other layers.

    The sorter runs PhysicsVoter before ShapeVoter, so physics layers need a
    source-name-blind structural read of the same measured shape facts that the
    diagnostic ShapeVoter later reports.
    """
    return shape_vote(facts)


def measured_roles(facts: SharedAudioFacts) -> dict[str, Any]:
    if not isinstance(getattr(facts, "evidence", None), dict):
        return {}
    roles = facts.evidence.get("measured_roles", {})
    return roles if isinstance(roles, dict) else {}


def role_value(roles: dict[str, Any], name: str) -> float:
    value = roles.get(name, 0.0)
    if isinstance(value, dict):
        return 0.0
    return safe_float(value, 0.0)


def number(values: dict[str, Any], name: str, default: float = 0.0) -> float:
    return safe_float(values.get(name, default), default)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def ramp(value: float, start: float, full: float) -> float:
    if full <= start:
        return 1.0 if value >= full else 0.0
    return clamp01((float(value) - start) / (full - start))


def inverse_ramp(value: float, good_at_or_below: float, bad_at_or_above: float) -> float:
    if bad_at_or_above <= good_at_or_below:
        return 1.0 if value <= good_at_or_below else 0.0
    return clamp01((bad_at_or_above - float(value)) / (bad_at_or_above - good_at_or_below))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def normalized_path(folder_path: str) -> str:
    return str(folder_path or "").replace("\\", "/")


def expm1_value(value: float) -> float:
    import math

    try:
        return float(math.expm1(max(0.0, float(value))))
    except Exception:
        return 0.0


def first_arrival(facts: SharedAudioFacts) -> dict[str, Any]:
    if not isinstance(getattr(facts, "evidence", None), dict):
        return {}
    data = facts.evidence.get("first_arrival_telemetry", {})
    return data if isinstance(data, dict) else {}


def physics_subpanel_flat(facts: SharedAudioFacts) -> dict[str, Any]:
    """Return flattened source-name-blind physics subpanels from shared facts."""
    evidence = getattr(facts, "evidence", {})
    if not isinstance(evidence, dict):
        return {}
    block = evidence.get("physics_subpanels", {})
    if not isinstance(block, dict):
        return {}
    flat = block.get("flat", {})
    return flat if isinstance(flat, dict) else {}


def apply_source_panel_lifts(
    branch_scores: dict[str, float],
    panel_scores: dict[str, Any],
    *,
    scale: float,
    floor: float,
) -> dict[str, float]:
    """Lift branch scores from measured source panels without hard routing.

    The panels are not candidate labels.  They only make a voter branch less
    timid when a broad measured panel strongly agrees with that branch.
    """
    lifted = dict(branch_scores)
    for branch, raw_score in panel_scores.items():
        panel = safe_float(raw_score, 0.0)
        if panel < floor:
            continue
        lifted[branch] = max(safe_float(lifted.get(branch, 0.0), 0.0), clamp01(panel * scale))
    return lifted


def candidate_is_broad_drum(folder: str) -> bool:
    low = folder.lower()
    return (
        "/drum loops/" in low
        or low.endswith("/drum loops/loops")
        or "/full drum loops/" in low
        or "/top loops/" in low
        or "/generic percussion/" in low
        or low.endswith("/generic percussion/one shots")
        or "/percussion/other percussion" in low
        or "/percussion/generic" in low
    )


def drum_candidate_matches_branch(folder: str, branch: str) -> bool:
    low = folder.lower()
    if branch == "Kick":
        return "/kick drums/" in low or "/kick/" in low or " kick" in low
    if branch == "TomOrConga":
        return any(
            token in low
            for token in ("/toms/", "/tom/", "/conga/", "/congas/", "/bongo/", "/bongos/", "latin percussion")
        )
    if branch == "Snare":
        return "/snares/" in low or "/snare" in low
    if branch == "Clap":
        return "/claps snaps slaps/" in low or "/clap" in low or "/snap" in low or "/slap" in low
    if branch == "Hat":
        return "/hi hats/" in low or " hat" in low or "/hat" in low
    if branch == "Cymbal":
        return "/cymbals/" in low or "cymbal" in low or "crash" in low or "ride" in low or "splash" in low
    if branch == "RimOrStick":
        return "/rims and sticks/" in low or "rimshot" in low or "sidestick" in low or "stick" in low
    if branch == "ShakerTambourine":
        return "shaker" in low or "tambourine" in low
    if branch == "ScrapeGuiro":
        return "guiro" in low or "scrape" in low or "rasp" in low
    if branch == "MetallicPercussion":
        return "metallic percussion" in low or "/bells and metallic" in low or "bell" in low
    return False


def candidate_is_broad_instrument(folder: str) -> bool:
    low = folder.lower()
    return (
        "/instrument loops/" in low
        or "/mixed musical" in low
        or "/multi instrument" in low
        or low.endswith("/instruments")
    )


def instrument_candidate_matches_branch(folder: str, branch: str) -> bool:
    low = folder.lower()
    if branch == "Bass":
        return candidate_is_bass(folder)
    if branch == "KeysPiano":
        return (
            any(token in low for token in ("/keys/", "piano", "rhodes", "electric piano", "organ", "clav", "keyboard"))
            and "coins" not in low
        )
    if branch == "PluckedString":
        return any(token in low for token in ("/guitar/", "guitar", "pluck", "banjo", "koto", "harp"))
    if branch in {"Woodwinds", "ReedWoodwind"}:
        return any(
            token in low
            for token in ("/woodwinds/", "woodwind", "sax", "saxophone", "flute", "clarinet", "bassoon", "reed")
        )
    if branch == "Brass":
        return any(token in low for token in ("/brass/", "brass", "horn", "trumpet", "trombone", "saxophone"))
    if branch == "Voice":
        return any(token in low for token in ("/voice/", "voice", "vocal", "choir", "spoken"))
    if branch == "Synth":
        return any(
            token in low for token in ("/synth", "synth", "lead", "pad", "arp", "electronic")
        ) and not candidate_is_bass(folder)
    if branch == "Strings":
        return any(token in low for token in ("/strings", "string", "violin", "viola", "cello", "bowed"))
    if branch == "MalletBell":
        return any(
            token in low for token in ("mallet", "bell", "bells", "vibraphone", "marimba", "glockenspiel", "celesta")
        )
    if branch == "MixedInstrument":
        return candidate_is_broad_instrument(folder)
    return False


def instrument_candidate_matches_subpanel(folder: str, branch: str, subpanel: str) -> bool:
    """Return True when a folder path agrees with a measured instrument subpanel.

    This is still path matching on the candidate label, not the source file name.
    It only shapes candidate scores after measured audio selected a branch.
    """
    low = folder.lower()
    sub = str(subpanel or "")
    if branch == "Bass":
        if sub == "808Sub":
            return "808" in low or "sub bass" in low
        if sub == "SynthBass":
            return "synth bass" in low
        if sub == "ElectricBass":
            return "electric bass" in low or (
                "/bass/" in low
                and "synth" not in low
                and "808" not in low
                and "sub" not in low
                and "upright" not in low
            )
        if sub == "UprightBass":
            return "upright" in low
        if sub == "BassLoop":
            return "bass" in low and "loop" in low
    if branch == "KeysPiano":
        if sub == "AcousticPiano":
            return "piano" in low and "electric piano" not in low and "rhodes" not in low
        if sub == "ElectricPiano":
            return "electric piano" in low or "rhodes" in low or "keys" in low
        if sub == "Organ":
            return "organ" in low
        if sub == "ClavinetHarpsichord":
            return "clav" in low or "harpsichord" in low
    if branch == "PluckedString":
        if sub == "AcousticGuitar":
            return "acoustic guitar" in low or "nylon guitar" in low
        if sub == "ElectricGuitar":
            return "electric guitar" in low
        if sub == "NylonOrSoftPluck":
            return "nylon" in low or "soft pluck" in low
        if sub == "WorldPluck":
            return any(
                token in low for token in ("banjo", "koto", "harp", "mandolin", "ukulele", "oud", "sitar", "pluck")
            )
    if branch == "Woodwinds":
        if sub == "Sax":
            return "sax" in low or "saxophone" in low
        if sub == "Flute":
            return "flute" in low or "pan pipe" in low
        if sub == "Clarinet":
            return "clarinet" in low
        if sub == "Bassoon":
            return "bassoon" in low
        if sub == "AiryWoodwind":
            return "woodwind" in low or "reed" in low or "flute" in low
    if branch == "Brass":
        if sub == "TrumpetHorn":
            return "trumpet" in low or "horn" in low
        if sub == "LowBrass":
            return "trombone" in low or "tuba" in low or "low brass" in low
        if sub == "BrassStab":
            return "stab" in low or "hit" in low or "one shot" in low
        if sub == "BrassSustainLoop":
            return "brass" in low or "horn" in low
    if branch == "Strings":
        if sub == "BowedSustain":
            return any(token in low for token in ("violin", "viola", "cello", "bowed", "sustain"))
        if sub == "StringPluck":
            return "pluck" in low or "pizz" in low
        if sub == "StringDrone":
            return "drone" in low
        if sub == "StringLoop":
            return "string" in low and "loop" in low
    if branch == "Synth":
        if sub == "SynthLead":
            return "lead" in low
        if sub == "SynthPad":
            return "pad" in low
        if sub == "SynthPluck":
            return "pluck" in low
        if sub == "SynthArp":
            return "arp" in low
        if sub == "SynthDrone":
            return "drone" in low
    if branch == "MalletBell":
        if sub == "BellChime":
            return "bell" in low or "chime" in low or "glock" in low
        if sub == "MalletKeys":
            return "mallet" in low or "marimba" in low or "vibe" in low or "xylophone" in low
        if sub == "SteelPanHandpan":
            return "steel" in low or "pan" in low or "handpan" in low or "tongue" in low
        if sub == "KalimbaMbira":
            return "kalimba" in low or "mbira" in low
        if sub == "SingingBowlGong":
            return "bowl" in low or "gong" in low
    return False


def instrument_candidate_has_specific_subbranch(folder: str, branch: str) -> bool:
    """Return True when a candidate is a specific leaf inside the measured branch."""
    low = folder.lower()
    if branch == "Bass":
        return any(token in low for token in ("808", "sub bass", "synth bass", "electric bass", "upright bass"))
    if branch == "KeysPiano":
        return any(token in low for token in ("piano", "rhodes", "electric piano", "organ", "clav", "harpsichord"))
    if branch == "PluckedString":
        return any(
            token in low
            for token in (
                "acoustic guitar",
                "electric guitar",
                "nylon",
                "banjo",
                "koto",
                "harp",
                "mandolin",
                "ukulele",
                "sitar",
            )
        )
    if branch == "Woodwinds":
        return any(token in low for token in ("sax", "saxophone", "flute", "clarinet", "bassoon", "reed"))
    if branch == "Brass":
        return any(token in low for token in ("trumpet", "horn", "trombone", "tuba", "brass"))
    if branch == "Strings":
        return any(token in low for token in ("violin", "viola", "cello", "string", "bowed", "pizz"))
    if branch == "Synth":
        return any(token in low for token in ("lead", "pad", "pluck", "arp", "drone", "synth"))
    if branch == "MalletBell":
        return any(
            token in low
            for token in (
                "bell",
                "chime",
                "mallet",
                "marimba",
                "vibe",
                "xylophone",
                "steel",
                "pan",
                "kalimba",
                "mbira",
                "bowl",
                "gong",
            )
        )
    return False


def candidate_is_broad_fx(folder: str) -> bool:
    low = folder.lower()
    return low == "fx" or low.endswith("/fx") or "/hybrid designed" in low or "/designed noise fx/" in low


def fx_candidate_matches_branch(folder: str, branch: str) -> bool:
    low = folder.lower()
    if branch == "RiserBuild":
        return any(token in low for token in ("riser", "build", "uplifter", "sweep up"))
    if branch == "DropDownlifter":
        return any(token in low for token in ("drop", "downlifter", "downlift", "fall", "sub drop"))
    if branch == "WhooshSweep":
        return any(token in low for token in ("whoosh", "swoosh", "swish", "sweep", "pass by", "air"))
    if branch == "ReverseSwell":
        return any(token in low for token in ("reverse", "swell", "suckback", "backspin"))
    if branch == "ImpactHit":
        return any(token in low for token in ("impact", "boom", "slam", "hit", "crash", "thud", "sub hit"))
    if branch == "GlitchStutter":
        return any(token in low for token in ("glitch", "stutter", "buffer", "digital", "error", "skip"))
    if branch == "BlipBeep":
        return any(token in low for token in ("blip", "beep", "ui", "button", "zap", "laser", "chirp"))
    if branch == "SirenAlarm":
        return any(token in low for token in ("siren", "alarm"))
    if branch == "FormantFX":
        return any(token in low for token in ("formant", "vowel", "vocal fx", "voice fx", "talkbox"))
    if branch == "RadioElectrical":
        return any(
            token in low for token in ("radio", "static", "electrical", "electric", "buzz", "hum", "interference")
        )
    if branch == "TextureAmbience":
        return any(
            token in low
            for token in (
                "texture",
                "ambience",
                "ambiance",
                "atmosphere",
                "drone",
                "rain",
                "water",
                "ocean",
                "wave",
                "wind",
                "fire",
                "thunder",
                "hiss",
                "static",
                "white noise",
                "vinyl noise",
                "granular",
            )
        )
    if branch == "MachineMechanical":
        return any(token in low for token in ("machine", "motor", "engine", "mechanical", "industrial", "transport"))
    if branch == "FoleyMaterial":
        return any(token in low for token in ("foley", "door", "footstep", "object", "slam", "splash", "water foley"))
    if branch == "SmallObjectCluster":
        return any(
            token in low for token in ("keys coins", "small object", "coin", "coins", "carkey", "carkeys", "keys/")
        )
    if branch == "HumanCreatureFX":
        return any(
            token in low
            for token in (
                "human and voice fx",
                "applause",
                "crowd",
                "spoken",
                "breath",
                "mouth",
                "scream",
                "animals and creatures",
                "bird",
                "dog",
                "cat",
                "cricket",
            )
        )
    if branch == "DesignedNoiseHybrid":
        return candidate_is_broad_fx(folder) or any(
            token in low for token in ("designed", "hybrid", "noise", "texture")
        )
    return False


def candidate_is_bass(folder: str) -> bool:
    low = folder.lower()
    return "/bass/" in low or low.endswith("/bass") or " bass/" in low


def candidate_is_sax(folder: str) -> bool:
    low = folder.lower()
    return "saxophone" in low or "/sax/" in low or low.endswith("/sax")
