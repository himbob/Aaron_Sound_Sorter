"""Source-name-blind CLAP semantic families for independent owner evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .contracts import EmbeddingRecord

SEMANTIC_FAMILY_PROMPTS: Mapping[str, tuple[str, ...]] = {
    "human_voice": (
        "an isolated natural human voice",
        "a person speaking or narrating",
        "a person singing or humming",
        "rap vocals, chanting, or beatboxing",
        "a shouted, whispered, crying, or laughing human voice",
        "a heavily processed, pitch-shifted, formant-shifted, vocoded, robotic, or altered human voice",
        "a chopped vocal sample or vocal sound effect",
    ),
    "drums": (
        "an acoustic or electronic drum hit",
        "a kick, snare, clap, rimshot, tom, or cymbal",
        "a percussion rhythm or drum fill",
        "a full drum beat or drum loop",
        "a layered drum kit performance",
    ),
    "drum_kick": (
        "an isolated acoustic kick drum or bass drum hit",
        "a deep electronic kick, sub kick, or 808 kick",
        "a short low-frequency drum thump",
    ),
    "drum_backbeat": (
        "an isolated snare drum, clap, snap, slap, or rimshot",
        "a sharp backbeat drum hit",
        "a layered snare and clap transient",
    ),
    "drum_cymbal": (
        "an isolated closed or open hi-hat",
        "a ride, crash, splash, or china cymbal",
        "a metallic cymbal rhythm or cymbal swell",
    ),
    "drum_percussion": (
        "hand percussion, shaker, conga, bongo, tambourine, or maraca",
        "a tom, wood block, cowbell, clave, gong, or world percussion sound",
        "an electronic percussion hit that is not a kick or snare",
    ),
    "drum_full": (
        "a full acoustic drum kit beat or drum loop",
        "a programmed electronic drum groove",
        "a layered kick snare hi-hat rhythm or drum fill",
    ),
    "bass": (
        "an electric or acoustic bass guitar",
        "a synthesized bass line or sub bass",
        "a low bass note, bass stab, or bass loop",
        "a distorted, processed, or resampled bass sound",
    ),
    "keys": (
        "an acoustic piano",
        "an electric piano, organ, or keyboard",
        "a keyboard chord, melody, stab, or loop",
        "a processed piano or sampled keyboard sound",
    ),
    "guitar_plucked": (
        "an acoustic or electric guitar",
        "a plucked string instrument such as guitar, banjo, harp, or ukulele",
        "a guitar chord, riff, melody, stab, or loop",
        "a distorted or heavily processed guitar sound",
    ),
    "strings_bowed": (
        "a bowed string instrument",
        "a violin, viola, cello, or double bass",
        "an orchestral string section, string stab, or string loop",
        "a processed or synthesized bowed-string sound",
    ),
    "woodwind_reed": (
        "a saxophone, clarinet, oboe, or bassoon",
        "a flute, recorder, or other woodwind instrument",
        "a reed instrument melody, stab, phrase, or loop",
        "a processed or synthetic woodwind-like sound",
    ),
    "brass": (
        "a trumpet, trombone, horn, or tuba",
        "a brass horn section, brass stab, or brass loop",
        "an isolated, muted, distorted, or processed brass instrument",
    ),
    "mallet_bell": (
        "a bell, chime, gong, or metallic tuned percussion sound",
        "a marimba, xylophone, vibraphone, glockenspiel, or steelpan",
        "a struck mallet instrument phrase, hit, or loop",
    ),
    "synth": (
        "an electronic synthesizer",
        "a synthesized pad, lead, pluck, chord, arpeggio, or stab",
        "an electronic musical tone, sequence, or synth loop",
        "a heavily processed synthetic sound that remains musical",
    ),
    "fx_impact": (
        "a cinematic impact, boom, explosion, slam, or hit sound effect",
        "a crash, break, smash, thud, or destruction sound effect",
        "a designed impact with a reverberant or sub-bass tail",
        "a weapon, gunshot, blast, or explosive transient",
    ),
    "fx_transition": (
        "a riser, build, downlifter, faller, or transition sound effect",
        "a whoosh, swoosh, sweep, pass-by, or movement sound effect",
        "a reversed build-up, suck-back, cymbal swell, or transition tail",
    ),
    "fx_glitch": (
        "a digital glitch, stutter, buffer error, or chopped sound effect",
        "a robotic, data, modem, computer, or electronic malfunction sound",
        "a bit-crushed, granular, corrupted, or synthetic digital effect",
    ),
    "fx_alert": (
        "an emergency siren, alarm, buzzer, bell, or warning horn",
        "a police, ambulance, fire, security, or industrial alarm",
        "an electronic warning beep, notification, or alert sound",
    ),
    "fx_machine": (
        "an engine, motor, machine, mechanism, tool, or appliance sound",
        "a mechanical servo, gear, hydraulic, or industrial mechanism",
        "a car, truck, train, aircraft, boat, or transport sound",
        "a household machine, power tool, or electrical device",
    ),
    "fx_nature": (
        "rain, thunder, wind, fire, ocean, river, or water ambience",
        "a natural weather, water, earth, or elemental sound effect",
        "an outdoor environmental nature recording with birds or insects",
    ),
    "fx_texture": (
        "an ambient texture, drone, rumble, hum, or noise bed",
        "static, vinyl noise, tape noise, electrical noise, or background atmosphere",
        "an abstract sustained soundscape, tonal texture, or dark ambience",
    ),
    "fx_foley": (
        "a real-world foley sound made by an object, body, surface, or material",
        "a door, footsteps, cloth, paper, glass, metal, wood, plastic, or household sound",
        "a small mechanical, kitchen, tool, handling, or movement sound effect",
        "a liquid splash, pour, drip, spray, or water foley sound",
    ),
    "animal_creature": (
        "an animal vocalization or animal movement sound",
        "a dog, cat, bird, farm animal, wild animal, insect, or amphibian",
        "a monster, creature, dinosaur, growl, roar, hiss, or fantasy beast sound effect",
        "a processed or designed animal-like creature sound",
    ),
}


@dataclass(frozen=True)
class SemanticFamilyPrediction:
    """Independent zero-shot source-family evidence for one audio embedding.

    Args:
        predicted_family: Strongest semantic family.
        second_family: Runner-up semantic family.
        top_score: Mean of the two strongest prompt similarities in the winner.
        second_score: Equivalent score for the runner-up.
        margin: Winner score minus runner-up score.
        family_scores: Score for every evaluated family.

    Side Effects:
        None.
    """

    predicted_family: str
    second_family: str
    top_score: float
    second_score: float
    margin: float
    family_scores: Mapping[str, float]


@dataclass(frozen=True)
class SemanticCompatibilityAssessment:
    """Explain whether independent audio semantics contradict a label.

    Args:
        contradictory: Whether decisive source-family evidence rejects the
            proposed trained label.
        reason: Stable diagnostic reason.
        compatible_families: Semantic families allowed for the label's broad
            audible source.
        best_compatible_score: Strongest score among compatible families.
        contradiction_gap: Winning score minus best compatible score.

    Side Effects:
        None.
    """

    contradictory: bool
    reason: str
    compatible_families: tuple[str, ...]
    best_compatible_score: float
    contradiction_gap: float


def flattened_semantic_prompts(
    prompts_by_family: Mapping[str, Sequence[str]] = SEMANTIC_FAMILY_PROMPTS,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return parallel prompt and family sequences for CLAP text embedding.

    Args:
        prompts_by_family: Audible prompt descriptions grouped by source family.

    Returns:
        ``(prompts, families)`` with matching positions.

    Raises:
        ValueError: If no non-empty prompt is supplied.

    Side Effects:
        None.
    """
    prompts: list[str] = []
    families: list[str] = []
    for family, family_prompts in prompts_by_family.items():
        for prompt in family_prompts:
            cleaned_prompt = str(prompt).strip()
            if cleaned_prompt:
                prompts.append(cleaned_prompt)
                families.append(str(family))
    if not prompts:
        raise ValueError("semantic family panel requires at least one prompt")
    return tuple(prompts), tuple(families)


def predict_semantic_family(
    audio_embedding: EmbeddingRecord,
    text_embeddings: np.ndarray,
    prompt_families: Sequence[str],
) -> SemanticFamilyPrediction:
    """Rank broad audible families without using training labels or filenames.

    Args:
        audio_embedding: Source-name-blind normalized CLAP audio embedding.
        text_embeddings: Normalized CLAP prompt embeddings.
        prompt_families: Family name corresponding to each prompt row.

    Returns:
        Strongest and runner-up semantic families with all family scores.

    Raises:
        ValueError: If dimensions or row counts do not match.

    Side Effects:
        None.
    """
    matrix = np.asarray(text_embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(prompt_families):
        raise ValueError("semantic prompt embeddings and family names must align")
    if matrix.shape[1] != audio_embedding.dimension:
        raise ValueError("semantic prompt and audio embedding dimensions differ")
    similarities = np.sum(matrix.astype(np.float64) * audio_embedding.vector[None, :], axis=1)
    grouped: dict[str, list[float]] = {}
    for family, similarity in zip(prompt_families, similarities):
        grouped.setdefault(str(family), []).append(float(similarity))
    family_scores = {
        family: float(np.mean(sorted(scores, reverse=True)[: min(2, len(scores))]))
        for family, scores in grouped.items()
    }
    ranked = sorted(family_scores.items(), key=lambda row: (-row[1], row[0]))
    top_family, top_score = ranked[0]
    second_family, second_score = ranked[1] if len(ranked) > 1 else ("", -1.0)
    return SemanticFamilyPrediction(
        predicted_family=top_family,
        second_family=second_family,
        top_score=top_score,
        second_score=second_score,
        margin=top_score - second_score,
        family_scores=family_scores,
    )


def assess_semantic_label_compatibility(
    label: str,
    *,
    predicted_family: str,
    top_score: float,
    family_scores: Mapping[str, float],
    minimum_top_score: float = 0.10,
    minimum_contradiction_gap: float = 0.10,
) -> SemanticCompatibilityAssessment:
    """Check a trained label against an independent CLAP text/audio panel.

    Args:
        label: Internal taxonomy label proposed by learned audio memory.
        predicted_family: Strongest zero-shot audible family.
        top_score: Similarity score for ``predicted_family``.
        family_scores: Similarity scores for all semantic families.
        minimum_top_score: Minimum winning score needed for a veto.
        minimum_contradiction_gap: Required lead over the strongest compatible
            family. The same ambiguity rule is applied to every family.

    Returns:
        Compatibility assessment. Broad labels with no defensible source
        contract remain unblocked.

    Raises:
        ValueError: If either threshold is negative.

    Side Effects:
        None.

    Important Constraints:
        This is only a catastrophic-family guardrail. It may send conflicting
        learned memory to Review; it never invents a detailed category.
    """
    if minimum_top_score < 0.0 or minimum_contradiction_gap < 0.0:
        raise ValueError("semantic compatibility thresholds cannot be negative")
    compatible = compatible_semantic_families(label)
    if not compatible or not predicted_family or not family_scores:
        return SemanticCompatibilityAssessment(False, "semantic_evidence_absent", compatible, 0.0, 0.0)
    best_compatible_score = max(float(family_scores.get(family, -1.0)) for family in compatible)
    contradiction_gap = float(top_score) - best_compatible_score
    required_gap = minimum_contradiction_gap
    if predicted_family in compatible:
        reason = "semantic_family_compatible"
        contradictory = False
    elif float(top_score) < minimum_top_score:
        reason = "semantic_winner_too_weak"
        contradictory = False
    elif contradiction_gap < required_gap:
        reason = "semantic_family_ambiguous"
        contradictory = False
    else:
        reason = "semantic_family_contradiction"
        contradictory = True
    return SemanticCompatibilityAssessment(
        contradictory,
        reason,
        compatible,
        best_compatible_score,
        contradiction_gap,
    )


def compatible_semantic_families(label: str) -> tuple[str, ...]:
    """Return broad audible families compatible with one taxonomy label.

    Args:
        label: Internal taxonomy label. Only taxonomy text is inspected; no
            source filename or source path is accepted.

    Returns:
        Semantic family names suitable for contradiction checking. An empty
        tuple means the label is too broad for a safe check.

    Side Effects:
        None.
    """
    normalized = str(label).strip().replace("\\", "/")
    if normalized.startswith("Instruments/Voice/") or normalized.startswith("FX/Human and Voice FX/"):
        return ("human_voice",)
    if normalized.startswith("Drums/"):
        common_drum_families = (
            "drums",
            "drum_kick",
            "drum_backbeat",
            "drum_cymbal",
            "drum_percussion",
            "drum_full",
            "fx_impact",
            "fx_foley",
        )
        if normalized.startswith("Drums/Kick Drums/"):
            return ("drum_kick", *common_drum_families)
        if normalized.startswith(("Drums/Snares/", "Drums/Claps Snaps Slaps/", "Drums/Rims and Sticks/")):
            return ("drum_backbeat", *common_drum_families)
        if normalized.startswith(("Drums/Cymbals/", "Drums/Hi Hats/")):
            return ("drum_cymbal", *common_drum_families, "fx_transition")
        if normalized.startswith(
            ("Drums/Percussion/", "Drums/Percussion Loops/", "Drums/Toms/", "Drums/World Percussion/")
        ):
            return ("drum_percussion", *common_drum_families)
        if normalized.startswith(("Drums/Drum Loops/", "Drums/Drum Fills and Rolls/")):
            return ("drum_full", *common_drum_families)
        return common_drum_families
    if normalized.startswith("Instruments/Bass/"):
        return ("bass", "synth")
    if normalized.startswith("Instruments/Keys/"):
        return ("keys", "synth", "mallet_bell")
    if normalized.startswith(("Instruments/Guitar/", "Instruments/Plucked Strings/")):
        return ("guitar_plucked",)
    if normalized.startswith(("Instruments/Strings/", "Instruments/Strings Bowed/")):
        return ("strings_bowed", "guitar_plucked")
    if normalized.startswith(("Instruments/Winds/", "Instruments/Woodwinds/")):
        return ("woodwind_reed", "brass")
    if normalized.startswith("Instruments/Brass/"):
        return ("brass", "woodwind_reed")
    if normalized.startswith(("Instruments/Mallets/", "Instruments/Mallets and Bells/")):
        return ("mallet_bell", "keys")
    if normalized.startswith("Instruments/Synths/"):
        return ("synth", "keys", "bass")
    if normalized.startswith(("Instruments/Instrument Loops/", "Instruments/Mixed Musical Loops/")):
        return ()
    if normalized.startswith("FX/Structural and Transitional FX/"):
        if "/Reverses and Tails/" in normalized:
            return ("fx_transition", "drum_cymbal", "drum_backbeat", "drums", "synth")
        if "/Risers and Builds/" in normalized:
            return ("fx_transition", "synth", "drum_cymbal", "fx_texture")
        return ("fx_transition", "synth", "fx_texture")
    if normalized.startswith(("FX/Impacts and Hits/", "FX/Crashes and Breaks/")):
        return ("fx_impact", "fx_foley", "drums")
    if normalized.startswith("FX/Everyday Foley/"):
        if "/Machines/" in normalized:
            return ("fx_machine", "fx_foley")
        if "/Water Foley/" in normalized:
            return ("fx_nature", "fx_foley")
        return ("fx_foley", "fx_impact")
    if normalized.startswith("FX/Animals and Creatures/"):
        return ("animal_creature",)
    if normalized.startswith(("FX/Textures/", "FX/Ambiences and Environments/")):
        if "/Natural Ambience/" in normalized:
            return ("fx_nature", "fx_texture")
        return ("fx_texture", "fx_nature", "fx_foley")
    if normalized.startswith("FX/Digital Mechanical Industrial Transport/"):
        if "/Machines/" in normalized:
            return ("fx_machine", "fx_foley", "fx_texture")
        return ("fx_glitch", "fx_machine", "fx_foley", "fx_texture", "fx_impact")
    if normalized.startswith("FX/Designed Noise FX/"):
        if "/Alarm/" in normalized or "/Siren/" in normalized:
            return ("fx_alert", "fx_glitch", "synth")
        return ("fx_glitch", "fx_texture", "fx_transition", "fx_impact", "synth")
    if normalized.startswith("FX/Weapons Explosions and Destruction/"):
        return ("fx_impact", "fx_foley")
    if normalized.startswith("FX/Nature Weather and Elements/"):
        return ("fx_nature", "fx_foley", "fx_texture")
    return ()
