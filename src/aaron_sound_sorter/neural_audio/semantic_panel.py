"""Source-name-blind CLAP semantic families for independent owner evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .contracts import EmbeddingRecord

SEMANTIC_FAMILY_PROMPTS: Mapping[str, tuple[str, ...]] = {
    "human_voice": (
        "an isolated human voice",
        "a person speaking",
        "a person singing",
        "a human vocal shout or chant",
        "rap vocals",
    ),
    "drums": (
        "an acoustic drum hit",
        "a kick or snare drum",
        "a cymbal or hi-hat",
        "a percussion rhythm",
        "a full drum beat",
    ),
    "drum_kick": (
        "an isolated kick drum or bass drum hit",
        "a deep electronic kick drum",
    ),
    "drum_backbeat": (
        "an isolated snare drum, clap, snap, or rimshot",
        "a sharp backbeat drum hit",
    ),
    "drum_cymbal": (
        "an isolated hi-hat, ride, or crash cymbal",
        "a metallic cymbal rhythm",
    ),
    "drum_percussion": (
        "hand percussion, shaker, conga, bongo, or tambourine",
        "a tom, wood block, cowbell, or world percussion sound",
    ),
    "drum_full": (
        "a full drum kit beat or drum loop",
        "a layered kick snare and hi-hat rhythm",
    ),
    "bass": (
        "an electric bass instrument",
        "a synthesized bass line",
        "a low bass note",
    ),
    "keys": (
        "an acoustic piano",
        "an electric piano or keyboard",
        "a keyboard melody",
    ),
    "guitar_plucked": (
        "an acoustic or electric guitar",
        "a plucked string instrument",
        "a guitar chord or melody",
    ),
    "strings_bowed": (
        "a bowed string instrument",
        "a violin or cello",
        "an orchestral string section",
    ),
    "woodwind_reed": (
        "a saxophone",
        "a flute or woodwind instrument",
        "a reed instrument melody",
    ),
    "brass": (
        "a trumpet or trombone",
        "a brass horn section",
        "an isolated brass instrument",
    ),
    "mallet_bell": (
        "a bell or chime",
        "a marimba or xylophone",
        "a struck mallet instrument",
    ),
    "synth": (
        "an electronic synthesizer",
        "a synthesized pad or lead",
        "an electronic musical tone",
    ),
    "fx_impact": (
        "a cinematic impact or boom sound effect",
        "a crash, slam, or hit sound effect",
        "a designed impact with a reverberant tail",
    ),
    "fx_transition": (
        "a riser, downlifter, or transition sound effect",
        "a whoosh or sweep sound effect",
        "a reversed build-up sound effect",
    ),
    "fx_glitch": (
        "a digital glitch or stutter sound effect",
        "a chopped electronic malfunction sound",
        "a robotic digital sound effect",
    ),
    "fx_alert": (
        "an emergency siren or alarm sound effect",
        "a police siren or warning horn",
        "an electronic warning alert sound",
    ),
    "fx_machine": (
        "an engine, motor, or machine sound effect",
        "a mechanical servo or industrial mechanism",
        "a vehicle engine or transport sound",
    ),
    "fx_nature": (
        "rain, thunder, wind, fire, or ocean ambience",
        "a natural water, weather, or elemental sound effect",
        "an outdoor environmental nature recording",
    ),
    "fx_texture": (
        "an ambient texture, drone, or noise bed",
        "static, vinyl noise, or background atmosphere",
        "an abstract sustained soundscape",
    ),
    "fx_foley": (
        "a real-world foley sound",
        "a small object, surface, or material sound",
        "a mechanical or household sound effect",
    ),
    "animal_creature": (
        "an animal or creature vocalization",
        "a monster, growl, or creature sound effect",
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
            family. Voice labels use a stricter 0.05 boundary because confusing
            voice with another source is a high-cost failure.

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
    required_gap = min(minimum_contradiction_gap, 0.05) if compatible == ("human_voice",) else minimum_contradiction_gap
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
