"""Placement-depth policy for weak/tiny category profiles and family-rescue decisions.

Architecture note for future AI helpers
---------------------------------------
This module implements the "deepest safe placement" idea.

The sorter must not treat a terminal folder as proven just because it is the
nearest remaining legal candidate after a higher-level sanity check rejects a
raw winner.  A label such as Guitar/Loops may have lots of training examples,
but that is profile support, not proof that the current file is guitar.  When
ShapeVoter/role/top-family evidence only proves a broad role, consensus should
place at the broad bucket and keep raw voter evidence visible in the manifest.

This is deliberately not a sax, string, or guitar patch.  It is a general rule:
terminal identity is opt-in.  Weak terminal profiles, top-family rescues, and
close terminal-vs-broad cases degrade to broad buckets instead of inventing a
specific identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.engine.family_claims import ConsensusClaim, claim_from_folder_path
from aaron_sound_sorter.voters.scoring_tools import detected_parent_role_name, role_strength


@dataclass(frozen=True)
class PlacementDepthPolicy:
    """Thresholds for deciding whether terminal placement is reliable."""

    tiny_effective_count: int = 4
    weak_effective_count: int = 12
    strong_role_strength: float = 0.84
    max_broad_combined_rank_score: float = 36.0
    max_broad_physics_rank: int = 35
    max_broad_brain_rank: int = 45
    terminal_rank_slack: float = 15.0

    # Terminal identity is opt-in.  These are intentionally stricter than the
    # raw shared-candidate thresholds.  A terminal label needs direct support
    # from both voters, not merely survival after FX/Drums were rejected.
    terminal_identity_primary_rank: int = 2
    terminal_identity_secondary_rank: int = 5
    terminal_identity_max_combined_rank_score: float = 8.0
    family_rescue_broad_slack: float = 18.0


class PlacementDepthDecider:
    """Choose terminal, broad bucket, or review based on evidence reliability.

    The central contract:
      * voters stay honest and report raw rankings;
      * ShapeVoter and measured roles constrain structure/top-family legality;
      * tiny terminal profiles are allowed to help, but not to overrule stronger
        broad role evidence;
      * top-family rescue cannot jump into a deep terminal folder unless direct
        terminal identity is proven;
      * when evidence supports only a broad bucket, stop there instead of forcing
        a brittle terminal category.
    """

    FAMILY_RESCUE_STATUSES = {
        "top_family_sanity_consensus",
        "role_sanity_consensus",
        "shape_sanity_consensus",
    }

    def __init__(self, policy: PlacementDepthPolicy | None = None) -> None:
        self.policy = policy or PlacementDepthPolicy()

    def refine_claim(
        self,
        *,
        raw_claim: ConsensusClaim,
        shared: list[dict[str, Any]],
        facts: SharedAudioFacts,
    ) -> ConsensusClaim | None:
        """Return a broad placement-depth claim when terminal placement is unsafe."""
        if not shared or raw_claim.family == "_TO_REVIEW":
            return None

        current = self._row_for_label(shared, raw_claim.final_label)
        current_depth = label_depth(current) if current is not None else label_depth_from_label(raw_claim.final_label)

        parent = facts.evidence.get("parent_eligibility_v2", {}) if isinstance(facts.evidence, dict) else {}
        if isinstance(parent, dict):
            allowed = parent.get("allowed_top_families") or ()
            parent_path = str(parent.get("broad_folder_path") or "")
            if (
                raw_claim.family == "Instruments"
                and "Drums" in allowed
                and "Instruments" not in allowed
                and parent_path.startswith("Drums/")
            ):
                return claim_from_folder_path(
                    folder_path=parent_path,
                    source="placement_depth_parent_eligible_broad_bucket",
                    reason=(
                        "stopped broad Instrument placement because parent eligibility "
                        "had already blocked Instruments and proved a broad Drums lane"
                    ),
                    shared=shared,
                    raw_candidate_score=raw_claim.raw_candidate_score,
                    brain_rank=raw_claim.brain_rank,
                    physics_rank=raw_claim.physics_rank,
                    shared_winner=parent_path,
                    can_override=True,
                    strength=max(0.92, raw_claim.strength),
                    is_real_candidate=False,
                )

        if raw_claim.family == "Instruments" and self.facts_support_short_low_percussive_drum_parent(facts):
            return claim_from_folder_path(
                folder_path="Drums/Kick Drums/Generic Kick/One Shots",
                source="placement_depth_low_percussive_parent_broad_bucket",
                reason=(
                    "stopped Instrument broadening because measured roles and shape evidence "
                    "showed a short low percussive one-shot, not an instrument loop"
                ),
                shared=shared,
                raw_candidate_score=raw_claim.raw_candidate_score,
                brain_rank=raw_claim.brain_rank,
                physics_rank=raw_claim.physics_rank,
                shared_winner="Drums/Kick Drums/Generic Kick/One Shots",
                can_override=True,
                strength=max(0.90, raw_claim.strength),
                is_real_candidate=False,
            )

        if current_depth <= 3:
            return None

        if raw_claim.source not in self.FAMILY_RESCUE_STATUSES and self.has_direct_terminal_identity(current):
            return None

        roles = facts.evidence.get("measured_roles", {}) if isinstance(facts.evidence, dict) else {}
        if not isinstance(roles, dict):
            roles = {}
        parent_role = detected_parent_role_name(roles)
        parent_strength = role_strength(roles, parent_role) if parent_role != "unknown" else 0.0

        if self.row_prefers_compound_broad_loop(current):
            broad = self.best_broad_bucket(
                shared=shared,
                top_family=raw_claim.family,
                parent_role=parent_role,
                current_score=float(current.get("combined_rank_score", 9999.0)),
                current_label=raw_claim.final_label,
                allow_synthetic=True,
            )
            if broad is not None:
                return self._broad_claim(
                    broad=broad,
                    shared=shared,
                    status="placement_depth_compound_music_broad_bucket",
                    reason=(
                        "stopped at broad bucket because PhysicsVoter measured a compound musical loop; "
                        f"specific terminal identity was too narrow for {raw_claim.final_label}"
                    ),
                )

        if raw_claim.source in self.FAMILY_RESCUE_STATUSES and not self.has_direct_terminal_identity(current):
            broad = self.best_broad_bucket(
                shared=shared,
                top_family=raw_claim.family,
                parent_role=parent_role,
                current_score=float(current.get("combined_rank_score", 9999.0)) if current else 9999.0,
                current_label=raw_claim.final_label,
                allow_synthetic=True,
            )
            if broad is not None:
                return self._broad_claim(
                    broad=broad,
                    shared=shared,
                    status="placement_depth_family_rescue_broad_bucket",
                    reason=(
                        "stopped at broad bucket because top-family/role rescue only proved a broad placement; "
                        f"terminal identity was not directly corroborated for {raw_claim.final_label}"
                    ),
                )

        if current is None:
            return None

        current_reliability = profile_reliability_from_row(current)
        if current_reliability["tier"] not in {"tiny", "weak", "tentative"}:
            return None

        if parent_strength < self.policy.strong_role_strength:
            return None

        broad = self.best_broad_bucket(
            shared=shared,
            top_family=raw_claim.family,
            parent_role=parent_role,
            current_score=float(current.get("combined_rank_score", 9999.0)),
            current_label=raw_claim.final_label,
            allow_synthetic=True,
        )
        if broad is None:
            return None

        return self._broad_claim(
            broad=broad,
            shared=shared,
            status="placement_depth_broad_bucket",
            reason=(
                "stopped at broad bucket because terminal profile was under-supported "
                f"({current_reliability['tier']}, effective_count={current_reliability['effective_count']}) "
                f"while measured role was strong: {parent_role}={parent_strength:.2f}"
            ),
        )

    def has_direct_terminal_identity(self, row: dict[str, Any] | None) -> bool:
        """Return True only when a deep terminal label has direct voter proof."""
        if not row:
            return False
        brain_rank = int(row.get("brain_rank", 9999))
        physics_rank = int(row.get("physics_rank", 9999))
        combined = float(row.get("combined_rank_score", 9999.0))
        label = str(row.get("label", row.get("folder_path", ""))).replace("\\", "/").lower()
        if combined <= self.policy.terminal_identity_max_combined_rank_score and (
            min(brain_rank, physics_rank) <= self.policy.terminal_identity_primary_rank
            and max(brain_rank, physics_rank) <= self.policy.terminal_identity_secondary_rank
        ):
            return True
        concrete_melodic_branch = any(
            fragment in label
            for fragment in (
                "/keys/",
                "/piano/",
                "/processed keys/",
                "/electric piano/",
                "/woodwinds/saxophone/",
                "/brass/",
            )
        )
        if not concrete_melodic_branch:
            return False
        return bool(
            (combined <= 12.0 and brain_rank <= 5 and physics_rank <= 5)
            or (
                combined <= 16.0
                and brain_rank <= 3
                and physics_rank <= 12
                and any(fragment in label for fragment in ("/keys/", "/piano/", "/processed keys/", "/electric piano/"))
            )
        )

    @staticmethod
    def row_prefers_compound_broad_loop(row: dict[str, Any] | None) -> bool:
        """Return True when PhysicsVoter says exact leaf identity is too narrow."""
        if not row:
            return False
        evidence: dict[str, Any] = {}
        for side in ("physics_evidence", "brain_evidence"):
            raw = row.get(side, {})
            if isinstance(raw, dict):
                evidence.update(raw)
        branch = str(
            evidence.get("physics_layer_branch")
            or evidence.get("physics_branch_layer_selected")
            or evidence.get("instrument_branch_selected")
            or ""
        )
        if branch != "MixedInstrument":
            return False
        return bool(evidence.get("compound_music_prefer_broad_loop"))

    def best_broad_bucket(
        self,
        *,
        shared: list[dict[str, Any]],
        top_family: str,
        parent_role: str,
        current_score: float,
        current_label: str = "",
        allow_synthetic: bool = False,
    ) -> dict[str, Any] | None:
        """Return the best shallower candidate in the same top family.

        If voters did not share a broad bucket, synthesize a role-level bucket.
        This is not terminal classification.  It is a safe fallback that says,
        for example, "instrument loop" when the exact instrument identity is not
        proven.
        """
        candidates: list[dict[str, Any]] = []
        for row in shared:
            if str(row.get("top_family", "")) != str(top_family):
                continue
            depth = label_depth(row)
            if depth > 3:
                continue
            combined = float(row.get("combined_rank_score", 9999.0))
            if combined > self.policy.max_broad_combined_rank_score:
                continue
            if combined > current_score + self.policy.family_rescue_broad_slack:
                continue
            if int(row.get("physics_rank", 9999)) > self.policy.max_broad_physics_rank:
                continue
            if int(row.get("brain_rank", 9999)) > self.policy.max_broad_brain_rank:
                continue
            signature = row.get("candidate_role_signature", {})
            role_fit = role_strength(signature if isinstance(signature, dict) else {}, parent_role)
            candidates.append({**row, "_role_fit": role_fit, "_depth": depth, "_synthetic": False})
        branch_broad = None
        if allow_synthetic:
            branch_broad = synthetic_branch_broad_bucket(
                top_family=top_family,
                parent_role=parent_role,
                current_label=current_label,
            )
        if candidates:
            candidates.sort(
                key=lambda row: (
                    int(row.get("_depth", 9999)),
                    -float(row.get("_role_fit", 0.0)),
                    float(row.get("combined_rank_score", 9999.0)),
                    int(row.get("physics_rank", 9999)),
                    int(row.get("brain_rank", 9999)),
                    str(row.get("label", "")),
                )
            )
            best = candidates[0]
            if branch_broad is not None and is_generic_instrument_loop_bucket(best):
                return branch_broad
            return best
        if allow_synthetic:
            if branch_broad is not None:
                return branch_broad
            return synthetic_broad_bucket(top_family=top_family, parent_role=parent_role)
        return None

    @staticmethod
    def facts_support_short_low_percussive_drum_parent(facts: SharedAudioFacts) -> bool:
        """Return True for short low drum hits misread as pitched instrument loops.

        This is placement-depth safety, not final leaf ID.  If the measured role
        layer already says a file is a strong one-shot hit with almost all
        energy in the low band, broad Instrument Loop placement is unsafe even
        when a bass/synth-bass candidate survived the shared-candidate list.
        """
        evidence = facts.evidence if isinstance(getattr(facts, "evidence", None), dict) else {}
        roles = evidence.get("measured_roles", {}) if isinstance(evidence, dict) else {}
        roles = roles if isinstance(roles, dict) else {}
        role_evidence = roles.get("evidence", {}) if isinstance(roles.get("evidence", {}), dict) else {}
        shape = evidence.get("shape_vote", {}) if isinstance(evidence.get("shape_vote", {}), dict) else {}

        duration = _number(evidence.get("duration_sec"), facts.feature_values_by_name.get("duration_sec", 0.0))
        events = max(
            _number(evidence.get("event_count_estimate")),
            _number(evidence.get("onset_count")),
            _number(shape.get("onset_count")),
        )
        percussive = max(
            _number(roles.get("percussive_one_shot")),
            _number(roles.get("low_kick_like_hit")),
            _number(role_evidence.get("low_pitched_hit_raw")),
        )
        bass_loop = _number(roles.get("bass_loop"))
        pitched_loop = _number(roles.get("pitched_music_loop"))
        low_total = _number(role_evidence.get("low_total"))
        if low_total <= 0.0:
            low_total = _number(facts.feature_values_by_name.get("sub_bass_ratio_lt_150hz")) + _number(
                facts.feature_values_by_name.get("bass_ratio_150_500hz")
            )
        low_event = _number(shape.get("low_event_ratio"), _number(facts.feature_values_by_name.get("low_event_ratio")))
        high_event = _number(
            shape.get("high_event_ratio"), _number(facts.feature_values_by_name.get("high_event_ratio"))
        )
        pitch_conf = _number(
            role_evidence.get("pitch_confidence"), _number(facts.feature_values_by_name.get("pitch_confidence"))
        )
        f0_voiced = _number(
            role_evidence.get("f0_voiced_ratio"), _number(facts.feature_values_by_name.get("f0_voiced_ratio"))
        )
        attack = _number(role_evidence.get("attack_rise_time_norm"), _number(shape.get("attack_rise_time_norm")))
        temporal = _number(role_evidence.get("temporal_centroid_ratio"), _number(shape.get("temporal_centroid_ratio")))
        tail = _number(role_evidence.get("tail_energy_ratio"), _number(shape.get("tail_ratio")))
        primary_shape = str(shape.get("primary_shape") or shape.get("shape") or "")

        return bool(
            percussive >= 0.86
            and bass_loop <= 0.10
            and pitched_loop <= 0.12
            and 0.0 < duration <= 0.80
            and 0.0 < events <= 3.0
            and max(low_total, low_event) >= 0.84
            and high_event <= 0.12
            and pitch_conf >= 0.55
            and f0_voiced <= 0.08
            and attack <= 0.16
            and temporal <= 0.58
            and tail <= 0.72
            and primary_shape
            in {"solo_phrase", "hit_with_tail", "single_hit", "impact_with_tail", "echo_tail_hit", "bass_phrase"}
        )

    @staticmethod
    def _row_for_label(shared: list[dict[str, Any]], label: str) -> dict[str, Any] | None:
        for row in shared:
            if str(row.get("label", "")) == str(label):
                return row
        return None

    @staticmethod
    def _broad_claim(
        *,
        broad: dict[str, Any],
        shared: list[dict[str, Any]],
        status: str,
        reason: str,
    ) -> ConsensusClaim:
        """Return a broad placement-depth claim without creating a final decision."""
        folder_path = str(broad["folder_path"])
        is_synthetic = bool(broad.get("brain_evidence", {}).get("synthetic_broad_bucket"))
        can_override = True
        if is_synthetic and _path_has_any(folder_path, ("human and voice", "voice", "vocal")):
            can_override = False
        return claim_from_folder_path(
            folder_path=folder_path,
            source=status,
            reason=reason,
            shared=shared,
            raw_candidate_score=float(broad.get("combined_rank_score", 9999.0)),
            brain_rank=int(broad.get("brain_rank", 9999)),
            physics_rank=int(broad.get("physics_rank", 9999)),
            shared_winner=str(broad["label"]),
            can_override=can_override,
            strength=0.86,
            is_real_candidate=not is_synthetic,
        )


def _number(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def is_generic_instrument_loop_bucket(row: dict[str, Any]) -> bool:
    """Return True for the broad all-instruments loop fallback."""
    label = str(row.get("label", row.get("folder_path", ""))).replace("\\", "/").lower()
    return label in {
        "instruments/instrument loops/loops",
        "instruments/mixed musical loops/multi instrument/loops",
    }


def synthetic_branch_broad_bucket(top_family: str, parent_role: str, current_label: str) -> dict[str, Any] | None:
    """Return a branch fallback only when branch identity is truly safe.

    Earlier versions preserved many instrument branches during placement-depth
    broadening.  That still over-claimed weak pitched loops as Keys, Mallets,
    Guitar, or Brass/Woodwinds when the only safe conclusion was simply
    "instrument loop."  Keep the generic broad bucket as the default.  Synth
    is the one branch currently allowed here because a separate Synth profile
    claim also requires direct synth candidate support.
    """
    top = str(top_family)
    role = str(parent_role)
    label = str(current_label or "").replace("\\", "/").lower()
    if top == "Drums" and role in {"percussive_one_shot", "protected_percussive_one_shot", "low_kick_like_hit"}:
        if "/kick drums/" in label:
            return synthetic_row(label="Drums/Kick Drums/Generic Kick/One Shots", top_family="Drums")
        if "/snares/" in label:
            return synthetic_row(label="Drums/Snares/Generic Snare/One Shots", top_family="Drums")
        if "/claps snaps slaps/" in label:
            return synthetic_row(label="Drums/Claps Snaps Slaps/Generic Clap/One Shots", top_family="Drums")
        if "/hi hats/" in label:
            return synthetic_row(label="Drums/Hi Hats/Generic Hat/One Shots", top_family="Drums")
        if "/cymbals/" in label:
            return synthetic_row(label="Drums/Cymbals/Generic Cymbal/One Shots", top_family="Drums")
        if "/toms/" in label:
            return synthetic_row(label="Drums/Toms/Generic Tom/One Shots", top_family="Drums")
        if "/rims and sticks/" in label:
            return synthetic_row(label="Drums/Rims and Sticks/Generic Rim or Stick/One Shots", top_family="Drums")
        return None
    if top != "Instruments":
        return None
    loop_like = role in {
        "pitched_music_loop",
        "pitched_music_phrase",
        "mixed_music_loop",
        "clean_sustained_tonal_instrument_loop",
        "bass_loop",
    }
    if not loop_like:
        return None
    if "/synths/" in label:
        return synthetic_row(label="Instruments/Synths/Synth Loops/Loops", top_family="Instruments")
    return None


def synthetic_broad_bucket(top_family: str, parent_role: str) -> dict[str, Any] | None:
    """Return a safe broad bucket for a measured structure/family pair."""
    top = str(top_family)
    role = str(parent_role)
    if top == "Instruments":
        if role == "bass_loop":
            label = "Instruments/Bass/Bass Loops"
        elif role in {"voiced_one_shot", "vocal_music_phrase"}:
            label = "Instruments/Voice/Phrase/One Shots"
        else:
            label = "Instruments/Instrument Loops/Loops"
        return synthetic_row(label=label, top_family="Instruments")
    if top == "Drums":
        if role in {"bright_drum_loop", "percussive_drum_loop", "low_rhythmic_drum_loop"}:
            return synthetic_row(label="Drums/Drum Loops/Loops", top_family="Drums")
        if role == "percussive_one_shot":
            return synthetic_row(label="Drums/Percussion/Generic Percussion/One Shots", top_family="Drums")
        return None
    if top == "FX":
        if role in {"voiced_one_shot", "vocal_music_phrase"}:
            return synthetic_row(label="FX/Human and Voice FX", top_family="FX")
        if role in {"transition_riser", "riser", "build"}:
            return synthetic_row(label="FX/Structural and Transitional FX/Risers and Builds", top_family="FX")
        if role in {"transition_drop", "drop", "downlifter"}:
            return synthetic_row(label="FX/Structural and Transitional FX/Drops and Downlifters", top_family="FX")
        return synthetic_row(label="FX/Hybrid Designed FX", top_family="FX")
    if top == "Textures":
        return synthetic_row(label="Textures/Hybrid Textures", top_family="Textures")
    return None


def synthetic_row(label: str, top_family: str) -> dict[str, Any]:
    return {
        "label": label,
        "folder_path": label,
        "top_family": top_family,
        "brain_rank": 999,
        "physics_rank": 999,
        "combined_rank_score": 999.0,
        "brain_score": 0.0,
        "physics_score": 0.0,
        "brain_confidence": 0.0,
        "physics_confidence": 0.0,
        "brain_evidence": {"synthetic_broad_bucket": True},
        "physics_evidence": {"synthetic_broad_bucket": True},
        "candidate_role_signature": {},
        "candidate_role_distance": None,
        "family_compatibility": {},
    }


def profile_reliability_from_row(row: dict[str, Any]) -> dict[str, Any]:
    """Return merged support/reliability diagnostics for a shared row."""
    evidence: dict[str, Any] = {}
    for side in ("physics_evidence", "brain_evidence"):
        raw = row.get(side, {})
        if isinstance(raw, dict):
            evidence.update(raw)
    return profile_reliability_from_evidence(evidence)


def profile_reliability_from_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    """Return a simple support tier from voter evidence."""

    def number(name: str, default: float = 0.0) -> float:
        try:
            return float(evidence.get(name, default) or default)
        except Exception:
            return default

    effective_count = number("profile_effective_count", number("effective_count", 0.0))
    raw_count = number("profile_raw_count", number("raw_count", effective_count))
    strength = str(
        evidence.get("profile_fact_profile_strength", evidence.get("fact_profile_strength", "")) or ""
    ).lower()
    if effective_count <= 4:
        tier = "tiny"
    elif effective_count <= 12 or strength in {"tentative", "weak"}:
        tier = "weak"
    elif effective_count <= 30:
        tier = "moderate"
    else:
        tier = "strong"
    if strength == "tentative":
        tier = "tentative"
    return {
        "tier": tier,
        "effective_count": int(effective_count),
        "raw_count": int(raw_count),
        "fact_profile_strength": strength,
    }


def label_depth(row: dict[str, Any]) -> int:
    """Return path depth from a shared-candidate row."""
    return label_depth_from_label(str(row.get("label", "")))


def label_depth_from_label(label: str) -> int:
    return len([part for part in str(label).split("/") if part])


def _path_has_any(path: str, fragments: tuple[str, ...]) -> bool:
    """Return True when an internal category path contains any fragment."""
    normalized = str(path or "").replace("\\", "/").lower()
    return any(fragment in normalized for fragment in fragments)
