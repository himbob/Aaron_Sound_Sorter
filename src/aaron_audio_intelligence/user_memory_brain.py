"""User correction memory brain utilities.

This module keeps human-approved GUI corrections as a small, explicit brain
lane. It is intentionally source-name blind: paths are copied only for audit and
deduplication, while matching uses measured fingerprints stored by the training
flow.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

USER_MEMORY_BRAIN_NAME = "stage4_folder_brain_user_memory.json"

_METADATA_KEYS = (
    "version",
    "feature_names",
    "feature_size",
    "feature_weights",
    "scaler_mean",
    "scaler_std",
    "scalers_by_structure",
    "max_anchors_per_label",
)
_LABEL_MAP_KEYS = (
    "top_by_label",
    "structure_by_label",
    "centroid_counts",
    "label_models_by_label",
    "label_reliability_by_label",
    "centroids",
    "exemplars_by_label",
    "anchors_by_label",
    "examples_by_label",
    "training_examples_detailed_by_label",
    "counts",
    "effective_counts",
    "raw_counts",
    "effective_training_balance_by_label",
    "category_fact_profiles",
)


def build_empty_user_memory_brain(base_brain: dict[str, Any]) -> dict[str, Any]:
    """Return an empty brain JSON ready to store GUI correction evidence.

    Args:
        base_brain: Active full brain used as the source of scaler and feature
            metadata.

    Returns:
        Brain-shaped dictionary with no labels and empty prototype maps.

    Side Effects:
        None.

    Important Constraints:
        The returned brain contains no filename-derived category evidence. It
        only preserves numeric feature metadata required to score future audio
        fingerprints in the same coordinate space as the full brain.
    """
    memory: dict[str, Any] = {key: deepcopy(base_brain[key]) for key in _METADATA_KEYS if key in base_brain}
    memory.update(
        {
            "brain_type": "user_correction_memory_brain",
            "labels": [],
            "user_memory_policy": "explicit_gui_correction_teacher_lane",
            "user_memory_source": "gui_train_brains_from_corrections",
            "incremental_global_heads_need_full_rebuild": False,
        }
    )
    for key in _LABEL_MAP_KEYS:
        memory[key] = {}
    return memory


def is_user_memory_brain(brain: dict[str, Any] | None) -> bool:
    """Return true when a loaded brain is the GUI correction memory lane."""
    return bool(isinstance(brain, dict) and brain.get("brain_type") == "user_correction_memory_brain")


def merge_user_memory_into_brain(
    brain: dict[str, Any],
    user_memory_brain: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge correction-memory labels into an in-memory full-brain view.

    Args:
        brain: Loaded full brain used for the current sort run.
        user_memory_brain: Optional loaded correction-memory brain.

    Returns:
        ``brain`` after in-memory merge.

    Side Effects:
        Mutates only the passed ``brain`` object for the current sort run. No
        files are written.

    Important Constraints:
        This is not a final-folder rescue. It only makes human-taught measured
        fingerprints visible to BrainVoter and PhysicsVoter before normal
        arbitration.
    """
    if not is_user_memory_brain(user_memory_brain):
        return brain
    memory_labels = [str(label) for label in user_memory_brain.get("labels", []) if str(label)]
    if not memory_labels:
        brain["_user_memory_brain_loaded"] = True
        brain["_user_memory_brain_label_count"] = 0
        return brain

    labels = [str(label) for label in brain.get("labels", []) if str(label)]
    label_set = set(labels)
    for label in memory_labels:
        if label not in label_set:
            labels.append(label)
            label_set.add(label)
    brain["labels"] = sorted(labels)

    for key in _LABEL_MAP_KEYS:
        source_map = user_memory_brain.get(key, {})
        if not isinstance(source_map, dict):
            continue
        target_map = brain.get(key)
        if not isinstance(target_map, dict):
            target_map = {}
            brain[key] = target_map
        for label in memory_labels:
            if label not in source_map:
                continue
            if key == "training_examples_detailed_by_label":
                target_map[label] = merged_training_examples(target_map.get(label), source_map.get(label))
            else:
                target_map[label] = deepcopy(source_map[label])

    brain["_user_memory_brain_loaded"] = True
    brain["_user_memory_brain_label_count"] = len(memory_labels)
    brain["_user_memory_brain_policy"] = "merged_for_voter_level_teacher_recall_only"
    return brain


def merged_training_examples(existing_value: object, memory_value: object) -> list[dict[str, Any]]:
    """Return existing and memory correction examples with duplicate paths removed."""
    merged: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for value in (existing_value, memory_value):
        if not isinstance(value, list):
            continue
        for example in value:
            if not isinstance(example, dict):
                continue
            source_path = str(example.get("source_path", ""))
            if source_path and source_path in seen_paths:
                continue
            if source_path:
                seen_paths.add(source_path)
            merged.append(deepcopy(example))
    return merged
