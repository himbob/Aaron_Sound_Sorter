"""Shared GUI brain-family filename policy.

The GUI sorter, backup step, and correction trainer all need the same view of
the active brain family.  Keeping the list here prevents the UI from backing up
one set of brains while training another.
"""

from __future__ import annotations

from aaron_audio_intelligence.physics_memory_brain import PHYSICS_MEMORY_BRAIN_NAME
from aaron_audio_intelligence.shape_memory_brain import SHAPE_MEMORY_BRAIN_NAME, SHAPE_STARTER_MEMORY_BRAIN_NAME
from aaron_audio_intelligence.user_memory_brain import USER_MEMORY_BRAIN_NAME
from aaron_audio_intelligence.voter_memory_brain import VOTER_MEMORY_BRAIN_NAME

ACTIVE_GUI_BRAIN_FILE_NAMES: tuple[str, ...] = (
    "stage4_folder_brain.json",
    "stage4_folder_brain_baby.json",
    "stage4_folder_brain_core_baby.json",
    "stage4_folder_brain_spread_baby.json",
    "stage4_folder_brain_outlier_baby.json",
    USER_MEMORY_BRAIN_NAME,
    PHYSICS_MEMORY_BRAIN_NAME,
    VOTER_MEMORY_BRAIN_NAME,
    SHAPE_STARTER_MEMORY_BRAIN_NAME,
    SHAPE_MEMORY_BRAIN_NAME,
    "stage4_folder_brain_harmonic_core_baby.json",
    "stage4_folder_brain_harmonic_spread_baby.json",
    "stage4_folder_brain_harmonic_outlier_baby.json",
)

INCREMENTAL_TRAINING_BRAIN_FILE_NAMES: tuple[str, ...] = (
    USER_MEMORY_BRAIN_NAME,
    PHYSICS_MEMORY_BRAIN_NAME,
    VOTER_MEMORY_BRAIN_NAME,
    SHAPE_MEMORY_BRAIN_NAME,
)


def active_gui_brain_file_names() -> list[str]:
    """Return every brain JSON filename the GUI may read, back up, or update.

    Returns:
        Ordered brain filenames from the GUI brain family. Missing files are
        skipped by callers so optional lanes can be listed safely.

    Side Effects:
        None.
    """

    return list(ACTIVE_GUI_BRAIN_FILE_NAMES)


def incremental_training_brain_names() -> tuple[str, ...]:
    """Return brain filenames that receive human correction evidence.

    Returns:
        Ordered dedicated memory-brain filenames for GUI and trusted-panel
        incremental updates. Full/core/spread/outlier and starter brains are
        rebuild-only and never receive sparse one-off corrections.

    Side Effects:
        None.
    """

    return INCREMENTAL_TRAINING_BRAIN_FILE_NAMES
