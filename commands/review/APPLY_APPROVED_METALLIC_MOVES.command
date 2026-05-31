#!/bin/zsh
set -e

PROJECT="/Volumes/T9/testbed/Aaron_Sound_Sorter"
APPROVE_ROOT="$PROJECT/reports/metallic_percussion_review/_APPROVED_MOVES"

TARGET_HIGH="$PROJECT/training/locked_curated_v1/Drums/Percussion/Bells and Metallic Percussion/High Rings and Chimes/One Shots"
TARGET_COW="$PROJECT/training/locked_curated_v1/Drums/Percussion/Bells and Metallic Percussion/Cowbells and Blocks/One Shots"
TARGET_OTHER="$PROJECT/training/locked_curated_v1/Drums/Percussion/Bells and Metallic Percussion/Other Metallic Percussion/One Shots"

mkdir -p "$TARGET_HIGH" "$TARGET_COW" "$TARGET_OTHER"

LOG_DIR="$PROJECT/reports/metallic_percussion_review/apply_logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/apply_metallic_moves_$(date +%Y%m%d_%H%M%S).csv"

echo "approval_folder,source_real_path,target_path,status" > "$LOG"

move_symlink_targets () {
  local APPROVAL_FOLDER="$1"
  local TARGET="$2"

  if [ ! -d "$APPROVAL_FOLDER" ]; then
    return
  fi

  find "$APPROVAL_FOLDER" -type l | while read LINK_PATH; do
    REAL_PATH="$(python3 - <<PY
import os
print(os.path.realpath("$LINK_PATH"))
PY
)"
    BASENAME="$(basename "$REAL_PATH")"
    TARGET_PATH="$TARGET/$BASENAME"

    if [ ! -f "$REAL_PATH" ]; then
      echo "\"$APPROVAL_FOLDER\",\"$REAL_PATH\",\"$TARGET_PATH\",\"MISSING_SOURCE\"" >> "$LOG"
      continue
    fi

    if [ -e "$TARGET_PATH" ]; then
      echo "\"$APPROVAL_FOLDER\",\"$REAL_PATH\",\"$TARGET_PATH\",\"TARGET_EXISTS_SKIPPED\"" >> "$LOG"
      continue
    fi

    mv "$REAL_PATH" "$TARGET_PATH"
    echo "\"$APPROVAL_FOLDER\",\"$REAL_PATH\",\"$TARGET_PATH\",\"MOVED\"" >> "$LOG"
  done
}

move_symlink_targets "$APPROVE_ROOT/High_Rings_and_Chimes" "$TARGET_HIGH"
move_symlink_targets "$APPROVE_ROOT/Cowbells_and_Blocks" "$TARGET_COW"
move_symlink_targets "$APPROVE_ROOT/Other_Metallic_Percussion" "$TARGET_OTHER"

echo
echo "DONE"
echo "Move log:"
echo "  $LOG"
echo
echo "Moved counts:"
grep ",MOVED$" "$LOG" | wc -l | sed 's/^/  /'
echo
open "$LOG"
