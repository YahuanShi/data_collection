#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# pipeline.sh — Data processing pipeline for robot demonstration episodes
#
# Processes raw HDF5 episodes recorded at high resolution (e.g. 480×480) into
# a training-ready dataset at the target model input size (e.g. 224×224).
# Compatible with openpi / pi0.5 and other robot-learning frameworks.
#
# Pipeline steps (processing only — no visualisation):
#   1. check_dataset      — quality report (short/frozen/spike detection)
#   2. drop_front_camera  — remove front_image_1 stream  [skip with --keep-front]
#   3. smooth_episodes    — Savitzky-Golay trajectory smoothing
#   4. trim_episodes      — cut start/end frames via movement detection
#   5. resize_images      — resize/crop images to target training resolution
#   → <out_dir>/          — final training-ready dataset
#
# Visualisation tools (run independently at any stage):
#   viz/viz_episode.py       — interactive episode playback / deletion
#   viz/viz_trajectory.py    — before/after trajectory comparison
#
# Usage:
#   ./pipeline.sh <raw_dataset_dir> [out_dir] [options]
#
# Options:
#   --out         DIR    final output directory                (default: processed_data/resized/<date>)
#   --size        N      target image size in pixels           (default: 224)
#   --method      STR    resize strategy: center_crop | resize | pad_resize
#                                                              (default: center_crop)
#   --window      N      SG filter window, must be odd        (default: 15)
#   --poly        N      SG polynomial order                  (default: 3)
#   --keep-front         skip the drop_front_camera step
#   --workers     N      parallel workers for resize step     (default: 4)
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── defaults ──────────────────────────────────────────────────────────────────
RAW_DIR=""
OUT_DIR=""
IMG_SIZE=224
IMG_METHOD="center_crop"
WINDOW=15
POLY=3
KEEP_FRONT=0
WORKERS=4

# ── parse args ────────────────────────────────────────────────────────────────
usage() {
    grep '^#' "$0" | sed 's/^# \{0,1\}//' | head -35
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out)         OUT_DIR="$2";     shift 2 ;;
        --size)        IMG_SIZE="$2";    shift 2 ;;
        --method)      IMG_METHOD="$2";  shift 2 ;;
        --window)      WINDOW="$2";      shift 2 ;;
        --poly)        POLY="$2";        shift 2 ;;
        --keep-front)  KEEP_FRONT=1;     shift ;;
        --workers)     WORKERS="$2";     shift 2 ;;
        -h|--help) usage ;;
        -*) echo "Unknown option: $1"; usage ;;
        *)
            if [[ -z "$RAW_DIR" ]]; then
                RAW_DIR="$1"
            elif [[ -z "$OUT_DIR" ]]; then
                OUT_DIR="$1"
            fi
            shift ;;
    esac
done

[[ -z "$RAW_DIR" ]] && { echo "ERROR: raw_dataset_dir is required."; usage; }
[[ -d "$RAW_DIR" ]] || { echo "ERROR: '$RAW_DIR' is not a directory."; exit 1; }

RAW_DIR="${RAW_DIR%/}"
DATASET_NAME="$(basename "$RAW_DIR")"
# Extract date suffix (last sequence of digits at end of name)
DATASET_DATE="$(echo "$DATASET_NAME" | grep -oP '\d+$' || echo "$DATASET_NAME")"

# All processing outputs go under dataset/processed/
PROCESSED_BASE="${PROJECT_ROOT}/dataset/processed"
STAGE_CAM="${PROCESSED_BASE}/no_front/${DATASET_DATE}"
STAGE_SMOOTH="${PROCESSED_BASE}/smoothed/${DATASET_DATE}"
STAGE_TRIM="${PROCESSED_BASE}/trimmed/${DATASET_DATE}"

# Default final output: dataset/processed/resized/<date>/
# dataset/for_training/ is reserved for LeRobot-converted datasets (convert_ur5_data_to_lerobot.py)
[[ -z "$OUT_DIR" ]] && OUT_DIR="${PROCESSED_BASE}/resized/${DATASET_DATE}"

# ── helper ────────────────────────────────────────────────────────────────────
banner() {
    echo
    echo "════════════════════════════════════════════════════════"
    echo " $1"
    echo "════════════════════════════════════════════════════════"
}
confirm() {
    echo
    read -rp ">>> $1  Continue? [y/N] " ans
    [[ "$ans" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }
}

# ── Step 1: quality check ─────────────────────────────────────────────────────
banner "Step 1 / 5 — Quality check"
python3 "$SCRIPT_DIR/01_check_dataset.py" "$RAW_DIR" || true
confirm "Review the report above."

# ── Step 2: drop front camera (optional) ──────────────────────────────────────
if [[ "$KEEP_FRONT" -eq 1 ]]; then
    banner "Step 2 / 5 — Drop front camera  [SKIPPED — --keep-front]"
    STAGE_CAM="$RAW_DIR"
else
    banner "Step 2 / 5 — Drop front camera  →  $STAGE_CAM"
    python3 "$SCRIPT_DIR/02_drop_front_camera.py" "$RAW_DIR" "$STAGE_CAM"
fi

# ── Step 3: smooth trajectories ───────────────────────────────────────────────
banner "Step 3 / 5 — Smooth trajectories  →  $STAGE_SMOOTH"
python3 "$SCRIPT_DIR/03_smooth_episodes.py" "$STAGE_CAM" "$STAGE_SMOOTH" \
    --window "$WINDOW" --poly "$POLY"

# ── Step 4: trim episodes ─────────────────────────────────────────────────────
banner "Step 4 / 5 — Trim episodes  →  $STAGE_TRIM"
python3 "$SCRIPT_DIR/04_trim_episodes.py" "$STAGE_SMOOTH" "$STAGE_TRIM"

# ── Step 5: resize images ─────────────────────────────────────────────────────
banner "Step 5 / 5 — Resize images  ${IMG_SIZE}×${IMG_SIZE} [${IMG_METHOD}]  →  $OUT_DIR"
python3 "$SCRIPT_DIR/05_resize_images.py" "$STAGE_TRIM" "$OUT_DIR" \
    --size "$IMG_SIZE" --method "$IMG_METHOD" --workers "$WORKERS"

# ── done ──────────────────────────────────────────────────────────────────────
echo
echo "✓  Pipeline complete."
echo "   Final dataset   : $OUT_DIR  (${IMG_SIZE}×${IMG_SIZE}, method=${IMG_METHOD})"
echo "   Intermediates   : $STAGE_CAM | $STAGE_SMOOTH | $STAGE_TRIM  (safe to delete)"
