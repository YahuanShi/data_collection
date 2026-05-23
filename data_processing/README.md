# Data Processing

Tools for converting raw HDF5 episodes (recorded by `recorder/episode_recorder.py`)
into training-ready datasets for **Pi0.5 / LeRobot, ACT, and Diffusion Policy**.

### Framework conversion paths

| Framework | Format needed | Tool |
|-----------|--------------|------|
| **Pi0.5 / LeRobot** | LeRobot dataset | `examples/ur5/convert_ur5_data_to_lerobot.py` *(project root)* |
| **ACT** | HDF5 directly | No conversion — `act-main/imitate_episodes_ur5.py` reads HDF5 |
| **Diffusion Policy** | Zarr store | `convert_hdf5_to_zarr.py` *(this directory)* |

The pipeline scripts below apply to all three: run them on the raw HDF5 dataset first,
then pass the processed output to whichever conversion/training tool you need.

## Quick Start

```bash
# Run the full pipeline
bash pipeline/pipeline.sh path/to/raw_dataset_dir

# Custom target size or resize method
bash pipeline/pipeline.sh path/to/raw_dataset_dir --size 224 --method center_crop
```

## Directory Structure

```
data_processing/
├── convert_hdf5_to_zarr.py               # HDF5 → Zarr conversion for Diffusion Policy
│
├── pipeline(optional)/                   # Sequential HDF5 processing steps
│   ├── 01_check_dataset.py
│   ├── 02_drop_front_camera.py
│   ├── 03_smooth_episodes.py
│   ├── 04_trim_episodes.py
│   ├── 05_resize_images.py
│   ├── split_episodes_by_gripper.py
│   └── pipeline.sh                       # Orchestrates all 5 steps end-to-end
│
└── viz/                                  # Standalone visualisation tools (no data modified)
    ├── viz_episode.py
    ├── viz_trajectory.py
    └── viz_EEFin3D.py
```

## Typical Workflow

```
recorder/episode_recorder.py
    → dataset/raw/ur5_dataset_YYYYMMDD/episode_N.hdf5
         (480×480 images, qpos[7]+qvel[7]+eef_pose[6]+action[7])

    │
    ├── [optional] viz/viz_episode.py      ← review & delete bad episodes
    │
    ▼  [optional] pipeline/pipeline.sh
    1. check_dataset        quality gate
    2. drop_front_camera    keep exterior + wrist only  [optional]
    3. smooth_episodes      Savitzky-Golay on qpos / action  (qvel + eef_pose passed through unchanged)
    4. trim_episodes        cut idle frames at start/end
    5. resize_images        480×480 → 224×224 (or any target size)
    │
    ├── [optional] viz/viz_trajectory.py   ← verify smoothing / trimming
    │
    ▼  dataset/processed/resized/  (HDF5, any resolution)
    │
    ├──────────────────────────────────────────┐──────────────────┐
    │                                          │                  │
    ▼  OpenPi / LeRobot                        ▼  ACT             ▼  Diffusion Policy
    examples/ur5/                              act/          convert_hdf5_to_zarr.py
    convert_ur5_data_to_lerobot.py             imitate_           → dataset/zarr/
    → dataset/for_training/                    episodes_ur5.py      ur5_dataset.zarr
```

---

## Pipeline Scripts

### `pipeline/01_check_dataset.py` — quality report

Scans all episodes and prints a report of data issues. No files are written.

```bash
python3 pipeline/01_check_dataset.py path/to/dataset_dir
python3 pipeline/01_check_dataset.py path/to/dataset_dir --spike-thresh 0.10
```

| Flag | Meaning |
|------|---------|
| `cameras` | Number of camera streams detected (informational) |
| `static_action` | `action` never changes — likely a recording bug |
| `frozen_gripper` | Gripper dimension is constant for the whole episode |
| `qpos_eq_action` | `qpos ≈ action` — recorder may be duplicating state |
| `short` | Fewer than `--min-steps` timesteps |
| `spikes` | Joint step exceeds `--spike-thresh` rad (warning, not failure) |

Exit code is `1` if any structural issues are found, `0` otherwise.

---

### `pipeline/02_drop_front_camera.py` — remove front camera stream

Copies all episodes to a new directory, dropping `front_image_1` so downstream
tools only see the exterior and wrist cameras. Skip this step with `--keep-front`
in `pipeline.sh` when three cameras are needed.

```bash
python3 pipeline/02_drop_front_camera.py path/to/raw_dir           path/to/no_front_dir
python3 pipeline/02_drop_front_camera.py path/to/episode_0.hdf5    path/to/episode_0_nf.hdf5
```

---

### `pipeline/03_smooth_episodes.py` — trajectory smoothing

Applies Savitzky-Golay smoothing to `qpos` and `action` trajectories.
The gripper dimension is left unsmoothed to preserve open/close transitions.
`qvel` and `eef_pose` (if present) are copied through unchanged — smoothing is not appropriate for velocity or Cartesian pose fields.
Run **before** trimming so the smoother doesn't see pad frames at the edges.

```bash
python3 pipeline/03_smooth_episodes.py path/to/dataset_dir        smoothed/
python3 pipeline/03_smooth_episodes.py path/to/episode_0.hdf5     smoothed/episode_0.hdf5
python3 pipeline/03_smooth_episodes.py path/to/dataset_dir        smoothed/ --window 9 --poly 2
```

---

### `pipeline/04_trim_episodes.py` — cut start/end frames

**Interactive mode** (default) — loops through each episode, shows the frame
count, and asks you to type the keep range:

```bash
python3 pipeline/04_trim_episodes.py path/to/dataset_dir  trimmed/
```

```
  episode_0.hdf5  [250 frames]
  Keep range (start end, negative ok, Enter = keep all): 10 -8
  → keeps frames 10-242  (233 frames)
```

- `10 230`  — keep frames 10 to 230 (inclusive)
- `10 -5`   — keep frames 10 to T−5 (negative counts from end)
- Enter      — keep all frames unchanged

**Batch mode** — apply the same range to every episode:

```bash
python3 pipeline/04_trim_episodes.py path/to/dataset_dir  trimmed/ --start 10 --end -8
```

**Dry-run** — preview the plan without writing anything (omit dst):

```bash
python3 pipeline/04_trim_episodes.py path/to/dataset_dir
```

All fields (`qpos`, `qvel`, `eef_pose`, `action`, images) are sliced to the kept frame range. `qvel` and `eef_pose` are included only when present in the source file (backward compatible with older datasets).

---

### `pipeline/05_resize_images.py` — resize images to training resolution

Converts all camera streams in HDF5 episodes from the recorded resolution
(e.g. 480×480) to the target training resolution (default 224×224).
Three preprocessing strategies are available:

| `--method` | Behaviour | Use when |
|---|---|---|
| `center_crop` (default) | Square-crop centre → resize | Task object always centred |
| `resize` | Squish full image to target | Global context matters |
| `pad_resize` | Aspect-ratio resize + black padding | Source is not square |

```bash
# Default: 224×224 center crop  (pi0.5)
python3 pipeline/05_resize_images.py trimmed/              dataset/processed/resized/

# Single file
python3 pipeline/05_resize_images.py episode_0.hdf5        episode_0_224.hdf5

# Custom size and method
python3 pipeline/05_resize_images.py trimmed/              dataset/processed/resized_256/ --size 256 --method resize

# Dry-run (omit dst)
python3 pipeline/05_resize_images.py trimmed/ --dry-run
```

Trajectory data (`qpos`, `qvel`, `eef_pose`, `action`) is copied unchanged — only image streams are resized. `qvel` and `eef_pose` are included only when present (backward compatible). Parallelised over episodes via `--workers` (default 4).

---

### `pipeline/pipeline.sh` — end-to-end pipeline

Runs all 5 steps sequentially with confirmation prompts between stages.

```bash
./pipeline/pipeline.sh path/to/raw_dataset_dir
./pipeline/pipeline.sh path/to/raw_dataset_dir --size 224 --method center_crop
./pipeline/pipeline.sh path/to/raw_dataset_dir --keep-front --size 256 --method resize
```

| Option | Default | Description |
|--------|---------|-------------|
| `--out DIR` | `dataset/processed/resized/<date>` | Final output directory |
| `--size N` | `224` | Target image size in pixels |
| `--method STR` | `center_crop` | Resize strategy |
| `--window N` | `15` | SG filter window (must be odd) |
| `--poly N` | `3` | SG polynomial order |
| `--keep-front` | — | Skip the drop_front_camera step |
| `--workers N` | `4` | Parallel workers for resize step |

All outputs go under `dataset/processed/`:
```
dataset/processed/no_front/<date>/    after step 2
dataset/processed/smoothed/<date>/    after step 3
dataset/processed/trimmed/<date>/     after step 4
dataset/processed/resized/<date>/     final HDF5 output (step 5)
```
Intermediates are safe to delete once the pipeline completes successfully.
`dataset/for_training/` is reserved for LeRobot-converted datasets (`convert_ur5_data_to_lerobot.py`).

---

## Visualisation Tools

These tools never modify data and can be run at any point independently.

### `viz/viz_episode.py` — video playback viewer

Plays back all camera streams side-by-side (2 or 3 cameras, auto-detected) with
per-joint trajectory strips and a scrubbing progress bar.

```bash
python3 viz/viz_episode.py path/to/dataset_dir
python3 viz/viz_episode.py path/to/episode_0.hdf5 --fps 30 --scale 2.0
```

| Key | Action |
|-----|--------|
| `SPACE` | Pause / resume |
| `←` / `→` | Step ±5 frames |
| `↑` / `↓` | Previous / next episode |
| `F` | Toggle 2× speed |
| `R` | Restart |
| `D` | Arm delete (shows red confirmation banner) |
| `Y` | Confirm delete — removes file from disk |
| `Q` | Quit |

Mouse drag on the progress bar or trajectory strips to scrub.

---

---

## `convert_hdf5_to_zarr.py` — HDF5 → Zarr for Diffusion Policy

Reads all `episode_N.hdf5` files from a dataset directory and concatenates them
into a single Zarr store in the layout that Diffusion Policy's `UR5ImageDataset` expects.

```
Output Zarr layout:
    data/
        exterior_image   (T_total, H, W, 3)  uint8
        wrist_image      (T_total, H, W, 3)  uint8   [omit with --no-wrist]
        robot_qpos       (T_total, 7)         float32
        eef_pose         (T_total, 6)         float32  [omit with --no-eef or if absent in source]
        action           (T_total, 7)         float32
    meta/
        episode_ends     (n_episodes,)        int64
```

`eef_pose` is automatically included when the source HDF5 files contain `/observations/eef_pose`. Use `--no-eef` for datasets recorded before EEF support was added.

```bash
# Convert at recorded resolution (480×480 or whatever was recorded)
python3 convert_hdf5_to_zarr.py \
    --raw-dir  dataset/processed/resized/ur5_dataset_20260415 \
    --zarr-out dataset/zarr/ur5_dataset_20260415.zarr

# Resize images to 224×224 during conversion (saves disk space)
python3 convert_hdf5_to_zarr.py \
    --raw-dir  dataset/processed/resized/ur5_dataset_20260415 \
    --zarr-out dataset/zarr/ur5_dataset_20260415.zarr \
    --image-size 224

# Exclude wrist camera
python3 convert_hdf5_to_zarr.py \
    --raw-dir  dataset/processed/resized/ur5_dataset_20260415 \
    --zarr-out dataset/zarr/ur5_dataset_20260415.zarr \
    --no-wrist

# Old dataset without eef_pose (recorded before EEF support)
python3 convert_hdf5_to_zarr.py \
    --raw-dir  dataset/processed/resized/ur5_dataset_20260101 \
    --zarr-out dataset/zarr/ur5_dataset_20260101.zarr \
    --no-eef

# Overwrite an existing zarr
python3 convert_hdf5_to_zarr.py ... --overwrite
```

After converting, set `task.zarr_path` in
`diffusion_policy-main/diffusion_policy/config/task/ur5_image.yaml`
to the output path, then run training:

```bash
cd diffusion_policy-main
python train.py --config-name=train_diffusion_unet_ur5_image_workspace.yaml
```

---

### `viz/viz_trajectory.py` — original vs processed trajectory comparison

Overlays original and processed joint trajectories in 7 vertical subplots —
one per joint. Useful for verifying smoothing and trim decisions.

```bash
python3 viz/viz_trajectory.py original_dir/ training_dataset/
python3 viz/viz_trajectory.py original_dir/ training_dataset/ --no-norm
```

`--no-norm` plots absolute frame numbers instead of a normalised 0–1 x-axis.

| Key | Action |
|-----|--------|
| `↑` / `←` / `P` | Previous episode |
| `↓` / `→` / `N` | Next episode |
| `S` | Save figure as PNG |
| `Q` / `Escape` | Quit |

---

### `viz/viz_EEFin3D.py` — 3D end-effector trajectory viewer

3D visualisation of `observations/eef_pose` (x, y, z, rx, ry, rz).

Left panel: 3D trajectory coloured by time (plasma), with sampled orientation
frames shown as RGB axis arrows (X=red, Y=green, Z=blue).  
Right panel: time-series strips for all 6 channels, with a frame cursor.

```bash
python3 viz/viz_EEFin3D.py path/to/dataset_dir
python3 viz/viz_EEFin3D.py path/to/episode_0.hdf5
python3 viz/viz_EEFin3D.py path/to/dataset_dir --orient-step 20
```

| Key | Action |
|-----|--------|
| `N` / `↓` | Next episode |
| `P` / `↑` | Previous episode |
| `→` / `.` | Step +1 frame |
| `←` / `,` | Step −1 frame |
| `Home` / `End` | First / last frame |
| `O` | Toggle orientation arrows |
| `S` | Save figure as PNG |
| `Q` / `Escape` | Quit |
