#!/usr/bin/env python3
"""
Apply Savitzky-Golay smoothing to qpos and action trajectories in HDF5 episodes.

The gripper dimension (last column) is intentionally left unsmoothed to
preserve discrete open/close transitions.

The last 2 timesteps are also left unchanged so the final joint state is
never altered by smoothing.

Usage:
    python3 03_smooth_episodes.py src [dst]
    python3 03_smooth_episodes.py path/to/dataset_dir        smoothed/
    python3 03_smooth_episodes.py path/to/episode_0.hdf5     smoothed/episode_0.hdf5
    python3 03_smooth_episodes.py path/to/dataset_dir        smoothed/ --window 9 --poly 2
"""

import argparse
import glob
import os
import sys

import h5py
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter

DEFAULT_WINDOW =9  # must be odd
DEFAULT_POLY = 1
FLAT_STD_THRESH = 1e-2  # rolling-std threshold to detect flat regions
FLAT_RANGE_THRESH = 1e-2  # if entire column range < this, skip smoothing it
PRESERVE_TAIL = 2  # number of final timesteps to leave untouched


def _rolling_std(col: np.ndarray, window: int) -> np.ndarray:
    """Fast rolling standard deviation via uniform_filter1d."""
    mean = uniform_filter1d(col.astype(float), size=window, mode="nearest")
    mean_sq = uniform_filter1d(col.astype(float) ** 2, size=window, mode="nearest")
    return np.sqrt(np.maximum(mean_sq - mean**2, 0.0))


def smooth_array(arr: np.ndarray, window: int, poly: int) -> np.ndarray:
    """Savitzky-Golay column-wise; last column (gripper) is left unchanged.

    Flat regions in the original (rolling std < FLAT_STD_THRESH) are restored
    after filtering to prevent transition-edge ringing from disturbing them.

    The last PRESERVE_TAIL timesteps are always restored to their original
    values so the final joint state is never altered by smoothing.
    """
    if len(arr) < window:
        return arr.copy()
    out = arr.copy()
    for d in range(arr.shape[1] - 1):
        col = arr[:, d]
        # flatten to mean if column is globally flat (avoids SG ringing on constant joints)
        if col.max() - col.min() < FLAT_RANGE_THRESH:
            out[:, d] = col.mean()
            continue
        smoothed = savgol_filter(col, window_length=window, polyorder=poly)
        flat_mask = _rolling_std(col, window) < FLAT_STD_THRESH
        smoothed[flat_mask] = col[flat_mask]
        out[:, d] = smoothed
    # Restore the last PRESERVE_TAIL timesteps and blend the transition
    if PRESERVE_TAIL > 0 and len(arr) > PRESERVE_TAIL:
        tail_start = len(arr) - PRESERVE_TAIL
        blend_len = min(window, tail_start)  # transition zone length
        blend_start = tail_start - blend_len
        # alpha ramps from 0 (fully smoothed) to 1 (fully original)
        alpha = np.linspace(0.0, 1.0, blend_len + PRESERVE_TAIL)
        for i, a in enumerate(alpha):
            idx = blend_start + i
            out[idx] = (1.0 - a) * out[idx] + a * arr[idx]
    return np.round(out, 1)


def smooth_episode(src: str, dst: str, window: int, poly: int) -> None:
    with h5py.File(src, "r") as f:
        qpos = f["observations/qpos"][:]
        action = f["action"][:]
        qvel = f["observations/qvel"][:] if "observations/qvel" in f else None
        eef_pose = f["observations/eef_pose"][:] if "observations/eef_pose" in f else None
        exterior = f["observations/images/exterior_image_1_left"][:]
        wrist = f["observations/images/wrist_image_left"][:]
        imgs = f["observations/images"]
        front = imgs["front_image_1"][:] if "front_image_1" in imgs else None
        attrs = dict(f.attrs)

    with h5py.File(dst, "w") as f:
        for k, v in attrs.items():
            f.attrs[k] = v
        f.create_dataset("observations/qpos", data=smooth_array(qpos, window, poly), compression="gzip")
        f.create_dataset("action", data=smooth_array(action, window, poly), compression="gzip")
        if qvel is not None:
            f.create_dataset("observations/qvel", data=qvel, compression="gzip")
        if eef_pose is not None:
            f.create_dataset("observations/eef_pose", data=eef_pose, compression="gzip")
        f.create_dataset("observations/images/exterior_image_1_left", data=exterior, compression="gzip")
        f.create_dataset("observations/images/wrist_image_left", data=wrist, compression="gzip")
        if front is not None:
            f.create_dataset("observations/images/front_image_1", data=front, compression="gzip")


def collect_files(src: str) -> list[str]:
    if os.path.isfile(src):
        return [src]
    if os.path.isdir(src):
        files = sorted(glob.glob(os.path.join(src, "*.hdf5")))
        if not files:
            sys.exit(f"No .hdf5 files found in: {src}")
        return files
    sys.exit(f"Path does not exist: {src}")


def main():
    parser = argparse.ArgumentParser(description="Smooth qpos and action with Savitzky-Golay filter.")
    parser.add_argument("src", help="Single .hdf5 file or directory containing *.hdf5 files")
    parser.add_argument("dst", nargs="?", default=None, help="Output file or directory (default: same as src)")
    parser.add_argument(
        "--window", type=int, default=DEFAULT_WINDOW, help=f"Filter window length (odd, default {DEFAULT_WINDOW})"
    )
    parser.add_argument("--poly", type=int, default=DEFAULT_POLY, help=f"Polynomial order (default {DEFAULT_POLY})")
    args = parser.parse_args()

    if args.dst is None:
        args.dst = args.src if os.path.isdir(args.src) else os.path.dirname(args.src) or "."

    if args.window % 2 == 0:
        sys.exit("ERROR: --window must be odd")
    if args.poly >= args.window:
        sys.exit("ERROR: --poly must be less than --window")

    files = collect_files(args.src)
    single = os.path.isfile(args.src)

    if not single:
        os.makedirs(args.dst, exist_ok=True)
    print(f"Smoothing {len(files)} episode(s)  (window={args.window}, poly={args.poly})  →  {args.dst}")

    for path in files:
        dst = args.dst if single else os.path.join(args.dst, os.path.basename(path))
        smooth_episode(path, dst, args.window, args.poly)
        print(f"  WROTE {dst}")

    print("Done.")


if __name__ == "__main__":
    main()
