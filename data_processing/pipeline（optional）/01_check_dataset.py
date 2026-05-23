#!/usr/bin/env python3
"""
Quality-check all episodes in an HDF5 dataset and print a report.

Checks per episode:
  - static_action  : action never changes (likely recording bug)
  - frozen_gripper : gripper dimension is constant throughout
  - qpos_eq_action : qpos ≈ action (recorder may be duplicating state)
  - short          : fewer than --min-steps timesteps
  - spikes         : joint step exceeds --spike-thresh rad  (warning only)

Exit code: 0 if no failures, 1 if any structural issues found.

Usage:
    python3 check_dataset.py path/to/dataset_dir
    python3 check_dataset.py path/to/dataset_dir --spike-thresh 0.10
    python3 check_dataset.py path/to/dataset_dir --min-steps 50
"""

import argparse
import glob
import os
import sys

import h5py
import numpy as np

DEFAULT_MIN_STEPS = 80
DEFAULT_SPIKE_THRESH = 0.15  # rad
DEFAULT_SAME_THRESH = 1e-4


_IMAGE_KEYS = ["exterior_image_1_left", "wrist_image_left", "front_image_1"]


def check_episode(path: str, min_steps: int, spike_thresh: float, same_thresh: float) -> dict:
    issues = {}
    with h5py.File(path, "r") as f:
        qpos = f["observations/qpos"][:]
        action = f["action"][:]
        imgs = f.get("observations/images", {})
        num_cams = sum(1 for k in _IMAGE_KEYS if k in imgs)
        issues["cameras"] = num_cams

    T = len(qpos)

    if min_steps > T:
        issues["short"] = f"{T} steps < {min_steps}"

    if np.abs(np.diff(action, axis=0)).max() < 1e-9:
        issues["static_action"] = "action never changes"

    gripper = action[:, -1]
    if np.unique(gripper).size == 1:
        issues["frozen_gripper"] = f"gripper fixed at {gripper[0]:.4f}"

    err = np.abs(qpos - action).mean()
    if err < same_thresh:
        issues["qpos_eq_action"] = f"mean |qpos-action| = {err:.2e}"

    deltas = np.abs(np.diff(qpos[:, :-1], axis=0))
    spike_frames = np.where(deltas.max(axis=1) > spike_thresh)[0]
    if len(spike_frames):
        issues["spikes"] = f"{len(spike_frames)} frames exceed {spike_thresh:.3f} rad: {spike_frames[:5].tolist()}"
    return issues


_INFO_KEYS = {"cameras"}  # informational — not failures


def print_report(results: list[tuple[str, dict]]) -> int:
    BAD = "\033[91m✗\033[0m"
    OK = "\033[92m✓\033[0m"
    WARN = "\033[93m⚠\033[0m"

    n_bad = 0
    print()
    print(f"{'Episode':<30}  Cams  Status   Issues")
    print("─" * 86)
    for name, issues in results:
        num_cams = issues.get("cameras", "?")
        real_issues = {k: v for k, v in issues.items() if k not in _INFO_KEYS}
        structural = {k: v for k, v in real_issues.items() if k != "spikes"}
        if structural:
            icon = BAD
            n_bad += 1
        elif "spikes" in real_issues:
            icon = WARN
        else:
            icon = OK
        txt = "  |  ".join(f"{k}: {v}" for k, v in real_issues.items()) or "—"
        print(f"  {icon}  {name:<28}  {num_cams!s:>4}  {txt}")

    print("─" * 86)
    n_warn = sum(1 for _, i in results if "spikes" in i and not {k for k in i if k not in _INFO_KEYS | {"spikes"}})
    n_clean = sum(1 for _, i in results if not {k for k in i if k not in _INFO_KEYS})
    print(f"Total: {len(results)}  |  bad: {n_bad}  |  warned: {n_warn}  |  clean: {n_clean}")
    print()
    return n_bad


def main():
    parser = argparse.ArgumentParser(description="Quality-check UR5 HDF5 episodes.")
    parser.add_argument("dataset_dir")
    parser.add_argument("--min-steps", type=int, default=DEFAULT_MIN_STEPS)
    parser.add_argument("--spike-thresh", type=float, default=DEFAULT_SPIKE_THRESH)
    parser.add_argument("--same-thresh", type=float, default=DEFAULT_SAME_THRESH)
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.dataset_dir, "*.hdf5")))
    if not files:
        sys.exit(f"No .hdf5 files found in: {args.dataset_dir}")

    print(f"Checking {len(files)} episode(s) in: {args.dataset_dir}")
    results = [
        (os.path.basename(p), check_episode(p, args.min_steps, args.spike_thresh, args.same_thresh)) for p in files
    ]

    n_bad = print_report(results)
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
