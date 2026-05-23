#!/usr/bin/env python3
"""
Auto-trim episode frames based on joint movement.

Logic:
  HEAD (first --head_window steps):
    In the diff sequence, find the FIRST index where ANY joint changes
    >= --threshold degrees.  Keep --pad frames before it and everything
    after.  Delete all earlier frames.

  TAIL (last --tail_window steps):
    Search BACKWARD from the end to find the last frame with movement,
    then scan forward to find where ALL joints settle below --threshold.
    Keep --tail_pad frames after that settling point.  Delete the rest.
    (Backward search avoids cutting at mid-episode pauses.)

Note: joint 6 (gripper, range 0-1) is excluded by default.

Usage:
    # ── Single episode ────────────────────────────────────────────────
    python3 trim_episodes_auto.py data/episode_0.hdf5
    python3 trim_episodes_auto.py data/episode_0.hdf5 output/

    # ── Batch (whole directory) ───────────────────────────────────────
    python3 trim_episodes_auto.py raw/
    python3 trim_episodes_auto.py raw/ trimmed/

    # ── Tune parameters ──────────────────────────────────────────────
    python3 trim_episodes_auto.py raw/ trimmed/ --threshold 0.1 --pad 3
    python3 trim_episodes_auto.py raw/ trimmed/ --head_window 50 --tail_window 50
"""

import argparse
import glob
import os
import shutil
import sys

import h5py
import numpy as np

# ─── I/O helpers ──────────────────────────────────────────────────────────────


def _image_keys(f: h5py.File) -> list[str]:
    imgs = f.get("observations/images", {})
    return list(imgs.keys())


def trim_episode(src: str, dst: str, start: int, end: int) -> int:
    """Copy src → dst keeping frames [start:end+1]. Returns kept frame count."""
    with h5py.File(src, "r") as f:
        qpos = f["observations/qpos"][:]
        action = f["action"][:]
        qvel = f["observations/qvel"][:] if "observations/qvel" in f else None
        eef_pose = f["observations/eef_pose"][:] if "observations/eef_pose" in f else None
        attrs = dict(f.attrs)
        img_data = {}
        for k in _image_keys(f):
            img_data[k] = f[f"observations/images/{k}"][:]

    sl = slice(start, end + 1)

    with h5py.File(dst, "w") as f:
        for k, v in attrs.items():
            f.attrs[k] = v
        f.create_dataset("observations/qpos", data=qpos[sl], compression="gzip")
        f.create_dataset("action", data=action[sl], compression="gzip")
        if qvel is not None:
            f.create_dataset("observations/qvel", data=qvel[sl], compression="gzip")
        if eef_pose is not None:
            f.create_dataset("observations/eef_pose", data=eef_pose[sl], compression="gzip")
        for k, v in img_data.items():
            f.create_dataset(f"observations/images/{k}", data=v[sl], compression="gzip")

    return end - start + 1


# ─── auto-detect movement range ──────────────────────────────────────────────


def detect_movement_range(
    qpos: np.ndarray,
    joints: list[int],
    threshold_deg: float,
    pad: int,
    head_window: int,
    tail_window: int,
    tail_pad: int,
) -> tuple[int, int]:
    """
    Detect the active (moving) region of the episode.

    Parameters
    ----------
    qpos          : (T, num_joints) array of joint positions in degrees.
    joints        : which joint indices to monitor (default 0-5).
    threshold_deg : joint diff >= this (degrees) counts as "moving".
    pad           : frames to keep before first movement (head).
    head_window   : how many steps from the start to scan for first movement.
    tail_window   : how many steps from the end to scan for all-still.
    tail_pad      : frames to keep after first all-still point (tail).

    Returns
    -------
    (start, end) : inclusive frame indices of the region to keep.
    """
    T = qpos.shape[0]

    # frame-to-frame absolute difference for the selected joints (degrees)
    diff = np.abs(np.diff(qpos[:, joints], axis=0))  # shape: (T-1, len(joints))

    # ── HEAD ─────────────────────────────────────────────────────────────────
    # In the first head_window diffs, find the FIRST index where ANY joint
    # changes >= threshold.  Keep `pad` frames before it.
    start = 0
    head_search = min(head_window, len(diff))
    for i in range(head_search):
        if np.any(diff[i] >= threshold_deg):
            start = max(0, i - pad)
            break

    # ── TAIL ─────────────────────────────────────────────────────────────────
    # In the last tail_window diffs, search BACKWARD from the end to find
    # the last diff with movement, then the first all-still diff after it
    # is the settling point.  Keep `tail_pad` frames after that point.
    #
    # Why backward?  A forward scan would hit mid-episode pauses (e.g.
    # gripper-close wait) before the actual final settling.
    end = T - 1
    tail_start = max(0, len(diff) - tail_window)

    # Find the last diff in the tail region where ANY joint moves
    last_move_in_tail = None
    for j in range(len(diff) - 1, tail_start - 1, -1):
        if np.any(diff[j] >= threshold_deg):
            last_move_in_tail = j
            break

    if last_move_in_tail is not None:
        # Scan forward from last movement to find first all-still diff
        for j in range(last_move_in_tail + 1, len(diff)):
            if np.all(diff[j] < threshold_deg):
                end = min(T - 1, j + tail_pad)
                break

    return start, end


# ─── resolve input to file list ──────────────────────────────────────────────


def collect_files(input_path: str) -> list[str]:
    """Accept a single .hdf5 file or a directory; return sorted file list."""
    if os.path.isfile(input_path):
        if not input_path.endswith(".hdf5"):
            sys.exit(f"Not an HDF5 file: {input_path}")
        return [input_path]
    if os.path.isdir(input_path):
        files = sorted(glob.glob(os.path.join(input_path, "*.hdf5")))
        if not files:
            sys.exit(f"No .hdf5 files found in: {input_path}")
        return files
    sys.exit(f"Path does not exist: {input_path}")


# ─── main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Auto-trim episode frames by detecting joint movement.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", help="Single .hdf5 file OR directory containing *.hdf5 files")
    parser.add_argument("output", nargs="?", default=None, help="Output directory (omit for dry-run)")
    parser.add_argument(
        "--joints",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4, 5],
        help="Joint indices to monitor (default: 0 1 2 3 4 5, excluding gripper)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.1, help="Joint angle change threshold in degrees (default: 0.1)"
    )
    parser.add_argument("--pad", type=int, default=3, help="Frames to keep before first movement (default: 3)")
    parser.add_argument(
        "--head_window",
        type=int,
        default=50,
        help="How many steps from the start to scan for first movement (default: 50)",
    )
    parser.add_argument(
        "--tail_window", type=int, default=50, help="How many steps from the end to scan for all-still (default: 50)"
    )
    parser.add_argument(
        "--tail_pad", type=int, default=5, help="Frames to keep after first all-still point in tail (default: 5)"
    )
    args = parser.parse_args()

    files = collect_files(args.input)
    dry_run = args.output is None

    if dry_run:
        print("Dry-run — provide output directory to write files.\n")

    print(f"Input             : {args.input}  ({len(files)} episode{'s' if len(files) != 1 else ''})")
    print(f"Joints monitored  : {args.joints}  (qpos in degrees)")
    print(f"Threshold         : {args.threshold}°")
    print(f"Head              : first {args.head_window} steps, keep {args.pad} frames before first movement")
    print(f"Tail              : last {args.tail_window} steps, keep {args.tail_pad} frames after first all-still\n")

    if not dry_run:
        os.makedirs(args.output, exist_ok=True)

    # ── per-episode detection ─────────────────────────────────────────────────
    plan = []  # (src_path, name, T, start, end)

    for path in files:
        name = os.path.basename(path)
        with h5py.File(path, "r") as f:
            qpos = f["observations/qpos"][:]

        T = qpos.shape[0]
        s, e = detect_movement_range(
            qpos, args.joints, args.threshold, args.pad, args.head_window, args.tail_window, args.tail_pad
        )
        plan.append((path, name, T, s, e))

    # ── summary table ─────────────────────────────────────────────────────────
    print(f"  {'Episode':<28}  {'Total':>6}  {'Start':>6}  {'End':>6}  {'Kept':>6}  {'Trimmed':>8}")
    print("  " + "─" * 72)
    for _, name, T, s, e in plan:
        kept = e - s + 1
        trimmed = T - kept
        flag = "  ⚠ no move" if (s == 0 and e == T - 1 and T > 10) else ""
        print(f"  {name:<28}  {T:>6}  {s:>6}  {e:>6}  {kept:>6}  {trimmed:>8}{flag}")
    print("  " + "─" * 72)

    total_orig = sum(T for _, _, T, _, _ in plan)
    total_kept = sum(e - s + 1 for _, _, _, s, e in plan)
    total_trimmed = total_orig - total_kept
    print(
        f"  Total: {total_orig} frames → {total_kept} kept, {total_trimmed} trimmed "
        f"({100 * total_trimmed / max(total_orig, 1):.1f}% removed)\n"
    )

    if dry_run:
        return

    # ── write ─────────────────────────────────────────────────────────────────
    for path, name, T, s, e in plan:
        dst = os.path.join(args.output, name)
        if s == 0 and e == T - 1:
            shutil.copy2(path, dst)
            print(f"  COPY  {dst}  (unchanged)")
        else:
            trim_episode(path, dst, s, e)
            print(f"  WROTE {dst}  [frames {s}-{e}]")

    print(f"\nDone. {len(plan)} episodes → {args.output}")


if __name__ == "__main__":
    main()
