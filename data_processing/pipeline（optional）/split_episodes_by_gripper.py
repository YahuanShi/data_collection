#!/usr/bin/env python3
"""
Split episodes into sub-episodes based on gripper state transitions.

For each episode, find all gripper 0→1 transitions in the action sequence
(action[:, 6]).  Each transition except the last creates a breakpoint 15
steps after the transition, yielding 4 sub-episodes per episode.

Breakpoint rule:
  transition at step t  →  gripper first becomes 1 at step t+1
  breakpoint            =  t + 15  (end of current sub-episode, inclusive)

Each sub-episode is saved with a `prompt` attribute assigned by sub-episode index.
Episodes listed in --skip are excluded entirely.

Output naming:
  episode_N.hdf5  →  episode_N_sub_1.hdf5, episode_N_sub_2.hdf5, ...

Usage:
    # Dry-run (print plan, no files written):
    python3 split_episodes_by_gripper.py dataset/20260402_pnpa/

    # Write sub-episodes to output directory:
    python3 split_episodes_by_gripper.py dataset/20260402_pnpa/ dataset/20260402_pnpa_split/

    # Override prompts or skip list:
    python3 split_episodes_by_gripper.py dataset/20260402_pnpa/ out/ \\
        --skip episode_119 \\
        --prompts "task 1" "task 2" "task 3" "task 4"
"""

import argparse
import glob
import os
import sys

import h5py
import numpy as np

# Default prompts for the 20260402_pnpa dataset (indexed by sub-episode, 0-based)
DEFAULT_PROMPTS = [
    "place red cube into red bin",
    "pick white cube into yellow bin",
    "pick red bin into front",
    "assembly yellow bin to red bin",
]

# Episodes to skip by default
DEFAULT_SKIP = {"episode_119"}


# ─── I/O helpers ──────────────────────────────────────────────────────────────


def _image_keys(f: h5py.File) -> list[str]:
    imgs = f.get("observations/images", {})
    return list(imgs.keys())


def write_sub_episode(src: str, dst: str, start: int, end: int, prompt: str) -> None:
    """Copy frames [start, end] (inclusive) from src into a new HDF5 file at dst.
    Writes `prompt` as a file-level attribute. Excludes the source `prompt` and
    updates `n_steps` to reflect the actual sub-episode length."""
    with h5py.File(src, "r") as f:
        qpos = f["observations/qpos"][:]
        action = f["action"][:]
        qvel = f["observations/qvel"][:] if "observations/qvel" in f else None
        eef_pose = f["observations/eef_pose"][:] if "observations/eef_pose" in f else None
        attrs = {k: v for k, v in f.attrs.items() if k != "prompt"}
        img_data = {k: f[f"observations/images/{k}"][:] for k in _image_keys(f)}

    sl = slice(start, end + 1)
    n_steps = end - start + 1

    with h5py.File(dst, "w") as f:
        for k, v in attrs.items():
            f.attrs[k] = v
        f.attrs["n_steps"] = n_steps
        f.attrs["prompt"] = prompt
        f.create_dataset("observations/qpos", data=qpos[sl], compression="gzip")
        f.create_dataset("action", data=action[sl], compression="gzip")
        if qvel is not None:
            f.create_dataset("observations/qvel", data=qvel[sl], compression="gzip")
        if eef_pose is not None:
            f.create_dataset("observations/eef_pose", data=eef_pose[sl], compression="gzip")
        for k, v in img_data.items():
            f.create_dataset(f"observations/images/{k}", data=v[sl], compression="gzip")


# ─── core logic ───────────────────────────────────────────────────────────────


def find_breakpoints(action: np.ndarray, gripper_col: int = 6, offset: int = 15) -> list[int]:
    """
    Return breakpoint indices (inclusive end of each sub-episode except the last).

    For each 0→1 gripper transition except the last, the breakpoint is at
    transition_index + offset.  With 4 transitions this yields 3 breakpoints
    and therefore 4 sub-episodes.
    """
    gripper = action[:, gripper_col]
    transitions = list(np.where((gripper[:-1] == 0) & (gripper[1:] == 1))[0])

    if len(transitions) < 2:
        return []

    active = transitions[:-1]
    return [t + offset for t in active]


def plan_splits(path: str, gripper_col: int, offset: int) -> list[tuple[int, int]]:
    """Return list of (start, end) frame ranges for each sub-episode."""
    with h5py.File(path, "r") as f:
        action = f["action"][:]

    T = action.shape[0]
    breakpoints = find_breakpoints(action, gripper_col, offset)

    if not breakpoints:
        return []

    boundaries = [0] + [bp + 1 for bp in breakpoints] + [T]
    segments = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1] - 1
        if start > end:
            return []
        segments.append((start, end))

    return segments


# ─── collect input files ──────────────────────────────────────────────────────


def collect_files(input_path: str) -> list[str]:
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
        description="Split episodes into sub-episodes at gripper 0→1 transitions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input",
        help="Single .hdf5 file OR directory containing *.hdf5 files",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Output directory for sub-episode files (omit for dry-run)",
    )
    parser.add_argument(
        "--gripper_col",
        type=int,
        default=6,
        help="Column index of the gripper in the action array (default: 6)",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=15,
        help="Steps after each 0→1 transition where the breakpoint falls (default: 15)",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=DEFAULT_PROMPTS,
        metavar="PROMPT",
        help=(
            "Prompt string for each sub-episode, in order. "
            f"Defaults: {DEFAULT_PROMPTS}"
        ),
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        default=list(DEFAULT_SKIP),
        metavar="STEM",
        help=(
            "Episode stems (without .hdf5) to exclude entirely. "
            f"Defaults: {sorted(DEFAULT_SKIP)}"
        ),
    )
    args = parser.parse_args()

    skip_set = set(args.skip)
    all_files = collect_files(args.input)
    files = [p for p in all_files if os.path.splitext(os.path.basename(p))[0] not in skip_set]
    n_skipped_by_name = len(all_files) - len(files)

    dry_run = args.output is None

    if dry_run:
        print("Dry-run — provide an output directory to write files.\n")

    print(f"Input        : {args.input}  ({len(all_files)} found, {n_skipped_by_name} skipped by name)")
    print(f"Gripper col  : {args.gripper_col}")
    print(f"Offset       : {args.offset} steps after each 0→1 transition")
    print(f"Skip list    : {sorted(skip_set)}")
    for i, p in enumerate(args.prompts, start=1):
        print(f"Prompt sub_{i} : {p}")
    print()

    if not dry_run:
        os.makedirs(args.output, exist_ok=True)

    # ── plan ──────────────────────────────────────────────────────────────────
    plan = []  # (src_path, stem, T, segments)

    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        with h5py.File(path, "r") as f:
            T = f["action"].shape[0]
        segments = plan_splits(path, args.gripper_col, args.offset)
        plan.append((path, stem, T, segments))

    # ── summary table ─────────────────────────────────────────────────────────
    col_w = 28
    print(f"  {'Episode':<{col_w}}  {'T':>6}  {'Segs':>5}  Ranges")
    print("  " + "─" * 80)

    n_bad = 0
    for _path, stem, T, segments in plan:
        if not segments:
            print(f"  {stem:<{col_w}}  {T:>6}  {'SKIP':>5}  (insufficient gripper transitions)")
            n_bad += 1
            continue
        ranges = "  ".join(f"[{s},{e}]({e-s+1})" for s, e in segments)
        warn = f"  ⚠ expected {len(args.prompts)}" if len(segments) != len(args.prompts) else ""
        print(f"  {stem:<{col_w}}  {T:>6}  {len(segments):>5}  {ranges}{warn}")

    print("  " + "─" * 80)
    total_subs = sum(len(segs) for _, _, _, segs in plan if segs)
    print(f"  {len(plan) - n_bad} episodes → {total_subs} sub-episodes  ({n_bad} skipped)\n")

    if dry_run:
        return

    # ── write ─────────────────────────────────────────────────────────────────
    written = 0
    for path, stem, T, segments in plan:
        if not segments:
            print(f"  SKIP  {stem} (insufficient gripper transitions)")
            continue
        for i, (start, end) in enumerate(segments, start=1):
            prompt = args.prompts[i - 1] if i - 1 < len(args.prompts) else ""
            out_name = f"{stem}_sub_{i}.hdf5"
            dst = os.path.join(args.output, out_name)
            write_sub_episode(path, dst, start, end, prompt)
            print(f"  WROTE {dst}  [frames {start}-{end}, {end-start+1} steps]  prompt={prompt!r}")
            written += 1

    print(f"\nDone. {written} sub-episode files written to: {args.output}")


if __name__ == "__main__":
    main()
