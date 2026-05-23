#!/usr/bin/env python3
"""
Before/after trajectory comparison viewer.

Each subplot overlays:
  • faint line  : original (before processing)  — only when --original is given
  • solid line  : processed (after)

Hover tooltip shows step, joint value, original value, and diff.
Vertical cursor synced across all subplots.

Navigate episodes
─────────────────
  N / ↓ / →   next episode
  P / ↑ / ←   previous episode

Other
─────
  S           save current figure as PNG
  Q / Escape  quit

Usage:
    python data_processing/viz/viz_trajectory.py processed_dir
    python data_processing/viz/viz_trajectory.py processed_dir --original original_dir
    python data_processing/viz/viz_trajectory.py processed_file.hdf5
    python data_processing/viz/viz_trajectory.py processed_file.hdf5 --original original_dir
"""

import argparse
import glob
import os
import sys

import h5py
import matplotlib as mpl
import numpy as np

mpl.use("TkAgg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

# ─── constants ────────────────────────────────────────────────────────────────
JOINT_COLORS = [
    "#F87171",  # J0 - red    (BGR 113,113,248)
    "#4ADE80",  # J1 - green  (BGR 128,222, 74)
    "#60A5FA",  # J2 - blue   (BGR 250,165, 96)
    "#FACC15",  # J3 - yellow (BGR  21,204,250)
    "#E879F9",  # J4 - pink   (BGR 249,121,232)
    "#22D3EE",  # J5 - cyan   (BGR 238,211, 34)
    "#94A3B8",  # J6 - slate  (BGR 184,163,148)
]
JOINT_LABELS = ["J0", "J1", "J2", "J3", "J4", "J5", "Gripper"]


# ─── helpers ──────────────────────────────────────────────────────────────────


def collect_episodes(path: str) -> tuple[list[str], int]:
    """Return (sorted hdf5 list, start index).  Accepts file or directory."""
    if os.path.isfile(path):
        parent = os.path.dirname(os.path.abspath(path))
        files = sorted(glob.glob(os.path.join(parent, "*.hdf5")))
        start = next((i for i, f in enumerate(files) if os.path.abspath(path) == f), 0)
        return files or [os.path.abspath(path)], start
    files = sorted(glob.glob(os.path.join(path, "*.hdf5")))
    if not files:
        sys.exit(f"No .hdf5 files found in: {path}")
    return files, 0


def load_qpos(path: str) -> np.ndarray:
    with h5py.File(path, "r") as f:
        return f["observations/qpos"][:]


def find_original(proc_path: str, orig_dir: str) -> str | None:
    """Try to locate the matching original file by episode name."""
    if orig_dir is None:
        return None
    name = os.path.basename(proc_path)
    candidate = os.path.join(orig_dir, name)
    if os.path.exists(candidate):
        return candidate
    # fallback: search recursively
    matches = glob.glob(os.path.join(orig_dir, "**", name), recursive=True)
    return matches[0] if matches else None


# ─── figure builder ───────────────────────────────────────────────────────────


def build_figure(n_joints: int, has_original: bool):
    fig = plt.figure(figsize=(13, 14))
    fig.patch.set_facecolor("#0d0d0d")

    gs = gridspec.GridSpec(n_joints, 1, hspace=0.06, left=0.08, right=0.97, top=0.93, bottom=0.04)
    axes = []
    lines_orig = []
    lines_proc = []

    for j in range(n_joints):
        ax = fig.add_subplot(gs[j])
        ax.set_facecolor("#111111")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2a2a")
        ax.tick_params(colors="#777777", labelsize=7)
        ax.grid(axis="y", color="#1e1e1e", linewidth=0.6)

        col = JOINT_COLORS[j]
        (lo,) = ax.plot([], [], color=col, alpha=0.55, linewidth=1.2, linestyle="--", label="original")
        (lp,) = ax.plot([], [], color=col, alpha=0.95, linewidth=1.5, label="processed")

        ax.set_ylabel(JOINT_LABELS[j], color=col, fontsize=9, rotation=0, labelpad=30, va="center")
        if j < n_joints - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("frame", color="#777777", fontsize=8)
        if j == 0:
            handles = [lp, lo] if has_original else [lp]
            ax.legend(
                handles=handles,
                loc="upper right",
                fontsize=7,
                facecolor="#1a1a1a",
                edgecolor="#3a3a3a",
                labelcolor="#cccccc",
                framealpha=0.85,
            )

        axes.append(ax)
        lines_orig.append(lo)
        lines_proc.append(lp)

    return fig, axes, lines_orig, lines_proc


# ─── draw ─────────────────────────────────────────────────────────────────────


def redraw(fig, axes, lines_orig, lines_proc, proc_episodes, orig_dir, state):
    ep_idx = state["ep_idx"]
    proc_path = proc_episodes[ep_idx]
    orig_path = find_original(proc_path, orig_dir)

    proc = load_qpos(proc_path)
    orig = load_qpos(orig_path) if orig_path else None

    T_proc = len(proc)
    T_orig = len(orig) if orig is not None else T_proc

    state["proc"] = proc
    state["orig"] = orig
    state["T_proc"] = T_proc
    state["T_orig"] = T_orig

    for j, ax in enumerate(axes):
        p_vals = proc[:, j]
        t_proc = np.arange(T_proc)

        lines_proc[j].set_data(t_proc, p_vals)

        if orig is not None:
            o_vals = orig[:, j]
            t_orig = np.arange(T_orig)
            lines_orig[j].set_data(t_orig, o_vals)
            lines_orig[j].set_visible(True)
            all_vals = np.concatenate([p_vals, o_vals])
        else:
            lines_orig[j].set_data([], [])
            lines_orig[j].set_visible(False)
            all_vals = p_vals

        ax.set_xlim(0, max(T_proc, T_orig) - 1)
        vmin, vmax = all_vals.min(), all_vals.max()
        margin = max((vmax - vmin) * 0.08, 0.05)
        ax.set_ylim(vmin - margin, vmax + margin)

    ep_name = os.path.basename(proc_path)
    orig_tag = f"  orig: {os.path.basename(orig_path)}" if orig_path else "  (no original)"
    nav_tag = f"  [{ep_idx + 1}/{len(proc_episodes)}]  ←/→ episode"
    fig.suptitle(f"{ep_name}{orig_tag}{nav_tag}", color="#dddddd", fontsize=9, fontweight="bold")
    fig.canvas.draw_idle()


# ─── main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Before/after trajectory comparison viewer.")
    parser.add_argument("processed", help="Processed HDF5 file or directory")
    parser.add_argument(
        "original_pos", nargs="?", default=None, help="Original (before) directory for comparison (optional positional)"
    )
    parser.add_argument(
        "--original",
        "-o",
        default=None,
        dest="original_flag",
        help="Original (before) directory for comparison (optional)",
    )
    args = parser.parse_args()
    args.original = args.original_pos or args.original_flag

    proc_episodes, ep_idx = collect_episodes(args.processed)
    has_original = args.original is not None

    print(f"Processed episodes : {len(proc_episodes)}")
    if has_original:
        print(f"Original dir       : {args.original}")
    print("  ←/→ D/A  prev/next episode   S  save PNG   Q  quit")

    state = {"ep_idx": ep_idx, "proc": None, "orig": None, "T_proc": 0, "T_orig": 0}

    fig, axes, lines_orig, lines_proc = build_figure(7, has_original)

    # ── hover tooltip ─────────────────────────────────────────────────────────
    tooltip = fig.text(
        0.01,
        0.97,
        "",
        va="top",
        ha="left",
        fontsize=7.5,
        color="#eeeeee",
        family="monospace",
        bbox={"boxstyle": "round,pad=0.5", "fc": "#1a1a1a", "ec": "#555555", "alpha": 0.92},
        transform=fig.transFigure,
        visible=False,
        zorder=10,
    )
    vlines = [ax.axvline(x=0, color="#ffffff", linewidth=0.7, alpha=0.4, visible=False) for ax in axes]

    def on_motion(event):
        if event.inaxes not in axes:
            tooltip.set_visible(False)
            for vl in vlines:
                vl.set_visible(False)
            fig.canvas.draw_idle()
            return
        xdata = event.xdata
        if xdata is None:
            return
        proc = state["proc"]
        orig = state["orig"]
        if proc is None:
            return
        T_proc = state["T_proc"]
        step_p = int(np.clip(round(xdata), 0, T_proc - 1))
        j = axes.index(event.inaxes)

        info = [
            f"step      : {step_p}/{T_proc - 1}",
            f"joint     : {JOINT_LABELS[j]}",
            f"processed : {proc[step_p, j]:.4f}",
        ]
        if orig is not None:
            T_orig = state["T_orig"]
            step_o = int(np.clip(round(xdata), 0, T_orig - 1))
            info.append(f"original  : {orig[step_o, j]:.4f}")
            info.append(f"diff      : {proc[step_p, j] - orig[step_o, j]:+.4f}")

        tooltip.set_text("\n".join(info))
        tooltip.set_visible(True)
        for vl in vlines:
            vl.set_xdata([xdata, xdata])
            vl.set_visible(True)
        fig.canvas.draw_idle()

    # ── keyboard handler ──────────────────────────────────────────────────────
    def on_key(event):
        key = event.key
        if key in ("right", "down", "n"):
            state["ep_idx"] = (state["ep_idx"] + 1) % len(proc_episodes)
        elif key in ("left", "up", "p"):
            state["ep_idx"] = (state["ep_idx"] - 1) % len(proc_episodes)
        elif key == "s":
            ep_name = os.path.basename(proc_episodes[state["ep_idx"]])
            out = os.path.splitext(ep_name)[0] + "_traj.png"
            fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            print(f"Figure saved → {out}")
            return
        elif key in ("q", "escape"):
            plt.close(fig)
            return
        else:
            return
        redraw(fig, axes, lines_orig, lines_proc, proc_episodes, args.original, state)

    redraw(fig, axes, lines_orig, lines_proc, proc_episodes, args.original, state)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


if __name__ == "__main__":
    main()
