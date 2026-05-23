#!/usr/bin/env python3
"""
3D End-Effector trajectory visualiser.

Left panel  — 3D trajectory (uniform colour), green start dot, red end dot.
Right panel — time-series strips for all 6 EEF channels: x, y, z, rx, ry, rz.

Reads:  observations/eef_pose  (T, 6)  [x, y, z, rx, ry, rz]  m / rad (rotvec)

Navigate episodes
─────────────────
  N / ↓   next episode
  P / ↑   previous episode

Other
─────
  S       save figure as PNG
  Q       Escape  quit

Usage:
    python data_processing/viz/viz_EEFin3D.py path/to/dataset_dir
    python data_processing/viz/viz_EEFin3D.py path/to/episode_0.hdf5
"""

import argparse
import glob
import os
import re
import sys

import h5py
import matplotlib as mpl

mpl.use("TkAgg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection

# ─── style constants ──────────────────────────────────────────────────────────
BG = "#0d0d0d"
AX_BG = "#111111"
PANE_EDGE = "#2a2a2a"
GRID_COL = "#1e1e1e"
TICK_COL = "#777777"

CHAN_COLORS = [
    "#F87171",  # x  — red
    "#4ADE80",  # y  — green
    "#60A5FA",  # z  — blue
    "#FACC15",  # rx — yellow
    "#E879F9",  # ry — pink
    "#22D3EE",  # rz — cyan
]
CHAN_LABELS = ["x (m)", "y (m)", "z (m)", "rx (rad)", "ry (rad)", "rz (rad)"]

TRAJ_COLOR = "#FB923C"       # orange — distinct from all 6 joint strip colors


# ─── helpers ──────────────────────────────────────────────────────────────────


def _natural_key(p: str):
    """Sort key: split filename into text/number chunks so episode_2 < episode_10."""
    name = os.path.basename(p)
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", name)]


def collect_episodes(path: str) -> tuple[list[str], int]:
    if os.path.isfile(path):
        parent = os.path.dirname(os.path.abspath(path))
        files = sorted(glob.glob(os.path.join(parent, "*.hdf5")), key=_natural_key)
        if not files:
            files = [os.path.abspath(path)]
        start = files.index(os.path.abspath(path)) if os.path.abspath(path) in files else 0
        return files, start
    files = sorted(glob.glob(os.path.join(path, "**/*.hdf5"), recursive=True), key=_natural_key)
    if not files:
        sys.exit(f"No .hdf5 files found in: {path}")
    return files, 0


def load_eef(path: str) -> np.ndarray:
    with h5py.File(path, "r") as f:
        if "observations/eef_pose" not in f:
            sys.exit(f"eef_pose not found in {path}")
        return f["observations/eef_pose"][:]


# ─── figure ───────────────────────────────────────────────────────────────────


def build_figure():
    fig = plt.figure(figsize=(15, 9))
    fig.patch.set_facecolor(BG)

    gs = gridspec.GridSpec(
        6, 2,
        width_ratios=[2, 1],
        hspace=0.05,
        wspace=0.30,
        left=0.05, right=0.97, top=0.93, bottom=0.06,
    )

    # 3D axis spans all 6 rows on the left
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    _style_3d(ax3d)

    ts_axes = []
    for i in range(6):
        ax = fig.add_subplot(gs[i, 1])
        ax.set_facecolor(AX_BG)
        for sp in ax.spines.values():
            sp.set_edgecolor(PANE_EDGE)
        ax.tick_params(colors=TICK_COL, labelsize=7)
        ax.grid(axis="y", color=GRID_COL, linewidth=0.6)
        ax.set_ylabel(
            CHAN_LABELS[i], color=CHAN_COLORS[i],
            fontsize=8, rotation=0, labelpad=55, va="center",
        )
        if i < 5:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("frame", color=TICK_COL, fontsize=8)
        ts_axes.append(ax)

    return fig, ax3d, ts_axes


def _style_3d(ax):
    ax.set_facecolor(AX_BG)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor(PANE_EDGE)
    ax.tick_params(colors=TICK_COL, labelsize=7)
    ax.set_xlabel("x (m)", color=CHAN_COLORS[0], fontsize=8)
    ax.set_ylabel("y (m)", color=CHAN_COLORS[1], fontsize=8)
    ax.set_zlabel("z (m)", color=CHAN_COLORS[2], fontsize=8)
    ax.xaxis.label.set_color(CHAN_COLORS[0])
    ax.yaxis.label.set_color(CHAN_COLORS[1])
    ax.zaxis.label.set_color(CHAN_COLORS[2])


# ─── draw ─────────────────────────────────────────────────────────────────────


def redraw(fig, ax3d, ts_axes, episodes, state):
    eef = load_eef(episodes[state["ep_idx"]])
    T = len(eef)

    xyz = eef[:, :3]
    t_idx = np.arange(T)

    # ── 3D panel ──────────────────────────────────────────────────────────────
    ax3d.cla()
    _style_3d(ax3d)

    # trajectory line (uniform orange)
    ax3d.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2],
              color=TRAJ_COLOR, linewidth=1.8, alpha=0.9, zorder=2)

    # start (green) / end (red) markers
    ax3d.scatter(*xyz[0],  color="#22C55E", s=35, zorder=4, label="start")
    ax3d.scatter(*xyz[-1], color="#EF4444", s=35, zorder=4, label="end")

    ax3d.legend(
        loc="upper left", fontsize=7,
        facecolor="#1a1a1a", edgecolor="#3a3a3a",
        labelcolor="#cccccc", framealpha=0.85,
    )

    # equal aspect ratio for 3D
    x_r = xyz[:, 0].max() - xyz[:, 0].min()
    y_r = xyz[:, 1].max() - xyz[:, 1].min()
    z_r = xyz[:, 2].max() - xyz[:, 2].min()
    half = max(x_r, y_r, z_r, 0.01) / 2
    mid = xyz.mean(axis=0)
    ax3d.set_xlim(mid[0] - half, mid[0] + half)
    ax3d.set_ylim(mid[1] - half, mid[1] + half)
    ax3d.set_zlim(mid[2] - half, mid[2] + half)

    # ── time-series strips ────────────────────────────────────────────────────
    for i, ax in enumerate(ts_axes):
        ax.cla()
        ax.set_facecolor(AX_BG)
        for sp in ax.spines.values():
            sp.set_edgecolor(PANE_EDGE)
        ax.tick_params(colors=TICK_COL, labelsize=7)
        ax.grid(axis="y", color=GRID_COL, linewidth=0.6)
        ax.set_ylabel(
            CHAN_LABELS[i], color=CHAN_COLORS[i],
            fontsize=8, rotation=0, labelpad=55, va="center",
        )
        if i < 5:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("frame", color=TICK_COL, fontsize=8)

        vals = eef[:, i]
        ax.plot(t_idx, vals, color=CHAN_COLORS[i], linewidth=1.3, alpha=0.9)

        vmin, vmax = float(vals.min()), float(vals.max())
        margin = max((vmax - vmin) * 0.08, 1e-4)
        ax.set_xlim(0, T - 1)
        ax.set_ylim(vmin - margin, vmax + margin)

    fig.suptitle(
        f"{os.path.basename(episodes[state['ep_idx']])}   "
        f"[{state['ep_idx'] + 1}/{len(episodes)}]",
        color="#dddddd", fontsize=9, fontweight="bold",
    )
    fig.canvas.draw_idle()


# ─── main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="3D EEF trajectory visualiser.")
    parser.add_argument("path", help="HDF5 file or directory")
    args = parser.parse_args()

    episodes, ep_idx = collect_episodes(args.path)
    print(f"Episodes : {len(episodes)}")
    print("  N/↓  next episode   P/↑  prev episode   S  save   Q  quit")

    state = {"ep_idx": ep_idx}

    fig, ax3d, ts_axes = build_figure()

    def on_key(event):
        k = event.key
        if k in ("n", "down"):
            state["ep_idx"] = (state["ep_idx"] + 1) % len(episodes)
        elif k in ("p", "up"):
            state["ep_idx"] = (state["ep_idx"] - 1) % len(episodes)
        elif k == "s":
            out = os.path.splitext(os.path.basename(episodes[state["ep_idx"]]))[0] + "_eef3d.png"
            fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            print(f"Saved → {out}")
            return
        elif k in ("q", "escape"):
            plt.close(fig)
            return
        else:
            return
        redraw(fig, ax3d, ts_axes, episodes, state)

    redraw(fig, ax3d, ts_axes, episodes, state)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


if __name__ == "__main__":
    main()
