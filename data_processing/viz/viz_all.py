#!/usr/bin/env python3
"""
Integrated UR5 episode viewer — fullscreen, all panels in one window.

Layout
------
  Top    : camera views (exterior / wrist [/ front])  |  EEF 3D trajectory
  Middle : progress bar
  Bottom : per-joint position strips (J0–Gripper)     |  EEF channel strips (x,y,z,rx,ry,rz)

The EEF 3D trail grows frame-by-frame in sync with playback.

Keyboard
--------
  SPACE       pause / resume
  ← / →       step −5 / +5 frames
  ↑ / ↓       previous / next episode
  f           toggle 1× / 2× speed
  r           restart current episode
  s           save screenshot PNG
  q / ESC     quit

Usage
-----
  python viz/viz_all.py  <hdf5_file_or_directory>
  python viz/viz_all.py  <path>  --fps 20
"""

import argparse
import glob
import os
import re
import sys

import h5py
import numpy as np
import matplotlib as mpl

mpl.use("TkAgg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3-D projection

# ─── palette ──────────────────────────────────────────────────────────────────
BG        = "#0d0d0d"
AX_BG     = "#111111"
PANE_EDGE = "#2a2a2a"
GRID_COL  = "#1e1e1e"
TICK_COL  = "#777777"
PROG_FG   = "#3B82F6"
CURSOR_COL = "#ffffff"

JOINT_COLORS = ["#F87171", "#4ADE80", "#60A5FA", "#FACC15", "#E879F9", "#22D3EE", "#94A3B8"]
JOINT_LABELS = ["J0", "J1", "J2", "J3", "J4", "J5", "Gripper"]
EEF_COLORS   = ["#F87171", "#4ADE80", "#60A5FA", "#FACC15", "#E879F9", "#22D3EE"]
EEF_LABELS   = ["x", "y", "z", "rx", "ry", "rz"]
TRAJ_COLOR   = "#FB923C"


# ─── style helpers ────────────────────────────────────────────────────────────

def _style_3d(ax):
    ax.set_facecolor(AX_BG)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor(PANE_EDGE)
    ax.tick_params(colors=TICK_COL, labelsize=7)
    ax.set_xlabel("x", color=EEF_COLORS[0], fontsize=8)
    ax.set_ylabel("y", color=EEF_COLORS[1], fontsize=8)
    ax.set_zlabel("z", color=EEF_COLORS[2], fontsize=8)


def _strip_style(ax, hide_xlabels: bool):
    ax.set_facecolor(AX_BG)
    for sp in ax.spines.values():
        sp.set_edgecolor(PANE_EDGE)
    ax.tick_params(colors=TICK_COL, labelsize=6)
    ax.grid(axis="y", color=GRID_COL, linewidth=0.5)
    if hide_xlabels:
        ax.set_xticklabels([])


def _set_ylim(ax, vals: np.ndarray):
    lo, hi = float(vals.min()), float(vals.max())
    pad = max((hi - lo) * 0.10, 1e-4)
    ax.set_ylim(lo - pad, hi + pad)


# ─── data helpers ─────────────────────────────────────────────────────────────

def _natural_key(p: str):
    name = os.path.basename(p)
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", name)]


def collect_episodes(path: str) -> tuple[list[str], int]:
    if os.path.isfile(path):
        parent = os.path.dirname(os.path.abspath(path))
        files  = sorted(glob.glob(os.path.join(parent, "*.hdf5")), key=_natural_key)
        if not files:
            files = [os.path.abspath(path)]
        start = files.index(os.path.abspath(path)) if os.path.abspath(path) in files else 0
        return files, start
    files = sorted(glob.glob(os.path.join(path, "**/*.hdf5"), recursive=True), key=_natural_key)
    if not files:
        sys.exit(f"No .hdf5 files found in: {path}")
    return files, 0


def load_episode(path: str) -> dict:
    with h5py.File(path, "r") as f:
        ep: dict = {
            "qpos":     f["observations/qpos"][:],
            "exterior": f["observations/images/exterior_image_1_left"][:],
            "wrist":    f["observations/images/wrist_image_left"][:],
        }
        imgs = f["observations/images"]
        if "front_image_1" in imgs:
            ep["front"] = imgs["front_image_1"][:]
        if "observations/eef_pose" in f:
            ep["eef"] = f["observations/eef_pose"][:]
        return ep


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Integrated UR5 episode viewer")
    ap.add_argument("path", help="HDF5 file or directory")
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args()

    episodes, start_idx = collect_episodes(args.path)
    print(f"Found {len(episodes)} episode(s).")

    S: dict = {
        "ep_idx": start_idx,
        "t":      0,
        "T":      0,
        "paused": False,
        "fast":   False,
        "data":   None,
    }

    def _load(idx: int):
        S["ep_idx"] = idx
        print(f"  [{idx + 1}/{len(episodes)}] loading {os.path.basename(episodes[idx])} …")
        S["data"] = load_episode(episodes[idx])
        S["T"]    = len(S["data"]["exterior"])
        S["t"]    = 0

    _load(start_idx)

    has_front = "front" in S["data"]
    has_eef   = "eef"   in S["data"]
    n_joints  = S["data"]["qpos"].shape[1]
    num_cams  = 3 if has_front else 2
    cam_keys  = ["exterior", "wrist"] + (["front"] if has_front else [])
    cam_names = ["EXTERIOR", "WRIST"] + (["FRONT"] if has_front else [])

    # ── figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(19.2, 10.8), dpi=96)
    fig.patch.set_facecolor(BG)

    try:
        plt.get_current_fig_manager().full_screen_toggle()
    except Exception:
        pass

    # ── outer grid: top | progress | bottom ───────────────────────────────────
    outer = gridspec.GridSpec(
        3, 1,
        height_ratios=[50, 3, 47],
        hspace=0.05,
        left=0.03, right=0.99, top=0.95, bottom=0.02,
    )

    # ── top row ───────────────────────────────────────────────────────────────
    top_gs = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[0],
        width_ratios=[2, 1], wspace=0.03,
    )
    cam_gs = gridspec.GridSpecFromSubplotSpec(
        1, num_cams, subplot_spec=top_gs[0], wspace=0.02,
    )

    cam_axes, cam_imgs = [], []
    for i in range(num_cams):
        ax = fig.add_subplot(cam_gs[i])
        ax.set_facecolor(BG)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        im = ax.imshow(S["data"][cam_keys[i]][0], aspect="equal")
        ax.set_title(cam_names[i], color=TICK_COL, fontsize=9, pad=3)
        cam_axes.append(ax)
        cam_imgs.append(im)

    ax3d = fig.add_subplot(top_gs[1], projection="3d")
    _style_3d(ax3d)

    # ── progress bar ──────────────────────────────────────────────────────────
    ax_prog = fig.add_subplot(outer[1])
    ax_prog.set_facecolor("#161616")
    ax_prog.set_xticks([]); ax_prog.set_yticks([])
    for sp in ax_prog.spines.values():
        sp.set_edgecolor(PANE_EDGE)
    ax_prog.set_xlim(0, 1)
    ax_prog.set_ylim(0, 1)
    prog_rect = mpatches.Rectangle((0, 0.08), 0.0, 0.84, color=PROG_FG, alpha=0.75)
    ax_prog.add_patch(prog_rect)
    prog_head = ax_prog.axvline(x=0, color="white", linewidth=2.0, alpha=0.9)

    # ── bottom row ────────────────────────────────────────────────────────────
    if has_eef:
        bot_gs = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer[2],
            width_ratios=[n_joints, 6], wspace=0.06,
        )
        joint_subplot = bot_gs[0]
        eef_subplot   = bot_gs[1]
    else:
        joint_subplot = outer[2]
        eef_subplot   = None

    j_gs = gridspec.GridSpecFromSubplotSpec(n_joints, 1, subplot_spec=joint_subplot, hspace=0.04)
    joint_axes, joint_lines, joint_vlines = [], [], []
    for j in range(n_joints):
        ax = fig.add_subplot(j_gs[j])
        _strip_style(ax, j < n_joints - 1)
        ax.set_ylabel(
            JOINT_LABELS[j], color=JOINT_COLORS[j],
            fontsize=7, rotation=0, labelpad=28, va="center",
        )
        vals = S["data"]["qpos"][:, j]
        (ln,) = ax.plot(np.arange(S["T"]), vals, color=JOINT_COLORS[j], lw=1.1, alpha=0.9)
        vl = ax.axvline(x=0, color=CURSOR_COL, lw=0.8, alpha=0.55)
        _set_ylim(ax, vals)
        ax.set_xlim(0, S["T"] - 1)
        if j == n_joints - 1:
            ax.set_xlabel("frame", color=TICK_COL, fontsize=7)
        joint_axes.append(ax); joint_lines.append(ln); joint_vlines.append(vl)

    eef_axes, eef_lines, eef_vlines = [], [], []
    if has_eef:
        e_gs = gridspec.GridSpecFromSubplotSpec(6, 1, subplot_spec=eef_subplot, hspace=0.04)
        for i in range(6):
            ax = fig.add_subplot(e_gs[i])
            _strip_style(ax, i < 5)
            ax.set_ylabel(
                EEF_LABELS[i], color=EEF_COLORS[i],
                fontsize=7, rotation=0, labelpad=25, va="center",
            )
            vals = S["data"]["eef"][:, i]
            (ln,) = ax.plot(np.arange(S["T"]), vals, color=EEF_COLORS[i], lw=1.1, alpha=0.9)
            vl = ax.axvline(x=0, color=CURSOR_COL, lw=0.8, alpha=0.55)
            _set_ylim(ax, vals)
            ax.set_xlim(0, S["T"] - 1)
            if i == 5:
                ax.set_xlabel("frame", color=TICK_COL, fontsize=7)
            eef_axes.append(ax); eef_lines.append(ln); eef_vlines.append(vl)

    # ── 3-D artists — rebuilt on episode change ───────────────────────────────
    _3d: dict = {"trail": None, "head": None}

    def _setup_3d():
        ax3d.cla()
        _style_3d(ax3d)
        if not has_eef or "eef" not in S["data"]:
            ax3d.set_title("no eef_pose", color=TICK_COL, fontsize=9)
            _3d["trail"] = _3d["head"] = None
            return
        xyz = S["data"]["eef"][:, :3]

        # faint full-trajectory backdrop
        ax3d.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                  color=TRAJ_COLOR, alpha=0.18, lw=1.0, zorder=1)
        ax3d.scatter(*xyz[0],  color="#22C55E", s=40, zorder=4, label="start")
        ax3d.scatter(*xyz[-1], color="#EF4444", s=40, zorder=4, label="end")

        # animated growing trail + moving head dot
        (trail,) = ax3d.plot([], [], [], color=TRAJ_COLOR, lw=2.2, alpha=0.95, zorder=3)
        (head,)  = ax3d.plot([], [], [], "o", color="white", ms=6, zorder=5)
        _3d["trail"] = trail
        _3d["head"]  = head

        # equal-aspect bounding box
        ranges = [xyz[:, i].max() - xyz[:, i].min() for i in range(3)]
        half   = max(*ranges, 0.01) / 2
        mid    = xyz.mean(axis=0)
        ax3d.set_xlim(mid[0] - half, mid[0] + half)
        ax3d.set_ylim(mid[1] - half, mid[1] + half)
        ax3d.set_zlim(mid[2] - half, mid[2] + half)
        ax3d.legend(
            loc="upper left", fontsize=7,
            facecolor="#1a1a1a", edgecolor="#3a3a3a",
            labelcolor="#cccccc", framealpha=0.85,
        )

    _setup_3d()

    # ── reload after episode switch ───────────────────────────────────────────
    def _reload():
        data = S["data"]; T = S["T"]
        for im, key in zip(cam_imgs, cam_keys):
            im.set_data(data[key][0])
        for j in range(n_joints):
            vals = data["qpos"][:, j]
            joint_lines[j].set_data(np.arange(T), vals)
            joint_vlines[j].set_xdata([0, 0])
            _set_ylim(joint_axes[j], vals)
            joint_axes[j].set_xlim(0, T - 1)
        if has_eef and "eef" in data:
            for i in range(6):
                vals = data["eef"][:, i]
                eef_lines[i].set_data(np.arange(T), vals)
                eef_vlines[i].set_xdata([0, 0])
                _set_ylim(eef_axes[i], vals)
                eef_axes[i].set_xlim(0, T - 1)
        _setup_3d()

    # ── animation ─────────────────────────────────────────────────────────────
    def _animate(_fn):
        if not S["paused"]:
            S["t"] = (S["t"] + (2 if S["fast"] else 1)) % S["T"]

        t    = S["t"]; T = S["T"]; data = S["data"]
        frac = t / max(T - 1, 1)

        for im, key in zip(cam_imgs, cam_keys):
            im.set_data(data[key][t])

        prog_rect.set_width(frac)
        prog_head.set_xdata([frac, frac])

        for vl in joint_vlines:
            vl.set_xdata([t, t])
        for vl in eef_vlines:
            vl.set_xdata([t, t])

        if _3d["trail"] is not None:
            xyz = data["eef"][:t + 1, :3]
            _3d["trail"].set_data_3d(xyz[:, 0], xyz[:, 1], xyz[:, 2])
            _3d["head"].set_data_3d([xyz[-1, 0]], [xyz[-1, 1]], [xyz[-1, 2]])

        status = "PAUSED" if S["paused"] else ("2×" if S["fast"] else "1×")
        fig.suptitle(
            f"{os.path.basename(episodes[S['ep_idx']])}   "
            f"[{S['ep_idx'] + 1}/{len(episodes)}]   frame {t} / {T - 1}   {status}   "
            "│   SPC:pause  ←→:±5f  ↑↓:episode  f:2×  r:reset  s:save  q:quit",
            color="#cccccc", fontsize=8,
        )
        return []

    # ── keyboard ──────────────────────────────────────────────────────────────
    def _on_key(ev):
        k = ev.key
        if k == " ":
            S["paused"] = not S["paused"]
        elif k == "r":
            S["t"] = 0
        elif k == "f":
            S["fast"] = not S["fast"]
        elif k == "right":
            S["t"] = min(S["T"] - 1, S["t"] + 5)
        elif k == "left":
            S["t"] = max(0, S["t"] - 5)
        elif k in ("down", "n"):
            _load((S["ep_idx"] + 1) % len(episodes))
            _reload()
        elif k in ("up", "p"):
            _load((S["ep_idx"] - 1) % len(episodes))
            _reload()
        elif k == "s":
            out = os.path.splitext(os.path.basename(episodes[S["ep_idx"]]))[0] + "_viz.png"
            fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
            print(f"Saved → {out}")
        elif k in ("q", "escape"):
            ani.event_source.stop()
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", _on_key)

    interval_ms = max(1, int(1000 / args.fps))
    ani = FuncAnimation(fig, _animate, interval=interval_ms, blit=False, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    main()
