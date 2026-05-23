#!/usr/bin/env python3
"""
Visualize UR5 episodes from HDF5 dataset  (optimized layout).

Layout (top -> bottom):
    header  ->  video row  ->  progress bar  ->  per-joint trajectory strips

Usage:
    python data_processing/viz/viz_episode.py <path>              # file or directory
    python data_processing/viz/viz_episode.py <path> --scale 2.0
    python data_processing/viz/viz_episode.py <path> --fps 20

Keyboard:
    SPACE           pause / resume
    Left / Right    step -5 / +5 frames
    Up / Down       previous / next episode
    n               next episode
    p               previous episode
    f               toggle 1x / 2x speed
    r               restart current episode
    q / ESC         quit

Mouse:
    Click / drag on progress bar or trajectory area to scrub timeline.
"""

import argparse
import glob
import os
import sys

# Suppress Qt font warnings (C-level stderr) during cv2 import
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype")


def _import_cv2_quiet():
    """Import cv2 while suppressing C-level stderr (Qt font warnings)."""

    _stderr_fd = sys.stderr.fileno()
    _old = os.dup(_stderr_fd)
    _devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_devnull, _stderr_fd)
    os.close(_devnull)
    try:
        import cv2
    finally:
        os.dup2(_old, _stderr_fd)
        os.close(_old)
    return cv2


cv2 = _import_cv2_quiet()
import h5py
import numpy as np

# ─── palette (all BGR) ────────────────────────────────────────────────────────
BG = (10, 10, 10)
PANEL = (20, 20, 20)
BORDER = (40, 40, 40)
TEXT_PRI = (240, 240, 240)
TEXT_SEC = (160, 160, 160)
TEXT_DIM = (100, 100, 100)
CURSOR = (58, 190, 255)
PROG_FG = (58, 190, 255)
PROG_BG = (30, 30, 30)
PLAY_COL = (80, 210, 90)
PAUSE_COL = (230, 170, 50)
FAST_COL = (80, 180, 255)
DELETE_COL = (50, 50, 220)  # red (BGR)

# joint colors (BGR)
JOINT_BGR = [
    (113, 113, 248),  # red
    (128, 222, 74),  # green
    (250, 165, 96),  # blue
    (21, 204, 250),  # yellow
    (249, 121, 232),  # pink
    (238, 211, 34),  # cyan
    (184, 163, 148),  # slate
]

# ─── layout constants ─────────────────────────────────────────────────────────
HEADER_H = 56
PROGRESS_H = 24
STRIP_H = 100
STRIP_PAD_L = 80
STRIP_PAD_R = 14


# ─── cross-platform arrow key detection ──────────────────────────────────────


def decode_arrow(key_raw):
    """
    Return 'left', 'right', 'up', 'down' or None.
    Works across Linux GTK/Qt, macOS, and Windows OpenCV backends.
    """
    if key_raw < 0:
        return None
    # Linux GTK
    if key_raw == 65361:
        return "left"
    if key_raw == 65363:
        return "right"
    if key_raw == 65362:
        return "up"
    if key_raw == 65364:
        return "down"
    # Windows (and some Qt backends)
    if key_raw == 2424832:
        return "left"
    if key_raw == 2555904:
        return "right"
    if key_raw == 2490368:
        return "up"
    if key_raw == 2621440:
        return "down"
    # macOS special: some builds encode as 0x00XXYY
    code = key_raw & 0xFFFF
    if code in {65361, 63234}:
        return "left"
    if code in {65363, 63235}:
        return "right"
    if code in {65362, 63232}:
        return "up"
    if code in {65364, 63233}:
        return "down"
    return None


# ══════════════════════════════════════════════════════════════════════════════
# data
# ══════════════════════════════════════════════════════════════════════════════


def _natural_key(p: str):
    """Sort key: split filename into text/number chunks so episode_2 < episode_10."""
    import re

    name = os.path.basename(p)
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", name)]


def find_episodes(path: str) -> tuple[list[str], int]:
    if os.path.isfile(path):
        parent = os.path.dirname(os.path.abspath(path))
        files = sorted(glob.glob(os.path.join(parent, "*.hdf5")), key=_natural_key)
        if not files:
            files = [os.path.abspath(path)]
        start = files.index(os.path.abspath(path)) if os.path.abspath(path) in files else 0
        return files, start
    files = sorted(glob.glob(os.path.join(path, "**/*.hdf5"), recursive=True), key=_natural_key)
    if not files:
        sys.exit(f"No .hdf5 files found under: {path}")
    return files, 0


def load_episode(path: str) -> dict:
    with h5py.File(path, "r") as f:
        ep = {
            "qpos": f["observations/qpos"][:],
            "action": f["action"][:],
            "exterior": f["observations/images/exterior_image_1_left"][:],
            "wrist": f["observations/images/wrist_image_left"][:],
        }
        imgs = f["observations/images"]
        if "front_image_1" in imgs:
            ep["front"] = imgs["front_image_1"][:]
        return ep


# ══════════════════════════════════════════════════════════════════════════════
# rendering helpers
# ══════════════════════════════════════════════════════════════════════════════


def _put(img, text, x, y, scale, color, thickness=1, shadow=True):
    """
    Clean anti-aliased text.  Optional soft shadow (single-pixel offset,
    same thickness) avoids the thick blurry halo of the old approach.
    """
    if shadow:
        cv2.putText(img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _text_size(text, scale, thickness=1):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    return w, h


# ── header (two rows) ────────────────────────────────────────────────────────


def make_header(ep_idx, n_eps, ep_name, t, T, paused, fast, canvas_w, confirm_delete=False):
    bar = np.full((HEADER_H, canvas_w, 3), PANEL, dtype=np.uint8)
    bar[-1, :] = BORDER

    row1_y = 22
    row2_y = 44

    if confirm_delete:
        # full-width red warning banner
        bar[:, :] = (30, 20, 20)
        bar[-1, :] = DELETE_COL
        warn = f"DELETE  {ep_name}  ?   Press Y to confirm  /  any other key to cancel"
        ww, _ = _text_size(warn, 0.52)
        _put(bar, warn, max(8, (canvas_w - ww) // 2), row1_y + 4, 0.52, DELETE_COL)
        return bar

    # ── ROW 1 ──
    # status dot + text
    dot_color = PAUSE_COL if paused else PLAY_COL
    cv2.circle(bar, (18, row1_y - 4), 6, dot_color, -1, cv2.LINE_AA)
    status_txt = "PAUSED" if paused else "PLAYING"
    _put(bar, status_txt, 32, row1_y, 0.50, dot_color)

    # speed indicator
    sw, _ = _text_size(status_txt, 0.50)
    speed_x = 32 + sw + 12
    if fast:
        _put(bar, "2x", speed_x, row1_y, 0.50, FAST_COL)
        sx, _ = _text_size("2x", 0.50)
        ep_start = speed_x + sx + 16
    else:
        _put(bar, "1x", speed_x, row1_y, 0.50, TEXT_DIM)
        sx, _ = _text_size("1x", 0.50)
        ep_start = speed_x + sx + 16

    # episode name
    ep_txt = "%s  [%d/%d]" % (ep_name, ep_idx + 1, n_eps)
    _put(bar, ep_txt, ep_start, row1_y, 0.50, TEXT_PRI)

    # frame counter (right-aligned)
    t_txt = "frame %d / %d" % (t, T - 1)
    tw, _ = _text_size(t_txt, 0.50)
    _put(bar, t_txt, canvas_w - tw - 20, row1_y, 0.50, TEXT_SEC)

    # ── ROW 2: key hints ──
    hints = "SPC:pause  Left/Right:+5  Up/Down:episode  f:2x  r:reset  d:delete episode  q:quit  |  mouse: drag scrub"
    _put(bar, hints, 18, row2_y, 0.38, TEXT_DIM, shadow=False)

    return bar


# ── video row ─────────────────────────────────────────────────────────────────


def make_video_row(cam_frames: list, disp_w: int, disp_h: int, sep_w: int = 4):
    """cam_frames: list of (rgb_array, label_str) — 2 or 3 cameras."""
    interp = cv2.INTER_LINEAR
    sep = np.full((disp_h, sep_w, 3), BG, dtype=np.uint8)
    panels = []

    for i, (rgb, label) in enumerate(cam_frames):
        bgr = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (disp_w, disp_h), interpolation=interp)
        font_scale = 0.55
        pad_x, pad_y = 10, 6
        tw, th = _text_size(label, font_scale)
        badge_w = tw + pad_x * 2
        badge_h = th + pad_y * 2 + 4
        roi = bgr[:badge_h, :badge_w]
        roi[:] = (roi.astype(np.int16) * 3 // 10).clip(0, 255).astype(np.uint8)
        _put(bgr, label, pad_x, pad_y + th + 2, font_scale, (220, 220, 220))
        panels.append(bgr)
        if i < len(cam_frames) - 1:
            panels.append(sep)

    return np.concatenate(panels, axis=1)


# ── progress bar (aligned with trajectory plot_x0..plot_x1) ─────────────────


def make_progress(t, T, canvas_w, plot_x0, plot_x1):
    bar = np.full((PROGRESS_H, canvas_w, 3), PANEL, dtype=np.uint8)
    bar[0, :] = BORDER
    bar[-1, :] = BORDER
    bar[1:-1, :plot_x0] = (18, 18, 18)
    bar[2:-2, plot_x0:plot_x1] = PROG_BG

    plot_w = plot_x1 - plot_x0
    frac = t / max(T - 1, 1)
    filled_x = plot_x0 + int(frac * plot_w)

    bar[5:-5, plot_x0:filled_x] = PROG_FG
    tip_start = max(plot_x0, filled_x - 4)
    bar[5:-5, tip_start:filled_x] = (255, 240, 200)

    # playhead marker
    ph_x0 = max(plot_x0, filled_x - 2)
    ph_x1 = min(plot_x1, filled_x + 2)
    cv2.rectangle(bar, (ph_x0, 2), (ph_x1, PROGRESS_H - 3), (255, 255, 255), -1)

    _put(bar, "%d" % t, 8, PROGRESS_H - 6, 0.36, TEXT_PRI, shadow=False)

    return bar


# ── per-joint trajectory strips ───────────────────────────────────────────────


def build_strips_bg(qpos, canvas_w, n_joints):
    total_h = n_joints * STRIP_H
    img = np.full((total_h, canvas_w, 3), BG, dtype=np.uint8)

    T = len(qpos)
    plot_x0 = STRIP_PAD_L
    plot_x1 = canvas_w - STRIP_PAD_R
    plot_w = plot_x1 - plot_x0

    strip_y_offsets = []
    margin_top = 12
    margin_bot = 12

    for j in range(n_joints):
        y_off = j * STRIP_H
        strip_y_offsets.append(y_off)

        # backgrounds
        img[y_off : y_off + STRIP_H, :] = (16, 16, 16)
        img[y_off : y_off + STRIP_H, :plot_x0] = (20, 20, 20)
        img[y_off + STRIP_H - 1, :] = BORDER
        cv2.line(img, (plot_x0, y_off), (plot_x0, y_off + STRIP_H - 1), BORDER, 1)

        col = JOINT_BGR[j % len(JOINT_BGR)]

        # joint label
        label = "joint %d" % j
        _put(img, label, 6, y_off + STRIP_H // 2 + 4, 0.42, col, shadow=False)

        vals = qpos[:, j]
        vmin, vmax = vals.min(), vals.max()
        span = max(vmax - vmin, 1e-6)

        # min/max in label margin
        _put(img, f"{vmax:.1f}", 6, y_off + margin_top + 7, 0.26, TEXT_DIM, shadow=False)
        _put(img, f"{vmin:.1f}", 6, y_off + STRIP_H - margin_bot + 5, 0.26, TEXT_DIM, shadow=False)

        draw_h = STRIP_H - margin_top - margin_bot

        # subtle horizontal grid
        for frac in (0.25, 0.50, 0.75):
            gy = y_off + margin_top + int(frac * draw_h)
            cv2.line(img, (plot_x0 + 1, gy), (plot_x1, gy), (26, 26, 26), 1)

        # trajectory polyline (vectorized)
        t_idx = np.arange(T)
        xs = plot_x0 + (t_idx * plot_w / max(T - 1, 1)).astype(np.int32)
        norms = (vals - vmin) / span
        ys = (y_off + margin_top + ((1.0 - norms) * draw_h)).astype(np.int32)
        pts = np.stack([xs, ys], axis=1)

        # soft glow + main line
        dim_col = tuple(int(c * 0.25) for c in col)
        cv2.polylines(img, [pts], False, dim_col, 2, cv2.LINE_AA)
        cv2.polylines(img, [pts], False, col, 1, cv2.LINE_AA)

    return img, plot_x0, plot_x1, strip_y_offsets


def draw_strip_cursor(strips, t, T, qpos_t, qpos, plot_x0, plot_x1, strip_y_offsets, n_joints):
    plot_w = plot_x1 - plot_x0
    cx = plot_x0 + int(t * plot_w / max(T - 1, 1))

    margin_top = 12
    margin_bot = 12
    draw_h = STRIP_H - margin_top - margin_bot
    total_h = n_joints * STRIP_H

    # cursor line
    cv2.line(strips, (cx, 0), (cx, total_h), CURSOR, 1, cv2.LINE_AA)

    for j in range(n_joints):
        y_off = strip_y_offsets[j]
        col = JOINT_BGR[j % len(JOINT_BGR)]

        vals = qpos[:, j]
        vmin, vmax = vals.min(), vals.max()
        span = max(vmax - vmin, 1e-6)
        norm = float(np.clip((qpos_t[j] - vmin) / span, 0, 1))
        cy = y_off + margin_top + int((1.0 - norm) * draw_h)

        # dot
        cv2.circle(strips, (cx, cy), 5, (16, 16, 16), -1, cv2.LINE_AA)
        cv2.circle(strips, (cx, cy), 4, col, -1, cv2.LINE_AA)
        cv2.circle(strips, (cx, cy), 4, (220, 220, 220), 1, cv2.LINE_AA)

        # value label
        val = float(qpos_t[j])
        txt = f"{val:+.1f}"
        tx = cx + 8
        tw, _ = _text_size(txt, 0.38)
        if tx + tw + 4 > plot_x1:
            tx = cx - tw - 8
        _put(strips, txt, tx, cy + 4, 0.38, col)


# ══════════════════════════════════════════════════════════════════════════════
# mouse scrubbing
# ══════════════════════════════════════════════════════════════════════════════


class MouseState:
    def __init__(self):
        self.dragging = False
        self.scrub_t = None

    def reset(self):
        self.dragging = False
        self.scrub_t = None


def make_mouse_callback(mouse, layout):
    def _cb(event, x, y, flags, param):
        in_progress = layout["progress_y0"] <= y <= layout["progress_y1"]
        in_strips = layout["strips_y0"] <= y <= layout["strips_y1"]
        scrub_zone = in_progress or in_strips

        if event == cv2.EVENT_LBUTTONDOWN and scrub_zone:
            mouse.dragging = True
        elif event == cv2.EVENT_LBUTTONUP:
            mouse.dragging = False
            return
        elif event == cv2.EVENT_MOUSEMOVE and not mouse.dragging:
            return

        if mouse.dragging:
            T = layout["T"]
            px0 = layout["plot_x0"]
            px1 = layout["plot_x1"]
            frac = max(0.0, min(1.0, (x - px0) / max(px1 - px0, 1)))
            mouse.scrub_t = int(frac * (T - 1))

    return _cb


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="HDF5 file or directory of episodes")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--scale", type=float, default=2.0, help="Video display scale (default 2.0)")
    args = parser.parse_args()

    episodes, ep_idx = find_episodes(args.path)
    print("Found %d episode(s). Starting at [%d/%d]" % (len(episodes), ep_idx + 1, len(episodes)))

    win_name = "Episode Viewer"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    mouse = MouseState()
    layout = {}

    running = True
    fast_mode = False  # 2x speed toggle
    base_delay = max(1, int(1000 / args.fps))

    while running:
        path = episodes[ep_idx]
        print("\n[%d/%d] Loading %s ..." % (ep_idx + 1, len(episodes), os.path.basename(path)))
        data = load_episode(path)

        exterior = data["exterior"]
        wrist = data["wrist"]
        front = data.get("front")
        qpos = data["qpos"]
        T = len(exterior)
        n_joints = qpos.shape[1]

        src_h, src_w = exterior.shape[1], exterior.shape[2]
        disp_w = int(src_w * args.scale)
        disp_h = int(src_h * args.scale)
        num_cams = 3 if front is not None else 2
        sep_w = 4
        canvas_w = disp_w * num_cams + sep_w * (num_cams - 1)

        print("  Building trajectory strips ...")
        strips_bg, plot_x0, plot_x1, strip_y_offsets = build_strips_bg(qpos, canvas_w, n_joints)
        strips_total_h = n_joints * STRIP_H

        # vertical layout
        video_y0 = HEADER_H
        video_y1 = video_y0 + disp_h
        progress_y0 = video_y1
        progress_y1 = progress_y0 + PROGRESS_H
        strips_y0 = progress_y1
        strips_y1 = strips_y0 + strips_total_h
        total_h = strips_y1

        layout.update(
            {
                "progress_y0": progress_y0,
                "progress_y1": progress_y1,
                "strips_y0": strips_y0,
                "strips_y1": strips_y1,
                "plot_x0": plot_x0,
                "plot_x1": plot_x1,
                "T": T,
                "canvas_w": canvas_w,
            }
        )

        cv2.setMouseCallback(win_name, make_mouse_callback(mouse, layout))
        mouse.reset()
        cv2.resizeWindow(win_name, canvas_w, total_h)

        t = 0
        paused = False
        nav = None
        confirm_delete = False

        while nav is None and running:
            # mouse scrub
            if mouse.scrub_t is not None:
                t = max(0, min(T - 1, mouse.scrub_t))
                if not mouse.dragging:
                    mouse.scrub_t = None

            # ── render ───────────────────────────────────────────────────
            header = make_header(
                ep_idx,
                len(episodes),
                os.path.basename(path),
                t,
                T,
                paused,
                fast_mode,
                canvas_w,
                confirm_delete=confirm_delete,
            )
            cam_frames = [(exterior[t], "EXTERIOR"), (wrist[t], "WRIST")]
            if front is not None:
                cam_frames.append((front[t], "FRONT"))
            video = make_video_row(cam_frames, disp_w, disp_h)
            prog = make_progress(t, T, canvas_w, plot_x0, plot_x1)

            strips = strips_bg.copy()
            draw_strip_cursor(strips, t, T, qpos[t], qpos, plot_x0, plot_x1, strip_y_offsets, n_joints)

            canvas = np.concatenate([header, video, prog, strips], axis=0)
            cv2.imshow(win_name, canvas)

            # ── input ────────────────────────────────────────────────────
            delay = (
                0 if confirm_delete else (1 if (paused or mouse.dragging) else (base_delay // 2 if fast_mode else base_delay))
            )
            key_raw = cv2.waitKeyEx(delay)
            k = key_raw & 0xFF
            arrow = decode_arrow(key_raw)

            if confirm_delete:
                if k == ord("y"):
                    print(f"  Deleting {os.path.basename(path)} ...")
                    os.remove(path)
                    episodes.pop(ep_idx)
                    if not episodes:
                        print("No episodes left.")
                        running = False
                    else:
                        ep_idx = min(ep_idx, len(episodes) - 1)
                        nav = 0  # reload at same index
                else:
                    print("  Delete cancelled.")
                confirm_delete = False
                continue

            if k == ord("q") or k == 27:
                running = False
            elif k == ord(" "):
                paused = not paused
            elif k == ord("r"):
                t = 0
            elif k == ord("f"):
                fast_mode = not fast_mode
            elif k == ord("n"):
                nav = +1
            elif k == ord("p"):
                nav = -1
            elif k == ord("d"):
                confirm_delete = True
                paused = True
            elif arrow == "left":
                t = max(0, t - 5)
            elif arrow == "right":
                t = min(T - 1, t + 5)
            elif arrow == "up":
                nav = -1
            elif arrow == "down":
                nav = +1

            # auto-advance
            if not paused and not mouse.dragging:
                step = 2 if fast_mode else 1
                t = (t + step) % T

        if nav is not None:
            if nav == 0:
                pass  # stay at current ep_idx (after delete)
            else:
                ep_idx = (ep_idx + nav) % len(episodes)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
