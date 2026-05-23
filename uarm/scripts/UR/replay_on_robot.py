#!/usr/bin/env python3
"""
Replay recorded HDF5 episodes on the real UR5.

Overlays live RTDE joint readings on the trajectory strips so you can compare
recorded vs actual motion in real time.

Usage:
    python uarm/scripts/UR/replay_on_robot.py <episode.hdf5>
    python uarm/scripts/UR/replay_on_robot.py <dataset_dir/>
    python uarm/scripts/UR/replay_on_robot.py <episode.hdf5> --no-gripper

Keyboard:
    SPACE       pause / resume
    Up / Down   previous / next episode  (robot moves to new start automatically)
    Q / ESC     stop and disconnect

Safety:
    - Type "yes" once at startup before the robot moves.
    - Keep E-stop within reach at all times.
    - Ctrl-C stops servoJ and disconnects cleanly.

HDF5 convention:
    action[:, 0:6]   commanded joint angles, degrees
    action[:, 6]     gripper  0 = closed, 1 = open  (raw HDF5)
    qpos[:, 0:6]     actual measured joint angles, degrees
"""

import argparse
import glob
import os
import re
import sys
import threading
import time

os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype")


def _import_cv2_quiet():
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

try:
    import rtde_control
    import rtde_receive
except ImportError:
    sys.exit("ur_rtde not installed: pip install ur-rtde")

try:
    import serial
except ImportError:
    sys.exit("pyserial not installed: pip install pyserial")

# ── Robot config ─────────────────────────────────────────────────────────────

UR5_IP = "10.0.0.1"
GRIPPER_PORT = "/dev/ttyACM0"
GRIPPER_BAUDRATE = 9600
GRIPPER_MAX_MM = 50
GRIPPER_CLOSE_MM = 3

CONTROL_HZ = 60
SERVO_J_LOOKAHEAD = 0.2
SERVO_J_GAIN = 300

# ── Palette (BGR, matches viz_episode.py) ────────────────────────────────────

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
LIVE_COL = (50, 50, 230)  # red — live RTDE dot
DELETE_COL = (50, 50, 220)  # red — delete confirmation banner

JOINT_BGR = [
    (113, 113, 248),
    (128, 222, 74),
    (250, 165, 96),
    (21, 204, 250),
    (249, 121, 232),
    (238, 211, 34),
    (184, 163, 148),
]

HEADER_H = 56
PROGRESS_H = 24
STRIP_H = 100
STRIP_PAD_L = 80
STRIP_PAD_R = 14


# ── Weiss CRG 30-050 gripper driver ──────────────────────────────────────────


class WeissCRGGripper:
    FLAG_OPEN = 0
    FLAG_IDLE = 3

    def __init__(self, port, baudrate):
        self._lock = threading.Lock()
        self._ser = serial.Serial(port, baudrate, timeout=0.1)
        self._flags = 0
        print(f"[Gripper] Opened {port} @ {baudrate} baud")

    def _send(self, cmd, wait=0.3):
        with self._lock:
            try:
                self._ser.reset_input_buffer()
                self._ser.write((cmd + "\n").encode("ascii"))
            except Exception as e:
                print(f"[Gripper] serial error: {e}")
        time.sleep(wait)

    def _parse_pdin(self, line):
        try:
            inner = line[7:].split("]")[0]
            parts = [int(x, 16) for x in inner.split(",")]
            self._flags = parts[3] if len(parts) >= 4 else 0
            return True
        except Exception:
            return False

    def _read_pdin(self, timeout=1.0):
        t0 = time.monotonic()
        with self._lock:
            saved, self._ser.timeout = self._ser.timeout, 0.15
        try:
            while time.monotonic() - t0 < timeout:
                with self._lock:
                    line = self._ser.readline().decode("ascii", errors="ignore").strip()
                if line.startswith("@PDIN=[") and self._parse_pdin(line):
                    return True
        finally:
            with self._lock:
                self._ser.timeout = saved
        return False

    def _wait_flag_any(self, flag_bits, timeout=15.0):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self._read_pdin(0.5) and (self._flags & flag_bits):
                return True
        return False

    @staticmethod
    def _pos_to_bytes(mm):
        val = int(mm * 100)
        return f"[{(val >> 8) & 0xFF:02x},{val & 0xFF:02x}]"

    def initialise(self):
        for cmd in ["ID?", "ID?", "FALLBACK(1)", "MODE?"]:
            self._send(cmd, 0.5)
        self._send("RESTART()", 0.5)
        time.sleep(1.5)
        self._send("OPERATE()", 0.5)
        self._send("PDOUT=[00,00]", 0.5)
        print("[Gripper] Initialisation complete.")

    def home(self):
        print("[Gripper] Homing...")
        self._send(f"SETPARAM(96, 2, {self._pos_to_bytes(GRIPPER_MAX_MM)})", 0.3)
        self._send(f"SETPARAM(96, 1, {self._pos_to_bytes(GRIPPER_CLOSE_MM)})", 0.3)
        self._send("SETPARAM(96, 3, [64])", 0.3)
        self._send("PDOUT=[07,00]", 0.2)
        done_mask = (1 << self.FLAG_OPEN) | (1 << self.FLAG_IDLE)
        ok = self._wait_flag_any(done_mask, timeout=15.0)
        if not ok:
            print("[Gripper] WARNING: homing timed out.")
            return False
        print(f"[Gripper] Homing done — flags=0x{self._flags:02x}")
        return True

    def set_open(self, want_open: bool):
        self._send("PDOUT=[02,00]" if want_open else "PDOUT=[03,00]", 0.05)

    def close_port(self):
        self._send("PDOUT=[00,00]", 0.3)
        self._send("FALLBACK(1)", 0.3)
        if self._ser.is_open:
            self._ser.close()


# ── Episode discovery / loading ───────────────────────────────────────────────


def _natural_key(p):
    name = os.path.basename(p)
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", name)]


def find_episodes(path):
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


def load_episode(path):
    with h5py.File(path, "r") as f:
        ep = {
            "action": f["action"][:],
            "qpos": f["observations/qpos"][:],
            "exterior": f["observations/images/exterior_image_1_left"][:],
            "wrist": f["observations/images/wrist_image_left"][:],
            "hz": float(f.attrs.get("hz", 10)),
            "prompt": str(f.attrs.get("prompt", "")),
        }
        imgs = f["observations/images"]
        if "front_image_1" in imgs:
            ep["front"] = imgs["front_image_1"][:]
        ep["n_steps"] = len(ep["action"])
    return ep


# ── Rendering (viz_episode.py style) ─────────────────────────────────────────


def _put(img, text, x, y, scale, color, thickness=1, shadow=True):
    if shadow:
        cv2.putText(img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _text_size(text, scale, thickness=1):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    return w, h


def make_header(ep_idx, n_eps, ep_name, t, T, paused, canvas_w, prompt, confirm_delete=False):
    bar = np.full((HEADER_H, canvas_w, 3), PANEL, dtype=np.uint8)
    bar[-1, :] = BORDER

    if confirm_delete:
        bar[:, :] = (30, 20, 20)
        bar[-1, :] = DELETE_COL
        warn = f"DELETE  {ep_name}  ?   Press S to confirm  /  any other key to cancel"
        ww, _ = _text_size(warn, 0.52)
        _put(bar, warn, max(8, (canvas_w - ww) // 2), 26, 0.52, DELETE_COL)
        return bar

    dot_color = PAUSE_COL if paused else PLAY_COL
    cv2.circle(bar, (18, 18), 6, dot_color, -1, cv2.LINE_AA)
    status_txt = "PAUSED" if paused else "REPLAYING"
    _put(bar, status_txt, 32, 22, 0.50, dot_color)

    sw, _ = _text_size(status_txt, 0.50)
    ep_txt = f"{ep_name}  [{ep_idx + 1}/{n_eps}]   frame {t}/{T - 1}"
    _put(bar, ep_txt, 32 + sw + 16, 22, 0.50, TEXT_PRI)

    tw, _ = _text_size(ep_txt, 0.50)
    prompt_s = (prompt[:60] + "...") if len(prompt) > 60 else prompt
    _put(bar, f'"{prompt_s}"', 32 + sw + 16 + tw + 20, 22, 0.45, TEXT_SEC)

    hints = "SPC:pause   Up/Down:episode   d:delete   Q:quit   |   red dot = live RTDE"
    _put(bar, hints, 18, 44, 0.38, TEXT_DIM, shadow=False)
    return bar


def make_video_row(cam_frames, disp_w, disp_h, sep_w=4):
    sep = np.full((disp_h, sep_w, 3), BG, dtype=np.uint8)
    panels = []
    for i, (rgb, label) in enumerate(cam_frames):
        bgr = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
        tw, th = _text_size(label, 0.55)
        roi = bgr[: th + 16, : tw + 20]
        roi[:] = (roi.astype(np.int16) * 3 // 10).clip(0, 255).astype(np.uint8)
        _put(bgr, label, 10, th + 8, 0.55, (220, 220, 220))
        panels.append(bgr)
        if i < len(cam_frames) - 1:
            panels.append(sep)
    return np.concatenate(panels, axis=1)


def make_progress(t, T, canvas_w, plot_x0, plot_x1):
    bar = np.full((PROGRESS_H, canvas_w, 3), PANEL, dtype=np.uint8)
    bar[0, :] = BORDER
    bar[-1, :] = BORDER
    bar[2:-2, plot_x0:plot_x1] = PROG_BG

    plot_w = plot_x1 - plot_x0
    frac = t / max(T - 1, 1)
    filled_x = plot_x0 + int(frac * plot_w)
    bar[5:-5, plot_x0:filled_x] = PROG_FG
    bar[5:-5, max(plot_x0, filled_x - 4) : filled_x] = (255, 240, 200)
    ph_x0 = max(plot_x0, filled_x - 2)
    ph_x1 = min(plot_x1, filled_x + 2)
    cv2.rectangle(bar, (ph_x0, 2), (ph_x1, PROGRESS_H - 3), (255, 255, 255), -1)
    _put(bar, str(t), 8, PROGRESS_H - 6, 0.36, TEXT_PRI, shadow=False)
    return bar


def build_strips_bg(qpos, action, canvas_w, n_joints):
    """Draw recorded qpos (dim) and action (bright) trajectories."""
    total_h = n_joints * STRIP_H
    img = np.full((total_h, canvas_w, 3), BG, dtype=np.uint8)
    T = len(qpos)
    plot_x0 = STRIP_PAD_L
    plot_x1 = canvas_w - STRIP_PAD_R
    plot_w = plot_x1 - plot_x0
    strip_y_offsets = []
    margin_top, margin_bot = 12, 12

    for j in range(n_joints):
        y_off = j * STRIP_H
        strip_y_offsets.append(y_off)
        img[y_off : y_off + STRIP_H, :] = (16, 16, 16)
        img[y_off : y_off + STRIP_H, :plot_x0] = (20, 20, 20)
        img[y_off + STRIP_H - 1, :] = BORDER
        cv2.line(img, (plot_x0, y_off), (plot_x0, y_off + STRIP_H - 1), BORDER, 1)

        col = JOINT_BGR[j % len(JOINT_BGR)]
        label = f"joint {j}" if j < 6 else "gripper"
        _put(img, label, 6, y_off + STRIP_H // 2 + 4, 0.42, col, shadow=False)

        # use qpos for axis range
        vals_q = qpos[:, j]
        vals_a = action[:, j]
        vmin = min(vals_q.min(), vals_a.min())
        vmax = max(vals_q.max(), vals_a.max())
        span = max(vmax - vmin, 1e-6)
        _put(img, f"{vmax:.1f}", 6, y_off + margin_top + 7, 0.26, TEXT_DIM, shadow=False)
        _put(img, f"{vmin:.1f}", 6, y_off + STRIP_H - margin_bot + 5, 0.26, TEXT_DIM, shadow=False)

        draw_h = STRIP_H - margin_top - margin_bot
        for frac in (0.25, 0.50, 0.75):
            gy = y_off + margin_top + int(frac * draw_h)
            cv2.line(img, (plot_x0 + 1, gy), (plot_x1, gy), (26, 26, 26), 1)

        t_idx = np.arange(T)
        xs = plot_x0 + (t_idx * plot_w / max(T - 1, 1)).astype(np.int32)

        # qpos: dim background line
        norms_q = (vals_q - vmin) / span
        ys_q = (y_off + margin_top + (1.0 - norms_q) * draw_h).astype(np.int32)
        dim_col = tuple(int(c * 0.25) for c in col)
        cv2.polylines(img, [np.stack([xs, ys_q], axis=1)], False, dim_col, 2, cv2.LINE_AA)

        # action: bright main line
        norms_a = (vals_a - vmin) / span
        ys_a = (y_off + margin_top + (1.0 - norms_a) * draw_h).astype(np.int32)
        cv2.polylines(img, [np.stack([xs, ys_a], axis=1)], False, col, 1, cv2.LINE_AA)

    return img, plot_x0, plot_x1, strip_y_offsets


def draw_cursor(strips, t, T, qpos_t, action_t, live_deg, qpos, action, plot_x0, plot_x1, strip_y_offsets, n_joints):
    """Draw cursor for current step + optional live RTDE dot (red)."""
    plot_w = plot_x1 - plot_x0
    cx = plot_x0 + int(t * plot_w / max(T - 1, 1))
    margin_top, margin_bot = 12, 12
    draw_h = STRIP_H - margin_top - margin_bot
    cv2.line(strips, (cx, 0), (cx, n_joints * STRIP_H), CURSOR, 1, cv2.LINE_AA)

    for j in range(n_joints):
        y_off = strip_y_offsets[j]
        col = JOINT_BGR[j]

        vals_q = qpos[:, j]
        vals_a = action[:, j]
        vmin = min(vals_q.min(), vals_a.min())
        vmax = max(vals_q.max(), vals_a.max())
        span = max(vmax - vmin, 1e-6)

        # action cursor dot
        norm_a = float(np.clip((action_t[j] - vmin) / span, 0, 1))
        cy_a = y_off + margin_top + int((1.0 - norm_a) * draw_h)
        cv2.circle(strips, (cx, cy_a), 5, (16, 16, 16), -1, cv2.LINE_AA)
        cv2.circle(strips, (cx, cy_a), 4, col, -1, cv2.LINE_AA)
        cv2.circle(strips, (cx, cy_a), 4, (220, 220, 220), 1, cv2.LINE_AA)
        txt = f"{action_t[j]:+.1f}"
        tx = cx + 8
        if tx + _text_size(txt, 0.38)[0] + 4 > plot_x1:
            tx = cx - _text_size(txt, 0.38)[0] - 8
        _put(strips, txt, tx, cy_a + 4, 0.38, col)

        # live RTDE dot (red) — joints only
        if live_deg is not None and j < 6:
            norm_l = float(np.clip((live_deg[j] - vmin) / span, 0, 1))
            cy_l = y_off + margin_top + int((1.0 - norm_l) * draw_h)
            cv2.circle(strips, (cx, cy_l), 5, (16, 16, 16), -1, cv2.LINE_AA)
            cv2.circle(strips, (cx, cy_l), 4, LIVE_COL, -1, cv2.LINE_AA)
            cv2.circle(strips, (cx, cy_l), 4, (255, 255, 255), 1, cv2.LINE_AA)
            err = live_deg[j] - action_t[j]
            err_col = (50, 200, 50) if abs(err) < 3.0 else LIVE_COL
            _put(strips, f"Δ{err:+.1f}", cx + 8, cy_l + 4, 0.34, err_col)


# ── Mouse scrub ───────────────────────────────────────────────────────────────


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
        if event == cv2.EVENT_LBUTTONDOWN and (in_progress or in_strips):
            mouse.dragging = True
        elif event == cv2.EVENT_LBUTTONUP:
            mouse.dragging = False
            return
        elif event == cv2.EVENT_MOUSEMOVE and not mouse.dragging:
            return
        if mouse.dragging:
            frac = max(0.0, min(1.0, (x - layout["plot_x0"]) / max(layout["plot_x1"] - layout["plot_x0"], 1)))
            mouse.scrub_t = int(frac * (layout["T"] - 1))

    return _cb


# ── Arrow key decoding (cross-platform, same as viz_episode.py) ──────────────


def decode_arrow(key_raw):
    if key_raw < 0:
        return None
    lut = {
        65361: "left",
        65363: "right",
        65362: "up",
        65364: "down",
        2424832: "left",
        2555904: "right",
        2490368: "up",
        2621440: "down",
    }
    if key_raw in lut:
        return lut[key_raw]
    code = key_raw & 0xFFFF
    mac = {
        65361: "left",
        63234: "left",
        65363: "right",
        63235: "right",
        65362: "up",
        63232: "up",
        65364: "down",
        63233: "down",
    }
    return mac.get(code)


# ── Main replay loop ──────────────────────────────────────────────────────────


def run(args):
    episodes, ep_idx = find_episodes(args.path)
    print(f"Found {len(episodes)} episode(s). Starting at [{ep_idx + 1}/{len(episodes)}]")

    ep0 = load_episode(episodes[ep_idx])
    print(f"  Dataset  : {os.path.dirname(os.path.abspath(episodes[ep_idx]))}")
    print(f"  Episodes : {len(episodes)}")
    print(
        f"  First ep : {os.path.basename(episodes[ep_idx])}  "
        f"({ep0['n_steps']} steps @ {ep0['hz']} Hz = {ep0['n_steps'] / ep0['hz']:.1f} s)"
    )
    print(f"  Gripper  : {'disabled (--no-gripper)' if args.no_gripper else 'enabled'}")

    # ── Connect ───────────────────────────────────────────────────────────────
    print(f"\nConnecting to UR5 at {UR5_IP} ...")
    rtde_c = rtde_control.RTDEControlInterface(UR5_IP)
    rtde_r = rtde_receive.RTDEReceiveInterface(UR5_IP)
    print("RTDE connected.")

    gripper = None
    gripper_open = True
    if not args.no_gripper:
        print(f"Initialising gripper on {GRIPPER_PORT} (~15 s) ...")
        gripper = WeissCRGGripper(GRIPPER_PORT, GRIPPER_BAUDRATE)
        gripper.initialise()
        gripper.home()

    win = "Episode Replay"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.moveWindow(win, 30, 30)
    mouse = MouseState()
    layout: dict = {}

    running = True

    try:
        while running and ep_idx < len(episodes):
            # ── Load episode ──────────────────────────────────────────────────
            path = episodes[ep_idx]
            print(f"\n[{ep_idx + 1}/{len(episodes)}] Loading {os.path.basename(path)} ...")
            ep = load_episode(path)
            action = ep["action"]
            T = ep["n_steps"]
            dt_step = 1.0 / ep["hz"]
            dt_servo = 1.0 / CONTROL_HZ
            n_servo = max(1, round(dt_step / dt_servo))

            exterior = ep["exterior"]
            wrist = ep["wrist"]
            front = ep.get("front")

            src_h, src_w = exterior.shape[1], exterior.shape[2]
            disp_w = int(src_w * args.scale)
            disp_h = int(src_h * args.scale)
            num_cams = 3 if front is not None else 2
            sep_w = 4
            canvas_w = disp_w * num_cams + sep_w * (num_cams - 1)

            print("  Building trajectory strips ...")
            strips_bg, plot_x0, plot_x1, strip_y_offsets = build_strips_bg(
                ep["qpos"], action, canvas_w, action.shape[1]
            )
            n_joints = action.shape[1]
            strips_total_h = n_joints * STRIP_H

            video_y0 = HEADER_H
            progress_y0 = video_y0 + disp_h
            strips_y0 = progress_y0 + PROGRESS_H
            strips_y1 = strips_y0 + strips_total_h

            layout.update(
                {
                    "progress_y0": progress_y0,
                    "progress_y1": progress_y0 + PROGRESS_H,
                    "strips_y0": strips_y0,
                    "strips_y1": strips_y1,
                    "plot_x0": plot_x0,
                    "plot_x1": plot_x1,
                    "T": T,
                }
            )
            cv2.setMouseCallback(win, make_mouse_callback(mouse, layout))
            mouse.reset()
            cv2.resizeWindow(win, canvas_w, strips_y1)

            # ── Move to episode start ─────────────────────────────────────────
            start_rad = np.deg2rad(action[0, :6]).tolist()
            print(f"  Moving to start: {action[0, :6].round(1).tolist()} deg ...")
            rtde_c.moveJ(start_rad, speed=0.3, acceleration=0.3)
            print("  At start position.")

            if gripper:
                want_open = action[0, 6] >= 0.5
                gripper.set_open(want_open)
                gripper_open = want_open
                time.sleep(0.3)

            # ── Replay ────────────────────────────────────────────────────────
            t = 0
            paused = False
            nav = None
            live_deg = None
            confirm_delete = False

            while nav is None and running:
                if mouse.scrub_t is not None:
                    t = max(0, min(T - 1, mouse.scrub_t))
                    if not mouse.dragging:
                        mouse.scrub_t = None

                # render — clamp so the last frame stays visible after episode ends
                rt = min(t, T - 1)
                live_deg = np.degrees(rtde_r.getActualQ())
                header = make_header(
                    ep_idx,
                    len(episodes),
                    os.path.basename(path),
                    rt,
                    T,
                    paused,
                    canvas_w,
                    ep["prompt"],
                    confirm_delete=confirm_delete,
                )
                cam_frames = [(exterior[rt], "EXTERIOR"), (wrist[rt], "WRIST")]
                if front is not None:
                    cam_frames.append((front[rt], "FRONT"))
                video = make_video_row(cam_frames, disp_w, disp_h)
                prog = make_progress(rt, T, canvas_w, plot_x0, plot_x1)
                strips = strips_bg.copy()
                draw_cursor(
                    strips,
                    rt,
                    T,
                    ep["qpos"][rt],
                    action[rt],
                    live_deg,
                    ep["qpos"],
                    action,
                    plot_x0,
                    plot_x1,
                    strip_y_offsets,
                    n_joints,
                )
                canvas = np.concatenate([header, video, prog, strips], axis=0)
                cv2.imshow(win, canvas)

                key_raw = cv2.waitKeyEx(1)
                k = key_raw & 0xFF
                arrow = decode_arrow(key_raw)

                if confirm_delete:
                    if k == ord("s"):
                        print(f"  Deleting {os.path.basename(path)} ...")
                        rtde_c.servoStop()
                        os.remove(path)
                        episodes.pop(ep_idx)
                        if not episodes:
                            print("No episodes left.")
                            running = False
                        else:
                            ep_idx = min(ep_idx, len(episodes) - 1)
                            nav = 0
                    else:
                        print("  Delete cancelled.")
                    confirm_delete = False
                    continue

                if k in (ord("q"), 27):
                    running = False
                elif k == ord(" "):
                    paused = not paused
                    if paused:
                        rtde_c.servoStop()
                elif k == ord("d"):
                    confirm_delete = True
                    paused = True
                    rtde_c.servoStop()
                elif arrow == "up":
                    nav = -1
                elif arrow == "down":
                    nav = +1

                if paused or mouse.dragging:
                    time.sleep(0.02)
                    continue

                if t >= T:
                    # episode finished — stay on last frame, paused
                    paused = True
                    rtde_c.servoStop()
                    continue

                # send servoJ: linearly interpolate to next step
                t0_rad = np.deg2rad(action[t, :6])
                t1_rad = np.deg2rad(action[min(t + 1, T - 1), :6])
                deadline = time.monotonic() + dt_step

                for sub in range(n_servo):
                    alpha = sub / n_servo
                    target = ((1.0 - alpha) * t0_rad + alpha * t1_rad).tolist()
                    t_call = time.monotonic()
                    rtde_c.servoJ(target, 0.0, 0.0, dt_servo, SERVO_J_LOOKAHEAD, SERVO_J_GAIN)
                    remaining = (t_call + dt_servo) - time.monotonic()
                    if remaining > 0:
                        time.sleep(remaining)

                if gripper:
                    want_open = action[t, 6] >= 0.5
                    if want_open != gripper_open:
                        gripper.set_open(want_open)
                        gripper_open = want_open

                slack = deadline - time.monotonic()
                if slack > 0:
                    time.sleep(slack)

                t += 1

            rtde_c.servoStop()

            if nav is not None and episodes and nav != 0:
                ep_idx = (ep_idx + nav) % len(episodes)

    except KeyboardInterrupt:
        print("\nInterrupted — stopping servoJ.")
        rtde_c.servoStop()

    finally:
        cv2.destroyAllWindows()
        if gripper:
            gripper.close_port()
        rtde_c.disconnect()
        rtde_r.disconnect()
        print("Disconnected.")


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Replay HDF5 episodes on the real UR5 with viz_episode.py-style display."
    )
    parser.add_argument("path", help="HDF5 episode file or dataset directory")
    parser.add_argument("--scale", type=float, default=1.5, help="Video display scale (default 1.5)")
    parser.add_argument("--no-gripper", action="store_true", help="Skip gripper initialisation")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
