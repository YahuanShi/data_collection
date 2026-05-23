#!/usr/bin/env python3
"""
Franka Panda + Franka Hand Teleoperation Node

Subscribes: /servo_angles  (Float64MultiArray, 7 floats in degrees relative to home,
                             published by Uarm_teleop master arm)
Publishes:  /robot_action  (Float64MultiArray, 8 floats:
                             joints 0-6 in degrees + gripper normalised 0-1)

Hardware:
    - Franka Panda connected via Ethernet (franka_ros2)
    - Franka Hand (integrated gripper, controlled via franka_gripper action servers)

Control:
    - Joint position streaming via forward_position_controller
      topic: /forward_position_controller/commands  (Float64MultiArray, 7 values in rad)
    - Gripper open/close via franka_gripper Move / Grasp actions

Prerequisites (launch before running this script):
    ros2 launch franka_bringup franka.launch.py \
        robot_ip:=<IP> use_fake_hardware:=false load_gripper:=true

Dependencies:
    apt install ros-humble-franka-ros2 ros-humble-franka-msgs
    pip install numpy
"""

import threading
import time

import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

try:
    from franka_msgs.action import Grasp, Homing, Move
except ImportError as err:
    raise ImportError(
        "franka_msgs not installed. Run: apt install ros-humble-franka-msgs"
    ) from err

# ══════════════════════════════ Configuration ══════════════════════════════

# ← [ADJUST] set your Panda's IP when launching franka_bringup (not used here directly)
FRANKA_IP = "192.168.1.1"

# Panda home position (degrees).  [joint1 … joint7]
# ← [ADJUST] tune for your workspace; verify collision-free before moving
FRANKA_HOME_DEG = [0.0, -45.0, 0.0, -135.0, 0.0, 90.0, 45.0]

# Master arm (Uarm) has 6 servo channels (indices 0-5); Panda has 7 joints.
# JOINT_MAP[panda_i] = master servo index that drives Panda joint panda_i.
# Use -1 to fix that Panda joint at its home value (no master channel assigned).
# ← [ADJUST] remap if your master arm servo ordering differs
JOINT_MAP = [0, 1, 2, 3, 4, 5, -1]  # panda_joint7 fixed at home

# Per-joint scale factors applied after index remapping.
# ← [ADJUST] flip sign or scale magnitude to match master arm kinematics
JOINT_SCALE = [1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 0.0]

# forward_position_controller command topic (configured in franka_bringup)
POSITION_CMD_TOPIC = "/forward_position_controller/commands"

# Franka Hand parameters
GRIPPER_MAX_WIDTH = 0.08  # m, maximum finger spread (Panda Hand spec)
# ← [ADJUST] speed and force for your task
GRIPPER_SPEED   = 0.05    # m/s, open/close speed
GRIPPER_FORCE   = 20.0    # N,   grasp force
GRIPPER_EPSILON = 0.01    # m,   inner/outer position tolerance for grasp

# Master-arm gripper channel (index 6): servo[6] is angle offset from home (deg).
# ← [ADJUST] tune thresholds to match your master arm finger travel range
GRIPPER_OPEN_DEG  = -7.0   # offset > this → open
GRIPPER_CLOSE_DEG = -14.0  # offset < this → close; dead zone in between keeps state

# Control rates
CONTROL_HZ = 60  # joint command loop rate
GRIPPER_HZ  = 5  # gripper action rate — keep low; actions are slow, don't queue up

# EMA smoothing — fills 60 Hz loop gaps between 20 Hz servo updates
# ← [ADJUST] 0.1 = very smooth/laggy, 1.0 = raw (no smoothing)
CMD_SMOOTH_ALPHA = 0.30

# Joint velocity safety cap (rad/s)
# ← [ADJUST] Panda hardware max ~2.175 rad/s per joint; keep conservative for teleop
MAX_JOINT_VEL_RAD = 0.5

# ══════════════════════════════ Franka Teleop Node ════════════════════════════


class FrankaTeleopNode(Node):
    """
    Reads Uarm master-arm joint angles, maps them to the Panda frame,
    streams joint position commands via forward_position_controller,
    and drives the Franka Hand gripper.
    """

    def __init__(self):
        super().__init__("franka_teleop_node")

        self.home_rad = np.radians(FRANKA_HOME_DEG)

        # ── Joint position publisher ──────────────────────────────
        self._cmd_pub = self.create_publisher(Float64MultiArray, POSITION_CMD_TOPIC, 10)
        self.get_logger().info(f"[FrankaTeleop] Publishing joint commands → {POSITION_CMD_TOPIC}")

        # ── Franka Hand action clients ────────────────────────────
        self._homing_client = ActionClient(self, Homing, "/franka_gripper/homing")
        self._move_client   = ActionClient(self, Move,   "/franka_gripper/move")
        self._grasp_client  = ActionClient(self, Grasp,  "/franka_gripper/grasp")

        self.get_logger().info("[FrankaTeleop] (1/4) Waiting for franka_gripper action servers...")
        for client, name in [
            (self._homing_client, "homing"),
            (self._move_client,   "move"),
            (self._grasp_client,  "grasp"),
        ]:
            if not client.wait_for_server(timeout_sec=10.0):
                self.get_logger().warning(f"[FrankaTeleop] Timeout waiting for /franka_gripper/{name}")
        self.get_logger().info("[FrankaTeleop] Gripper action servers ready.")

        # ── Home gripper ──────────────────────────────────────────
        self.get_logger().info("[FrankaTeleop] (2/4) Homing gripper (closes then opens, ~5 s)...")
        self._send_gripper_homing()
        self.get_logger().info("[FrankaTeleop] Gripper homed.")

        # ── Move robot to home position ───────────────────────────
        self.get_logger().info(
            f"[FrankaTeleop] (3/4) Sending home position: {FRANKA_HOME_DEG} deg..."
        )
        self._publish_joint_cmd(self.home_rad)
        time.sleep(3.0)  # allow controller to reach home before teleop starts
        self.get_logger().info("[FrankaTeleop] At home position.")

        # ── Shared state ──────────────────────────────────────────
        self._lock             = threading.Lock()
        self._cmd_joints_rad   = self.home_rad.copy()
        self._cmd_gripper_open = True
        self._last_cmd_rad     = self.home_rad.copy()
        self._smooth_rad       = self.home_rad.copy()
        self._servo_warned     = False

        # ── ROS 2 interfaces ──────────────────────────────────────
        self._action_pub = self.create_publisher(Float64MultiArray, "/robot_action", 10)
        self.create_subscription(Float64MultiArray, "/servo_angles", self._servo_callback, 10)

        # ── Gripper thread ────────────────────────────────────────
        self._gripper_thread = threading.Thread(target=self._gripper_loop, daemon=True)
        self._gripper_thread.start()

        self.get_logger().info(
            f"[FrankaTeleop] (4/4) *** READY *** — streaming at {CONTROL_HZ} Hz. "
            "Move the master arm now."
        )

    # ── Gripper helpers ───────────────────────────────────────────────────────

    def _send_gripper_homing(self):
        # Wait for goal accepted, then wait for result (homing takes ~5 s)
        goal_future = self._homing_client.send_goal_async(Homing.Goal())
        rclpy.spin_until_future_complete(self, goal_future, timeout_sec=10.0)
        handle = goal_future.result()
        if handle and handle.accepted:
            result_future = handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future, timeout_sec=15.0)

    def _send_gripper_open(self):
        goal = Move.Goal()
        goal.width = GRIPPER_MAX_WIDTH
        goal.speed = GRIPPER_SPEED
        self._move_client.send_goal_async(goal)

    def _send_gripper_close(self):
        goal = Grasp.Goal()
        goal.width = 0.0
        goal.speed = GRIPPER_SPEED
        goal.force = GRIPPER_FORCE
        goal.epsilon.inner = GRIPPER_EPSILON
        goal.epsilon.outer = GRIPPER_EPSILON
        self._grasp_client.send_goal_async(goal)

    def _publish_joint_cmd(self, rad: np.ndarray):
        msg = Float64MultiArray()
        msg.data = rad.tolist()
        self._cmd_pub.publish(msg)

    # ── Joint mapping ─────────────────────────────────────────────────────────

    def _map_joints(self, servo_deg: np.ndarray) -> np.ndarray:
        target = self.home_rad.copy()
        for panda_i, master_i in enumerate(JOINT_MAP):
            if master_i < 0:
                continue  # joint fixed at home
            delta_rad = np.deg2rad(servo_deg[master_i]) * JOINT_SCALE[panda_i]
            target[panda_i] = self.home_rad[panda_i] + delta_rad
        return target

    def _map_gripper(self, servo_deg: float):
        """Returns True (open), False (close), or None (dead zone → keep state)."""
        if servo_deg > GRIPPER_OPEN_DEG:
            return True
        if servo_deg < GRIPPER_CLOSE_DEG:
            return False
        return None

    def _limit_joint_vel(self, target: np.ndarray, dt: float) -> np.ndarray:
        max_step = MAX_JOINT_VEL_RAD * dt
        delta = target - self._last_cmd_rad
        return self._last_cmd_rad + np.clip(delta, -max_step, max_step)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _servo_callback(self, msg: Float64MultiArray):
        data = np.asarray(msg.data, dtype=np.float64)
        if data.size < 7:
            if not self._servo_warned:
                self.get_logger().warning(
                    "[FrankaTeleop] /servo_angles needs 7 values (6 joints + gripper)"
                )
                self._servo_warned = True
            return
        gripper_cmd = self._map_gripper(float(data[6]))
        with self._lock:
            self._cmd_joints_rad = self._map_joints(data[:6])
            if gripper_cmd is not None:
                self._cmd_gripper_open = gripper_cmd

    # ── Gripper thread ────────────────────────────────────────────────────────

    def _gripper_loop(self):
        dt = 1.0 / GRIPPER_HZ
        last_open = True
        self._send_gripper_open()
        while rclpy.ok():
            with self._lock:
                want_open = self._cmd_gripper_open
            if want_open != last_open:
                if want_open:
                    self._send_gripper_open()
                    self.get_logger().info("[FrankaTeleop] Gripper opening")
                else:
                    self._send_gripper_close()
                    self.get_logger().info("[FrankaTeleop] Gripper closing")
                last_open = want_open
            time.sleep(dt)

    # ── Main control loop ─────────────────────────────────────────────────────

    def run(self):
        dt = 1.0 / CONTROL_HZ
        while rclpy.ok():
            t0 = time.monotonic()

            with self._lock:
                target_rad   = self._cmd_joints_rad.copy()
                gripper_open = self._cmd_gripper_open

            # EMA: smooth the servo-mapped target across the faster control loop
            self._smooth_rad = (
                CMD_SMOOTH_ALPHA * target_rad
                + (1.0 - CMD_SMOOTH_ALPHA) * self._smooth_rad
            )
            cmd_rad = self._limit_joint_vel(self._smooth_rad, dt)
            self._last_cmd_rad = cmd_rad.copy()

            self._publish_joint_cmd(cmd_rad)

            gripper_norm = 1.0 if gripper_open else 0.0
            action_msg = Float64MultiArray()
            action_msg.data = [*np.degrees(cmd_rad).tolist(), gripper_norm]
            self._action_pub.publish(action_msg)

            self.get_logger().debug(
                f"[FrankaTeleop] action: {[f'{v:.2f}' for v in action_msg.data]}"
            )

            elapsed = time.monotonic() - t0
            sleep_t = dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)


# ══════════════════════════════ Entry point ══════════════════════════════


def main():
    rclpy.init()
    node = FrankaTeleopNode()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
