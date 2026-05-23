#!/usr/bin/env python3
"""
Franka Panda State Publisher

Published topics:
    /robot_state    (Float64MultiArray, 8 floats:
                     joints 0-6 in degrees (actual) + gripper width 0-1 normalised)
    /robot_vel      (Float64MultiArray, 8 floats:
                     joints 0-6 velocity in deg/s + index 7 = 0.0 placeholder)
    /robot_eef_pose (Float64MultiArray, 6 floats:
                     TCP pose [x, y, z, rx, ry, rz] metres/rad via TF2)

Joint state from /joint_states (sensor_msgs/JointState, published by franka_ros2).
Gripper width from /franka_gripper/joint_states (finger positions summed, normalised).
EEF pose from TF2: panda_link0 → panda_EE.

Hardware:
    - Franka Panda running franka_ros2

Dependencies:
    apt install ros-humble-franka-ros2 ros-humble-tf2-ros
    pip install numpy scipy
"""

import threading

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

try:
    from scipy.spatial.transform import Rotation
except ImportError as err:
    raise ImportError("scipy not installed. Run: pip install scipy") from err

try:
    from tf2_ros import Buffer, TransformListener
except ImportError as err:
    raise ImportError("tf2_ros not available. Run: apt install ros-humble-tf2-ros") from err

# ══════════════════════════════ Configuration ══════════════════════════════

PUBLISH_HZ = 10

# TF frames for EEF pose (panda_EE is the flange; change to panda_hand_tcp for tool tip)
TF_BASE_FRAME = "panda_link0"
TF_EEF_FRAME  = "panda_EE"

# Panda arm joint names in order (franka_ros2 publishes these in /joint_states)
ARM_JOINT_NAMES = [
    "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
    "panda_joint5", "panda_joint6", "panda_joint7",
]

# Franka Hand finger joint names
FINGER_JOINT_NAMES = ["panda_finger_joint1", "panda_finger_joint2"]

# Max total finger spread for normalisation (Panda Hand spec: 2 × 40 mm = 80 mm)
GRIPPER_MAX_WIDTH = 0.08  # m

# ══════════════════════════════ State Publisher Node ══════════════════════════


class FrankaStatePublisher(Node):
    """
    Publishes actual Panda joint angles on /robot_state.
    Gripper normalised value (0=closed, 1=open) is derived from finger joint positions.
    EEF pose is obtained via TF2 (no dependency on franka_msgs).
    """

    def __init__(self):
        super().__init__("franka_state_pub")

        # ── TF2 for EEF pose ──────────────────────────────────────
        self._tf_buffer   = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # ── Arm joint state ───────────────────────────────────────
        self._arm_lock    = threading.Lock()
        self._arm_pos_deg = [0.0] * 7
        self._arm_vel_deg = [0.0] * 7
        self.create_subscription(JointState, "/joint_states", self._cb_joint_states, 10)

        # ── Gripper state ─────────────────────────────────────────
        self._gripper_lock = threading.Lock()
        self._gripper_norm = 0.0
        # franka_gripper publishes finger joints separately from /joint_states
        self.create_subscription(
            JointState, "/franka_gripper/joint_states", self._cb_gripper, 10
        )

        # ── Publishers + timer ────────────────────────────────────
        self._pub     = self.create_publisher(Float64MultiArray, "/robot_state",    10)
        self._vel_pub = self.create_publisher(Float64MultiArray, "/robot_vel",      10)
        self._eef_pub = self.create_publisher(Float64MultiArray, "/robot_eef_pose", 10)
        self.create_timer(1.0 / PUBLISH_HZ, self._publish_state)

        self.get_logger().info(
            f"[FrankaPub] Publishing /robot_state, /robot_vel, /robot_eef_pose at {PUBLISH_HZ} Hz."
        )

    def _cb_joint_states(self, msg: JointState):
        """Parse arm joint positions and velocities by name (order not guaranteed)."""
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        pos = [0.0] * 7
        vel = [0.0] * 7
        for arm_i, jname in enumerate(ARM_JOINT_NAMES):
            if jname not in name_to_idx:
                continue
            idx = name_to_idx[jname]
            pos[arm_i] = float(np.degrees(msg.position[idx]))
            vel[arm_i] = float(np.degrees(msg.velocity[idx])) if msg.velocity else 0.0
        with self._arm_lock:
            self._arm_pos_deg = pos
            self._arm_vel_deg = vel

    def _cb_gripper(self, msg: JointState):
        """Normalise total finger spread: 0 (closed) → 1 (open)."""
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        width = sum(
            msg.position[name_to_idx[n]]
            for n in FINGER_JOINT_NAMES
            if n in name_to_idx and msg.position
        )
        with self._gripper_lock:
            self._gripper_norm = float(np.clip(width / GRIPPER_MAX_WIDTH, 0.0, 1.0))

    def _get_eef_pose(self):
        """Return [x, y, z, rx, ry, rz] in metres/rad, or None if TF unavailable."""
        try:
            tf = self._tf_buffer.lookup_transform(
                TF_BASE_FRAME, TF_EEF_FRAME, rclpy.time.Time()
            )
            t = tf.transform.translation
            q = tf.transform.rotation
            rotvec = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_rotvec()
            return [t.x, t.y, t.z, *rotvec.tolist()]
        except Exception:
            return None

    def _publish_state(self):
        with self._arm_lock:
            pos = list(self._arm_pos_deg)
            vel = list(self._arm_vel_deg)
        with self._gripper_lock:
            grip = self._gripper_norm

        state_msg = Float64MultiArray()
        state_msg.data = [*pos, grip]
        self._pub.publish(state_msg)

        vel_msg = Float64MultiArray()
        vel_msg.data = [*vel, 0.0]  # index 7: gripper has no velocity sensor
        self._vel_pub.publish(vel_msg)

        eef = self._get_eef_pose()
        if eef is not None:
            eef_msg = Float64MultiArray()
            eef_msg.data = eef
            self._eef_pub.publish(eef_msg)

        self.get_logger().debug(f"[FrankaPub] state: {[f'{v:.2f}' for v in state_msg.data]}")


# ══════════════════════════════ Entry point ══════════════════════════════


def main():
    rclpy.init()
    node = FrankaStatePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
