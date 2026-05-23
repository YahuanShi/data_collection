#!/usr/bin/env python3
"""
test_franka.py - Franka Panda connectivity and basic motion test (no ROS required beyond rclpy).

Steps:
    1. Check /joint_states is publishing (robot alive and franka_ros2 running)
    2. Check forward_position_controller has subscribers
    3. Check franka_gripper action servers are reachable
    4. Move to home position via forward_position_controller

Usage:
    python3 test_franka.py
    # Requires franka_ros2 already launched:
    #   ros2 launch franka_bringup franka.launch.py \
    #       robot_ip:=<IP> use_fake_hardware:=false load_gripper:=true

Press Ctrl+C to abort at any time.
"""

import sys
import time

import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

try:
    from franka_msgs.action import Homing
except ImportError:
    print("[FAIL] franka_msgs not installed. Run: apt install ros-humble-franka-msgs")
    sys.exit(1)

# ← [ADJUST] verify this pose is collision-free before moving on your setup
HOME_DEG = [0.0, -45.0, 0.0, -135.0, 0.0, 90.0, 45.0]

ARM_JOINT_NAMES = [
    "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
    "panda_joint5", "panda_joint6", "panda_joint7",
]

POSITION_CMD_TOPIC = "/forward_position_controller/commands"


class FrankaTestNode(Node):
    def __init__(self):
        super().__init__("franka_test_node")
        self._joint_received = False
        self._joint_pos_deg  = [0.0] * 7
        self.create_subscription(JointState, "/joint_states", self._cb_joints, 10)
        self._cmd_pub       = self.create_publisher(Float64MultiArray, POSITION_CMD_TOPIC, 10)
        self._homing_client = ActionClient(self, Homing, "/franka_gripper/homing")

    def _cb_joints(self, msg: JointState):
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        for arm_i, jname in enumerate(ARM_JOINT_NAMES):
            if jname in name_to_idx:
                self._joint_pos_deg[arm_i] = float(
                    np.degrees(msg.position[name_to_idx[jname]])
                )
        self._joint_received = True


def main():
    rclpy.init()
    node = FrankaTestNode()
    print("Franka Panda connectivity test")
    print("=" * 48)

    # ── Step 1: /joint_states ────────────────────────────────────────────────
    print("\n[1/4] Waiting for /joint_states (timeout 5 s)...")
    t0 = time.time()
    while not node._joint_received and time.time() - t0 < 5.0:
        rclpy.spin_once(node, timeout_sec=0.1)
    if not node._joint_received:
        print("[FAIL] /joint_states not received.")
        print("       Check: franka_ros2 launched, robot powered on and unlocked.")
        sys.exit(1)
    print("[OK]  /joint_states received. Current joints:")
    for name, deg in zip(ARM_JOINT_NAMES, node._joint_pos_deg):
        print(f"        {name}: {deg:+7.2f}°")

    # ── Step 2: forward_position_controller ─────────────────────────────────
    print(f"\n[2/4] Checking {POSITION_CMD_TOPIC}...")
    time.sleep(0.5)
    count = node._cmd_pub.get_subscription_count()
    if count == 0:
        print(f"[WARN] No subscribers on {POSITION_CMD_TOPIC}.")
        print("       Ensure forward_position_controller is loaded and active.")
        print("       Check: ros2 control list_controllers")
    else:
        print(f"[OK]  {count} subscriber(s) on {POSITION_CMD_TOPIC}.")

    # ── Step 3: Gripper action server ────────────────────────────────────────
    print("\n[3/4] Checking /franka_gripper/homing action server (timeout 5 s)...")
    available = node._homing_client.wait_for_server(timeout_sec=5.0)
    if not available:
        print("[WARN] /franka_gripper/homing not available.")
        print("       Is load_gripper:=true set in franka_bringup?")
    else:
        print("[OK]  franka_gripper action servers available.")

    # ── Step 4: Move to home ─────────────────────────────────────────────────
    print(f"\n[4/4] Moving to home position: {HOME_DEG} deg")
    print("      Ensure workspace is clear before proceeding.")
    choice = input("      Press Enter to move, or type 'Q' to abort: ")
    if choice.strip().lower() == "q":
        print("\n[ABORT] Cancelled by user.")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)

    home_rad = np.radians(HOME_DEG).tolist()
    msg = Float64MultiArray()
    msg.data = home_rad
    node._cmd_pub.publish(msg)
    print("[OK]  Home command sent. Watch the robot move slowly.")

    print("\nLive joint angles for 5 s...")
    header = "  " + "  ".join(f"{n[-6:]:>9}" for n in ARM_JOINT_NAMES)
    print(header)
    t0 = time.time()
    try:
        while time.time() - t0 < 5.0:
            rclpy.spin_once(node, timeout_sec=0.1)
            row = "  " + "  ".join(f"{v:+9.2f}" for v in node._joint_pos_deg)
            print(row, end="\r", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    print()

    print("\n[PASS] Franka test complete.")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
