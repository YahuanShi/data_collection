#!/usr/bin/env python3
"""
test_gripper.py - Franka Hand gripper test (no additional setup beyond franka_ros2).

Steps:
    1. Check franka_gripper action servers are available
    2. Home the gripper (closes then opens fully)
    3. Open fully
    4. Close (grasp at low force)
    5. Open again

Usage:
    python3 test_gripper.py
    # Requires franka_ros2 launched with load_gripper:=true

Protocol:
    - Open  : Move action  (position control to GRIPPER_MAX_WIDTH)
    - Close : Grasp action (force control)
    - Home  : Homing action (calibration, closes then opens)
"""

import sys

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

try:
    from franka_msgs.action import Grasp, Homing, Move
except ImportError:
    print("[FAIL] franka_msgs not installed. Run: apt install ros-humble-franka-msgs")
    sys.exit(1)

# ← [ADJUST] tune to your application
GRIPPER_MAX_WIDTH = 0.08   # m, Panda Hand maximum spread
GRIPPER_SPEED     = 0.05   # m/s
GRIPPER_FORCE     = 10.0   # N,  keep low for this test
GRIPPER_EPSILON   = 0.01   # m,  position tolerance for grasp


class GripperTestNode(Node):
    def __init__(self):
        super().__init__("franka_gripper_test")
        self._homing = ActionClient(self, Homing, "/franka_gripper/homing")
        self._move   = ActionClient(self, Move,   "/franka_gripper/move")
        self._grasp  = ActionClient(self, Grasp,  "/franka_gripper/grasp")

    def wait_servers(self, timeout: float = 10.0) -> bool:
        return all(
            c.wait_for_server(timeout_sec=timeout)
            for c in [self._homing, self._move, self._grasp]
        )

    def _send_sync(self, client, goal, label: str, timeout: float = 15.0) -> bool:
        """Send action goal and block until result."""
        future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        handle = future.result()
        if not handle or not handle.accepted:
            print(f"[FAIL] {label}: goal rejected")
            return False
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=timeout)
        print(f"[OK]  {label} complete.")
        return True

    def home(self) -> bool:
        return self._send_sync(self._homing, Homing.Goal(), "Homing")

    def open(self) -> bool:
        goal = Move.Goal()
        goal.width = GRIPPER_MAX_WIDTH
        goal.speed = GRIPPER_SPEED
        return self._send_sync(self._move, goal, f"Open (width={GRIPPER_MAX_WIDTH} m)")

    def close(self) -> bool:
        goal = Grasp.Goal()
        goal.width = 0.0
        goal.speed = GRIPPER_SPEED
        goal.force = GRIPPER_FORCE
        goal.epsilon.inner = GRIPPER_EPSILON
        goal.epsilon.outer = GRIPPER_EPSILON
        return self._send_sync(self._grasp, goal, f"Grasp (force={GRIPPER_FORCE} N)")


def main():
    rclpy.init()
    node = GripperTestNode()
    print("Franka Hand gripper test")
    print("=" * 48)

    print("\n[1/5] Checking action servers (timeout 10 s)...")
    if not node.wait_servers():
        print("[FAIL] Gripper action servers not available.")
        print("       Is franka_ros2 running with load_gripper:=true?")
        sys.exit(1)
    print("[OK]  All gripper action servers available.")

    print("\n[2/5] Homing (closes then opens fully, ~5 s)...")
    node.home()

    print("\n[3/5] Opening fully...")
    node.open()

    print("\n[4/5] Closing (grasp)...")
    node.close()

    print("\n[5/5] Opening again...")
    node.open()

    print("\n[PASS] Franka Hand test complete.")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
