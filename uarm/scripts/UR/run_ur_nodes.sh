#!/bin/bash
# run_ur_nodes.sh
# Launches the full UR + Weiss Robotics CRG 30-050 teleoperation pipeline.
#
# Node graph:
#   Uarm_teleop/servo_reader  →  /servo_angles  →  servo2ur.py  →  /robot_action
#   ur_pub.py                 →  /robot_state
#   cam_pub.py                →  /cam_1, /cam_2 [, /cam_3 if front camera connected]
#   episode_recorder.py  (subscribes to /robot_action, /robot_state, /cam_*)
#   joint_viz.py         (subscribes to /servo_angles, /robot_state, /is_recording)
#
# Window layout (1920×1080 monitor — 3-camera mode):
#   ┌─────────────────────────────────────────────────────────┐
#   │  Recorder  [exterior|wrist|front]  (30,30)  1920×596   │
#   ├─────────────────────────────────────────────────────────┤
#   │  Joint Trajectories               (30,660)  1300×380   │
#   └─────────────────────────────────────────────────────────┘
# In 2-camera mode the Recorder window is 1280×596.
#
# Usage:
#   bash run_ur_nodes.sh
#
# Requirements:
#   Ubuntu 22.04 + ROS 2 Humble
#   pip install ur-rtde pyserial pyrealsense2 opencv-python h5py numpy
#   UR reachable at UR_IP (edit servo2ur.py and ur_pub.py)  # ← [ADJUST] IP per robot
#   CRG 30-050 USB cable connected → /dev/ttyACM0 (edit GRIPPER_PORT in both .py files)
#   Serial port permission: sudo usermod -aG dialout $USER  (then re-login)

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
UARM_SCRIPTS="$SCRIPT_DIR/.."                  # → .../data_collection/uarm/scripts
DATA_COLLECTION="$UARM_SCRIPTS/../../recorder" # → .../data_collection/recorder

# ── ROS 2 environment ────────────────────────────────────────────────────────
source /opt/ros/humble/setup.bash

# Source colcon workspace overlay if built (adjust path if needed)
if [ -f "$UARM_SCRIPTS/../../install/setup.bash" ]; then
    source "$UARM_SCRIPTS/../../install/setup.bash"
fi

echo "[INFO] ROS_DOMAIN_ID: ${ROS_DOMAIN_ID:-0}"
echo "[INFO] Starting UR teleoperation nodes..."

# ── 1. Camera publisher ───────────────────────────────────────────────────────
python3 "$DATA_COLLECTION/cam_pub.py" &
PID_CAM=$!
echo "[INFO] cam_pub.py  → PID $PID_CAM"

# ── 2. UR state publisher ────────────────────────────────────────────────────
python3 "$SCRIPT_DIR/ur_pub.py" &
PID_STATE=$!
echo "[INFO] ur_pub.py  → PID $PID_STATE"

# ── 3. Uarm master-arm servo reader ─────────────────────────────────────────
python3 "$UARM_SCRIPTS/Uarm_teleop/Zhonglin_servo/servo_reader.py" &
PID_SERVO=$!
echo "[INFO] servo_reader.py (Zhonglin)  → PID $PID_SERVO"

# ── 4. UR + Weiss CRG 30-050 teleoperation controller ───────────────────────
python3 "$SCRIPT_DIR/servo2ur.py" &
PID_TELEOP=$!
echo "[INFO] servo2ur.py  → PID $PID_TELEOP"

# ── 5. Episode recorder ──────────────────────────────────────────────────────
python3 "$DATA_COLLECTION/episode_recorder.py" &
PID_REC=$!
echo "[INFO] episode_recorder.py  → PID $PID_REC"

# ── 6. Joint trajectory visualizer ──────────────────────────────────────────
# python3 "$SCRIPT_DIR/joint_viz.py" &
# PID_VIZ=$!
# echo "[INFO] joint_viz.py  → PID $PID_VIZ"

# ── Cleanup on Ctrl+C ────────────────────────────────────────────────────────
trap "echo '';
      echo '[INFO] Ctrl+C — shutting down all nodes...';
      kill $PID_CAM $PID_STATE $PID_SERVO $PID_TELEOP $PID_REC 2>/dev/null;
      exit 0" SIGINT SIGTERM

echo ""
echo "[INFO] All nodes running. Press Ctrl+C to stop."
wait
