#!/bin/bash
# launch_gazebo.sh
# Sets up Franka Panda Gazebo simulation and activates forward_position_controller.
#
# Run this INSTEAD of (not alongside) the real-hardware franka_bringup launch.
# After this script is running, use run_franka_nodes.sh as normal.
#
# Terminal layout:
#   Terminal 1: bash launch_gazebo.sh       ← Gazebo + controller setup (this script)
#   Terminal 2: bash run_franka_nodes.sh    ← teleoperation pipeline
#
# One-time install:
#   sudo apt install ros-humble-franka-gazebo \
#                    ros-humble-ros2-controllers \
#                    ros-humble-gazebo-ros2-control
#
# Troubleshooting:
#   Joints oscillate  → lower p gain or raise d in config/franka_controllers.yaml
#   Joints sluggish   → raise p gain in config/franka_controllers.yaml
#   Controller fails  → check: ros2 control list_controllers

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CONTROLLER_CONFIG="$SCRIPT_DIR/config/franka_controllers.yaml"

source /opt/ros/humble/setup.bash

UARM_SCRIPTS="$SCRIPT_DIR/.."
if [ -f "$UARM_SCRIPTS/../../install/setup.bash" ]; then
    source "$UARM_SCRIPTS/../../install/setup.bash"
fi

echo "[INFO] Controller config: $CONTROLLER_CONFIG"
echo "[INFO] Starting Franka Panda Gazebo simulation..."

# ── 1. Launch Gazebo with Panda and ros2_control ──────────────────────────────
# franka_gazebo.launch.py starts Gazebo, loads the Panda URDF with
# gazebo_ros2_control plugin, and spawns joint_state_broadcaster by default.
ros2 launch franka_gazebo franka_gazebo.launch.py \
    load_gripper:=true &
PID_GAZEBO=$!
echo "[INFO] Gazebo PID: $PID_GAZEBO"

# ── 2. Wait for controller_manager to become available ────────────────────────
echo "[INFO] Waiting for controller_manager..."
until ros2 service list 2>/dev/null | grep -q "/controller_manager/list_controllers"; do
    sleep 1
done
echo "[INFO] controller_manager ready."
sleep 2  # extra buffer for all controllers to register

# ── 3. Activate joint_state_broadcaster (if not already active) ───────────────
echo "[INFO] Activating joint_state_broadcaster..."
ros2 control set_controller_state joint_state_broadcaster active 2>/dev/null || \
ros2 control load_controller --set-state active joint_state_broadcaster
sleep 1

# ── 4. Load and activate forward_position_controller ─────────────────────────
echo "[INFO] Loading forward_position_controller..."
ros2 control load_controller forward_position_controller \
    --param-file "$CONTROLLER_CONFIG"
sleep 1

echo "[INFO] Activating forward_position_controller..."
ros2 control set_controller_state forward_position_controller active
sleep 1

# ── 5. Verify ─────────────────────────────────────────────────────────────────
echo ""
echo "[INFO] Active controllers:"
ros2 control list_controllers

echo ""
echo "[INFO] Gazebo simulation ready."
echo "[INFO] Now run in a separate terminal:"
echo "         bash $SCRIPT_DIR/run_franka_nodes.sh"
echo ""
echo "[INFO] Press Ctrl+C to shut down Gazebo."

# ── Cleanup ───────────────────────────────────────────────────────────────────
trap "echo '';
      echo '[INFO] Shutting down Gazebo...';
      kill $PID_GAZEBO 2>/dev/null;
      exit 0" SIGINT SIGTERM

wait $PID_GAZEBO
