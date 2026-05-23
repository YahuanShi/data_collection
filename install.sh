#!/bin/bash
# One-shot setup for data_collection project.
# Run this once on a new machine before using either sub-project.
#
# Usage:
#   bash install.sh
#
# What it does:
#   1. Install ROS 2 Humble apt packages
#   2. Create Python venv at DaCo/  (project root)
#   3. Install all pip dependencies (requirements.txt)
#   4. Build the uarm ROS 2 package (teleoperation/uarm)
#   5. Add $USER to the dialout group (serial port access)
#   6. Install shell prompt hook (shows active venv in PS1)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/DaCo"

# ── 1. ROS 2 Humble (apt) ────────────────────────────────────────────────────
if [ -f /opt/ros/humble/setup.bash ]; then
    echo "[1/6] ROS 2 Humble already installed — skipping."
else
    echo "[1/6] Installing ROS 2 Humble apt packages..."
    sudo apt-get update -qq
    sudo apt-get install -y \
        ros-humble-desktop \
        python3-colcon-common-extensions \
        ros-humble-cv-bridge \
        ros-humble-sensor-msgs \
        ros-humble-std-msgs
fi

# ── 2. Python venv at DaCo/ ─────────────────────────────────────────────────
echo "[2/6] Creating Python venv at DaCo/ ..."
python3 -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

pip install --upgrade pip -q

# ── 3. pip dependencies ───────────────────────────────────────────────────────
echo "[3/6] Installing pip dependencies..."
pip install -r "$SCRIPT_DIR/requirements.txt"

# ── 4. Build ROS 2 package ────────────────────────────────────────────────────
echo "[4/6] Building uarm ROS 2 package..."
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
cd "$SCRIPT_DIR"
colcon build --packages-select uarm

# ── 5. Serial port permissions ────────────────────────────────────────────────
echo "[5/6] Adding $USER to dialout group (re-login required)..."
sudo usermod -aG dialout "$USER"

# ── 6. Shell prompt hook ──────────────────────────────────────────────────────
echo "[6/6] Adding virtualenv prompt hook to shell rc..."
HOOK='
# Show active direnv virtualenv in prompt (added by data_collection/install.sh)
_direnv_venv_ps1() { [[ -n "$VIRTUAL_ENV" && -n "$DIRENV_DIR" ]] && echo "($(basename "$VIRTUAL_ENV")) "; }
PS1='"'"'$(_direnv_venv_ps1)'"'"'$PS1'

SHELL_RC=""
if [[ "$SHELL" == */zsh ]]; then
    SHELL_RC="$HOME/.zshrc"
else
    SHELL_RC="$HOME/.bashrc"
fi

if ! grep -q "_direnv_venv_ps1" "$SHELL_RC" 2>/dev/null; then
    echo "$HOOK" >> "$SHELL_RC"
    echo "    Added prompt hook to $SHELL_RC"
else
    echo "    Prompt hook already present in $SHELL_RC"
fi

echo ""
echo "Setup complete. Next steps:"
echo "  1. Re-login (or run: newgrp dialout) for serial port access."
echo "  2. Reload shell:  source $SHELL_RC"
echo "  3. Allow direnv:  cd $SCRIPT_DIR && direnv allow"
echo "     cd into data_collection/ will now activate (DaCo) automatically."
