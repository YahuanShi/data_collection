# Data Collection

End-to-end robot demonstration data pipeline — from teleoperation to training-ready datasets.

**Platform:** Ubuntu 22.04 + ROS 2 Humble  
**Supported robots:** Universal Robots · Franka Panda *(pending hardware verification)*

---

## Getting Started

Run once on a new machine before using any part of this project:

```bash
bash install.sh
```

This installs ROS 2 Humble apt packages, creates the shared Python venv at `DaCo/`, installs all pip dependencies, builds the `uarm` ROS 2 package, and adds your user to the `dialout` group. **Re-login after** for serial port access.

After setup, entering this directory will automatically activate the environment via direnv (run `direnv allow` once).

---

## Project Structure

```
data_collection/
├── recorder/           # Camera publisher, episode recorder, replay viewer
├── uarm/               # ROS 2 package — UR and Franka teleoperation scripts
├── data_processing/    # HDF5 post-processing and training format conversion
├── install.sh          # One-shot setup (venv + ROS 2 + colcon build)
├── requirements.txt    # Unified Python dependencies
└── .envrc              # direnv: auto-activates DaCo + ROS 2 on cd
```

### Overall workflow

```
recorder/episode_recorder.py
      → dataset/raw/episode_N.hdf5
            │
            ├── [optional] data_processing/pipeline/pipeline.sh
            │             quality check · smooth · trim · resize
            │             → dataset/processed/resized/
            │
            ▼  (raw or processed HDF5)
            ├──► Pi0.5 / LeRobot   convert_ur5_data_to_lerobot.py
            ├──► ACT               HDF5 directly
            └──► Diffusion Policy  convert_hdf5_to_zarr.py → .zarr
```

---

## recorder & uarm — Teleoperation

### Hardware

#### Universal Robots setup

| Component | Hardware | Interface |
|---|---|---|
| Master arm | Uarm with Feetech or Zhonglin servos | USB serial |
| Follower arm | Universal Robots UR (any 6-axis model) | Ethernet (RTDE) |
| Gripper | Weiss Robotics CRG 30-050 | USB (`/dev/ttyACM0`) |
| Cameras | Intel RealSense D4xx × 2 or 3 (auto-detected) | USB |

#### Franka Panda setup *(pending hardware verification)*

| Component | Hardware | Interface |
|---|---|---|
| Master arm | Uarm with Feetech or Zhonglin servos | USB serial |
| Follower arm | Franka Panda (7-axis) | Ethernet (franka_ros2) |
| Gripper | Franka Hand (integrated) | franka_gripper action server |
| Cameras | Intel RealSense D4xx × 2 or 3 (auto-detected) | USB |

---

### Per-Machine Configuration

These values must be updated to match each machine's hardware before running.

#### Camera serial numbers — `recorder/cam_pub.py`

```python
SERIAL_1 = "105422061000"  # exterior camera (D415)  [required]
SERIAL_2 = "352122273671"  # wrist camera    (D405)  [required]
SERIAL_3 = "104122061227"  # front camera    (D415, USB 2.1) [optional — auto-detected]
```

Find your serial numbers:
```bash
python3 -c "import pyrealsense2 as rs; [print(d.get_info(rs.camera_info.serial_number), d.get_info(rs.camera_info.name)) for d in rs.context().query_devices()]"
```

#### UR constants — `uarm/scripts/UR/servo2ur.py` and `ur_pub.py`

| Constant | Default | Description |
|---|---|---|
| `UR_IP` | `10.0.0.1` | UR controller IP |
| `UR_HOME_DEG` | `[45,-20,-140,-40,-270,0]` | Home joint angles (deg) |
| `GRIPPER_PORT` | `/dev/ttyACM0` | CRG 30-050 USB port |
| `JOINT_MAP` | `[0,1,2,3,5,4]` | Master → UR joint index mapping |
| `JOINT_SCALE` | `[0.8,-1,-0.9,1,1,0.6]` | Sign/scale correction per joint |
| `MAX_JOINT_VEL_RAD` | `0.7` | Safety velocity cap (rad/s) |

> Keep `UR_IP` in sync between `servo2ur.py` and `ur_pub.py`.  
> `MAX_JOINT_VEL_RAD` hardware limit: UR3/UR5 ≈ 2.09 rad/s, UR10/UR16 ≈ 1.57 rad/s.

#### Franka constants — `uarm/scripts/Franka/servo2franka.py` *(pending hardware verification)*

| Constant | Default | Description |
|---|---|---|
| `FRANKA_HOME_DEG` | `[0,-45,0,-135,0,90,45]` | Home joint angles (deg) |
| `JOINT_MAP` | `[0,1,2,3,4,5,-1]` | Master → Panda joint index mapping (`-1` = fixed at home) |
| `JOINT_SCALE` | `[1,-1,1,-1,1,1,0]` | Sign/scale correction per joint |
| `MAX_JOINT_VEL_RAD` | `0.5` | Safety velocity cap (rad/s, Panda max ≈ 2.175) |
| `GRIPPER_SPEED` | `0.05` | Franka Hand open/close speed (m/s) |
| `GRIPPER_FORCE` | `20.0` | Franka Hand grasp force (N) |

---

### Quick Start

```bash
# Source colcon workspace (or use direnv — automatic on cd)
source install/setup.bash
```

#### Universal Robots

```bash
bash uarm/scripts/UR/run_ur_nodes.sh
```

#### Franka Panda — real hardware *(pending hardware verification)*

```bash
# Terminal 1: launch franka_ros2
ros2 launch franka_bringup franka.launch.py \
    robot_ip:=<IP> use_fake_hardware:=false load_gripper:=true

# Terminal 2: teleoperation pipeline
bash uarm/scripts/Franka/run_franka_nodes.sh
```

#### Franka Panda — Gazebo simulation *(pending hardware verification)*

```bash
# Terminal 1: Gazebo + controller setup
bash uarm/scripts/Franka/launch_gazebo.sh

# Terminal 2: teleoperation pipeline (after Terminal 1 prints "Gazebo simulation ready")
bash uarm/scripts/Franka/run_franka_nodes.sh
```

> **Gazebo PID tuning:** if joints are sluggish raise `p`; if they oscillate lower `p` or raise `d` in `uarm/scripts/Franka/config/franka_controllers.yaml`.

### Key operations

| Key | State | Action |
|-----|-------|--------|
| `P` | any | Set task language prompt *(once per session)* |
| `B` | WAITING | Begin recording *(requires prompt)* |
| `S` | RECORDING | Stop and save episode |
| `D` | WAITING | Delete last saved episode *(undo)* |
| `Q` | any | Quit |

---

### Node Graphs

#### Universal Robots

```
Uarm_teleop/servo_reader.py
        │
        │  /servo_angles  (Float64MultiArray, 7 — degrees)
        ▼
UR/servo2ur.py    ──→  /robot_action   (Float64MultiArray, 7)
                                        joints [deg ×6] + gripper [0-1]

UR/ur_pub.py      ──→  /robot_state    (Float64MultiArray, 7)
                  ──→  /robot_vel      (Float64MultiArray, 7)
                  ──→  /robot_eef_pose (Float64MultiArray, 6)

recorder/cam_pub.py  →  /cam_1, /cam_2 [, /cam_3]

recorder/episode_recorder.py
    subscribes: /cam_1, /cam_2, [/cam_3], /robot_state, /robot_vel,
                /robot_action, /robot_eef_pose
```

#### Franka Panda *(pending hardware verification)*

```
Uarm_teleop/servo_reader.py
        │
        │  /servo_angles  (Float64MultiArray, 7 — degrees)
        ▼
Franka/servo2franka.py  ──→  /forward_position_controller/commands  (7 rad)
                         ──→  /robot_action  (Float64MultiArray, 8)

Franka/franka_pub.py    ──→  /robot_state   (Float64MultiArray, 8)
                         ──→  /robot_vel     (Float64MultiArray, 8)
                         ──→  /robot_eef_pose (Float64MultiArray, 6)

franka_gripper  ←→  servo2franka.py  (Homing / Move / Grasp actions)

recorder/cam_pub.py  →  /cam_1, /cam_2 [, /cam_3]

recorder/episode_recorder.py
    subscribes: /cam_1, /cam_2, [/cam_3], /robot_state, /robot_vel,
                /robot_action, /robot_eef_pose
```

---

### Dataset Format

Episodes are saved as `episode_0.hdf5`, `episode_1.hdf5`, … in the configured dataset directory.

```
episode_N.hdf5
├── observations/
│   ├── images/
│   │   ├── exterior_image_1_left   (T, H, W, 3)  uint8   lzf   RGB
│   │   ├── wrist_image_left        (T, H, W, 3)  uint8   lzf   RGB
│   │   └── front_image_1           (T, H, W, 3)  uint8   lzf   RGB  [3-camera only]
│   ├── qpos                        (T, D)         float64 gzip
│   ├── qvel                        (T, D)         float64 gzip
│   └── eef_pose                    (T, 6)         float64 gzip
└── action                          (T, D)         float64 gzip

attrs: sim=False, prompt, task, hz, n_steps, timestamp, num_cameras
```

| Robot | D | `qpos`/`action` layout | `qvel` layout |
|---|---|---|---|
| UR (any) | 7 | `[joint_0…joint_5 (deg), gripper (0-1)]` | `[joint_0…joint_5 (deg/s), 0.0]` |
| Franka Panda | 8 | `[joint_0…joint_6 (deg), gripper (0-1)]` | `[joint_0…joint_6 (deg/s), 0.0]` |

- Gripper: `0 = closed, 1 = open`; last `qvel` index is always `0.0`
- `eef_pose`: `[x, y, z, rx, ry, rz]` — TCP pose in metres / radians
- Default recording resolution: **480 × 480** (square center-crop)

---

## data_processing — Post-processing

HDF5 post-processing pipeline and training format converters. All steps are optional — raw HDF5 from `recorder/` can be used directly for training.

See **[data_processing/README.md](data_processing/README.md)** for full documentation.

### Quick start

```bash
# Run full pipeline on a raw dataset
bash data_processing/pipeline/pipeline.sh path/to/raw_dataset_dir

# Convert processed HDF5 → Zarr for Diffusion Policy
python3 data_processing/convert_hdf5_to_zarr.py \
    --raw-dir  dataset/processed/resized/ur5_dataset_YYYYMMDD \
    --zarr-out dataset/zarr/ur5_dataset_YYYYMMDD.zarr
```

### Pipeline steps

| Step | Script | What it does |
|------|--------|-------------|
| 1 | `01_check_dataset.py` | Quality report — spikes, frozen gripper, short episodes |
| 2 | `02_drop_front_camera.py` | Remove front camera stream (optional) |
| 3 | `03_smooth_episodes.py` | Savitzky-Golay smoothing on `qpos` / `action` |
| 4 | `04_trim_episodes.py` | Cut idle frames at start / end |
| 5 | `05_resize_images.py` | Resize images to training resolution (default 224×224) |

### Framework conversion

| Framework | Tool | Input |
|-----------|------|-------|
| Pi0.5 / LeRobot | `convert_ur5_data_to_lerobot.py` | processed HDF5 |
| ACT | — | HDF5 directly |
| Diffusion Policy | `convert_hdf5_to_zarr.py` | processed HDF5 → Zarr |

---

## Folder Structure

```
data_collection/
├── install.sh                              # One-shot setup for a new machine
├── requirements.txt                        # Unified Python dependencies
├── .envrc                                  # direnv: auto-activates DaCo + ROS 2 + workspace
├── DaCo/                                  # Python venv (git-ignored)
│
├── recorder/                               # Robot-agnostic data tools
│   ├── episode_recorder.py                 # HDF5 dataset recorder
│   ├── cam_pub.py                          # Adaptive RealSense publisher (2 or 3 cams)
│   ├── replay_episode.py                   # Offline HDF5 viewer
│   └── test_cam.py                         # Camera hardware check
│
├── uarm/                                   # ROS 2 package (ament_python)
│   ├── package.xml
│   ├── setup.py / setup.cfg
│   └── scripts/
│       ├── UR/                             # Universal Robots (UR3/5/10/16)
│       │   ├── servo2ur.py                 # Teleop controller (60 Hz servoJ + Weiss gripper)
│       │   ├── ur_pub.py                   # State publisher
│       │   ├── joint_viz.py                # Real-time joint trajectory visualizer
│       │   ├── test_ur.py                  # Connectivity and motion test
│       │   ├── test_gripper.py             # Weiss CRG gripper test
│       │   ├── replay_on_robot.py          # Replay recorded episode on real UR5
│       │   └── run_ur_nodes.sh             # Launch all UR nodes
│       │
│       ├── Franka/                         # Franka Panda
│       │   ├── servo2franka.py             # Teleop controller (forward_position_controller)
│       │   ├── franka_pub.py               # State publisher (joint_states + TF2)
│       │   ├── joint_viz.py                # Real-time joint trajectory visualizer (8-ch)
│       │   ├── test_franka.py              # Connectivity and motion test
│       │   ├── test_gripper.py             # Franka Hand action test
│       │   ├── run_franka_nodes.sh         # Launch all Franka nodes
│       │   ├── launch_gazebo.sh            # Gazebo simulation + controller setup
│       │   └── config/
│       │       └── franka_controllers.yaml # forward_position_controller + Gazebo PID gains
│       │
│       ├── Uarm_teleop/
│       │   ├── Feetech_servo/              # Feetech servo master arm reader
│       │   └── Zhonglin_servo/             # Zhonglin servo master arm reader
│       └── add_permission.sh               # Grant serial port permissions
│
└── data_processing/                        # HDF5 post-processing
    ├── pipeline/                           # 5-step processing pipeline (optional)
    ├── viz/
    │   ├── viz_episode.py                  # Video + joint trajectory playback
    │   ├── viz_trajectory.py               # Before/after trajectory comparison
    │   └── viz_EEFin3D.py                  # 3D EEF trajectory viewer
    ├── convert_hdf5_to_zarr.py             # HDF5 → Zarr (Diffusion Policy)
    └── README.md
```
