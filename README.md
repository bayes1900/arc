# ARC

A compact ROS 2 stack for autonomous drone exploration, planning, mapping, and LiDAR/visual-inertial odometry.

## Packages

| Package | Description |
|---|---|
| `arc` | Core header-only libraries: exploration, planning, mapping, VIO, LIO |
| `arcmsgs` | ROS 2 messages and services |
| `arc_sim` | Simulation and runtime nodes: odometry, LiDAR render, LIO/VIO, exploration |
| `arc_bringup` | Launch files, parameters, RViz configuration |

## Requirements

- Ubuntu 22.04 and ROS 2 Humble
- C++17 and `colcon`
- PCL, Eigen3, OpenCV, NLopt, TBB, Sophus, fmt, yaml-cpp, nlohmann_json, OpenMP

## Build

```bash
bash scripts/build.sh
```

The script sources ROS 2 Humble (or `$ROS_SETUP`) and runs `colcon build --symlink-install`.

## Test

```bash
bash scripts/test.sh
```

## Run

Source the workspace first:

```bash
source install/setup.bash
```

Map and odometry demo:

```bash
ros2 launch arc_bringup map_pub.launch.py
```

Single-drone exploration:

```bash
ros2 launch arc_bringup exploration.launch.py
```

Swarm exploration:

```bash
ros2 launch arc_bringup swarm_exploration.launch.py drone_num:=4
```

Add `rviz:=false` for headless runs.

Optional LIO/VIO in the swarm launch:

```bash
ros2 launch arc_bringup swarm_exploration.launch.py use_lio:=true
ros2 launch arc_bringup swarm_exploration.launch.py use_vio:=true
```

Use `use_lio_for_planning:=true` or `use_vio_for_planning:=true` to feed the estimate back into planning instead of ground truth.
