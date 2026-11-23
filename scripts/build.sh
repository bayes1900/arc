#!/usr/bin/env bash
set -e

ARC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ARC_ROOT}"

unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE CONDA_PROMPT_MODIFIER CONDA_SHLVL

if [ -n "${ROS_SETUP:-}" ]; then
    export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    source "${ROS_SETUP}"
elif [ -f "/opt/ros/humble/setup.bash" ]; then
    export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    source "/opt/ros/humble/setup.bash"
elif [ -f "${HOME}/workspace/pixi/ros2_ws/.pixi/envs/default/setup.bash" ]; then
    export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    source "${HOME}/workspace/pixi/ros2_ws/.pixi/envs/default/setup.bash"
fi

if ! command -v colcon >/dev/null 2>&1; then
    echo "colcon not found; source a ROS 2 environment or set ROS_SETUP." >&2
    exit 1
fi

Python3Executable="${Python3_EXECUTABLE:-$(command -v python3)}"

colcon build --symlink-install \
  --cmake-args \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DPython3_EXECUTABLE="${Python3Executable}" \
  --parallel-workers "${COLCON_WORKERS:-14}"

"${Python3Executable}" scripts/merge_compile_commands.py
