#!/usr/bin/env bash
set -e

SCRIPT_PATH="${BASH_SOURCE[0]}"
while [ -L "${SCRIPT_PATH}" ]; do
    SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
    SCRIPT_PATH="$(readlink "${SCRIPT_PATH}")"
    if [[ "${SCRIPT_PATH}" != /* ]]; then
        SCRIPT_PATH="${SCRIPT_DIR}/${SCRIPT_PATH}"
    fi
done

ARC_ROOT="$(cd "$(dirname "${SCRIPT_PATH}")/../.." && pwd)"
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

if [ ! -f "${ARC_ROOT}/install/setup.bash" ]; then
    echo "install/setup.bash not found; run scripts/build.sh first" >&2
    exit 1
fi

source "${ARC_ROOT}/install/setup.bash"

exec ros2 launch arc_bringup map_pub.launch.py "$@"
