#!/usr/bin/env bash
set -e

ARC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ARC_ROOT}"

unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE CONDA_PROMPT_MODIFIER CONDA_SHLVL
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/ros/humble/bin"
export LD_LIBRARY_PATH="/opt/ros/humble/lib"

source /opt/ros/humble/setup.bash
source install/setup.bash

colcon test --event-handlers console_direct+
colcon test-result --verbose
