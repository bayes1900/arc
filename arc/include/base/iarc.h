#pragma once

#include <base/base.h>
#include <base/misc.h>

namespace arc {

struct OdomState {
    Eigen::Vector3d pos{Eigen::Vector3d::Zero()};
    Eigen::Vector3d vel{Eigen::Vector3d::Zero()};
    Eigen::Vector3d acc{Eigen::Vector3d::Zero()};
    Eigen::Vector3d yaw{Eigen::Vector3d::Zero()};
};

} // arc
