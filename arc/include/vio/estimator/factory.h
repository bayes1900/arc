#pragma once

#include <vio/estimator/base.h>
#include <vio/estimator/sqrtkeypoint.h>

namespace arc {

class VioEstimatorFactory {
public:
    static VioEstimatorBase::Ptr getVioEstimator(const VioConfig& config, const Calibration<double>& cam, const Eigen::Vector3d& g);
};

inline VioEstimatorBase::Ptr VioEstimatorFactory::getVioEstimator(
    const VioConfig& config, const Calibration<double>& cam, const Eigen::Vector3d& g) {
    if (cam.intrinsics.empty() || cam.intrinsics.size() != cam.resolution.size() ||
        cam.intrinsics.size() != cam.tIC.size()) {
        throw std::invalid_argument("invalid VIO calibration");
    }
    if (!g.allFinite() || g.norm() < 1e-6) {
        throw std::invalid_argument("invalid gravity vector");
    }
    return std::make_shared<SqrtKeypointVioEstimator>(g, cam, config);
}

} // arc
