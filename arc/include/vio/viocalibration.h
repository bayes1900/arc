#pragma once

#include <base/base.h>
#include <vio/viocalibbias.h>
#include <vio/camera/base.h>

namespace arc {

template <class Scalar>
struct Calibration {
    using Ptr = std::shared_ptr<Calibration>;
    using SE3 = Sophus::SE3<Scalar>;
    using Vec3 = Eigen::Matrix<Scalar, 3, 1>;

    std::vector<SE3> tIC;
    std::vector<GenericCamera<Scalar>> intrinsics;
    std::vector<Eigen::Vector2i> resolution;
    int64_t camTimeOffsetNs;
    CalibAccelBias<Scalar> calibAccelBias;
    CalibGyroBias<Scalar> calibGyroBias;
    Scalar imuUpdateRate;
    Vec3 gyroNoiseStd;
    Vec3 accelNoiseStd;
    Vec3 gyroBiasStd;
    Vec3 accelBiasStd;

    Calibration() {
        camTimeOffsetNs = 0;
        imuUpdateRate = 200;
        gyroNoiseStd.setConstant(0.000282);
        accelNoiseStd.setConstant(0.016);
        accelBiasStd.setConstant(0.001);
        gyroBiasStd.setConstant(0.0001);
    }
    template <class Scalar2>
    Calibration<Scalar2> cast() const {
        Calibration<Scalar2> newCam;
        for (const auto& v : tIC) {
            newCam.tIC.emplace_back(v.template cast<Scalar2>());
        }
        for (const auto& v : intrinsics) {
            newCam.intrinsics.emplace_back(v.template cast<Scalar2>());
        }
        newCam.resolution = resolution;
        newCam.camTimeOffsetNs = camTimeOffsetNs;
        newCam.calibAccelBias.getParam() = calibAccelBias.getParam().template cast<Scalar2>();
        newCam.calibGyroBias.getParam() = calibGyroBias.getParam().template cast<Scalar2>();
        newCam.imuUpdateRate = imuUpdateRate;
        newCam.gyroNoiseStd = gyroNoiseStd.template cast<Scalar2>();
        newCam.accelNoiseStd = accelNoiseStd.template cast<Scalar2>();
        newCam.gyroBiasStd = gyroBiasStd.template cast<Scalar2>();
        newCam.accelBiasStd = accelBiasStd.template cast<Scalar2>();
        return newCam;
    }
    Vec3 dicreteTimeGyroNoiseStd() const {
        return gyroNoiseStd * std::sqrt(imuUpdateRate);
    }
    Vec3 dicreteTimeAccelNoiseStd() const {
        return accelNoiseStd * std::sqrt(imuUpdateRate);
    }
};

} // arc
