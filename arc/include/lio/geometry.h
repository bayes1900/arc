#pragma once

#include <base/base.h>
#include <lio/liotypes.h>

namespace arc {

inline constexpr int kNumMatchPoints = 5;
inline constexpr double kGravity = 9.81;

inline float calcDist(const LioPoint& p1, const LioPoint& p2) {
    return (p1.x - p2.x) * (p1.x - p2.x)
        + (p1.y - p2.y) * (p1.y - p2.y)
        + (p1.z - p2.z) * (p1.z - p2.z);
}

template <typename T>
inline bool estiPlane(
    Eigen::Matrix<T, 4, 1>& plane,
    const LioPointVector& points,
    const T& threshold
) {
    Eigen::Matrix<T, kNumMatchPoints, 3> a;
    Eigen::Matrix<T, kNumMatchPoints, 1> b;
    a.setZero();
    b.setOnes();
    b *= T(-1);

    for (int j = 0; j < kNumMatchPoints; ++ j) {
        a(j, 0) = points[j].x;
        a(j, 1) = points[j].y;
        a(j, 2) = points[j].z;
    }

    const Eigen::Matrix<T, 3, 1> norm = a.colPivHouseholderQr().solve(b);
    const T length = norm.norm();
    plane(0) = norm(0) / length;
    plane(1) = norm(1) / length;
    plane(2) = norm(2) / length;
    plane(3) = T(1) / length;

    for (int j = 0; j < kNumMatchPoints; ++ j) {
        if (std::fabs(
                plane(0) * points[j].x
                + plane(1) * points[j].y
                + plane(2) * points[j].z
                + plane(3)) > threshold) {
            return false;
        }
    }
    return true;
}

template <typename T>
inline ImuPose setPose6D(
    double time,
    const Eigen::Matrix<T, 3, 1>& acc,
    const Eigen::Matrix<T, 3, 1>& gyr,
    const Eigen::Matrix<T, 3, 1>& vel,
    const Eigen::Matrix<T, 3, 1>& pos,
    const Eigen::Matrix<T, 3, 3>& rot
) {
    ImuPose pose;
    pose.offsetTime = time;
    pose.acc = acc.template cast<double>();
    pose.gyr = gyr.template cast<double>();
    pose.vel = vel.template cast<double>();
    pose.pos = pos.template cast<double>();
    pose.rot = rot.template cast<double>();
    return pose;
}

} // arc
