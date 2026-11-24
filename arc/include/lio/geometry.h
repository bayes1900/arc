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

inline bool estiPlane(
    Eigen::Matrix<float, 4, 1>& plane,
    const LioPointVector& points,
    float threshold
) {
    Eigen::Matrix<float, kNumMatchPoints, 3> a;
    Eigen::Matrix<float, kNumMatchPoints, 1> b;
    a.setZero();
    b.setOnes();
    b *= -1.0f;

    for (int j = 0; j < kNumMatchPoints; ++ j) {
        a(j, 0) = points[j].x;
        a(j, 1) = points[j].y;
        a(j, 2) = points[j].z;
    }

    const Eigen::Matrix<float, 3, 1> norm = a.colPivHouseholderQr().solve(b);
    const float length = norm.norm();
    plane(0) = norm(0) / length;
    plane(1) = norm(1) / length;
    plane(2) = norm(2) / length;
    plane(3) = 1.0f / length;

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

inline ImuPose setPose6D(
    double time,
    const Vec3d& acc,
    const Vec3d& gyr,
    const Vec3d& vel,
    const Vec3d& pos,
    const Mat3d& rot
) {
    ImuPose pose;
    pose.offsetTime = time;
    pose.acc = acc;
    pose.gyr = gyr;
    pose.vel = vel;
    pose.pos = pos;
    pose.rot = rot;
    return pose;
}

} // arc
