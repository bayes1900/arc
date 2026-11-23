#pragma once

#include <base/base.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/register_point_struct.h>

namespace arc {

enum class LidarType {
    Generic = 0,
    Sim = 1,
    Ouster = 2,
    Avia = 3, // reserved for a node-side Livox CustomMsg adapter
};

struct LidarPoint {
    float x{0.0f};
    float y{0.0f};
    float z{0.0f};
    float intensity{0.0f};
    float time{0.0f};
    uint16_t ring{0};
    uint8_t tag{0};
};

struct LidarFrame {
    double stamp{0.0};
    double endTime{0.0};
    bool hasTime{false};
    std::vector<LidarPoint> points;
};

struct ImuSample {
    double time{0.0};
    Vec3d accel{Vec3d::Zero()};
    Vec3d gyro{Vec3d::Zero()};
};

struct EIGEN_ALIGN16 LioPoint {
    PCL_ADD_POINT4D;
    float intensity{0.0f};
    float time{0.0f};

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

using LioCloud = pcl::PointCloud<LioPoint>;
using LioPointVector = LioCloud::VectorType;

struct MeasurementGroup {
    double lidarBeginTime{0.0};
    double lidarEndTime{0.0};
    LioCloud::Ptr lidar;
    std::deque<ImuSample> imu;
};

struct ImuPose {
    double offsetTime{0.0};
    Vec3d acc{Vec3d::Zero()};
    Vec3d gyr{Vec3d::Zero()};
    Vec3d vel{Vec3d::Zero()};
    Vec3d pos{Vec3d::Zero()};
    Mat3d rot{Mat3d::Identity()};
};

struct LioOutput {
    bool valid{false};
    double stamp{0.0};
    Vec3d position{Vec3d::Zero()};
    Mat3d rotation{Mat3d::Identity()};
    Vec3d velocity{Vec3d::Zero()};
    MatXd covariance;
    LioCloud::Ptr cloud;
};

} // arc

POINT_CLOUD_REGISTER_POINT_STRUCT(arc::LioPoint,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (float, time, time)
)
