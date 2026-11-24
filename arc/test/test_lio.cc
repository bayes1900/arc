#include <gtest/gtest.h>

#include <lio/lio.h>

#include <chrono>
#include <cmath>
#include <thread>
#include <vector>

namespace {

struct Pose {
    Eigen::Vector3d position;
    Eigen::Matrix3d rotation;
};

std::vector<Eigen::Vector3d> makeRoom() {
    std::vector<Eigen::Vector3d> points;
    const double step = 0.3;
    const double xMax = 6.0;
    const double yMax = 6.0;
    const double zMax = 3.0;
    for (double x = -xMax; x <= xMax; x += step) {
        for (double y = -yMax; y <= yMax; y += step) {
            points.emplace_back(x, y, 0.0);
            points.emplace_back(x, y, zMax);
        }
    }
    for (double x = -xMax; x <= xMax; x += step) {
        for (double z = 0.0; z <= zMax; z += step) {
            points.emplace_back(x, -yMax, z);
            points.emplace_back(x, yMax, z);
        }
    }
    for (double y = -yMax; y <= yMax; y += step) {
        for (double z = 0.0; z <= zMax; z += step) {
            points.emplace_back(-xMax, y, z);
            points.emplace_back(xMax, y, z);
        }
    }
    return points;
}

Pose poseAt(double time) {
    Pose pose;
    pose.position = Eigen::Vector3d(0.0, 0.0, 1.0);
    if (time > 1.0) {
        pose.position.x() = time - 1.0;
    }
    pose.rotation.setIdentity();
    return pose;
}

arc::LidarFrame makeScan(const std::vector<Eigen::Vector3d>& world, double time) {
    arc::LidarFrame frame;
    frame.stamp = time;
    frame.endTime = time + 0.1;
    frame.hasTime = true;
    for (const Eigen::Vector3d& point : world) {
        const double azimuth = std::atan2(point.y(), point.x());
        const double fraction = azimuth < 0.0
            ? (azimuth + 2.0 * M_PI) / (2.0 * M_PI)
            : azimuth / (2.0 * M_PI);
        const double pointTime = fraction * 0.1;
        const Pose pose = poseAt(time + pointTime);
        const Eigen::Vector3d local = pose.rotation.transpose() * (point - pose.position);
        const double range = local.norm();
        if (range > 12.0 || range < 0.3) {
            continue;
        }
        arc::LidarPoint lidar;
        lidar.x = static_cast<float>(local.x());
        lidar.y = static_cast<float>(local.y());
        lidar.z = static_cast<float>(local.z());
        lidar.intensity = 100.0f;
        lidar.time = static_cast<float>(pointTime);
        const double elevation = std::asin(local.z() / range);
        lidar.ring = static_cast<uint16_t>(std::clamp<int>(
            static_cast<int>((elevation + 0.4) / 0.8 * 15.0), 0, 15));
        frame.points.push_back(lidar);
    }
    return frame;
}

arc::LioConfig simConfig() {
    arc::LioConfig config;
    config.lidarType = arc::LidarType::Generic;
    config.scanLine = 16;
    config.scanRate = 10;
    config.blind = 0.3;
    config.pointFilterNum = 3;
    config.filterSizeSurf = 0.4;
    config.filterSizeMap = 0.4;
    config.cubeSideLength = 40.0;
    config.detRange = 15.0;
    config.maxIterations = 4;
    return config;
}

} // namespace

TEST(LioMath, So3ExpMatchesAngleAxis) {
    const Eigen::Vector3d angle(0.2, -0.1, 0.3);
    const Eigen::Matrix3d rotation = arc::manifold::so3Exp(angle);
    const Eigen::Matrix3d expected = Eigen::AngleAxisd(
        angle.norm(), angle.normalized()).toRotationMatrix();
    EXPECT_TRUE(rotation.isApprox(expected, 1e-9));
}

TEST(LioManifold, So3BoxPlusMinus) {
    const Eigen::Vector3d delta(0.02, -0.01, 0.03);
    arc::manifold::ManifoldSo3<double> orientation;
    orientation.boxPlus(delta, 1.0);
    Eigen::Vector3d recovered;
    orientation.boxMinus(recovered, arc::manifold::ManifoldSo3<double>());
    EXPECT_TRUE(recovered.isApprox(delta, 1e-9));
}

TEST(LioManifold, S2BoxPlusMinus) {
    using S2 = arc::manifold::ManifoldS2<double, 98090, 10000, 1>;
    const S2 origin;
    const Eigen::Vector2d delta(0.01, -0.02);
    S2 direction = origin;
    direction.boxPlus(delta, 1.0);
    Eigen::Vector2d recovered;
    direction.boxMinus(recovered, origin);
    EXPECT_TRUE(recovered.isApprox(delta, 1e-9));
    EXPECT_NEAR(direction.vector().norm(), S2::LENGTH, 1e-9);
}

TEST(LioState, BoxPlusMinus) {
    arc::LioState origin;
    arc::LioState state = origin;
    Eigen::Matrix<double, arc::LioState::DOF, 1> delta =
        Eigen::Matrix<double, arc::LioState::DOF, 1>::Zero();
    delta.segment<3>(0) = Eigen::Vector3d(0.1, -0.2, 0.3);
    delta.segment<3>(3) = Eigen::Vector3d(0.01, 0.02, -0.03);
    delta.segment<3>(6) = Eigen::Vector3d(-0.01, 0.02, 0.01);
    delta.segment<3>(9) = Eigen::Vector3d(0.2, 0.1, -0.1);
    delta.segment<3>(12) = Eigen::Vector3d(0.3, -0.2, 0.1);
    delta.segment<3>(15) = Eigen::Vector3d(0.001, -0.002, 0.003);
    delta.segment<3>(18) = Eigen::Vector3d(-0.001, 0.002, -0.003);
    delta.segment<2>(21) = Eigen::Vector2d(0.01, -0.02);
    state.boxPlus(delta, 1.0);
    Eigen::Matrix<double, arc::LioState::DOF, 1> recovered;
    state.boxMinus(recovered, origin);
    EXPECT_TRUE(recovered.isApprox(delta, 1e-9));
}

TEST(LioMath, PlaneFit) {
    arc::LioPointVector points;
    points.resize(5);
    for (int i = 0; i < 5; ++ i) {
        points[static_cast<size_t>(i)].x = static_cast<float>(i);
        points[static_cast<size_t>(i)].y = static_cast<float>(i * i % 3);
        points[static_cast<size_t>(i)].z = 1.0f;
    }
    Eigen::Matrix<float, 4, 1> plane;
    EXPECT_TRUE(arc::estiPlane(plane, points, 0.1f));
    EXPECT_NEAR(std::fabs(plane(2)), 1.0f, 1e-5f);
}

TEST(LioPreprocessor, TimedCloudKeepsTime) {
    arc::LioConfig config;
    config.lidarType = arc::LidarType::Generic;
    config.blind = 0.5;
    config.pointFilterNum = 1;
    arc::PointPreprocessor preprocessor(config);

    arc::LidarFrame frame;
    frame.hasTime = true;
    arc::LidarPoint inside;
    inside.x = 1.0f;
    inside.time = 0.05f;
    arc::LidarPoint outside;
    outside.x = 0.1f;
    outside.time = 0.02f;
    frame.points = {inside, outside};

    arc::LioCloud cloud;
    preprocessor.process(frame, cloud);
    ASSERT_EQ(cloud.size(), 1u);
    EXPECT_FLOAT_EQ(cloud.points[0].time, 0.05f);
}

TEST(LioPreprocessor, UntimedCloudInfersSeconds) {
    arc::LioConfig config;
    config.lidarType = arc::LidarType::Generic;
    config.scanLine = 16;
    config.scanRate = 10;
    config.blind = 0.0;
    config.pointFilterNum = 1;
    arc::PointPreprocessor preprocessor(config);

    arc::LidarFrame frame;
    frame.hasTime = false;
    const auto addPoint = [&frame](float x, float y) {
        arc::LidarPoint point;
        point.x = x;
        point.y = y;
        point.ring = 0;
        frame.points.push_back(point);
    };
    addPoint(1.0f, 0.0f);
    addPoint(0.0f, -1.0f);
    addPoint(-1.0f, 0.0f);

    arc::LioCloud cloud;
    preprocessor.process(frame, cloud);
    ASSERT_EQ(cloud.size(), 2u);
    EXPECT_NEAR(cloud.points[0].time, 0.025f, 1e-5f);
    EXPECT_NEAR(cloud.points[1].time, 0.05f, 1e-5f);
}

TEST(LioState, ProcessModel) {
    arc::LioState state;
    state.pos = arc::LioVect3(Eigen::Vector3d(1.0, 2.0, 3.0));
    state.rot = arc::LioSo3(Eigen::Matrix3d::Identity());
    state.vel = arc::LioVect3(Eigen::Vector3d(0.5, 0.0, 0.0));
    state.grav = arc::LioS2(arc::LioVect3(Eigen::Vector3d(0.0, 0.0, -9.81)));

    arc::LioInput input;
    input.acc = arc::LioVect3(Eigen::Vector3d(0.0, 0.0, 9.81));
    input.gyro = arc::LioVect3(Eigen::Vector3d::Zero());

    const Eigen::Matrix<double, 24, 1> f = arc::lioGetF(state, input);
    EXPECT_DOUBLE_EQ(f(0), 0.5);
    EXPECT_DOUBLE_EQ(f(1), 0.0);
    EXPECT_DOUBLE_EQ(f(12), 0.0);

    const arc::LioProcessNoiseCov cov = arc::processNoiseCov();
    EXPECT_EQ(cov.rows(), 12);
    EXPECT_EQ(cov.cols(), 12);
}

TEST(LioMap, BuildAndSearch) {
    arc::LioConfig config;
    config.filterSizeMap = 0.3;
    arc::IncrementalMap map(config);

    arc::LioPointVector points;
    for (int i = 0; i < 3; ++ i) {
        arc::LioPoint point;
        point.x = static_cast<float>(i);
        point.y = 0.0f;
        point.z = 0.0f;
        point.intensity = 1.0f;
        points.push_back(point);
    }
    map.build(points);
    EXPECT_FALSE(map.empty());
    EXPECT_EQ(map.size(), 3);

    arc::LioPoint query;
    query.x = 0.1f;
    query.y = 0.0f;
    query.z = 0.0f;
    arc::LioPointVector nearest;
    std::vector<float> distances;
    map.nearestSearch(query, 3, nearest, distances);
    ASSERT_EQ(nearest.size(), 3u);
    EXPECT_NEAR(nearest[0].x, 0.0f, 1e-5f);
}

TEST(LioEstimator, SyntheticRoom) {
    const std::vector<Eigen::Vector3d> world = makeRoom();
    arc::LioConfig config = simConfig();
    arc::LioPipeline pipeline(config);
    pipeline.start();

    const double imuStep = 0.005;
    double imuTime = 0.0;
    for (int scan = 0; scan < 30; ++ scan) {
        const double time = scan * 0.1;
        pipeline.addLidar(makeScan(world, time));
        while (imuTime <= time + 0.1 + 1e-9) {
            arc::ImuSample imu;
            imu.time = imuTime;
            imu.accel = Eigen::Vector3d(0.0, 0.0, 9.81);
            imu.gyro = Eigen::Vector3d::Zero();
            pipeline.addImu(imu);
            imuTime += imuStep;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    std::this_thread::sleep_for(std::chrono::seconds(2));
    arc::LioOutput output;
    ASSERT_TRUE(pipeline.getOutput(output));
    EXPECT_TRUE(output.valid);
    const double groundTruthX = std::max(0.0, output.stamp - 1.0);
    EXPECT_NEAR(output.position.x(), groundTruthX, 0.4);
    EXPECT_NEAR(output.position.z(), 0.0, 0.3);
    pipeline.stop();
}
