#include <gtest/gtest.h>

#include <vio/vio.h>


int vioOdrHelper();

namespace {

TEST(Vio, ConfigDefaults) {
    const arc::VioConfig config;
    EXPECT_EQ(config.opticalFlowType, "frame_to_frame");
    EXPECT_EQ(config.opticalFlowDetectionGridSize, 50);
    EXPECT_EQ(config.opticalFlowPattern, 51);
    EXPECT_EQ(config.vioMaxStates, 3);
    EXPECT_EQ(config.vioMaxKfs, 7);
    EXPECT_EQ(config.vioLinearizationType, arc::LinearizationType::ABS_QR);
    EXPECT_TRUE(config.vioSqrtMarg);
}

TEST(Vio, CalibrationDefaults) {
    const arc::Calibration<double> calibration;
    EXPECT_EQ(calibration.imuUpdateRate, 200.0);
    EXPECT_EQ(calibration.camTimeOffsetNs, 0);
    EXPECT_NEAR(calibration.gyroNoiseStd(0), 0.000282, 1e-12);
    EXPECT_NEAR(calibration.accelNoiseStd(0), 0.016, 1e-12);
}

TEST(Vio, PinholeProjectRoundTrip) {
    arc::PinholeCamera<double> camera(
        Eigen::Vector4d(500.0, 500.0, 320.0, 240.0));
    const Eigen::Vector3d point(1.0, -0.5, 3.0);
    Eigen::Vector2d pixel;
    ASSERT_TRUE(camera.project(point, pixel));

    Eigen::Vector3d ray;
    ASSERT_TRUE(camera.unproject(pixel, ray));
    EXPECT_NEAR(ray.normalized().x(), point.normalized().x(), 1e-9);
    EXPECT_NEAR(ray.normalized().y(), point.normalized().y(), 1e-9);
    EXPECT_NEAR(ray.normalized().z(), point.normalized().z(), 1e-9);
}

TEST(Vio, GenericCameraFromString) {
    auto camera = arc::GenericCamera<double>::fromString("pinhole");
    EXPECT_EQ(camera.getName(), "pinhole");
    EXPECT_EQ(camera.getN(), 4);
}

TEST(Vio, ImageInterpolation) {
    arc::ManagedImage<uint16_t> image(4, 4);
    for (size_t y = 0; y < image.h; ++ y) {
        for (size_t x = 0; x < image.w; ++ x) {
            image(x, y) = static_cast<uint16_t>(x + 10 * y);
        }
    }
    const double value = image.interp<double>(1.0, 2.0);
    EXPECT_NEAR(value, 21.0, 1e-9);
    const double half = image.interp<double>(1.5, 2.0);
    EXPECT_NEAR(half, 21.5, 1e-9);
}

TEST(Vio, KeypointDetectorFindsCorners) {
    arc::ManagedImage<uint16_t> image(128, 128);
    uint32_t seed = 1;
    for (size_t y = 0; y < image.h; ++ y) {
        for (size_t x = 0; x < image.w; ++ x) {
            seed = seed * 1664525u + 1013904223u;
            image(x, y) = static_cast<uint16_t>(seed >> 16);
        }
    }

    const arc::Image<const uint16_t> imageView(
        image.ptr, image.w, image.h, image.pitch);
    arc::KeypointsData keypoints;
    arc::detectKeypoints(imageView, keypoints, 32, 1);
    EXPECT_GT(keypoints.corners.size(), 8u);
}

TEST(Vio, OpticalFlowFactoryCreatesAndStops) {
    arc::Calibration<double> calibration;
    auto camera = arc::GenericCamera<double>::fromString("pinhole");
    camera.setFromInit(Eigen::Vector4d(500.0, 500.0, 320.0, 240.0));
    calibration.intrinsics.push_back(camera);
    calibration.resolution.emplace_back(640, 480);
    calibration.tIC.emplace_back(Sophus::SE3d());

    arc::VioConfig config;
    auto flow = arc::OpticalFlowFactory::getOpticalFlow(config, calibration);
    ASSERT_TRUE(flow != nullptr);
    flow->inputQueue.push(nullptr);
    flow.reset();
}

TEST(Vio, VioEstimatorFactoryCreates) {
    arc::Calibration<double> calibration;
    auto camera = arc::GenericCamera<double>::fromString("pinhole");
    camera.setFromInit(Eigen::Vector4d(500.0, 500.0, 320.0, 240.0));
    calibration.intrinsics.push_back(camera);
    calibration.resolution.emplace_back(640, 480);
    calibration.tIC.emplace_back(Sophus::SE3d());

    arc::VioConfig config;
    auto estimator = arc::VioEstimatorFactory::getVioEstimator(
        config, calibration, Eigen::Vector3d(0.0, 0.0, -9.81));
    ASSERT_TRUE(estimator != nullptr);
    EXPECT_FALSE(estimator->finished.load());
}

TEST(Vio, OdrLinks) {
    EXPECT_EQ(vioOdrHelper(), 42);
}

} // namespace
