#include <gtest/gtest.h>

#include <base/iarc.h>
#include <env/sdfmap.h>
#include <planning/bsplineopt.h>

namespace {

class BSplineOptTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        if (!rclcpp::ok()) {
            rclcpp::init(0, nullptr);
        }
    }

    static void TearDownTestSuite() {
        if (rclcpp::ok()) {
            rclcpp::shutdown();
        }
    }

    void SetUp() override {
        rclcpp::NodeOptions options;
        options.parameter_overrides({
            rclcpp::Parameter("sdfmap.xlen", 4.0),
            rclcpp::Parameter("sdfmap.ylen", 4.0),
            rclcpp::Parameter("sdfmap.zlen", 3.0),
            rclcpp::Parameter("sdfmap.resolution", 0.2),
            rclcpp::Parameter("sdfmap.inflation", 0.1),
            rclcpp::Parameter("sdfmap.maxRayLength", 10.0),
            rclcpp::Parameter("sdfmap.phit", 0.65),
            rclcpp::Parameter("sdfmap.pmiss", 0.35),
            rclcpp::Parameter("sdfmap.pmin", 0.12),
            rclcpp::Parameter("sdfmap.pmax", 0.90),
            rclcpp::Parameter("sdfmap.pocc", 0.80),
        });
        node_ = std::make_shared<rclcpp::Node>("test_bsplineopt", options);
        map_ = std::make_unique<arc::SdfMap>(node_);
        fillFreeSpace();
    }

    void TearDown() override {
        map_.reset();
        node_.reset();
    }

    void fillFreeSpace() {
        const auto& mp = map_->params();
        auto& md = map_->data();
        for (size_t i = 0; i < md.numVoxels; ++i) {
            md.occupancyLogits[i] = 0.0;
        }
        md.localBoundMin.setZero();
        md.localBoundMax = mp.voxelShape - Eigen::Vector3i::Ones();
        md.updateInflateOccupancy();
    }

    rclcpp::Node::SharedPtr node_;
    std::unique_ptr<arc::SdfMap> map_;
};

TEST_F(BSplineOptTest, OptimizeSmoothTrajectory) {
    arc::BSplineOptRequest request;
    request.points.resize(8, 3);
    for (int i = 0; i < request.points.rows(); ++i) {
        request.points.row(i) << 0.4 * i - 1.4, 0.2 * std::sin(i), 1.0;
    }
    request.interval = 0.2;
    request.cost.SMOOTHNESS = 1;
    request.cost.START = 1;
    request.cost.END = 1;
    request.startState = {
        request.points.row(0).transpose(),
        Eigen::Vector3d::Zero(),
        Eigen::Vector3d::Zero(),
    };
    request.endState = {
        request.points.row(7).transpose(),
        Eigen::Vector3d::Zero(),
        Eigen::Vector3d::Zero(),
    };

    arc::BSplineOptimizer optimizer(node_, *map_);
    const arc::BSplineOptResult result = optimizer.optimize(request);
    EXPECT_TRUE(
        result.code == arc::BSplineOptResult::SUCCEEDED ||
        result.code == arc::BSplineOptResult::MAXEVALREACHED ||
        result.code == arc::BSplineOptResult::MAXTIMEREACHED);
    EXPECT_GT(result.iterations, 0);
    EXPECT_EQ(result.points.rows(), request.points.rows());
    EXPECT_EQ(result.points.cols(), 3);
    EXPECT_GT(result.interval, 0.0);
    EXPECT_TRUE(result.points.allFinite());
}

} // namespace