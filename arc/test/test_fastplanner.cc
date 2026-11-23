#include <gtest/gtest.h>

#include <base/iarc.h>
#include <env/sdfmap.h>
#include <fastplanner/fastplanner.h>
#include <planning/trajectory.h>
#include <planning/polytraj.h>

namespace {

using arc::BSpline;
using arc::FastPlanner;
using arc::FastPlannerOptions;
using arc::LocalTrajectory;
using arc::LocalTrajectoryServer;
using arc::LocalTrajectoryState;

class FastPlannerTest : public ::testing::Test {
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
        node_ = std::make_shared<rclcpp::Node>("test_fastplanner", options);
        map_ = std::make_unique<arc::SdfMap>(node_);
        fillFreeSpace();
        planner_ = std::make_unique<FastPlanner>(node_, *map_, FastPlannerOptions());
    }

    void TearDown() override {
        planner_.reset();
        map_.reset();
        node_.reset();
    }

    void fillFreeSpace() {
        auto& mp = map_->params();
        auto& md = map_->data();
        std::fill(md.occupancyLogits.begin(), md.occupancyLogits.end(), 0.0);
        md.localBoundMin.setZero();
        md.localBoundMax = mp.voxelShape - Eigen::Vector3i::Ones();
        md.updateInflateOccupancy();
    }

    void fillBox(const Eigen::Vector3d& low, const Eigen::Vector3d& high) {
        auto& mp = map_->params();
        auto& md = map_->data();
        const Eigen::Vector3i minIndex = mp.p2i(low);
        const Eigen::Vector3i maxIndex = mp.p2i(high);
        for (int x = minIndex(0); x <= maxIndex(0); ++ x) {
            for (int y = minIndex(1); y <= maxIndex(1); ++ y) {
                for (int z = minIndex(2); z <= maxIndex(2); ++ z) {
                    const Eigen::Vector3i index(x, y, z);
                    if (mp.isInMap(index)) {
                        md.occupancyLogits[mp.getVoxelIdx(index)] = mp.lmax;
                    }
                }
            }
        }
        md.localBoundMin.setZero();
        md.localBoundMax = mp.voxelShape - Eigen::Vector3i::Ones();
        md.updateInflateOccupancy();
    }

    LocalTrajectory makeTrajectory(int id, double startTime, double duration) {
        const double interval = duration / 2.0;
        Eigen::MatrixXd points(5, 3);
        Eigen::MatrixXd yawPoints(5, 1);
        for (int i = 0; i < 5; ++ i) {
            points.row(i) << 0.5 * i, 0.2 * i, 1.0;
            yawPoints(i, 0) = 0.1 * i;
        }

        LocalTrajectory trajectory;
        trajectory.id = id;
        trajectory.startTime = startTime;
        trajectory.duration = duration;
        trajectory.position.setUniformBSpline(points, 3, interval);
        trajectory.velocity = trajectory.position.getDerivativeBSpline();
        trajectory.acceleration = trajectory.velocity.getDerivativeBSpline();
        trajectory.yaw.setUniformBSpline(yawPoints, 3, interval);
        trajectory.yawRate = trajectory.yaw.getDerivativeBSpline();
        trajectory.yawAcceleration = trajectory.yawRate.getDerivativeBSpline();
        trajectory.startPosition = Eigen::Vector3d(trajectory.position.evalt(0.0).head<3>());
        return trajectory;
    }

    rclcpp::Node::SharedPtr node_;
    std::unique_ptr<arc::SdfMap> map_;
    std::unique_ptr<FastPlanner> planner_;
};

TEST_F(FastPlannerTest, PolynomialTrajectoryWaypoints) {
    Eigen::MatrixXd positions(3, 3);
    positions << 0.0, 0.0, 1.0,
                 1.0, 0.0, 1.0,
                 2.0, 0.0, 1.0;
    Eigen::VectorXd times(2);
    times << 1.0, 1.0;
    const Eigen::Vector3d zero = Eigen::Vector3d::Zero();

    arc::PolynomialTrajectory trajectory;
    ASSERT_TRUE(arc::PolynomialTrajectory::waypointsTraj(
        positions, zero, zero, zero, zero, times, trajectory));
    EXPECT_NEAR(trajectory.getTotalTime(), 2.0, 1e-9);
    EXPECT_NEAR(trajectory.getLength(), 2.0, 0.05);
    EXPECT_LT((trajectory.evaluate(0.0, 0) - positions.row(0).transpose()).norm(), 1e-6);
    EXPECT_LT((trajectory.evaluate(2.0, 0) - positions.row(2).transpose()).norm(), 1e-6);
    EXPECT_LT(trajectory.evaluate(0.0, 1).norm(), 1e-6);
    EXPECT_LT(trajectory.evaluate(0.0, 2).norm(), 1e-6);
}

TEST_F(FastPlannerTest, BSplineEvaluation) {
    Eigen::MatrixXd points(8, 3);
    for (int i = 0; i < points.rows(); ++ i) {
        points.row(i) << 0.3 * i, std::sin(0.5 * i), 1.0;
    }
    BSpline spline;
    spline.setUniformBSpline(points, 3, 0.2);

    EXPECT_TRUE(spline.valid());
    EXPECT_NEAR(spline.duration(), 1.0, 1e-9);
    EXPECT_NEAR(spline.evalt(0.0)(2), 1.0, 1e-9);

    const BSpline derivative = spline.getDerivativeBSpline();
    EXPECT_TRUE(derivative.valid());
    EXPECT_NEAR(derivative.duration(), 1.0, 1e-9);
}

TEST_F(FastPlannerTest, CalcNextYawAcrossPi) {
    double yaw = -arc::__pi__ + 0.05;
    FastPlanner::calcNextYaw(arc::__pi__ - 0.05, yaw);
    EXPECT_NEAR(yaw, arc::__pi__ + 0.05, 1e-9);
}

TEST_F(FastPlannerTest, LocalTrajectoryServerSwitches) {
    LocalTrajectoryServer server;
    server.addTrajectory(makeTrajectory(1, 100.0, 2.0));
    server.addTrajectory(makeTrajectory(2, 101.0, 2.0));

    LocalTrajectoryState state;
    ASSERT_TRUE(server.evaluate(100.5, state));
    EXPECT_EQ(state.id, 1);
    EXPECT_FALSE(state.finished);

    ASSERT_TRUE(server.evaluate(101.1, state));
    EXPECT_EQ(state.id, 2);
    EXPECT_FALSE(state.finished);

    ASSERT_TRUE(server.evaluate(110.0, state));
    EXPECT_EQ(state.id, 2);
    EXPECT_TRUE(state.finished);
    EXPECT_NEAR(state.velocity.norm(), 0.0, 1e-12);
    EXPECT_NEAR(state.acceleration.norm(), 0.0, 1e-12);
}

TEST_F(FastPlannerTest, ResetDurationOnlyTruncatesCurrent) {
    LocalTrajectoryServer server;
    server.addTrajectory(makeTrajectory(1, 10.0, 2.0));
    server.addTrajectory(makeTrajectory(2, 11.0, 2.0));
    server.resetDuration(10.5, 0.0);

    LocalTrajectoryState state;
    ASSERT_TRUE(server.evaluate(10.6, state));
    EXPECT_EQ(state.id, 1);
    EXPECT_TRUE(state.finished);

    ASSERT_TRUE(server.evaluate(11.1, state));
    EXPECT_EQ(state.id, 2);
    EXPECT_FALSE(state.finished);
}

TEST_F(FastPlannerTest, KinodynamicReplanReachesGoal) {
    const Eigen::Vector3d start(-1.5, 0.0, 1.0);
    const Eigen::Vector3d end(1.5, 0.0, 1.0);
    const Eigen::Vector3d zero = Eigen::Vector3d::Zero();

    const arc::PlannerResult result = planner_->kinodynamicReplan(
        start, zero, zero, end, zero);
    ASSERT_TRUE(result.succeeded());
    ASSERT_TRUE(planner_->localTrajectory().valid());
    EXPECT_GT(planner_->localTrajectory().duration, 1.0);
    EXPECT_FALSE(planner_->planningData().kinoPath.empty());

    const Eigen::Vector3d endPosition(planner_->localTrajectory().position.evalt(
        planner_->localTrajectory().duration).head<3>());
    EXPECT_LT((endPosition - end).norm(), 0.3);
}

TEST_F(FastPlannerTest, PlanExploreTrajectoryThroughWaypoints) {
    const std::vector<Eigen::Vector3d> tour = {
        Eigen::Vector3d(-1.0, 0.0, 1.0),
        Eigen::Vector3d(0.0, 1.0, 1.0),
        Eigen::Vector3d(1.0, 0.0, 1.0),
    };
    const Eigen::Vector3d zero = Eigen::Vector3d::Zero();

    const arc::PlannerResult result = planner_->planExploreTrajectory(tour, zero, zero);
    ASSERT_TRUE(result.succeeded());
    ASSERT_TRUE(planner_->localTrajectory().valid());

    const Eigen::Vector3d startPosition(planner_->localTrajectory().position.evalt(0.0).head<3>());
    const Eigen::Vector3d endPosition(planner_->localTrajectory().position.evalt(
        planner_->localTrajectory().duration).head<3>());
    EXPECT_LT((startPosition - tour.front()).norm(), 0.2);
    EXPECT_LT((endPosition - tour.back()).norm(), 0.5);
}

TEST_F(FastPlannerTest, YawPlanReachesEndYaw) {
    FastPlannerOptions options;
    options.yaw.maxTime = 0.05;
    options.yaw.startWeight = 100.0;
    options.yaw.endWeight = 100.0;
    FastPlanner planner(node_, *map_, options);

    const std::vector<Eigen::Vector3d> tour = {
        Eigen::Vector3d(-1.0, 0.0, 1.0),
        Eigen::Vector3d(1.0, 0.0, 1.0),
    };
    const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
    ASSERT_TRUE(planner.planExploreTrajectory(tour, zero, zero).succeeded());

    const Eigen::Vector3d startYaw(0.5 * arc::__pi__, 0.0, 0.0);
    const arc::PlannerResult result = planner.planYawExplore(startYaw, 0.0, true, 1.5);
    ASSERT_TRUE(result.succeeded());
    ASSERT_TRUE(planner.localTrajectory().yaw.valid());
    EXPECT_NEAR(planner.localTrajectory().yaw.duration(),
        planner.localTrajectory().duration, 1e-6);

    const double yawBegin = planner.localTrajectory().yaw.evalt(0.0)(0);
    const double yawEnd = planner.localTrajectory().yaw.evalt(
        planner.localTrajectory().duration)(0);
    EXPECT_NEAR(yawBegin, 0.5 * arc::__pi__, 0.35);
    EXPECT_LT(std::fabs(yawEnd), 0.35);

    double maxYawRate = 0.0;
    for (double t = 0.0; t <= planner.localTrajectory().duration; t += 0.01) {
        maxYawRate = std::max(maxYawRate,
            std::fabs(planner.localTrajectory().yawRate.evalt(t)(0)));
    }
    EXPECT_LT(maxYawRate, 1.5 * options.maxYawRate);
}

TEST_F(FastPlannerTest, CheckTrajectoryCollisionDetectsObstacle) {
    const Eigen::Vector3d start(-1.5, 0.0, 1.0);
    const Eigen::Vector3d end(1.5, 0.0, 1.0);
    const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
    ASSERT_TRUE(planner_->kinodynamicReplan(start, zero, zero, end, zero).succeeded());
    planner_->setLocalTrajectoryStartTime(node_->now().seconds());

    double distance = -1.0;
    EXPECT_TRUE(planner_->checkTrajectoryCollision(distance));

    const Eigen::Vector3d middle(planner_->localTrajectory().position.evalt(
        planner_->localTrajectory().duration * 0.5).head<3>());
    fillBox(middle - Eigen::Vector3d::Constant(0.2),
        middle + Eigen::Vector3d::Constant(0.2));

    EXPECT_FALSE(planner_->checkTrajectoryCollision(distance));
    EXPECT_GE(distance, 0.0);
}

} // namespace
