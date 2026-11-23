#include <gtest/gtest.h>

#include <filesystem>
#include <memory>

#include <base/iarc.h>
#include <env/sdfmap.h>
#include <explore/exploration/explorationfsm.h>
#include <explore/exploration/explorationmanager.h>
#include <explore/exploration/planretry.h>
#include <explore/frontier/frontiermanager.h>
#include <explore/frontier/viewnode.h>
#include <explore/grid/hgrid.h>
#include <explore/tsp/tspsolver.h>
#include <fastplanner/fastplanner.h>

namespace {

constexpr double RES = 0.2;

class ExplorationTest : public ::testing::Test {
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
            rclcpp::Parameter("sdfmap.xlen", 8.0),
            rclcpp::Parameter("sdfmap.ylen", 8.0),
            rclcpp::Parameter("sdfmap.zlen", 2.0),
            rclcpp::Parameter("sdfmap.resolution", RES),
            rclcpp::Parameter("sdfmap.inflation", 0.2),
            rclcpp::Parameter("sdfmap.maxRayLength", 10.0),
            rclcpp::Parameter("sdfmap.phit", 0.65),
            rclcpp::Parameter("sdfmap.pmiss", 0.35),
            rclcpp::Parameter("sdfmap.pmin", 0.12),
            rclcpp::Parameter("sdfmap.pmax", 0.90),
            rclcpp::Parameter("sdfmap.pocc", 0.80),
        });
        node_ = std::make_shared<rclcpp::Node>("test_exploration", options);
        map_ = std::make_unique<arc::SdfMap>(node_);

        arc::FrontierParams frontierParams;
        frontierParams.clusterMin = 10;
        frontierParams.minVisibleNum = 5;
        arc::AstarOptions astarOptions;
        astarOptions.resolution = RES;
        viewNode_ = std::make_unique<arc::ViewNodeContext>(*map_, arc::ViewNodeParams(), astarOptions);
        frontierManager_ = std::make_unique<arc::FrontierManager>(*map_, *viewNode_, frontierParams);

        arc::GridParams gridParams;
        gridParams.gridSize = 2.0;
        gridParams.minUnknown = 20;
        gridParams.minFree = 10;
        gridParams.boxMin = Eigen::Vector3d(-4.0, -4.0, -0.5);
        gridParams.boxMax = Eigen::Vector3d(4.0, 4.0, 1.5);
        hgrid_ = std::make_unique<arc::HGrid>(*map_, gridParams, astarOptions);

        const std::string tspDir = (std::filesystem::temp_directory_path() / "arc_exploration_tsp").string();
        std::filesystem::remove_all(tspDir);
        tspSolver_ = std::make_unique<arc::TspSolver>(tspDir, "exploration");

        planner_ = std::make_unique<arc::FastPlanner>(node_, *map_, arc::FastPlannerOptions{});

        arc::ExplorationParams params;
        params.planTrajectory = false;
        params.droneId = 1;
        params.droneNum = 1;
        manager_ = std::make_unique<arc::ExplorationManager>(
            node_, *map_, *frontierManager_, *hgrid_, *viewNode_, *tspSolver_, *planner_, params);
    }
    void TearDown() override {
        manager_.reset();
        planner_.reset();
        tspSolver_.reset();
        hgrid_.reset();
        frontierManager_.reset();
        viewNode_.reset();
        map_.reset();
        node_.reset();
    }

    std::shared_ptr<rclcpp::Node> node_;
    std::unique_ptr<arc::SdfMap> map_;
    std::unique_ptr<arc::ViewNodeContext> viewNode_;
    std::unique_ptr<arc::FrontierManager> frontierManager_;
    std::unique_ptr<arc::HGrid> hgrid_;
    std::unique_ptr<arc::TspSolver> tspSolver_;
    std::unique_ptr<arc::FastPlanner> planner_;
    std::unique_ptr<arc::ExplorationManager> manager_;
};

TEST_F(ExplorationTest, EmptyMapHasNoFrontier) {
    EXPECT_EQ(manager_->refreshFrontiers(Eigen::Vector3d::Zero()), 0u);
}

TEST(PlanRetryPolicyTest, ExponentialBackoff) {
    arc::PlanRetryPolicy policy;
    EXPECT_TRUE(policy.shouldAttempt(0.0));
    policy.onAttempt(0.0);
    policy.onFailure(0.1, 0.2, 0.8, 2.0);
    EXPECT_FALSE(policy.shouldAttempt(0.2));
    EXPECT_TRUE(policy.shouldAttempt(0.3));
    policy.onAttempt(0.3);
    policy.onFailure(0.3, 0.2, 0.8, 2.0);
    EXPECT_FALSE(policy.shouldAttempt(0.6));
    EXPECT_TRUE(policy.shouldAttempt(0.7));
    policy.onAttempt(0.7);
    policy.onFailure(0.7, 0.2, 0.8, 2.0);
    EXPECT_FALSE(policy.shouldAttempt(1.4));
    EXPECT_TRUE(policy.shouldAttempt(1.5));
    policy.onSuccess();
    EXPECT_TRUE(policy.shouldAttempt(1.6));
    EXPECT_EQ(policy.consecutiveFailures(), 0);
}

TEST_F(ExplorationTest, FsmStartsWaitsAndIdles) {
    arc::ExplorationFsmParams params;
    params.initWaitDuration = 2.0;
    params.idleDuration = 100.0;
    arc::ExplorationFsm fsm(*manager_, params);
    arc::OdomState odom;

    fsm.tick(0.0, odom);
    EXPECT_EQ(fsm.state(), arc::ExplorationState::INIT);

    fsm.tick(2.1, odom);
    EXPECT_EQ(fsm.state(), arc::ExplorationState::WAIT_TRIGGER);

    fsm.trigger();
    fsm.tick(2.2, odom);
    EXPECT_EQ(fsm.state(), arc::ExplorationState::PLAN_TRAJ);

    fsm.tick(2.3, odom);
    EXPECT_EQ(fsm.state(), arc::ExplorationState::IDLE);

    fsm.tick(102.4, odom);
    EXPECT_EQ(fsm.state(), arc::ExplorationState::FINISH);
}

} // namespace