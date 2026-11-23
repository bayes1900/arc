#include <gtest/gtest.h>

#include <filesystem>
#include <memory>

#include <base/iarc.h>
#include <env/sdfmap.h>
#include <explore/exploration/explorationdata.h>
#include <explore/grid/hgrid.h>
#include <explore/swarm/gridallocator.h>
#include <explore/swarm/swarmcoordinator.h>
#include <explore/swarm/swarmstate.h>
#include <explore/tsp/tspsolver.h>

namespace arc {

class HGridTestPeer {
public:
    static void deactivateExceptFirst(HGrid& grid) {
        auto& grids = grid.coarse_->grids_;
        for (size_t i = 1; i < grids.size(); ++i) {
            grids[i].isActive = false;
        }
    }
};

} // arc

namespace {

constexpr double RES = 0.2;

class SwarmTest : public ::testing::Test {
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
        node_ = std::make_shared<rclcpp::Node>("test_swarm", options);
        map_ = std::make_unique<arc::SdfMap>(node_);

        arc::GridParams gridParams;
        gridParams.gridSize = 2.0;
        gridParams.minUnknown = 20;
        gridParams.minFree = 10;
        gridParams.boxMin = Eigen::Vector3d(-4.0, -4.0, -0.5);
        gridParams.boxMax = Eigen::Vector3d(4.0, 4.0, 1.5);
        arc::AstarOptions astarOptions;
        astarOptions.resolution = RES;
        hgrid_ = std::make_unique<arc::HGrid>(*map_, gridParams, astarOptions);
        const std::string tspDir = (std::filesystem::temp_directory_path() / "arc_swarm_tsp").string();
        std::filesystem::remove_all(tspDir);
        tspSolver_ = std::make_unique<arc::TspSolver>(tspDir, "swarm");
        allocator_ = std::make_unique<arc::GridAllocator>(*hgrid_, *tspSolver_);
    }
    void TearDown() override {
        allocator_.reset();
        tspSolver_.reset();
        hgrid_.reset();
        map_.reset();
        node_.reset();
    }

    std::shared_ptr<rclcpp::Node> node_;
    std::unique_ptr<arc::SdfMap> map_;
    std::unique_ptr<arc::HGrid> hgrid_;
    std::unique_ptr<arc::TspSolver> tspSolver_;
    std::unique_ptr<arc::GridAllocator> allocator_;
};

TEST_F(SwarmTest, StateRejectsOldMessagesAndFindsMissedGrids) {
    arc::SwarmState state(1, 2);
    arc::DroneState message;
    message.droneId = 2;
    message.stamp = 1.0;
    message.assignedGridIds = {1};
    EXPECT_TRUE(state.updateState(message));

    message.stamp = 0.5;
    message.assignedGridIds = {2};
    EXPECT_FALSE(state.updateState(message));
    EXPECT_EQ(state.state(2).assignedGridIds.front(), 1);

    std::vector<int> missed;
    state.findUnallocated({0, 1, 2}, missed);
    ASSERT_EQ(missed.size(), 2u);
    EXPECT_EQ(missed[0], 0);
    EXPECT_EQ(missed[1], 2);
}

TEST_F(SwarmTest, AllocatorSplitsGrids) {
    std::vector<int> selfGridIds;
    std::vector<int> otherGridIds;
    const bool ok = allocator_->allocate(
        {Eigen::Vector3d(-3.0, -3.0, 0.5), Eigen::Vector3d(3.0, 3.0, 0.5)},
        {Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()},
        {{}, {}},
        {{}, {}},
        {0, 1},
        selfGridIds,
        otherGridIds);
    ASSERT_TRUE(ok);
    EXPECT_EQ(selfGridIds.size() + otherGridIds.size(), 2u);
    std::vector<int> all = selfGridIds;
    all.insert(all.end(), otherGridIds.begin(), otherGridIds.end());
    std::sort(all.begin(), all.end());
    EXPECT_EQ(all[0], 0);
    EXPECT_EQ(all[1], 1);
}

TEST_F(SwarmTest, AllocatorRepeatedSolves) {
    for (int iteration = 0; iteration < 5; ++ iteration) {
        std::vector<int> selfGridIds;
        std::vector<int> otherGridIds;
        const bool ok = allocator_->allocate(
            {Eigen::Vector3d(-3.0, -3.0, 0.5), Eigen::Vector3d(3.0, 3.0, 0.5)},
            {Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()},
            {{}, {}},
            {{}, {}},
            {0, 1},
            selfGridIds,
            otherGridIds);
        ASSERT_TRUE(ok);
        EXPECT_EQ(selfGridIds.size() + otherGridIds.size(), 2u);
    }
}

TEST_F(SwarmTest, CoordinatorExchangesGridIds) {
    arc::HGridTestPeer::deactivateExceptFirst(*hgrid_);
    arc::ExplorationData data1;
    arc::ExplorationData data2;
    data1.gridTour.assignedGridIds.assign({0});
    data2.gridTour.assignedGridIds.assign({0});

    arc::InMemorySwarmTransport transport1;
    arc::InMemorySwarmTransport transport2;
    arc::SwarmParams params;

    arc::SwarmCoordinator coordinator1(1, 2, *hgrid_, *allocator_, transport1, data1, params);
    arc::SwarmCoordinator coordinator2(2, 2, *hgrid_, *allocator_, transport2, data2, params);

    transport1.setDroneStateHandler([&](const arc::DroneState& state) {
        coordinator2.onDroneState(state);
    });
    transport2.setDroneStateHandler([&](const arc::DroneState& state) {
        coordinator1.onDroneState(state);
    });
    transport1.setPairOptHandler([&](const arc::PairOptRequest& request) {
        coordinator2.onPairOptRequest(request);
    });
    transport2.setPairOptHandler([&](const arc::PairOptRequest& request) {
        coordinator1.onPairOptRequest(request);
    });
    transport1.setPairOptResponseHandler([&](const arc::PairOptResponse& response) {
        coordinator2.onPairOptResponse(response);
    });
    transport2.setPairOptResponseHandler([&](const arc::PairOptResponse& response) {
        coordinator1.onPairOptResponse(response);
    });

    coordinator2.tick(1000.0);
    coordinator1.tick(1000.0);

    EXPECT_TRUE(data2.gridTour.assignedGridIds.empty());
    EXPECT_EQ(data1.gridTour.assignedGridIds.size(), 1u);
    EXPECT_EQ(data1.gridTour.assignedGridIds.front(), 0);
}

} // namespace