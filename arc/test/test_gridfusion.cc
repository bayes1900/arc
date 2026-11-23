#include <gtest/gtest.h>

#include <algorithm>
#include <memory>

#include <base/iarc.h>
#include <env/multimapmanager.h>
#include <env/sdfmap.h>
#include <explore/frontier/frontierdetector.h>
#include <explore/grid/hgrid.h>
#include <explore/grid/uniformgrid.h>

namespace arc {

class HGridTestPeer {
public:
    static GridInfo& coarseGrid(HGrid& grid, int id) {
        return grid.coarse_->grids_[id];
    }
};

} // arc

namespace {

constexpr double RES = 0.2;

class GridFusionTest : public ::testing::Test {
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
            rclcpp::Parameter("sdfmap.zlen", 4.0),
            rclcpp::Parameter("sdfmap.resolution", RES),
            rclcpp::Parameter("sdfmap.inflation", 0.2),
            rclcpp::Parameter("sdfmap.maxRayLength", 10.0),
            rclcpp::Parameter("sdfmap.phit", 0.65),
            rclcpp::Parameter("sdfmap.pmiss", 0.35),
            rclcpp::Parameter("sdfmap.pmin", 0.12),
            rclcpp::Parameter("sdfmap.pmax", 0.90),
            rclcpp::Parameter("sdfmap.pocc", 0.80),
        });
        node_ = std::make_shared<rclcpp::Node>("test_gridfusion", options);
        map_ = std::make_unique<arc::SdfMap>(node_);
        arc::MultiMapManagerParams mmParams;
        mmParams.droneId = 2;
        mmParams.droneNum = 2;
        mmParams.chunkSize = 10000;
        manager_ = std::make_unique<arc::MultiMapManager>(*map_, mmParams);
    }
    void TearDown() override {
        manager_.reset();
        map_.reset();
        node_.reset();
    }
    arc::GridParams gridParams() const {
        arc::GridParams params;
        params.gridSize = 2.0;
        params.minUnknown = 100;
        params.minFree = 100000;
        params.consistentCost = -5.0;
        params.consistentCost2 = 8.0;
        return params;
    }
    arc::AstarOptions astarOptions() const {
        arc::AstarOptions options;
        options.resolution = RES;
        return options;
    }
    std::vector<size_t> slabAddresses() const {
        const auto& mp = map_->params();
        std::vector<size_t> adrs;
        for (int x = 2; x <= 8; ++ x) {
            for (int y = 2; y <= 8; ++ y) {
                for (int z = 5; z <= 15; ++ z) {
                    adrs.push_back(mp.getVoxelIdx(x, y, z));
                }
            }
        }
        return adrs;
    }
    std::shared_ptr<rclcpp::Node> node_;
    std::unique_ptr<arc::SdfMap> map_;
    std::unique_ptr<arc::MultiMapManager> manager_;
};

TEST_F(GridFusionTest, ExternalChunkUpdatesMapAndGrid) {
    arc::FrontierParams frontierParams;
    frontierParams.clusterMin = 10;
    frontierParams.minFrontierZ = 0.0;
    arc::FrontierDetector detector(*map_, frontierParams);
    std::vector<arc::Frontier> frontiers;
    std::vector<arc::Frontier> dormant;
    std::vector<arc::Frontier> detected;
    std::vector<int> removed;
    detector.detect(frontiers, dormant, detected, removed);
    EXPECT_TRUE(detected.empty());

    arc::HGrid grid(*map_, gridParams(), astarOptions());
    grid.setMultiMapManager(manager_.get());
    EXPECT_EQ(grid.unknownCellsOf(0), 2000);

    auto& md = map_->data();
    md.updateMin = Eigen::Vector3d(-0.5, -3.5, 3.0);
    md.updateMax = Eigen::Vector3d(-0.5, -3.5, 3.0);

    const std::vector<size_t> adrs = slabAddresses();
    arc::ChunkData chunk;
    chunk.fromDroneId = 1;
    chunk.toDroneId = 2;
    chunk.chunkDroneId = 1;
    chunk.idx = 1;
    chunk.voxelAdrs = adrs;
    chunk.voxelOcc.assign(adrs.size(), 0);
    manager_->receiveChunk(chunk);
    manager_->processReceivedChunks();

    const auto& boxes = manager_->chunkBoxes();
    ASSERT_EQ(boxes.size(), 2u);
    ASSERT_TRUE(boxes[0].valid);
    EXPECT_NEAR(boxes[0].min(0), -3.5, 1e-9);
    EXPECT_NEAR(boxes[0].max(0), -2.3, 1e-9);
    EXPECT_NEAR(boxes[0].min(1), -3.5, 1e-9);
    EXPECT_NEAR(boxes[0].max(1), -2.3, 1e-9);
    EXPECT_NEAR(boxes[0].min(2), 0.6, 1e-9);
    EXPECT_NEAR(boxes[0].max(2), 2.6, 1e-9);
    EXPECT_GT(md.updateRevision, 0u);
    EXPECT_EQ(md.dirtyIndexes.size(), adrs.size());

    arc::GridIdList gridIds;
    std::vector<int> first;
    std::vector<int> second;
    grid.updateGridData(1, gridIds, {}, first, second);

    EXPECT_TRUE(arc::HGridTestPeer::coarseGrid(grid, 0).updatedThisPass);
    EXPECT_FALSE(arc::HGridTestPeer::coarseGrid(grid, 1).updatedThisPass);
    EXPECT_EQ(grid.unknownCellsOf(0), 2000 - 7 * 7 * 11);

    detector.detect(frontiers, dormant, detected, removed);
    ASSERT_FALSE(detected.empty());
    EXPECT_TRUE(md.dirtyIndexes.empty());
    const Eigen::Vector3d chunkMin(-3.5, -3.5, 0.6);
    const Eigen::Vector3d chunkMax(-2.3, -2.3, 2.6);
    bool cellInChunk = false;
    for (const auto& cell : detected.front().cells) {
        if ((cell.array() >= chunkMin.array() - 1e-6).all() &&
            (cell.array() <= chunkMax.array() + 1e-6).all()) {
            cellInChunk = true;
            break;
        }
    }
    EXPECT_TRUE(cellInChunk);

    std::vector<Eigen::Vector3d> averages;
    for (const auto& frontier : detected) {
        averages.push_back(frontier.center);
    }
    grid.inputFrontiers(averages);
    EXPECT_FALSE(arc::HGridTestPeer::coarseGrid(grid, 0).frontierIndices.empty());
    std::vector<int> frontierIndices;
    grid.frontiersInGrid({ 0 }, frontierIndices);
    EXPECT_EQ(
        frontierIndices.size(),
        arc::HGridTestPeer::coarseGrid(grid, 0).frontierIndices.size());
}

} // namespace