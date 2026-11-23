#include <gtest/gtest.h>

#include <base/iarc.h>
#include <env/sdfmap.h>
#include <explore/grid/uniformgrid.h>

namespace arc {

class UniformGridTestPeer {
public:
    static GridInfo& grid(UniformGrid& grid, int id) {
        return grid.grids_[id];
    }
    static const GridInfo& grid(const UniformGrid& grid, int id) {
        return grid.grids_[id];
    }
    static int gridNum(const UniformGrid& grid, int axis) {
        return grid.gridCounts_(axis);
    }
    static Eigen::Vector3d resolution(const UniformGrid& grid) {
        return grid.resolution_;
    }
    static void setInitialized(UniformGrid& grid, bool value) {
        grid.initialized_ = value;
    }
    static void updateGridInfo(UniformGrid& grid, const Eigen::Vector3i& id) {
        grid.updateGridInfo_(grid.toAddress_(id));
    }
    static int toAddress(UniformGrid& grid, const Eigen::Vector3i& id) {
        return grid.toAddress_(id);
    }
};

} // arc

namespace {

constexpr double RES = 0.2;
constexpr double GRID_SIZE = 3.0;

class UniformGridTest : public ::testing::Test {
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
            rclcpp::Parameter("sdfmap.xlen", 12.0),
            rclcpp::Parameter("sdfmap.ylen", 12.0),
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
        node_ = std::make_shared<rclcpp::Node>("test_uniformgrid", options);
        map_ = std::make_unique<arc::SdfMap>(node_);
    }
    void TearDown() override {
        map_.reset();
        node_.reset();
    }
    arc::GridParams gridParams() const {
        arc::GridParams params;
        params.gridSize = GRID_SIZE;
        params.minUnknown = 100;
        params.minFree = 10;
        params.consistentCost = -5.0;
        params.consistentCost2 = 8.0;
        params.boxMin = Eigen::Vector3d(0.0, 0.0, -0.5);
        params.boxMax = Eigen::Vector3d(6.0, 6.0, 1.5);
        return params;
    }
    void fillLogitBox(const Eigen::Vector3d& low, const Eigen::Vector3d& high, double logit) {
        const auto& mp = map_->params();
        auto& md = map_->data();
        Eigen::Vector3i minIndex = mp.p2i(low);
        Eigen::Vector3i maxIndex = mp.p2i(high);
        mp.boundIndex(minIndex);
        mp.boundIndex(maxIndex);
        for (int x = minIndex(0); x <= maxIndex(0); ++ x) {
            for (int y = minIndex(1); y <= maxIndex(1); ++ y) {
                for (int z = minIndex(2); z <= maxIndex(2); ++ z) {
                    md.occupancyLogits[mp.getVoxelIdx(Eigen::Vector3i(x, y, z))] = logit;
                }
            }
        }
    }
    std::shared_ptr<rclcpp::Node> node_;
    std::unique_ptr<arc::SdfMap> map_;
};

TEST_F(UniformGridTest, LayoutAndResolution) {
    arc::UniformGrid coarse(*map_, gridParams(), 1);
    coarse.initializeGrids();
    coarse.updateBaseCoor();

    EXPECT_EQ(arc::UniformGridTestPeer::gridNum(coarse, 0), 2);
    EXPECT_EQ(arc::UniformGridTestPeer::gridNum(coarse, 1), 2);
    EXPECT_EQ(arc::UniformGridTestPeer::gridNum(coarse, 2), 1);
    EXPECT_NEAR(arc::UniformGridTestPeer::resolution(coarse)[0], 3.0, 1e-9);
    EXPECT_NEAR(arc::UniformGridTestPeer::resolution(coarse)[1], 3.0, 1e-9);
    EXPECT_NEAR(arc::UniformGridTestPeer::resolution(coarse)[2], 2.0, 1e-9);

    arc::UniformGrid fine(*map_, gridParams(), 2);
    fine.initializeGrids();
    fine.updateBaseCoor();

    EXPECT_EQ(arc::UniformGridTestPeer::gridNum(fine, 0), 4);
    EXPECT_EQ(arc::UniformGridTestPeer::gridNum(fine, 1), 4);
    EXPECT_EQ(arc::UniformGridTestPeer::gridNum(fine, 2), 1);
    EXPECT_NEAR(arc::UniformGridTestPeer::resolution(fine)[0], 1.5, 1e-9);
    EXPECT_NEAR(arc::UniformGridTestPeer::resolution(fine)[1], 1.5, 1e-9);
    EXPECT_NEAR(arc::UniformGridTestPeer::resolution(fine)[2], 2.0, 1e-9);

    for (int i = 0; i < 4; ++ i) {
        auto& grid = arc::UniformGridTestPeer::grid(coarse, i);
        EXPECT_TRUE(grid.isActive);
        EXPECT_FALSE(grid.shouldPartition);
        EXPECT_EQ(grid.vertices.size(), 4u);
        EXPECT_EQ(grid.normals.size(), 4u);
        EXPECT_NEAR(grid.center(0), (i / 2) * 3.0 + 1.5, 1e-9);
        EXPECT_NEAR(grid.center(1), (i % 2) * 3.0 + 1.5, 1e-9);
        EXPECT_NEAR(grid.center(2), 0.5, 1e-9);
        EXPECT_NEAR(grid.boxMin(0), (i / 2) * 3.0, 1e-9);
        EXPECT_NEAR(grid.boxMax(0), (i / 2) * 3.0 + 3.0, 1e-9);
    }
    for (int i = 0; i < 16; ++ i) {
        EXPECT_FALSE(arc::UniformGridTestPeer::grid(fine, i).isActive);
    }
}

TEST_F(UniformGridTest, UnknownStatisticsAndCenter) {
    arc::UniformGrid coarse(*map_, gridParams(), 1);
    coarse.initializeGrids();
    coarse.updateBaseCoor();
    arc::UniformGridTestPeer::updateGridInfo(coarse, Eigen::Vector3i(0, 0, 0));

    auto& grid = arc::UniformGridTestPeer::grid(
        coarse, arc::UniformGridTestPeer::toAddress(coarse, Eigen::Vector3i(0, 0, 0)));
    EXPECT_EQ(grid.unknownCount, 15 * 15 * 10);
    EXPECT_NEAR(grid.center(0), 1.5, 1e-9);
    EXPECT_NEAR(grid.center(1), 1.5, 1e-9);
    EXPECT_NEAR(grid.center(2), 0.5, 1e-9);
    EXPECT_TRUE(grid.isRelevantNow);
    EXPECT_FALSE(grid.shouldPartition);

    arc::UniformGrid fine(*map_, gridParams(), 2);
    fine.initializeGrids();
    fine.updateBaseCoor();
    arc::UniformGridTestPeer::updateGridInfo(fine, Eigen::Vector3i(0, 0, 0));
    auto& fineGrid = arc::UniformGridTestPeer::grid(fine, 0);
    EXPECT_EQ(fineGrid.unknownCount, 8 * 8 * 10);
    EXPECT_TRUE(fineGrid.isRelevantNow);
}

TEST_F(UniformGridTest, FreeVoxelsTriggerDivision) {
    arc::UniformGrid coarse(*map_, gridParams(), 1);
    coarse.initializeGrids();
    coarse.updateBaseCoor();
    fillLogitBox(Eigen::Vector3d(0.1, 0.1, -0.4), Eigen::Vector3d(1.4, 2.9, 1.4), 0.0);

    arc::UniformGridTestPeer::updateGridInfo(coarse, Eigen::Vector3i(0, 0, 0));
    auto& grid = arc::UniformGridTestPeer::grid(coarse, 0);
    EXPECT_TRUE(grid.shouldPartition);
    EXPECT_EQ(grid.unknownCount, 15 * 15 * 10 - 8 * 15 * 10);
    EXPECT_NEAR(grid.center(0), 2.3, 1e-9);
}

TEST_F(UniformGridTest, InputFrontiersMapsToGrid) {
    arc::UniformGrid coarse(*map_, gridParams(), 1);
    coarse.initializeGrids();
    coarse.updateBaseCoor();
    arc::UniformGrid fine(*map_, gridParams(), 2);
    fine.initializeGrids();
    fine.updateBaseCoor();

    std::vector<Eigen::Vector3d> averages = {
        Eigen::Vector3d(1.0, 1.0, 0.5),
        Eigen::Vector3d(4.5, 1.0, 0.5),
        Eigen::Vector3d(4.5, 4.5, 0.5),
    };
    coarse.inputFrontiers(averages);
    fine.inputFrontiers(averages);

    EXPECT_EQ(arc::UniformGridTestPeer::grid(coarse, 0).frontierIndices.size(), 1u);
    EXPECT_EQ(arc::UniformGridTestPeer::grid(coarse, 0).frontierIndices.count(0), 1u);
    EXPECT_EQ(arc::UniformGridTestPeer::grid(coarse, 2).frontierIndices.count(1), 1u);
    EXPECT_EQ(arc::UniformGridTestPeer::grid(coarse, 3).frontierIndices.count(2), 1u);

    int found = 0;
    for (int i = 0; i < 16; ++ i) {
        found += static_cast<int>(arc::UniformGridTestPeer::grid(fine, i).frontierIndices.size());
    }
    EXPECT_EQ(found, 3);
}

TEST_F(UniformGridTest, UpdateRemovesDividedGrid) {
    arc::UniformGrid coarse(*map_, gridParams(), 1);
    coarse.initializeGrids();
    coarse.updateBaseCoor();
    arc::UniformGridTestPeer::setInitialized(coarse, true);
    fillLogitBox(Eigen::Vector3d(0.1, 0.1, -0.4), Eigen::Vector3d(1.4, 2.9, 1.4), 0.0);

    auto& md = map_->data();
    md.updateMin = Eigen::Vector3d(0.0, 0.0, -0.5);
    md.updateMax = Eigen::Vector3d(3.0, 3.0, 1.5);

    arc::GridIdList gridIds;
    gridIds.assign({ 0 });
    std::vector<int> cellsToPartition;
    std::vector<int> allCellsToPartition;
    coarse.updateGridData(1, gridIds, cellsToPartition, allCellsToPartition);

    EXPECT_EQ(cellsToPartition, (std::vector<int>{ 0 }));
    EXPECT_EQ(allCellsToPartition, (std::vector<int>{ 0 }));
    EXPECT_TRUE(gridIds.empty());
    EXPECT_FALSE(arc::UniformGridTestPeer::grid(coarse, 0).isActive);
}

} // namespace