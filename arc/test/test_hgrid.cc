#include <gtest/gtest.h>

#include <algorithm>

#include <base/iarc.h>
#include <env/sdfmap.h>
#include <explore/grid/hgrid.h>
#include <explore/grid/uniformgrid.h>

namespace arc {

class HGridTestPeer {
public:
    static UniformGrid& coarse(HGrid& grid) {
        return *grid.coarse_;
    }
    static UniformGrid& fine(HGrid& grid) {
        return *grid.fine_;
    }
    static GridInfo& coarseGrid(HGrid& grid, int id) {
        return grid.coarse_->grids_[id];
    }
    static GridInfo& fineGrid(HGrid& grid, int id) {
        return grid.fine_->grids_[id];
    }
    static int coarseNum(HGrid& grid) {
        return static_cast<int>(grid.coarse_->grids_.size());
    }
    static int fineNum(HGrid& grid) {
        return static_cast<int>(grid.fine_->grids_.size());
    }
    static void setInitialized(HGrid& grid, bool value) {
        grid.coarse_->initialized_ = value;
        grid.fine_->initialized_ = value;
    }
    static void coarseToFine(HGrid& grid, int coarseId, std::vector<int>& fines) {
        const auto values = grid.fineChildrenOf_(coarseId);
        fines.assign(values.begin(), values.end());
    }
    static void fineToCoarse(HGrid& grid, int fineId, int& coarseId) {
        coarseId = grid.coarseParentOfFine_(fineId);
    }
};

} // arc

namespace {

constexpr double RES = 0.2;

TEST(GridIdListTest, ReplaceOrderKeepsMembership) {
    arc::GridIdList ids;
    ids.insert(3);
    ids.insert(7);
    ids.insert(9);

    ids.replaceOrder({ 9, 3, 7 });

    EXPECT_EQ(ids.values(), (std::vector<int>{ 9, 3, 7 }));
    EXPECT_TRUE(ids.contains(3));
    EXPECT_TRUE(ids.contains(7));
    EXPECT_TRUE(ids.contains(9));
    EXPECT_FALSE(ids.contains(5));
}

class HGridTest : public ::testing::Test {
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
        node_ = std::make_shared<rclcpp::Node>("test_hgrid", options);
        map_ = std::make_unique<arc::SdfMap>(node_);
    }
    void TearDown() override {
        map_.reset();
        node_.reset();
    }
    arc::GridParams gridParams() const {
        arc::GridParams params;
        params.gridSize = 3.0;
        params.minUnknown = 100;
        params.minFree = 10;
        params.consistentCost = -5.0;
        params.consistentCost2 = 8.0;
        params.boxMin = Eigen::Vector3d(0.0, 0.0, -0.5);
        params.boxMax = Eigen::Vector3d(6.0, 6.0, 1.5);
        return params;
    }
    arc::AstarOptions astarOptions() const {
        arc::AstarOptions options;
        options.resolution = RES;
        return options;
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

TEST_F(HGridTest, IdRoundTrip) {
    arc::HGrid grid(*map_, gridParams(), astarOptions());
    const int coarseNum = arc::HGridTestPeer::coarseNum(grid);
    const int fineNum = arc::HGridTestPeer::fineNum(grid);
    EXPECT_EQ(coarseNum, 4);
    EXPECT_EQ(fineNum, 16);

    std::vector<int> allFines;
    for (int coarse = 0; coarse < coarseNum; ++ coarse) {
        std::vector<int> fines;
        arc::HGridTestPeer::coarseToFine(grid, coarse, fines);
        ASSERT_EQ(fines.size(), 4u);
        for (int fine : fines) {
            int back = -1;
            arc::HGridTestPeer::fineToCoarse(grid, fine, back);
            EXPECT_EQ(back, coarse);
            EXPECT_GE(fine, 0);
            EXPECT_LT(fine, fineNum);
        }
        allFines.insert(allFines.end(), fines.begin(), fines.end());
    }
    std::sort(allFines.begin(), allFines.end());
    for (int i = 0; i < fineNum; ++ i) {
        EXPECT_EQ(allFines[i], i);
    }
}

TEST_F(HGridTest, ConsistentGrid) {
    arc::HGrid grid(*map_, gridParams(), astarOptions());
    std::vector<int> first;
    std::vector<int> second;

    grid.buildConsistentSets({ 0, 1 }, { 0, 1 }, first, second);
    EXPECT_EQ(first, (std::vector<int>{ 0 }));
    EXPECT_EQ(second, (std::vector<int>{ 1 }));

    grid.buildConsistentSets({}, { 0, 1 }, first, second);
    EXPECT_TRUE(first.empty());
    EXPECT_TRUE(second.empty());

    std::vector<int> fineOf0;
    arc::HGridTestPeer::coarseToFine(grid, 0, fineOf0);
    const int coarseNum = arc::HGridTestPeer::coarseNum(grid);
    std::vector<int> curIds;
    for (int fine : fineOf0) {
        curIds.push_back(fine + coarseNum);
    }
    curIds.push_back(1);
    grid.buildConsistentSets({ 0, 1 }, curIds, first, second);
    EXPECT_EQ(first.size(), fineOf0.size());
    EXPECT_EQ(second, (std::vector<int>{ 1 }));

    std::vector<int> lastIds = { fineOf0[0] + coarseNum, fineOf0[1] + coarseNum, 1 };
    grid.buildConsistentSets(lastIds, curIds, first, second);
    EXPECT_EQ(first.size(), fineOf0.size());
    EXPECT_EQ(second, (std::vector<int>{ 1 }));
}

TEST_F(HGridTest, CostMatrixStructure) {
    arc::HGrid grid(*map_, gridParams(), astarOptions());
    std::vector<Eigen::Vector3d> positions = { Eigen::Vector3d(0.5, 0.5, 0.5) };
    std::vector<Eigen::Vector3d> velocities = { Eigen::Vector3d::Zero() };
    std::vector<std::vector<int>> firstIds = { { 2 } };
    std::vector<std::vector<int>> secondIds = { { 3 } };
    std::vector<int> gridIds = { 0, 1, 2, 3 };

    Eigen::MatrixXd mat;
    grid.buildCostMatrix(positions, velocities, firstIds, secondIds, gridIds, mat);

    const int dim = 1 + 1 + 4;
    EXPECT_EQ(mat.rows(), dim);
    EXPECT_EQ(mat.cols(), dim);
    EXPECT_DOUBLE_EQ(mat(0, 1), -1000.0);
    EXPECT_DOUBLE_EQ(mat(1, 0), 1000.0);
    for (int i = 0; i < 4; ++ i) {
        EXPECT_DOUBLE_EQ(mat(0, 2 + i), 1000.0);
        EXPECT_DOUBLE_EQ(mat(2 + i, 0), 0.0);
        EXPECT_DOUBLE_EQ(mat(2 + i, 1), 0.0);
        if (i != 2) {
            EXPECT_GT(mat(1, 2 + i), 0.0);
        }
    }
    EXPECT_DOUBLE_EQ(mat(1, 1), 1000.0);
    for (int i = 0; i < dim; ++ i) {
        EXPECT_DOUBLE_EQ(mat(i, i), 1000.0);
    }
    for (int i = 0; i < 4; ++ i) {
        for (int j = i + 1; j < 4; ++ j) {
            EXPECT_DOUBLE_EQ(mat(2 + i, 2 + j), mat(2 + j, 2 + i));
        }
    }
}


TEST_F(HGridTest, BuildGridCenters) {
    arc::HGrid grid(*map_, gridParams(), astarOptions());
    const Eigen::Vector3d pos(1.0, 2.0, 3.0);
    std::vector<Eigen::Vector3d> centers;
    grid.buildGridCenters({ 0, 2 }, pos, centers);
    ASSERT_EQ(centers.size(), 3u);
    EXPECT_TRUE(centers.front().isApprox(pos));
}

TEST_F(HGridTest, NextGrid) {
    arc::HGrid grid(*map_, gridParams(), astarOptions());
    Eigen::Vector3d pos;
    double yaw = 0.0;

    EXPECT_TRUE(grid.nextGrid({ 0, 2 }, pos, yaw));
    EXPECT_NEAR(pos(0), 4.5, 1e-9);
    EXPECT_NEAR(pos(1), 1.5, 1e-9);
    EXPECT_NEAR(yaw, 0.0, 1e-9);

    const int coarseNum = arc::HGridTestPeer::coarseNum(grid);
    std::vector<int> fines;
    arc::HGridTestPeer::coarseToFine(grid, 0, fines);
    std::vector<int> sameCoarse;
    for (int fine : fines) {
        sameCoarse.push_back(fine + coarseNum);
    }
    EXPECT_FALSE(grid.nextGrid(sameCoarse, pos, yaw));
    EXPECT_FALSE(grid.nextGrid({}, pos, yaw));
    EXPECT_FALSE(grid.nextGrid({ 0 }, pos, yaw));
}

TEST_F(HGridTest, FrontiersInGrid) {
    arc::HGrid grid(*map_, gridParams(), astarOptions());
    const int coarseNum = arc::HGridTestPeer::coarseNum(grid);

    arc::HGridTestPeer::coarseGrid(grid, 0).frontierIndices.insert(3);
    arc::HGridTestPeer::coarseGrid(grid, 0).frontierIndices.insert(7);
    arc::HGridTestPeer::coarseGrid(grid, 1).frontierIndices.insert(9);

    std::vector<int> frontierIndices;
    grid.frontiersInGrid({ 0 }, frontierIndices);
    std::sort(frontierIndices.begin(), frontierIndices.end());
    EXPECT_EQ(frontierIndices, (std::vector<int>{ 3, 7 }));

    std::vector<int> fines;
    arc::HGridTestPeer::coarseToFine(grid, 0, fines);
    arc::HGridTestPeer::fineGrid(grid, fines[0]).frontierIndices.insert(5);
    arc::HGridTestPeer::fineGrid(grid, fines[1]).frontierIndices.insert(6);
    arc::HGridTestPeer::fineGrid(grid, fines[2]).frontierIndices.insert(8);

    std::vector<int> allocated = { fines[0] + coarseNum, fines[1] + coarseNum };
    grid.frontiersInGrid(allocated, frontierIndices);
    std::sort(frontierIndices.begin(), frontierIndices.end());
    EXPECT_EQ(frontierIndices, (std::vector<int>{ 5, 6 }));

    grid.frontiersInGrid({}, frontierIndices);
    EXPECT_TRUE(frontierIndices.empty());
}

TEST_F(HGridTest, UpdateDividesCoarseGrid) {
    fillLogitBox(Eigen::Vector3d(0.1, 0.1, -0.4), Eigen::Vector3d(0.4, 2.9, 1.4), 0.0);
    auto& md = map_->data();
    md.updateMin = Eigen::Vector3d(0.0, 0.0, -0.5);
    md.updateMax = Eigen::Vector3d(3.0, 3.0, 1.5);

    arc::HGrid grid(*map_, gridParams(), astarOptions());
    arc::HGridTestPeer::setInitialized(grid, true);
    arc::GridIdList gridIds;
    gridIds.assign({ 0 });
    std::vector<int> first;
    std::vector<int> second;
    grid.updateGridData(1, gridIds, {}, first, second);

    const int coarseNum = arc::HGridTestPeer::coarseNum(grid);
    EXPECT_FALSE(arc::HGridTestPeer::coarseGrid(grid, 0).isActive);
    EXPECT_EQ(gridIds.size(), 4u);
    for (int id : gridIds) {
        EXPECT_GE(id, coarseNum);
        EXPECT_TRUE(arc::HGridTestPeer::fineGrid(grid, id - coarseNum).isActive);
    }

    std::vector<int> activeIds;
    grid.activeGrids(activeIds);
    EXPECT_EQ(activeIds.size(), 7u);
}

TEST_F(HGridTest, CenterAndUnknownQueries) {
    arc::HGrid grid(*map_, gridParams(), astarOptions());
    const int coarseNum = arc::HGridTestPeer::coarseNum(grid);
    for (int i = 0; i < coarseNum; ++ i) {
        EXPECT_GT(grid.unknownCellsOf(i), 0);
        const Eigen::Vector3d center = grid.centerOf(i);
        EXPECT_GE(center(0), 0.0);
        EXPECT_LE(center(0), 6.0);
        EXPECT_GE(center(1), 0.0);
        EXPECT_LE(center(1), 6.0);
    }
    EXPECT_TRUE(grid.sameCoarseParent(0, 0));
    EXPECT_FALSE(grid.sameCoarseParent(0, 1));
    EXPECT_TRUE(grid.sameCoarseParent(0, coarseNum));
}

} // namespace