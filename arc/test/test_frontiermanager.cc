#include <gtest/gtest.h>

#include <base/iarc.h>
#include <env/sdfmap.h>
#include <explore/frontier/frontiermanager.h>
#include <explore/frontier/viewnode.h>

namespace arc {

class FrontierManagerTestPeer {
public:
    static const std::vector<Frontier>& frontiers(const FrontierManager& manager) {
        return manager.activeFrontiers_;
    }
    static const std::vector<Frontier>& dormant(const FrontierManager& manager) {
        return manager.dormantFrontiers_;
    }
    static const std::vector<Frontier>& temporary(const FrontierManager& manager) {
        return manager.pendingFrontiers_;
    }
    static const std::vector<int>& removedIds(const FrontierManager& manager) {
        return manager.removedFrontierIndices_;
    }
};

} // arc

namespace {

constexpr double RES = 0.1;
constexpr double MAP_X = 8.0;
constexpr double MAP_Y = 8.0;
constexpr double MAP_Z = 4.0;

class FrontierManagerTest : public ::testing::Test {
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
            rclcpp::Parameter("sdfmap.xlen", MAP_X),
            rclcpp::Parameter("sdfmap.ylen", MAP_Y),
            rclcpp::Parameter("sdfmap.zlen", MAP_Z),
            rclcpp::Parameter("sdfmap.resolution", RES),
            rclcpp::Parameter("sdfmap.inflation", 0.0),
            rclcpp::Parameter("sdfmap.maxRayLength", 10.0),
            rclcpp::Parameter("sdfmap.phit", 0.65),
            rclcpp::Parameter("sdfmap.pmiss", 0.35),
            rclcpp::Parameter("sdfmap.pmin", 0.12),
            rclcpp::Parameter("sdfmap.pmax", 0.90),
            rclcpp::Parameter("sdfmap.pocc", 0.80),
        });
        node_ = std::make_shared<rclcpp::Node>("test_frontiermanager", options);
        map_ = std::make_unique<arc::SdfMap>(node_);

        arc::AstarOptions astarOptions;
        astarOptions.resolution = RES;
        viewNode_ = std::make_unique<arc::ViewNodeContext>(*map_, arc::ViewNodeParams(), astarOptions);
        manager_ = std::make_unique<arc::FrontierManager>(*map_, *viewNode_, params_);
    }

    void TearDown() override {
        manager_.reset();
        viewNode_.reset();
        map_.reset();
        node_.reset();
    }

    void setAll(uint8_t state) {
        auto& md = map_->data();
        const double value = logitOf(state);
        std::fill(md.occupancyLogits.begin(), md.occupancyLogits.end(), value);
        rebuildInflate_();
    }

    void fillBox(const Eigen::Vector3d& low, const Eigen::Vector3d& high, uint8_t state) {
        const auto& mp = map_->params();
        auto& md = map_->data();
        const Eigen::Vector3i minIndex = mp.p2i(low).cwiseMax(0);
        const Eigen::Vector3i maxIndex = mp.p2i(high).cwiseMin(mp.voxelShape - Eigen::Vector3i::Ones());
        const double value = logitOf(state);
        for (int x = minIndex(0); x <= maxIndex(0); ++ x) {
            for (int y = minIndex(1); y <= maxIndex(1); ++ y) {
                for (int z = minIndex(2); z <= maxIndex(2); ++ z) {
                    md.occupancyLogits[mp.getVoxelIdx(Eigen::Vector3i(x, y, z))] = value;
                }
            }
        }
        rebuildInflate_();
    }

    double logitOf(uint8_t state) const {
        const auto& mp = map_->params();
        if (state == arc::VoxelOccupancyCode::FREE) {
            return 0.0;
        }
        if (state == arc::VoxelOccupancyCode::OCCUPIED) {
            return mp.lmax;
        }
        return mp.lmin - 0.01;
    }

    void rebuildInflate_() {
        auto& md = map_->data();
        md.localBoundMin.setZero();
        md.localBoundMax = map_->params().voxelShape - Eigen::Vector3i::Ones();
        md.updateInflateOccupancy();
    }

    void commit() {
        rebuildInflate_();
        const auto& mp = map_->params();
        map_->data().updateMin = mp.origin + Eigen::Vector3d::Constant(RES);
        map_->data().updateMax = mp.endpoint - Eigen::Vector3d::Constant(RES);
    }

    void makeRoom() {
        setAll(arc::VoxelOccupancyCode::UNKNOWN);
        fillBox(Eigen::Vector3d(-3.5, -3.5, 0.6), Eigen::Vector3d(3.5, 3.5, 2.4), arc::VoxelOccupancyCode::FREE);
        commit();
    }

    void makeWall() {
        setAll(arc::VoxelOccupancyCode::UNKNOWN);
        fillBox(Eigen::Vector3d(-3.0, -0.2, 0.6), Eigen::Vector3d(3.0, 0.2, 2.4), arc::VoxelOccupancyCode::FREE);
        commit();
    }

    void search() {
        manager_->detectFrontiers();
        manager_->sampleNewFrontierViewpoints();
    }

    std::shared_ptr<rclcpp::Node> node_;
    std::unique_ptr<arc::SdfMap> map_;
    std::unique_ptr<arc::ViewNodeContext> viewNode_;
    std::unique_ptr<arc::FrontierManager> manager_;
    arc::FrontierParams params_;
};

TEST_F(FrontierManagerTest, EmptyMapHasNoFrontier) {
    setAll(arc::VoxelOccupancyCode::UNKNOWN);
    commit();
    search();

    EXPECT_TRUE(arc::FrontierManagerTestPeer::temporary(*manager_).empty());
    EXPECT_TRUE(arc::FrontierManagerTestPeer::frontiers(*manager_).empty());
    EXPECT_TRUE(arc::FrontierManagerTestPeer::dormant(*manager_).empty());
}

TEST_F(FrontierManagerTest, AllFreeMapHasNoFrontier) {
    setAll(arc::VoxelOccupancyCode::FREE);
    commit();
    search();

    EXPECT_TRUE(arc::FrontierManagerTestPeer::temporary(*manager_).empty());
    EXPECT_TRUE(arc::FrontierManagerTestPeer::frontiers(*manager_).empty());
}

TEST_F(FrontierManagerTest, FreeRoomHasFrontiers) {
    makeRoom();
    search();

    const auto& temporary = arc::FrontierManagerTestPeer::temporary(*manager_);
    ASSERT_FALSE(temporary.empty());
    for (const auto& frontier : temporary) {
        ASSERT_FALSE(frontier.cells.empty());
        EXPECT_FALSE(frontier.downsampledCells.empty());
        EXPECT_LE(frontier.downsampledCells.size(), frontier.cells.size());
        for (const auto& cell : frontier.cells) {
            EXPECT_EQ(map_->data().getOccupancy(cell), arc::VoxelOccupancyCode::FREE);
        }
        EXPECT_LT(std::fabs(frontier.center(0)), 3.6);
        EXPECT_LT(std::fabs(frontier.center(1)), 3.6);
        EXPECT_GT(frontier.center(2), 0.5);
        EXPECT_LT(frontier.center(2), 2.5);
    }
}

TEST_F(FrontierManagerTest, LongWallIsSplit) {
    makeWall();
    search();

    const auto& temporary = arc::FrontierManagerTestPeer::temporary(*manager_);
    ASSERT_GE(temporary.size(), 2u);
    for (const auto& frontier : temporary) {
        ASSERT_FALSE(frontier.downsampledCells.empty());
        for (const auto& cell : frontier.downsampledCells) {
            EXPECT_LE((cell.head<2>() - frontier.center.head<2>()).norm(), params_.clusterSizeXY + RES);
        }
    }
}

TEST_F(FrontierManagerTest, ViewpointsAreSafeAndVisible) {
    makeRoom();
    search();

    const auto& frontiers = arc::FrontierManagerTestPeer::frontiers(*manager_);
    ASSERT_FALSE(frontiers.empty());
    EXPECT_TRUE(arc::FrontierManagerTestPeer::dormant(*manager_).empty());
    for (const auto& frontier : frontiers) {
        ASSERT_FALSE(frontier.viewpoints.empty());
        for (const auto& viewpoint : frontier.viewpoints) {
            EXPECT_TRUE(map_->params().isInMap(viewpoint.pos));
            EXPECT_EQ(map_->data().getInflateOccupancy(viewpoint.pos), arc::VoxelInflateOccupancyCode::FREE);
            EXPECT_GT(viewpoint.visibleCellCount, params_.minVisibleNum);
            EXPECT_LE(viewpoint.yaw, arc::__pi__ + 1e-6);
            EXPECT_GE(viewpoint.yaw, -arc::__pi__ - 1e-6);
        }
        for (size_t i = 1; i < frontier.viewpoints.size(); ++ i) {
            EXPECT_GE(frontier.viewpoints[i - 1].visibleCellCount, frontier.viewpoints[i].visibleCellCount);
        }
    }
}

TEST_F(FrontierManagerTest, GainOfViewDependsOnYaw) {
    makeRoom();

    const Eigen::Vector3d pos(3.0, 0.0, 1.5);
    const int inward = manager_->viewGain(pos, arc::__pi__);
    const int outward = manager_->viewGain(pos, 0.0);
    EXPECT_GE(inward, 0);
    EXPECT_GT(outward, 0);
    EXPECT_GE(outward, inward);
}

TEST_F(FrontierManagerTest, CostMatrixLayout) {
    makeRoom();
    search();
    manager_->refreshCostMatrix();

    const int count = static_cast<int>(arc::FrontierManagerTestPeer::frontiers(*manager_).size());
    ASSERT_GE(count, 2);

    Eigen::MatrixXd matrix;
    manager_->buildFullCostMatrix(Eigen::Vector3d(0.0, 0.0, 1.5), Eigen::Vector3d::Zero(), 0.0, matrix);
    EXPECT_EQ(matrix.rows(), count + 1);
    EXPECT_EQ(matrix.cols(), count + 1);
    for (int i = 1; i <= count; ++ i) {
        EXPECT_NEAR(matrix(i, i), 0.0, 1e-9);
        for (int j = 1; j <= count; ++ j) {
            EXPECT_NEAR(matrix(i, j), matrix(j, i), 1e-9);
            EXPECT_GE(matrix(i, j), 0.0);
        }
        EXPECT_GE(matrix(0, i), 0.0);
    }

    Eigen::MatrixXd swarm;
    manager_->buildSwarmCostMatrix({Eigen::Vector3d(0.0, 0.0, 1.5)}, {Eigen::Vector3d::Zero()}, {0.0}, swarm);
    EXPECT_EQ(swarm.rows(), count + 2);
    EXPECT_EQ(swarm.cols(), count + 2);
    EXPECT_DOUBLE_EQ(swarm(0, 1), -1000.0);
    EXPECT_DOUBLE_EQ(swarm(1, 0), 1000.0);
    for (int i = 0; i < swarm.rows(); ++ i) {
        EXPECT_DOUBLE_EQ(swarm(i, i), 1000.0);
    }

    Eigen::MatrixXd subset;
    manager_->buildSwarmCostMatrix({Eigen::Vector3d(0.0, 0.0, 1.5)}, {Eigen::Vector3d::Zero()}, {0.0}, {0, 1}, {}, subset);
    EXPECT_EQ(subset.rows(), 4);
    EXPECT_EQ(subset.cols(), 4);
    for (int i = 0; i < subset.rows(); ++ i) {
        EXPECT_DOUBLE_EQ(subset(i, i), 1000.0);
    }
    for (int j = 0; j < 2; ++ j) {
        EXPECT_DOUBLE_EQ(subset(1, 2 + j), swarm(1, 2 + j));
        for (int i = 0; i < 2; ++ i) {
            EXPECT_DOUBLE_EQ(subset(2 + i, 2 + j), swarm(2 + i, 2 + j));
        }
    }
}

TEST_F(FrontierManagerTest, MapQueriesMatchOccupancy) {
    makeRoom();
    const auto& md = map_->data();
    const auto& mp = map_->params();
    int freeMismatch = 0;
    int neighborMismatch = 0;
    int frontierMismatch = 0;
    int blockedMismatch = 0;
    for (int x = 0; x < mp.voxelShape(0); ++ x) {
        for (int y = 0; y < mp.voxelShape(1); ++ y) {
            for (int z = 0; z < mp.voxelShape(2); ++ z) {
                const Eigen::Vector3i index(x, y, z);
                const bool free = md.getOccupancy(index) == arc::VoxelOccupancyCode::FREE;
                bool neighborUnknown = false;
                for (int axis = 0; axis < 3; ++ axis) {
                    for (int sign = -1; sign <= 1; sign += 2) {
                        Eigen::Vector3i neighbor = index;
                        neighbor(axis) += sign;
                        if (md.getOccupancy(neighbor) == arc::VoxelOccupancyCode::UNKNOWN) {
                            neighborUnknown = true;
                        }
                    }
                }
                const bool blocked = md.getInflateOccupancy(index) != arc::VoxelInflateOccupancyCode::FREE ||
                    md.getOccupancy(index) == arc::VoxelOccupancyCode::UNKNOWN;
                freeMismatch += static_cast<int>(md.isFree(index) != free);
                neighborMismatch += static_cast<int>(md.isNeighborUnknown(index) != neighborUnknown);
                frontierMismatch += static_cast<int>(md.isFrontierCell(index) != (free && neighborUnknown));
                blockedMismatch += static_cast<int>(md.isRayBlocked(index) != blocked);
            }
        }
    }
    EXPECT_EQ(freeMismatch, 0);
    EXPECT_EQ(neighborMismatch, 0);
    EXPECT_EQ(frontierMismatch, 0);
    EXPECT_EQ(blockedMismatch, 0);
    EXPECT_TRUE(md.isRayBlocked(Eigen::Vector3i(-1, 0, 0)));
    EXPECT_TRUE(md.isRayBlocked(mp.voxelShape));
}

TEST_F(FrontierManagerTest, DeleteFrontierShrinksMatrix) {
    makeRoom();
    search();
    manager_->refreshCostMatrix();

    const int count = static_cast<int>(arc::FrontierManagerTestPeer::frontiers(*manager_).size());
    ASSERT_GE(count, 3);

    manager_->removeFrontiers({0});
    manager_->refreshCostMatrix();

    const auto& frontiers = arc::FrontierManagerTestPeer::frontiers(*manager_);
    ASSERT_EQ(static_cast<int>(frontiers.size()), count - 1);
    for (int i = 0; i < static_cast<int>(frontiers.size()); ++ i) {
        EXPECT_EQ(static_cast<int>(frontiers[i].edgeCosts.size()), count - 1);
        EXPECT_EQ(static_cast<int>(frontiers[i].edgePaths.size()), count - 1);
    }

    Eigen::MatrixXd matrix;
    manager_->buildFullCostMatrix(Eigen::Vector3d(0.0, 0.0, 1.5), Eigen::Vector3d::Zero(), 0.0, matrix);
    EXPECT_EQ(matrix.rows(), count);
}

TEST_F(FrontierManagerTest, AddFrontierComputesPendingCost) {
    makeRoom();
    search();
    manager_->refreshCostMatrix();

    const int count = static_cast<int>(arc::FrontierManagerTestPeer::frontiers(*manager_).size());
    ASSERT_GE(count, 3);

    manager_->addFrontiersFromViews({{Eigen::Vector3d(0.0, 0.0, 1.5), 0.0}});
    manager_->refreshCostMatrix();

    const auto& frontiers = arc::FrontierManagerTestPeer::frontiers(*manager_);
    ASSERT_EQ(static_cast<int>(frontiers.size()), count + 1);
    for (int i = 0; i < static_cast<int>(frontiers.size()); ++ i) {
        EXPECT_FALSE(frontiers[i].needsCostUpdate);
        EXPECT_EQ(static_cast<int>(frontiers[i].edgeCosts.size()), count + 1);
        EXPECT_EQ(static_cast<int>(frontiers[i].edgePaths.size()), count + 1);
    }
    EXPECT_DOUBLE_EQ(frontiers.back().edgeCosts.back(), 0.0);

    Eigen::MatrixXd matrix;
    manager_->buildFullCostMatrix(Eigen::Vector3d(0.0, 0.0, 1.5), Eigen::Vector3d::Zero(), 0.0, matrix);
    EXPECT_EQ(matrix.rows(), count + 2);
}

TEST_F(FrontierManagerTest, PathForTourIsContinuous) {
    makeRoom();
    search();
    manager_->refreshCostMatrix();

    std::vector<Eigen::Vector3d> path;
    manager_->buildPathForTour(Eigen::Vector3d(0.0, 0.0, 1.5), {0, 1}, path);
    ASSERT_GE(path.size(), 2u);
    for (size_t i = 1; i < path.size(); ++ i) {
        EXPECT_LT((path[i] - path[i - 1]).norm(), 5.0);
    }
}

TEST_F(FrontierManagerTest, FrontierCoveredDetectsChange) {
    makeRoom();
    search();

    ASSERT_FALSE(arc::FrontierManagerTestPeer::frontiers(*manager_).empty());
    EXPECT_FALSE(manager_->isAnyFrontierCovered());

    setAll(arc::VoxelOccupancyCode::FREE);
    commit();
    EXPECT_TRUE(manager_->isAnyFrontierCovered());
}

TEST_F(FrontierManagerTest, FrontierRemovedWhenMapChanges) {
    makeRoom();
    search();
    ASSERT_FALSE(arc::FrontierManagerTestPeer::frontiers(*manager_).empty());

    setAll(arc::VoxelOccupancyCode::FREE);
    commit();
    manager_->detectFrontiers();
    manager_->sampleNewFrontierViewpoints();

    EXPECT_FALSE(arc::FrontierManagerTestPeer::removedIds(*manager_).empty());

    manager_->refreshCostMatrix();
    EXPECT_TRUE(arc::FrontierManagerTestPeer::frontiers(*manager_).empty());
}

TEST_F(FrontierManagerTest, DeletedFrontierCanBeRediscovered) {
    makeRoom();
    search();
    const size_t count = arc::FrontierManagerTestPeer::frontiers(*manager_).size();
    ASSERT_GE(count, 2u);

    manager_->removeFrontiers({0});
    EXPECT_EQ(arc::FrontierManagerTestPeer::frontiers(*manager_).size(), count - 1);

    manager_->detectFrontiers();
    manager_->sampleNewFrontierViewpoints();
    EXPECT_GE(arc::FrontierManagerTestPeer::frontiers(*manager_).size(), count);
}

TEST_F(FrontierManagerTest, IncrementalSeedsRespectRevision) {
    makeRoom();
    search();

    auto& md = map_->data();
    md.updateRevision = 1;
    md.needsFullScan = false;
    manager_->detectFrontiers();
    manager_->sampleNewFrontierViewpoints();

    const size_t count = arc::FrontierManagerTestPeer::frontiers(*manager_).size();
    const size_t dormantCount = arc::FrontierManagerTestPeer::dormant(*manager_).size();
    const auto& mp = map_->params();
    const Eigen::Vector3d low(-0.3, -0.3, 1.2);
    const Eigen::Vector3d high(0.3, 0.3, 1.8);
    const Eigen::Vector3i minIndex = mp.p2i(low);
    const Eigen::Vector3i maxIndex = mp.p2i(high);
    for (int x = minIndex(0); x <= maxIndex(0); ++ x) {
        for (int y = minIndex(1); y <= maxIndex(1); ++ y) {
            for (int z = minIndex(2); z <= maxIndex(2); ++ z) {
                const size_t address = mp.getVoxelIdx(Eigen::Vector3i(x, y, z));
                md.occupancyLogits[address] = mp.lmin - 0.01;
                md.dirtyIndexes.push_back(address);
            }
        }
    }
    rebuildInflate_();
    md.updateMin = low;
    md.updateMax = high;

    manager_->detectFrontiers();
    EXPECT_TRUE(arc::FrontierManagerTestPeer::temporary(*manager_).empty());
    EXPECT_EQ(arc::FrontierManagerTestPeer::frontiers(*manager_).size(), count);

    md.updateRevision += 1;
    manager_->detectFrontiers();
    EXPECT_FALSE(arc::FrontierManagerTestPeer::temporary(*manager_).empty());
    manager_->sampleNewFrontierViewpoints();
    EXPECT_GT(
        arc::FrontierManagerTestPeer::frontiers(*manager_).size() +
            arc::FrontierManagerTestPeer::dormant(*manager_).size(),
        count + dormantCount
    );
}

TEST_F(FrontierManagerTest, FrameUpdatePopulatesDirtySeeds) {
    makeRoom();
    auto& md = map_->data();
    const auto& mp = map_->params();
    const size_t revision = md.updateRevision;
    const size_t dirty = md.dirtyIndexes.size();
    const size_t address = mp.getVoxelIdx(Eigen::Vector3i(0, 0, mp.p2i(Eigen::Vector3d(0.0, 0.0, 1.5))(2)));
    md.setFrameOccupancy(address, arc::VoxelOccupancyCode::OCCUPIED, 7);
    md.mergeFrameOccupancy(7);

    EXPECT_GT(md.updateRevision, revision);
    EXPECT_GT(md.dirtyIndexes.size(), dirty);
}

TEST_F(FrontierManagerTest, SearchTimeBudget) {
    makeWall();

    const auto start = std::chrono::steady_clock::now();
    search();
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() * 1000.0;

    ASSERT_FALSE(arc::FrontierManagerTestPeer::temporary(*manager_).empty());
    EXPECT_LT(elapsed, 200.0);
}

TEST_F(FrontierManagerTest, FrontierPerformance) {
    constexpr int iterations = 15;
    std::vector<double> detectTimes;
    std::vector<double> sampleTimes;
    std::vector<double> costTimes;
    detectTimes.reserve(iterations);
    sampleTimes.reserve(iterations);
    costTimes.reserve(iterations);

    auto milliseconds = [](const auto& begin, const auto& end) {
        return std::chrono::duration<double>(end - begin).count() * 1000.0;
    };
    auto median = [](std::vector<double> values) {
        std::sort(values.begin(), values.end());
        return values[values.size() / 2];
    };

    for (int i = 0; i < iterations; ++ i) {
        makeRoom();
        manager_ = std::make_unique<arc::FrontierManager>(*map_, *viewNode_, params_);

        const auto t0 = std::chrono::steady_clock::now();
        manager_->detectFrontiers();
        const auto t1 = std::chrono::steady_clock::now();
        manager_->sampleNewFrontierViewpoints();
        const auto t2 = std::chrono::steady_clock::now();
        manager_->refreshCostMatrix();
        const auto t3 = std::chrono::steady_clock::now();

        detectTimes.push_back(milliseconds(t0, t1));
        sampleTimes.push_back(milliseconds(t1, t2));
        costTimes.push_back(milliseconds(t2, t3));
    }

    const double detect = median(detectTimes);
    const double sample = median(sampleTimes);
    const double cost = median(costTimes);
    size_t downsampledCells = 0;
    for (const auto& frontier : arc::FrontierManagerTestPeer::frontiers(*manager_)) {
        downsampledCells += frontier.downsampledCells.size();
    }
    std::cout << "[perf] frontiers " << arc::FrontierManagerTestPeer::frontiers(*manager_).size()
              << ", filtered " << downsampledCells
              << ", detect " << detect << " ms, sample " << sample
              << " ms, cost " << cost << " ms, total " << detect + sample + cost << " ms" << std::endl;

    EXPECT_FALSE(arc::FrontierManagerTestPeer::frontiers(*manager_).empty());
    EXPECT_LT(detect + sample + cost, 5000.0);
}

} // namespace