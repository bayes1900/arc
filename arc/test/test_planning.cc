#include <gtest/gtest.h>

#include <base/iarc.h>
#include <env/sdfmap.h>
#include <search/astar.h>
#include <search/graphsearch.h>
#include <explore/frontier/viewnode.h>

namespace {

class PlanningTest : public ::testing::Test {
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
        createMap(4.0, 4.0, 3.0, 0.2);
    }

    void TearDown() override {
        map_.reset();
        node_.reset();
    }

    void createMap(double xlen, double ylen, double zlen, double resolution) {
        rclcpp::NodeOptions options;
        options.parameter_overrides({
            rclcpp::Parameter("sdfmap.xlen", xlen),
            rclcpp::Parameter("sdfmap.ylen", ylen),
            rclcpp::Parameter("sdfmap.zlen", zlen),
            rclcpp::Parameter("sdfmap.resolution", resolution),
            rclcpp::Parameter("sdfmap.inflation", 0.0),
            rclcpp::Parameter("sdfmap.maxRayLength", 10.0),
            rclcpp::Parameter("sdfmap.phit", 0.65),
            rclcpp::Parameter("sdfmap.pmiss", 0.35),
            rclcpp::Parameter("sdfmap.pmin", 0.12),
            rclcpp::Parameter("sdfmap.pmax", 0.90),
            rclcpp::Parameter("sdfmap.pocc", 0.80),
        });
        map_.reset();
        node_.reset();
        node_ = std::make_shared<rclcpp::Node>("test_planning", options);
        map_ = std::make_unique<arc::SdfMap>(node_);
    }

    void fillAll(uint8_t state) {
        const auto& mp = map_->params();
        auto& md = map_->data();
        for (int x = 0; x < mp.voxelShape(0); ++ x) {
            for (int y = 0; y < mp.voxelShape(1); ++ y) {
                for (int z = 0; z < mp.voxelShape(2); ++ z) {
                    const Eigen::Vector3i idx(x, y, z);
                    const size_t addr = mp.getVoxelIdx(idx);
                    if (state == arc::VoxelOccupancyCode::OCCUPIED) {
                        md.occupancyLogits[addr] = mp.lmax;
                    } else if (state == arc::VoxelOccupancyCode::FREE) {
                        md.occupancyLogits[addr] = 0.0;
                    } else {
                        md.occupancyLogits[addr] = mp.lmin - 0.01;
                    }
                }
            }
        }
        rebuildInflate();
    }

    void fillBox(
        const Eigen::Vector3d& low,
        const Eigen::Vector3d& high,
        uint8_t state
    ) {
        const auto& mp = map_->params();
        auto& md = map_->data();
        const Eigen::Vector3i minIdx = mp.p2i(low);
        const Eigen::Vector3i maxIdx = mp.p2i(high);
        for (int x = std::max(0, minIdx(0)); x <= std::min(mp.voxelShape(0) - 1, maxIdx(0)); ++ x) {
            for (int y = std::max(0, minIdx(1)); y <= std::min(mp.voxelShape(1) - 1, maxIdx(1)); ++ y) {
                for (int z = std::max(0, minIdx(2)); z <= std::min(mp.voxelShape(2) - 1, maxIdx(2)); ++ z) {
                    const size_t addr = mp.getVoxelIdx(Eigen::Vector3i(x, y, z));
                    if (state == arc::VoxelOccupancyCode::OCCUPIED) {
                        md.occupancyLogits[addr] = mp.lmax;
                    } else if (state == arc::VoxelOccupancyCode::FREE) {
                        md.occupancyLogits[addr] = 0.0;
                    } else {
                        md.occupancyLogits[addr] = mp.lmin - 0.01;
                    }
                }
            }
        }
        rebuildInflate();
    }

    void rebuildInflate() {
        const auto& mp = map_->params();
        auto& md = map_->data();
        md.localBoundMin.setZero();
        md.localBoundMax = mp.voxelShape - Eigen::Vector3i::Ones();
        md.updateInflateOccupancy();
    }

    bool pathFree(const std::vector<Eigen::Vector3d>& path) const {
        for (const auto& point : path) {
            if (!map_->params().isInMap(point)) {
                return false;
            }
            if (map_->data().getInflateOccupancy(point) != arc::VoxelInflateOccupancyCode::FREE) {
                return false;
            }
        }
        return true;
    }

    arc::AstarOptions astarOptions(double resolution) const {
        arc::AstarOptions options;
        options.resolution = resolution;
        options.maxNodes = 200000;
        options.collisionCheckStep = 0.1;
        return options;
    }

    double bruteForceDijkstra(const Eigen::Vector3i& start, const Eigen::Vector3i& goal) const {
        const auto& mp = map_->params();
        const auto& md = map_->data();
        const int numX = mp.voxelShape(0);
        const int numY = mp.voxelShape(1);
        const int numZ = mp.voxelShape(2);
        std::vector<double> distance(md.numVoxels, std::numeric_limits<double>::infinity());
        std::vector<uint8_t> closed(md.numVoxels, 0);
        using Entry = std::pair<double, size_t>;
        std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> openSet;

        const size_t startAddress = mp.getVoxelIdx(start);
        const size_t goalAddress = mp.getVoxelIdx(goal);
        distance[startAddress] = 0.0;
        openSet.push({0.0, startAddress});

        while (!openSet.empty()) {
            const auto [cost, address] = openSet.top();
            openSet.pop();
            if (closed[address]) {
                continue;
            }
            closed[address] = 1;
            if (address == goalAddress) {
                return cost;
            }

            size_t remainder = address;
            const int z = remainder % numZ;
            remainder /= numZ;
            const int y = remainder % numY;
            const int x = remainder / numY;

            for (int dx = -1; dx <= 1; ++ dx) {
                for (int dy = -1; dy <= 1; ++ dy) {
                    for (int dz = -1; dz <= 1; ++ dz) {
                        if (dx == 0 && dy == 0 && dz == 0) {
                            continue;
                        }
                        const int nx = x + dx;
                        const int ny = y + dy;
                        const int nz = z + dz;
                        if (nx < 0 || ny < 0 || nz < 0 ||
                            nx >= numX || ny >= numY || nz >= numZ) {
                            continue;
                        }
                        const Eigen::Vector3i neighbor(nx, ny, nz);
                        if (md.getInflateOccupancy(neighbor) != arc::VoxelInflateOccupancyCode::FREE) {
                            continue;
                        }
                        const size_t neighborAddress = mp.getVoxelIdx(neighbor);
                        const double nextCost = cost + mp.resolution *
                            Eigen::Vector3d(dx, dy, dz).norm();
                        if (nextCost < distance[neighborAddress]) {
                            distance[neighborAddress] = nextCost;
                            openSet.push({nextCost, neighborAddress});
                        }
                    }
                }
            }
        }
        return std::numeric_limits<double>::infinity();
    }

    rclcpp::Node::SharedPtr node_;
    std::unique_ptr<arc::SdfMap> map_;
};

TEST_F(PlanningTest, AstarEmptyMapAndResolution) {
    fillAll(arc::VoxelOccupancyCode::FREE);
    arc::Astar astar(*map_, astarOptions(0.2));

    const Eigen::Vector3d start(-1.5, 0.0, 1.0);
    const Eigen::Vector3d end(1.5, 0.0, 1.0);
    ASSERT_EQ(astar.search(start, end, false), arc::AstarSearchStatus::DONE);
    auto path = astar.getPath();
    ASSERT_GE(path.size(), 2u);
    EXPECT_NEAR((path.front() - start).norm(), 0.0, 1e-9);
    EXPECT_NEAR((path.back() - end).norm(), 0.0, 1e-9);
    EXPECT_TRUE(pathFree(path));
    EXPECT_LT(std::fabs(arc::misc::pathLength(path) - 3.0), 0.05 * 3.0);

    arc::Astar coarseAstar(*map_, astarOptions(0.4));
    ASSERT_EQ(coarseAstar.search(start, end, false), arc::AstarSearchStatus::DONE);
    EXPECT_TRUE(pathFree(coarseAstar.getPath()));
}

TEST_F(PlanningTest, AstarRespectsBox) {
    fillAll(arc::VoxelOccupancyCode::FREE);
    arc::Astar astar(*map_, astarOptions(0.2));
    astar.setBox(
        Eigen::Vector3d(-0.4, -1.0, 0.5),
        Eigen::Vector3d(0.4, 1.0, 1.5));

    const Eigen::Vector3d inside(-0.2, 0.0, 1.0);
    const Eigen::Vector3d outside(1.0, 0.0, 1.0);
    EXPECT_TRUE(astar.isLineFree(inside, inside));
    EXPECT_FALSE(astar.isLineFree(inside, outside));
    EXPECT_EQ(astar.search(inside, outside, true), arc::AstarSearchStatus::OUTSIDE);
}

TEST_F(PlanningTest, AstarWallAndSealedRoom) {
    fillAll(arc::VoxelOccupancyCode::FREE);
    fillBox(
        Eigen::Vector3d(-0.1, -1.8, 0.4),
        Eigen::Vector3d(0.1, 0.6, 2.9),
        arc::VoxelOccupancyCode::OCCUPIED
    );

    arc::Astar astar(*map_, astarOptions(0.2));
    const Eigen::Vector3d start(-1.5, 0.0, 1.0);
    const Eigen::Vector3d end(1.5, 0.0, 1.0);
    ASSERT_EQ(astar.search(start, end, false), arc::AstarSearchStatus::DONE);
    const auto path = astar.getPath();
    EXPECT_GT(path.size(), 2u);
    EXPECT_TRUE(pathFree(path));
    EXPECT_GT(arc::misc::pathLength(path), 3.0 * 1.1);

    fillAll(arc::VoxelOccupancyCode::FREE);
    const double minX = 0.5;
    const double maxX = 1.5;
    const double minY = -0.5;
    const double maxY = 0.5;
    const double minZ = 0.5;
    const double maxZ = 1.5;
    const double thickness = 0.2;
    fillBox(Eigen::Vector3d(minX, minY, minZ), Eigen::Vector3d(minX + thickness, maxY, maxZ),
        arc::VoxelOccupancyCode::OCCUPIED);
    fillBox(Eigen::Vector3d(maxX - thickness, minY, minZ), Eigen::Vector3d(maxX, maxY, maxZ),
        arc::VoxelOccupancyCode::OCCUPIED);
    fillBox(Eigen::Vector3d(minX, minY, minZ), Eigen::Vector3d(maxX, minY + thickness, maxZ),
        arc::VoxelOccupancyCode::OCCUPIED);
    fillBox(Eigen::Vector3d(minX, maxY - thickness, minZ), Eigen::Vector3d(maxX, maxY, maxZ),
        arc::VoxelOccupancyCode::OCCUPIED);
    fillBox(Eigen::Vector3d(minX, minY, minZ), Eigen::Vector3d(maxX, maxY, minZ + thickness),
        arc::VoxelOccupancyCode::OCCUPIED);
    fillBox(Eigen::Vector3d(minX, minY, maxZ - thickness), Eigen::Vector3d(maxX, maxY, maxZ),
        arc::VoxelOccupancyCode::OCCUPIED);
    EXPECT_EQ(astar.search(start, Eigen::Vector3d(1.0, 0.0, 1.0), false),
        arc::AstarSearchStatus::NOPATH);
    EXPECT_TRUE(astar.getPath().empty());
}

TEST_F(PlanningTest, AstarUnknownSpace) {
    fillAll(arc::VoxelOccupancyCode::FREE);
    fillBox(
        Eigen::Vector3d(-0.2, -2.0, -0.5),
        Eigen::Vector3d(0.2, 2.0, 3.0),
        arc::VoxelOccupancyCode::UNKNOWN
    );
    EXPECT_EQ(map_->data().getOccupancy(Eigen::Vector3d(0.0, 0.0, 1.0)),
        arc::VoxelOccupancyCode::UNKNOWN);
    EXPECT_EQ(map_->data().getOccupancy(Eigen::Vector3d(-1.5, 0.0, 1.0)),
        arc::VoxelOccupancyCode::FREE);

    arc::Astar astar(*map_, astarOptions(0.2));
    const Eigen::Vector3d start(-1.5, 0.0, 1.0);
    const Eigen::Vector3d end(1.5, 0.0, 1.0);
    EXPECT_EQ(astar.search(start, end, false), arc::AstarSearchStatus::NOPATH);
    EXPECT_EQ(astar.search(start, end, true), arc::AstarSearchStatus::DONE);
    EXPECT_TRUE(pathFree(astar.getPath()));
}

TEST_F(PlanningTest, AstarMatchesDijkstra) {
    createMap(4.0, 4.0, 1.0, 0.2);
    fillAll(arc::VoxelOccupancyCode::FREE);

    const Eigen::Vector3d start(-1.5, 0.0, 0.0);
    const Eigen::Vector3d end(1.5, 0.0, 0.0);
    const auto& mp = map_->params();
    const Eigen::Vector3i startIndex = mp.p2i(start);
    const Eigen::Vector3i endIndex = mp.p2i(end);

    arc::Astar astar(*map_, astarOptions(0.2));
    ASSERT_EQ(astar.search(start, end, false), arc::AstarSearchStatus::DONE);
    const double astarCost = arc::misc::pathLength(astar.getPath());
    const double dijkstraCost = bruteForceDijkstra(startIndex, endIndex);
    ASSERT_TRUE(std::isfinite(dijkstraCost));
    EXPECT_LT(std::fabs(astarCost - dijkstraCost), 0.01 * std::max(1.0, dijkstraCost));
}

TEST_F(PlanningTest, DistanceGradientAndLineCheck) {
    fillAll(arc::VoxelOccupancyCode::FREE);
    const auto& mp = map_->params();
    auto& md = map_->data();
    for (int x = 0; x < mp.voxelShape(0); ++ x) {
        for (int y = 0; y < mp.voxelShape(1); ++ y) {
            for (int z = 0; z < mp.voxelShape(2); ++ z) {
                const Eigen::Vector3d pos = mp.i2p(Eigen::Vector3i(x, y, z));
                md.distance[mp.getVoxelIdx(x, y, z)] = pos.x() - mp.origin.x();
            }
        }
    }

    Eigen::Vector3d grad;
    const double dist = map_->getDistWithGrad(Eigen::Vector3d(-0.5, 0.0, 1.0), grad);
    EXPECT_GT(dist, 0.0);
    EXPECT_NEAR(grad.x(), 1.0, 0.1);
    EXPECT_NEAR(grad.y(), 0.0, 0.1);
    EXPECT_NEAR(grad.z(), 0.0, 0.1);

    EXPECT_TRUE(map_->isLineFree(
        Eigen::Vector3d(-1.0, 0.0, 1.0),
        Eigen::Vector3d(1.0, 0.0, 1.0)));
    fillBox(
        Eigen::Vector3d(-0.1, -0.5, 0.5),
        Eigen::Vector3d(0.1, 0.5, 1.5),
        arc::VoxelOccupancyCode::OCCUPIED
    );
    EXPECT_FALSE(map_->isLineFree(
        Eigen::Vector3d(-1.0, 0.0, 1.0),
        Eigen::Vector3d(1.0, 0.0, 1.0)));
}

TEST_F(PlanningTest, ViewNodeCostAndSearchPath) {
    fillAll(arc::VoxelOccupancyCode::FREE);
    std::vector<Eigen::Vector3d> path;

    const Eigen::Vector3d p1(-1.0, 0.0, 1.0);
    const Eigen::Vector3d p2(1.0, 0.0, 1.0);
    arc::ViewNodeParams params;
    arc::AstarOptions options;
    options.resolution = 0.4;
    arc::ViewNodeContext context(*map_, params, options);
    arc::ViewNode viewNode(context, p1, 0.0);
    EXPECT_NEAR(context.costBetween(
        p1, p2, 0.0, 0.0, Eigen::Vector3d::Zero(), 0.0, path),
        2.0 / context.params().vm, 1e-6);
    EXPECT_EQ(path.size(), 2u);

    arc::ViewNode secondNode(context, p2, 0.0);
    EXPECT_NEAR(viewNode.costTo(secondNode), 2.0 / context.params().vm, 1e-6);

    EXPECT_NEAR(context.costBetween(
        p1, p1, 0.0, 3.14159265358979323846 / 2.0,
        Eigen::Vector3d::Zero(), 0.0, path),
        (3.14159265358979323846 / 2.0) / context.params().yd, 1e-6);

    const double aligned = context.costBetween(
        p1, p2, 0.0, 0.0, Eigen::Vector3d(1.0, 0.0, 0.0), 0.0, path);
    const double reversed = context.costBetween(
        p1, p2, 0.0, 0.0, Eigen::Vector3d(-1.0, 0.0, 0.0), 0.0, path);
    EXPECT_NEAR(reversed - aligned, context.params().wDir * 3.14159265358979323846, 1e-6);

    EXPECT_NEAR(context.findPath(p1, p2, path), 2.0, 1e-6);
    EXPECT_EQ(path.size(), 2u);

    fillBox(
        Eigen::Vector3d(-0.1, -1.8, 0.4),
        Eigen::Vector3d(0.1, 0.6, 2.9),
        arc::VoxelOccupancyCode::OCCUPIED
    );
    EXPECT_GT(context.findPath(p1, p2, path), 2.0);
    EXPECT_GT(path.size(), 2u);

    const double normalized = context.costBetween(
        p1, p1, 0.0, 0.2, Eigen::Vector3d::Zero(), 0.0, path);
    const double wrapped = context.costBetween(
        p1, p1, 0.0, 2.0 * 3.14159265358979323846 + 0.2,
        Eigen::Vector3d::Zero(), 0.0, path);
    EXPECT_NEAR(normalized, wrapped, 1e-9);

    fillAll(arc::VoxelOccupancyCode::FREE);
    const double minX = 0.5;
    const double maxX = 1.5;
    const double minY = -0.5;
    const double maxY = 0.5;
    const double minZ = 0.5;
    const double maxZ = 1.5;
    const double thickness = 0.2;
    fillBox(Eigen::Vector3d(minX, minY, minZ), Eigen::Vector3d(minX + thickness, maxY, maxZ),
        arc::VoxelOccupancyCode::OCCUPIED);
    fillBox(Eigen::Vector3d(maxX - thickness, minY, minZ), Eigen::Vector3d(maxX, maxY, maxZ),
        arc::VoxelOccupancyCode::OCCUPIED);
    fillBox(Eigen::Vector3d(minX, minY, minZ), Eigen::Vector3d(maxX, minY + thickness, maxZ),
        arc::VoxelOccupancyCode::OCCUPIED);
    fillBox(Eigen::Vector3d(minX, maxY - thickness, minZ), Eigen::Vector3d(maxX, maxY, maxZ),
        arc::VoxelOccupancyCode::OCCUPIED);
    fillBox(Eigen::Vector3d(minX, minY, minZ), Eigen::Vector3d(maxX, maxY, minZ + thickness),
        arc::VoxelOccupancyCode::OCCUPIED);
    fillBox(Eigen::Vector3d(minX, minY, maxZ - thickness), Eigen::Vector3d(maxX, maxY, maxZ),
        arc::VoxelOccupancyCode::OCCUPIED);
    EXPECT_NEAR(context.findPath(
        p1, Eigen::Vector3d(1.0, 0.0, 1.0), path),
        context.params().noPathPenaltyLength, 1e-9);
}

TEST_F(PlanningTest, ViewNodeNoPathPenaltyScalesWithMap) {
    createMap(20.0, 20.0, 6.0, 0.2);
    fillAll(arc::VoxelOccupancyCode::FREE);

    arc::ViewNodeParams params;
    params.noPathPenaltyFactor = 10.0;
    params.minNoPathPenaltyLength = 100.0;

    arc::AstarOptions options;
    options.resolution = 0.4;
    arc::ViewNodeContext context(*map_, params, options);

    const double expected = std::max(
        100.0,
        params.noPathPenaltyFactor * map_->params().size.norm());
    EXPECT_NEAR(context.params().noPathPenaltyLength, expected, 1e-9);
    EXPECT_GT(context.params().noPathPenaltyLength, 100.0);
}

TEST_F(PlanningTest, GraphSearchChoosesBestSequence) {
    fillAll(arc::VoxelOccupancyCode::FREE);

    arc::ViewNodeParams params;
    params.vm = 1.0;
    params.wDir = 0.0;
    arc::AstarOptions options;
    options.resolution = 0.4;
    arc::ViewNodeContext context(*map_, params, options);

    arc::ViewNode start(context, Eigen::Vector3d(-1.0, 0.0, 1.0), 0.0);
    arc::ViewNode a(context, Eigen::Vector3d(0.0, 0.0, 1.0), 0.0);
    arc::ViewNode b(context, Eigen::Vector3d(0.0, 0.5, 1.0), 0.0);
    arc::ViewNode goal(context, Eigen::Vector3d(1.0, 0.0, 1.0), 0.0);

    arc::GraphSearch<arc::ViewNode> search;
    search.addNode(&start);   // 0
    search.addNode(&a);       // 1
    search.addNode(&b);       // 2
    search.addNode(&goal);    // 3

    search.addEdge(0, 1);     // 1.0
    search.addEdge(1, 3);     // 1.0
    search.addEdge(0, 2);     // sqrt(1.25) ≈ 1.118
    search.addEdge(2, 3);     // sqrt(1.25) ≈ 1.118

    EXPECT_EQ(search.nodeNum(), 4);
    EXPECT_EQ(search.edgeNum(), 4);

    std::vector<arc::ViewNode*> path;
    ASSERT_TRUE(search.dijkstraSearch(0, 3, path));
    ASSERT_EQ(path.size(), 3u);
    EXPECT_EQ(path[0], &start);
    EXPECT_EQ(path[1], &a);
    EXPECT_EQ(path[2], &goal);
    EXPECT_NEAR(search.gValue(path.back()), 2.0, 1e-6);

    std::vector<arc::ViewNode*> pathSecond;
    ASSERT_TRUE(search.dijkstraSearch(0, 3, pathSecond));
    ASSERT_EQ(pathSecond.size(), path.size());
}

TEST_F(PlanningTest, GraphSearchReportsUnreachable) {
    fillAll(arc::VoxelOccupancyCode::FREE);

    arc::ViewNodeParams params;
    params.vm = 1.0;
    params.wDir = 0.0;
    arc::AstarOptions options;
    options.resolution = 0.4;
    arc::ViewNodeContext context(*map_, params, options);

    arc::ViewNode start(context, Eigen::Vector3d(-1.0, 0.0, 1.0), 0.0);
    arc::ViewNode middle(context, Eigen::Vector3d(0.0, 0.0, 1.0), 0.0);
    arc::ViewNode goal(context, Eigen::Vector3d(1.0, 0.0, 1.0), 0.0);

    arc::GraphSearch<arc::ViewNode> search;
    search.addNode(&start);
    search.addNode(&middle);
    search.addNode(&goal);
    search.addEdge(0, 1);

    std::vector<arc::ViewNode*> path;
    EXPECT_FALSE(search.dijkstraSearch(0, 2, path));
    EXPECT_TRUE(path.empty());
    EXPECT_FALSE(search.dijkstraSearch(0, 42, path));
    EXPECT_FALSE(search.dijkstraSearch(-1, 0, path));
}

TEST_F(PlanningTest, AstarPerformance) {
    createMap(20.0, 20.0, 6.0, 0.2);
    fillAll(arc::VoxelOccupancyCode::FREE);
    arc::Astar astar(*map_, astarOptions(0.2));

    const Eigen::Vector3d start(-9.5, -9.5, 1.0);
    const Eigen::Vector3d end(9.5, 9.5, 5.0);
    double bestTime = std::numeric_limits<double>::max();
    for (int i = 0; i < 3; ++ i) {
        const auto t1 = std::chrono::steady_clock::now();
        ASSERT_EQ(astar.search(start, end, false), arc::AstarSearchStatus::DONE);
        const auto t2 = std::chrono::steady_clock::now();
        bestTime = std::min(
            bestTime,
            std::chrono::duration<double, std::milli>(t2 - t1).count());
    }
    std::cout << "A* 100x100x30 search: " << bestTime << " ms" << std::endl;
    EXPECT_LT(bestTime, 50.0);
}

} // namespace
