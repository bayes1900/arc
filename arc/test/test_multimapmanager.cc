#include <gtest/gtest.h>

#include <algorithm>
#include <memory>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <base/iarc.h>
#include <env/multimapmanager.h>
#include <env/sdfmap.h>

namespace {

constexpr double RES = 0.2;

class MultiMapManagerTest : public ::testing::Test {
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
        createMap(1);
        createMap(2);
    }
    void TearDown() override {
        managerB_.reset();
        managerA_.reset();
        mapB_.reset();
        mapA_.reset();
        nodeB_.reset();
        nodeA_.reset();
    }
    void createMap(int droneId) {
        auto& node = droneId == 1 ? nodeA_ : nodeB_;
        auto& map = droneId == 1 ? mapA_ : mapB_;
        auto& manager = droneId == 1 ? managerA_ : managerB_;
        manager.reset();
        map.reset();
        node.reset();
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
        node = std::make_shared<rclcpp::Node>("test_multimap_" + std::to_string(droneId), options);
        map = std::make_unique<arc::SdfMap>(node);
        arc::MultiMapManagerParams params;
        params.droneId = droneId;
        params.droneNum = 2;
        params.chunkSize = 4;
        manager = std::make_unique<arc::MultiMapManager>(*map, params);
    }
    std::vector<size_t> makeAdrs(int count, int startX = 5, int startY = 5, int startZ = 2) {
        const auto& mp = mapA_->params();
        std::vector<size_t> adrs;
        adrs.reserve(static_cast<size_t>(count));
        for (int i = 0; i < count; ++ i) {
            adrs.push_back(mp.getVoxelIdx(startX + i, startY, startZ));
        }
        return adrs;
    }
    void setLogit(arc::SdfMap& map, size_t adr, bool occupied) {
        auto& md = map.data();
        md.occupancyLogits[adr] = occupied ? map.params().lmax : map.params().lmin;
    }
    bool isOccupied(const arc::SdfMap& map, size_t adr) const {
        const auto& mp = map.params();
        return map.data().getOccupancy(mp.voxelIdxToIndex(adr)) == arc::VoxelOccupancyCode::OCCUPIED;
    }
    bool isFree(const arc::SdfMap& map, size_t adr) const {
        const auto& mp = map.params();
        return map.data().getOccupancy(mp.voxelIdxToIndex(adr)) == arc::VoxelOccupancyCode::FREE;
    }
    std::shared_ptr<rclcpp::Node> nodeA_;
    std::shared_ptr<rclcpp::Node> nodeB_;
    std::unique_ptr<arc::SdfMap> mapA_;
    std::unique_ptr<arc::SdfMap> mapB_;
    std::unique_ptr<arc::MultiMapManager> managerA_;
    std::unique_ptr<arc::MultiMapManager> managerB_;
};

TEST_F(MultiMapManagerTest, FindMissedChunkIds) {
    std::vector<int> missed;
    arc::MultiMapManager::findMissedChunkIds({ 1, 10 }, {}, missed);
    EXPECT_EQ(missed, (std::vector<int>{ 1, 10 }));

    arc::MultiMapManager::findMissedChunkIds({ 1, 10 }, { 1, 3, 5, 8 }, missed);
    EXPECT_EQ(missed, (std::vector<int>{ 4, 4, 9, 10 }));

    arc::MultiMapManager::findMissedChunkIds({ 1, 10 }, { 1, 10 }, missed);
    EXPECT_TRUE(missed.empty());

    arc::MultiMapManager::findMissedChunkIds({}, { 1, 10 }, missed);
    EXPECT_TRUE(missed.empty());

    arc::MultiMapManager::findMissedChunkIds({ 5, 8 }, { 1, 10 }, missed);
    EXPECT_TRUE(missed.empty());
}

TEST_F(MultiMapManagerTest, MergeChunkIds) {
    std::vector<int> merged;
    arc::MultiMapManager::mergeChunkIds({ 1, 1 }, { 1, 9 }, merged);
    EXPECT_EQ(merged, (std::vector<int>{ 1, 9 }));

    arc::MultiMapManager::mergeChunkIds({ 1, 1 }, { 3, 5 }, merged);
    EXPECT_EQ(merged, (std::vector<int>{ 1, 1, 3, 5 }));

    arc::MultiMapManager::mergeChunkIds({ 1, 3 }, { 4, 6 }, merged);
    EXPECT_EQ(merged, (std::vector<int>{ 1, 6 }));

    arc::MultiMapManager::mergeChunkIds({}, { 2, 4 }, merged);
    EXPECT_EQ(merged, (std::vector<int>{ 2, 4 }));

    arc::MultiMapManager::mergeChunkIds({ 1, 2, 5, 6 }, { 3, 4 }, merged);
    EXPECT_EQ(merged, (std::vector<int>{ 1, 6 }));
}

TEST_F(MultiMapManagerTest, UpdateMapChunkGroupsVoxels) {
    managerA_->updateMapChunk(makeAdrs(9));
    const auto& chunks = managerA_->chunks();
    ASSERT_EQ(chunks.size(), 2u);
    ASSERT_EQ(chunks[0].chunks.size(), 2u);
    EXPECT_EQ(chunks[0].chunks[0].idx, 1u);
    EXPECT_EQ(chunks[0].chunks[1].idx, 2u);
    EXPECT_EQ(chunks[0].idxList, (std::vector<int>{ 1, 2 }));
    for (const auto& chunk : chunks[0].chunks) {
        EXPECT_EQ(chunk.voxelAdrs.size(), 4u);
        EXPECT_TRUE(chunk.needQuery);
    }

    managerA_->updateMapChunk(makeAdrs(3, 20, 5, 2));
    EXPECT_EQ(managerA_->chunks()[0].chunks.size(), 2u);

    managerA_->updateMapChunk(makeAdrs(1, 25, 5, 2));
    EXPECT_EQ(managerA_->chunks()[0].chunks.size(), 3u);
    EXPECT_EQ(managerA_->chunks()[0].idxList, (std::vector<int>{ 1, 3 }));
    EXPECT_EQ(managerA_->chunks()[0].chunks.back().idx, 3u);
}

TEST_F(MultiMapManagerTest, StampRequestsMissingChunks) {
    const auto adrs = makeAdrs(5);
    managerA_->updateMapChunk(adrs);
    ASSERT_EQ(managerA_->chunks()[0].chunks.size(), 1u);
    std::vector<uint8_t> expected = { 1, 0, 1, 0 };
    for (size_t i = 0; i < expected.size(); ++ i) {
        setLogit(*mapA_, adrs[i], expected[i] != 0);
    }

    const arc::ChunkStamps stampB = managerB_->buildStamp(1.0);
    EXPECT_EQ(stampB.fromDroneId, 2);
    ASSERT_EQ(stampB.idxLists.size(), 2u);
    EXPECT_TRUE(stampB.idxLists[0].empty());

    std::vector<arc::ChunkData> outgoing = managerA_->handleStamp(stampB);
    ASSERT_EQ(outgoing.size(), 1u);
    EXPECT_EQ(outgoing[0].fromDroneId, 1);
    EXPECT_EQ(outgoing[0].toDroneId, 2);
    EXPECT_EQ(outgoing[0].chunkDroneId, 1);
    EXPECT_EQ(outgoing[0].idx, 1u);
    EXPECT_EQ(outgoing[0].voxelAdrs, std::vector<size_t>(adrs.begin(), adrs.begin() + 4));
    EXPECT_EQ(outgoing[0].voxelOcc, expected);
    EXPECT_FALSE(managerA_->chunks()[0].chunks[0].needQuery);

    arc::ChunkStamps throttled = stampB;
    throttled.time = 1.1;
    EXPECT_TRUE(managerA_->handleStamp(throttled).empty());

    arc::ChunkStamps later = stampB;
    later.time = 1.5;
    EXPECT_EQ(managerA_->handleStamp(later).size(), 1u);

    managerB_->receiveChunk(outgoing[0]);
    managerB_->processReceivedChunks();
    for (size_t i = 0; i < expected.size(); ++ i) {
        EXPECT_EQ(isOccupied(*mapB_, adrs[i]), expected[i] != 0);
        EXPECT_EQ(isFree(*mapB_, adrs[i]), expected[i] == 0);
    }
    EXPECT_TRUE(managerB_->chunkBoxes()[0].valid);
}

TEST_F(MultiMapManagerTest, ReceiveChunkUpdatesMapAndBox) {
    const auto adrs = makeAdrs(4);
    arc::ChunkData chunk;
    chunk.fromDroneId = 1;
    chunk.toDroneId = 2;
    chunk.chunkDroneId = 1;
    chunk.idx = 1;
    chunk.voxelAdrs = adrs;
    chunk.voxelOcc = { 1, 0, 1, 0 };

    managerB_->receiveChunk(chunk);
    auto& md = mapB_->data();
    md.updateMin = Eigen::Vector3d(2.0, 2.0, 3.0);
    md.updateMax = Eigen::Vector3d(2.0, 2.0, 3.0);
    managerB_->processReceivedChunks();

    for (size_t i = 0; i < adrs.size(); ++ i) {
        EXPECT_EQ(isOccupied(*mapB_, adrs[i]), chunk.voxelOcc[i] != 0);
    }

    EXPECT_GT(md.updateRevision, 0u);
    EXPECT_EQ(md.dirtyIndexes.size(), adrs.size());
    for (size_t adr : adrs) {
        EXPECT_NE(std::find(md.dirtyIndexes.begin(), md.dirtyIndexes.end(), adr), md.dirtyIndexes.end());
    }
    EXPECT_DOUBLE_EQ(md.updateMin(0), 2.0);
    EXPECT_DOUBLE_EQ(md.updateMax(0), 2.0);
    EXPECT_DOUBLE_EQ(md.updateMin(1), 2.0);
    EXPECT_DOUBLE_EQ(md.updateMin(2), 3.0);

    const auto& boxes = managerB_->chunkBoxes();
    ASSERT_EQ(boxes.size(), 2u);
    EXPECT_TRUE(boxes[0].valid);
    EXPECT_FALSE(boxes[1].valid);
    EXPECT_NEAR(boxes[0].min(0), -2.9, 1e-9);
    EXPECT_NEAR(boxes[0].max(0), -2.3, 1e-9);
    EXPECT_NEAR(boxes[0].min(1), -2.9, 1e-9);
    EXPECT_NEAR(boxes[0].min(2), 0.0, 1e-9);

    std::vector<Eigen::Vector3d> mins;
    std::vector<Eigen::Vector3d> maxs;
    managerB_->getChunkBoxes(mins, maxs, true);
    ASSERT_EQ(mins.size(), 1u);
    EXPECT_NEAR(mins[0](0), -2.9, 1e-9);
    EXPECT_NEAR(maxs[0](0), -2.3, 1e-9);
    mins.clear();
    maxs.clear();
    managerB_->getChunkBoxes(mins, maxs, true);
    EXPECT_TRUE(mins.empty());
}

TEST_F(MultiMapManagerTest, TwoMapsFuseIdentical) {
    const auto adrsA = makeAdrs(4, 5, 5, 2);
    const auto adrsB = makeAdrs(4, 5, 8, 2);
    std::vector<uint8_t> occA = { 1, 1, 0, 0 };
    std::vector<uint8_t> occB = { 0, 0, 1, 1 };
    for (size_t i = 0; i < 4; ++ i) {
        setLogit(*mapA_, adrsA[i], occA[i] != 0);
        setLogit(*mapB_, adrsB[i], occB[i] != 0);
    }

    arc::ChunkData aToB;
    aToB.fromDroneId = 1;
    aToB.toDroneId = 2;
    aToB.chunkDroneId = 1;
    aToB.idx = 1;
    aToB.voxelAdrs = adrsA;
    aToB.voxelOcc = occA;
    managerB_->receiveChunk(aToB);

    arc::ChunkData bToA;
    bToA.fromDroneId = 2;
    bToA.toDroneId = 1;
    bToA.chunkDroneId = 2;
    bToA.idx = 1;
    bToA.voxelAdrs = adrsB;
    bToA.voxelOcc = occB;
    managerA_->receiveChunk(bToA);

    managerA_->processReceivedChunks();
    managerB_->processReceivedChunks();

    for (size_t i = 0; i < 4; ++ i) {
        EXPECT_EQ(isOccupied(*mapB_, adrsA[i]), occA[i] != 0);
        EXPECT_EQ(isOccupied(*mapA_, adrsB[i]), occB[i] != 0);
        EXPECT_EQ(isOccupied(*mapA_, adrsA[i]), isOccupied(*mapB_, adrsA[i]));
        EXPECT_EQ(isOccupied(*mapA_, adrsB[i]), isOccupied(*mapB_, adrsB[i]));
    }
}

TEST_F(MultiMapManagerTest, NewVoxelsFeedChunks) {
    pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::PointXYZ point;
    point.x = 0.5;
    point.y = 0.5;
    point.z = 0.5;
    cloud.push_back(point);

    mapA_->updateFramePoints(cloud, 1, Eigen::Vector3d(0.0, 0.0, 0.5));
    const std::vector<size_t> newVoxels = mapA_->takeNewVoxelIndexes();
    EXPECT_FALSE(newVoxels.empty());
    EXPECT_TRUE(mapA_->data().newVoxelIndexes.empty());

    arc::MultiMapManagerParams params;
    params.droneId = 1;
    params.droneNum = 2;
    params.chunkSize = 1;
    arc::MultiMapManager manager(*mapA_, params);
    manager.updateMapChunk(newVoxels);
    EXPECT_FALSE(manager.chunks()[0].chunks.empty());
    EXPECT_EQ(manager.chunks()[0].idxList.front(), 1);
}

} // namespace