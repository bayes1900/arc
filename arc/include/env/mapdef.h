#pragma once

#include <base/ros.h>

#include <base/iarc.h>

namespace arc {

struct VoxelOccupancyCode {
    static constexpr uint8_t UNKNOWN  = 0;
    static constexpr uint8_t FREE     = 1;
    static constexpr uint8_t OCCUPIED = 2;
    static constexpr uint8_t OUTSIDE  = 255;
};

struct VoxelInflateOccupancyCode {
    static constexpr uint8_t FREE     = 0;
    static constexpr uint8_t OCCUPIED = 1;
    static constexpr uint8_t OUTSIDE  = 255;
};

struct MapParams {
    Eigen::Vector3d origin;
    Eigen::Vector3d endpoint;
    Eigen::Vector3d size;
    Eigen::Vector3d boxMin;
    Eigen::Vector3d boxMax;
    Eigen::Vector3i voxelShape;
    double resolution;
    double resolutionInv;
    double inflation;
    double maxRayLength;
    double phit;
    double pmiss;
    double pmin;
    double pmax;
    double pocc;
    double lhit;
    double lmiss;
    double lmin;
    double lmax;
    double locc;

    MapParams(rclcpp::Node::SharedPtr node) {
        double xlen = node->declare_parameter<double>("sdfmap.xlen");
        double ylen = node->declare_parameter<double>("sdfmap.ylen");
        double zlen = node->declare_parameter<double>("sdfmap.zlen");
        double groundHeight = node->declare_parameter<double>("sdfmap.groundHeight", -0.5);
        resolution = node->declare_parameter<double>("sdfmap.resolution");
        inflation = node->declare_parameter<double>("sdfmap.inflation");
        maxRayLength = node->declare_parameter<double>("sdfmap.maxRayLength");
        phit = node->declare_parameter<double>("sdfmap.phit");
        pmiss = node->declare_parameter<double>("sdfmap.pmiss");
        pmin = node->declare_parameter<double>("sdfmap.pmin");
        pmax = node->declare_parameter<double>("sdfmap.pmax");
        pocc = node->declare_parameter<double>("sdfmap.pocc");

        size << xlen, ylen, zlen;
        origin << -xlen / 2.0, -ylen / 2.0, groundHeight;
        endpoint = origin + size;
        const double boxMinX = node->declare_parameter<double>("sdfmap.boxMinX", origin(0));
        const double boxMinY = node->declare_parameter<double>("sdfmap.boxMinY", origin(1));
        const double boxMinZ = node->declare_parameter<double>("sdfmap.boxMinZ", origin(2));
        const double boxMaxX = node->declare_parameter<double>("sdfmap.boxMaxX", endpoint(0));
        const double boxMaxY = node->declare_parameter<double>("sdfmap.boxMaxY", endpoint(1));
        const double boxMaxZ = node->declare_parameter<double>("sdfmap.boxMaxZ", endpoint(2));
        boxMin << boxMinX, boxMinY, boxMinZ;
        boxMax << boxMaxX, boxMaxY, boxMaxZ;
        boxMin = boxMin.cwiseMax(origin);
        boxMax = boxMax.cwiseMin(endpoint);
        if ((boxMax.array() <= boxMin.array()).any()) {
            boxMin = origin;
            boxMax = endpoint;
        }
        voxelShape = (size / resolution).array().ceil().cast<int>();
        resolutionInv = 1.0 / resolution;
        lhit = misc::logit(phit);
        lmiss = misc::logit(pmiss);
        lmin = misc::logit(pmin);
        lmax = misc::logit(pmax);
        locc = misc::logit(pocc);
    }
    bool isInBox(const Eigen::Vector3d& pos) const {
        return (pos.array() >= boxMin.array()).all() && (pos.array() <= boxMax.array()).all();
    }
    bool isInBox(const Eigen::Vector3i& idx) const {
        return isInBox(i2p(idx));
    }
    bool isInMap(const Eigen::Vector3d& pos) const {
        if ((pos.array() < origin.array() + 1e-4).any() || (pos.array() > endpoint.array() - 1e-4).any()) {
            return false;
        }
        return true;
    }
    bool isInMap(const Eigen::Vector3i& idx) const {
        return (idx.array() >= 0).all() && (idx.array() < voxelShape.array()).all();
    }
    void boundIndex(Eigen::Vector3i& idx) const {
        idx = idx.cwiseMax(0).cwiseMin(voxelShape - Eigen::Vector3i::Ones());
    }
    int inflateStep() const {
        return std::max(0, static_cast<int>(std::ceil(inflation / resolution)));
    }
    Eigen::Vector3i p2i(const Eigen::Vector3d& pos) const {
        return ((pos - origin) * resolutionInv).array().floor().cast<int>();
    }
    Eigen::Vector3i p2i(const Eigen::Vector3d& pos, double resoInv) const {
        return ((pos - origin) * resoInv).array().floor().cast<int>();
    }
    Eigen::Vector3d i2p(const Eigen::Vector3i& idx) const {
        return (idx.array().cast<double>() + 0.5) * resolution + origin.array();
    }
    size_t getVoxelIdx(const Eigen::Vector3i& idx) const {
        return idx(0) * voxelShape(1) * voxelShape(2) + idx(1) * voxelShape(2) + idx(2);
    }
    size_t getVoxelIdx(int x, int y, int z) const {
        return x * voxelShape(1) * voxelShape(2) + y * voxelShape(2) + z;
    }
    Eigen::Vector3i voxelIdxToIndex(size_t voxelIdx) const {
        const size_t strideX = static_cast<size_t>(voxelShape(1)) * static_cast<size_t>(voxelShape(2));
        const size_t strideY = static_cast<size_t>(voxelShape(2));
        const size_t x = voxelIdx / strideX;
        voxelIdx %= strideX;
        const size_t y = voxelIdx / strideY;
        return Eigen::Vector3i(static_cast<int>(x), static_cast<int>(y), static_cast<int>(voxelIdx % strideY));
    }
};

struct MapData {
    const MapParams& mp;
    size_t numVoxels;
    std::vector<double> occupancyLogits;
    std::vector<uint8_t> occupied;
    std::vector<double> distance;
    std::vector<uint8_t> raycastIndexes;
    std::vector<uint8_t> frameOccupancy;
    std::vector<size_t> updatedIndexes;
    std::vector<size_t> dirtyIndexes;
    std::vector<size_t> newVoxelIndexes;
    size_t updateRevision{0};
    bool needsFullScan{true};
    bool resetUpdatedBox{true};
    Eigen::Vector3d allMin;
    Eigen::Vector3d allMax;
    Eigen::Vector3d updateMin;
    Eigen::Vector3d updateMax;
    Eigen::Vector3i localBoundMin;
    Eigen::Vector3i localBoundMax;
    std::vector<Eigen::Vector3i> occupiedVoxelsScratch;

    MapData(const MapParams& mp) : mp(mp) {
        numVoxels = mp.voxelShape(0) * mp.voxelShape(1) * mp.voxelShape(2);
        occupancyLogits.resize(numVoxels, mp.lmin - 0.01);
        occupied.resize(numVoxels, 0);
        distance.resize(numVoxels, mp.size.norm());
        raycastIndexes.resize(numVoxels, 0);
        frameOccupancy.resize(numVoxels, 0);
        allMin = Eigen::Vector3d::Constant(std::numeric_limits<double>::max());
        allMax = Eigen::Vector3d::Constant(-std::numeric_limits<double>::max());
        updateMin = mp.origin;
        updateMax = mp.endpoint;
        localBoundMin.setZero();
        localBoundMax = mp.voxelShape - Eigen::Vector3i::Ones();
        occupiedVoxelsScratch.reserve(1024);
    }
    void reset() {
        std::fill(occupancyLogits.begin(), occupancyLogits.end(), mp.lmin - 0.01);
        std::fill(occupied.begin(), occupied.end(), 0);
        std::fill(distance.begin(), distance.end(), mp.size.norm());
        std::fill(raycastIndexes.begin(), raycastIndexes.end(), 0);
        std::fill(frameOccupancy.begin(), frameOccupancy.end(), 0);
        updatedIndexes.clear();
        dirtyIndexes.clear();
        newVoxelIndexes.clear();
        ++ updateRevision;
        needsFullScan = true;
        resetUpdatedBox = true;
        allMin = Eigen::Vector3d::Constant(std::numeric_limits<double>::max());
        allMax = Eigen::Vector3d::Constant(-std::numeric_limits<double>::max());
        updateMin = mp.origin;
        updateMax = mp.endpoint;
        localBoundMin.setZero();
        localBoundMax = mp.voxelShape - Eigen::Vector3i::Ones();
        occupiedVoxelsScratch.clear();
    }
    void setFrameOccupancy(size_t voxelIdx, uint8_t occ, uint8_t raycastIndex) {
        uint8_t& frameOcc = frameOccupancy[voxelIdx];
        uint8_t& raycastIdxCurrent = raycastIndexes[voxelIdx];
        if (raycastIndex != raycastIdxCurrent) {
            frameOcc = occ;
            raycastIdxCurrent = raycastIndex;
            updatedIndexes.push_back(voxelIdx);
            return;
        }
        if (frameOcc == VoxelOccupancyCode::OCCUPIED) {
            return;
        }
        frameOcc = occ;
    }
    void mergeFrameOccupancy(uint8_t raycastIndex) {
        const bool hadUpdates = !updatedIndexes.empty();
        for (size_t i = 0; i < updatedIndexes.size(); ++ i) {
            size_t voxelIdx = updatedIndexes[i];
            updateVoxelOccupancyLogit(voxelIdx, frameOccupancy[voxelIdx], raycastIndex);
        }
        updatedIndexes.clear();
        if (hadUpdates) {
            ++ updateRevision;
        }
    }
    void updateVoxelOccupancyLogit(size_t voxelIdx, uint8_t occ, uint8_t raycastIndex) {
        double newLogit = occ == VoxelOccupancyCode::OCCUPIED ? mp.lhit : mp.lmiss;
        double& occLogit = occupancyLogits[voxelIdx];
        const double oldLogit = occLogit;
        if (occLogit < mp.lmin - 1e-6) {
            occLogit = mp.locc;
        }
        occLogit += newLogit;
        occLogit = std::max<double>(mp.lmin, occLogit);
        occLogit = std::min<double>(mp.lmax, occLogit);
        raycastIndexes[voxelIdx] = raycastIndex;
        if (oldLogit < mp.lmin - 1e-6 && occLogit >= mp.lmin - 1e-3) {
            newVoxelIndexes.push_back(voxelIdx);
        }
        if (occLogit != oldLogit) {
            dirtyIndexes.push_back(voxelIdx);
        }
    }
    uint8_t getOccupancy(const Eigen::Vector3i& idx) const {
        if (!mp.isInMap(idx)) {
            return VoxelOccupancyCode::OUTSIDE;
        }
        const double logit = occupancyLogits[mp.getVoxelIdx(idx)];
        if (logit < mp.lmin - 1e-3) {
            return VoxelOccupancyCode::UNKNOWN;
        }
        if (logit > mp.locc) {
            return VoxelOccupancyCode::OCCUPIED;
        }
        return VoxelOccupancyCode::FREE;
    }
    uint8_t getOccupancy(const Eigen::Vector3d& pos) const {
        if (!mp.isInMap(pos)) {
            return VoxelOccupancyCode::OUTSIDE;
        }
        return getOccupancy(mp.p2i(pos));
    }
    bool isFree(const Eigen::Vector3i& idx) const {
        if (!mp.isInMap(idx)) {
            return false;
        }
        const double logit = occupancyLogits[mp.getVoxelIdx(idx)];
        return logit >= mp.lmin - 1e-3 && logit <= mp.locc;
    }
    bool isNeighborUnknown(const Eigen::Vector3i& idx) const {
        if (!mp.isInMap(idx)) {
            return false;
        }
        const size_t base = mp.getVoxelIdx(idx);
        const int strideX = mp.voxelShape(1) * mp.voxelShape(2);
        const int strideY = mp.voxelShape(2);
        const double threshold = mp.lmin - 1e-3;
        if (idx(0) > 0 && occupancyLogits[base - strideX] < threshold) {
            return true;
        }
        if (idx(0) + 1 < mp.voxelShape(0) && occupancyLogits[base + strideX] < threshold) {
            return true;
        }
        if (idx(1) > 0 && occupancyLogits[base - strideY] < threshold) {
            return true;
        }
        if (idx(1) + 1 < mp.voxelShape(1) && occupancyLogits[base + strideY] < threshold) {
            return true;
        }
        if (idx(2) > 0 && occupancyLogits[base - 1] < threshold) {
            return true;
        }
        if (idx(2) + 1 < mp.voxelShape(2) && occupancyLogits[base + 1] < threshold) {
            return true;
        }
        return false;
    }
    bool isFrontierCell(const Eigen::Vector3i& idx) const {
        return isFree(idx) && isNeighborUnknown(idx);
    }
    bool isRayBlocked(const Eigen::Vector3i& idx) const {
        if (!mp.isInMap(idx)) {
            return true;
        }
        const size_t voxelIdx = mp.getVoxelIdx(idx);
        return occupied[voxelIdx] != 0 || occupancyLogits[voxelIdx] < mp.lmin - 1e-3;
    }
    uint8_t getInflateOccupancy(const Eigen::Vector3i& idx) const {
        if (!mp.isInMap(idx)) {
            return VoxelInflateOccupancyCode::OUTSIDE;
        }
        return occupied[mp.getVoxelIdx(idx)] ? VoxelInflateOccupancyCode::OCCUPIED : VoxelInflateOccupancyCode::FREE;
    }
    uint8_t getInflateOccupancy(const Eigen::Vector3d& pos) const {
        if (!mp.isInMap(pos)) {
            return VoxelInflateOccupancyCode::OUTSIDE;
        }
        return getInflateOccupancy(mp.p2i(pos));
    }
    double getDistance(const Eigen::Vector3i& idx) const {
        if (!mp.isInMap(idx)) {
            return -1.0;
        }
        return distance[mp.getVoxelIdx(idx)];
    }
    double getDistance(const Eigen::Vector3d& pos) const {
        if (!mp.isInMap(pos)) {
            return -1.0;
        }
        return getDistance(mp.p2i(pos));
    }
    void updateInflateOccupancy() {
        const int inflateStep = mp.inflateStep();
        const int zMin = localBoundMin[2];
        const int zMax = localBoundMax[2];
        const int zLen = zMax - zMin + 1;
        if (zLen <= 0) {
            return;
        }
        occupiedVoxelsScratch.clear();

        for (int x = localBoundMin[0]; x <= localBoundMax[0]; ++ x) {
            for (int y = localBoundMin[1]; y <= localBoundMax[1]; ++ y) {
                const size_t base = mp.getVoxelIdx(x, y, zMin);
                std::memset(occupied.data() + base, VoxelInflateOccupancyCode::FREE, zLen * sizeof(occupied[0]));
                for (int z = zMin; z <= zMax; ++ z) {
                    if (occupancyLogits[mp.getVoxelIdx(x, y, z)] > mp.locc) {
                        occupiedVoxelsScratch.emplace_back(x, y, z);
                    }
                }
            }
        }
        for (const auto& voxel : occupiedVoxelsScratch) {
            for (int dx = -inflateStep; dx <= inflateStep; ++ dx) {
                const int nx = voxel[0] + dx;
                if (nx < localBoundMin[0] || nx > localBoundMax[0]) {
                    continue;
                }
                for (int dy = -inflateStep; dy <= inflateStep; ++ dy) {
                    const int ny = voxel[1] + dy;
                    if (ny < localBoundMin[1] || ny > localBoundMax[1]) {
                        continue;
                    }
                    const int nzBegin = std::max(localBoundMin[2], voxel[2] - inflateStep);
                    const int nzEnd = std::min(localBoundMax[2], voxel[2] + inflateStep);
                    const int nzLen = nzEnd - nzBegin + 1;
                    const size_t base = mp.getVoxelIdx(nx, ny, nzBegin);
                    std::memset(occupied.data() + base, VoxelInflateOccupancyCode::OCCUPIED, nzLen * sizeof(occupied[0]));
                }
            }
        }
    }
};

} // arc
