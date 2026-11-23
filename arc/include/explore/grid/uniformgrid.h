#pragma once

#include <base/iarc.h>
#include <env/multimapmanager.h>
#include <env/sdfmap.h>
#include <explore/grid/griddef.h>
#include <explore/grid/grididlist.h>

namespace arc {

class HGrid;

class UniformGrid {
public:
    UniformGrid(SdfMap& map, const GridParams& params, int level, MultiMapManager* multiMap=nullptr)
        : map_(map)
        , mp_(map.params())
        , md_(map.data())
        , params_(params)
        , multiMap_(multiMap)
        , level_(std::clamp(level, 1, 2))
    {
        boundsMin_ = params_.boxMin.value_or(mp_.boxMin);
        boundsMax_ = params_.boxMax.value_or(mp_.boxMax);
        if ((boundsMax_.array() <= boundsMin_.array()).any()) {
            throw std::invalid_argument("invalid grid bounds");
        }
        params_.gridSize = std::max(1e-3, params_.gridSize);
        const Eigen::Vector3d size = boundsMax_ - boundsMin_;
        for (int i = 0; i < 2; ++ i) {
            const int baseNum = std::max(1, static_cast<int>(std::ceil(size(i) / params_.gridSize)));
            const int multiplier = 1 << (level_ - 1);
            resolution_(i) = size(i) / static_cast<double>(baseNum * multiplier);
        }
        resolution_(2) = size(2);
    }
    void initializeGrids() {
        const Eigen::Vector3d size = boundsMax_ - boundsMin_;
        for (int i = 0; i < 3; ++ i) {
            gridCounts_(i) = std::max(1, static_cast<int>(std::ceil(size(i) / resolution_(i))));
        }
        grids_.resize(static_cast<size_t>(gridCounts_(0)) * static_cast<size_t>(gridCounts_(1)) * static_cast<size_t>(gridCounts_(2)));
        const double voxelVolume = std::pow(mp_.resolution, 3.0);
        const double estimatedUnknown = resolution_(0) * resolution_(1) * resolution_(2) / voxelVolume;
        const double cappedUnknown = std::min(
            estimatedUnknown,
            static_cast<double>(std::numeric_limits<int>::max())
        );
        const int initialUnknown = static_cast<int>(std::lround(cappedUnknown));
        for (int x = 0; x < gridCounts_(0); ++ x) {
            for (int y = 0; y < gridCounts_(1); ++ y) {
                for (int z = 0; z < gridCounts_(2); ++ z) {
                    const Eigen::Vector3i index(x, y, z);
                    GridInfo& grid = grids_[toAddress_(index)];
                    indexToPos_(index, 0.5, grid.center);
                    grid.unknownCount = initialUnknown;
                    grid.wasRelevant = true;
                    grid.isRelevantNow = true;
                    grid.shouldPartition = false;
                    grid.isActive = level_ == 1;
                }
            }
        }
    }
    void updateBaseCoor() {
        for (size_t i = 0; i < grids_.size(); ++ i) {
            GridInfo& grid = grids_[i];
            const Eigen::Vector3i index = adrToIndex_(static_cast<int>(i));
            Eigen::Vector3d leftBottom;
            Eigen::Vector3d rightTop;
            indexToPos_(index, 0.0, leftBottom);
            indexToPos_(index, 1.0, rightTop);
            const Eigen::Vector3d rightBottom(rightTop(0), leftBottom(1), leftBottom(2));
            const Eigen::Vector3d leftTop(leftBottom(0), rightTop(1), leftBottom(2));
            grid.vertices = { leftBottom, rightBottom, rightTop, leftTop };
            grid.boxMin = leftBottom;
            grid.boxMax = rightTop;
            grid.normals.clear();
            for (int j = 0; j < 4; ++ j) {
                const Eigen::Vector3d edge = (grid.vertices[(j + 1) % 4] - grid.vertices[j]).normalized();
                grid.normals.emplace_back(-edge(1), edge(0), 0.0);
            }
        }
    }
    void updateGridData(
        int droneId,
        GridIdList& assignedIds,
        std::vector<int>& cellsToPartition,
        std::vector<int>& allCellsToPartition
    ) {
        resetPassFlags_();
        cellsToPartition.clear();

        const UpdateBoxes boxes = gatherUpdateBoxes_();
        const std::vector<int> rediscoveredCells = refreshChangedGrids_(boxes, allCellsToPartition);
        refreshPendingActivation_();
        rebuildRelevantSet_();

        if (initializeIfNeeded_(droneId, assignedIds)) {
            return;
        }
        pruneAssignedIds_(assignedIds, cellsToPartition);
        assignedIds.insertAll(rediscoveredCells);
    }
    void activateGrids(const std::vector<int>& ids) {
        for (int id : ids) {
            if (id < 0 || id >= static_cast<int>(grids_.size())) {
                continue;
            }
            grids_[id].isActive = true;
        }
        pendingActivation_ = ids;
    }
    void inputFrontiers(const std::vector<Eigen::Vector3d>& averages) {
        for (GridInfo& grid : grids_) {
            grid.frontierIndices.clear();
        }
        for (int i = 0; i < static_cast<int>(averages.size()); ++ i) {
            const Eigen::Vector3i index = posToIndex_(averages[i]);
            if (!insideGrid_(index)) {
                continue;
            }
            grids_[toAddress_(index)].frontierIndices.insert(i);
        }
    }
    void setMultiMapManager(MultiMapManager* multiMap) {
        multiMap_ = multiMap;
    }
    void buildGridTour(const std::vector<int>& ids, std::vector<Eigen::Vector3d>& tour) const {
        tour.clear();
        tour.reserve(ids.size());
        for (int id : ids) {
            if (id >= 0 && id < static_cast<int>(grids_.size())) {
                tour.push_back(grids_[id].center);
            }
        }
    }
    void frontiersInGrid(int gridId, std::vector<int>& frontierIndices) const {
        frontierIndices.clear();
        if (gridId < 0 || gridId >= static_cast<int>(grids_.size())) {
            return;
        }
        for (int id : grids_[gridId].frontierIndices) {
            frontierIndices.push_back(id);
        }
    }
    void buildWireframe(std::vector<Eigen::Vector3d>& edgeStarts, std::vector<Eigen::Vector3d>& edgeEnds) const {
        Eigen::Vector3d p1 = boundsMin_;
        Eigen::Vector3d p2 = boundsMin_ + Eigen::Vector3d(boundsMax_(0) - boundsMin_(0), 0.0, 0.0);
        for (int i = 0; i <= gridCounts_(1); ++ i) {
            edgeStarts.push_back(p1 + Eigen::Vector3d(0.0, resolution_(1) * i, 0.0));
            edgeEnds.push_back(p2 + Eigen::Vector3d(0.0, resolution_(1) * i, 0.0));
        }
        p1 = boundsMin_;
        p2 = boundsMin_ + Eigen::Vector3d(0.0, boundsMax_(1) - boundsMin_(1), 0.0);
        for (int i = 0; i <= gridCounts_(0); ++ i) {
            edgeStarts.push_back(p1 + Eigen::Vector3d(resolution_(0) * i, 0.0, 0.0));
            edgeEnds.push_back(p2 + Eigen::Vector3d(resolution_(0) * i, 0.0, 0.0));
        }
        for (auto& point : edgeStarts) {
            point(2) = 0.5;
        }
        for (auto& point : edgeEnds) {
            point(2) = 0.5;
        }
    }

private:
    struct UpdateBoxes {
        Eigen::Vector3d updateMin;
        Eigen::Vector3d updateMax;
        std::vector<Eigen::Vector3d> chunkMins;
        std::vector<Eigen::Vector3d> chunkMaxs;
    };
    void resetPassFlags_() {
        for (GridInfo& grid : grids_) {
            grid.updatedThisPass = false;
        }
    }
    UpdateBoxes gatherUpdateBoxes_() {
        UpdateBoxes boxes;
        boxes.updateMin = md_.updateMin;
        boxes.updateMax = md_.updateMax;
        if (level_ == 2) {
            map_.requestUpdatedBoxReset();
        }
        if (multiMap_ != nullptr) {
            multiMap_->getChunkBoxes(boxes.chunkMins, boxes.chunkMaxs, level_ == 2);
        }
        return boxes;
    }
    bool overlapsChunk_(const GridInfo& grid, const UpdateBoxes& boxes) const {
        for (size_t i = 0; i < boxes.chunkMins.size(); ++ i) {
            if (misc::boxesOverlap(grid.boxMin, grid.boxMax, boxes.chunkMins[i], boxes.chunkMaxs[i])) {
                return true;
            }
        }
        return false;
    }
    std::vector<int> refreshChangedGrids_(
        const UpdateBoxes& boxes,
        std::vector<int>& allCellsToPartition
    ) {
        std::vector<int> rediscoveredCells;
        for (int i = 0; i < static_cast<int>(grids_.size()); ++ i) {
            GridInfo& grid = grids_[i];
            if (!grid.isActive) {
                continue;
            }
            const bool overlapWithChunk = overlapsChunk_(grid, boxes);
            const bool overlapWithUpdate = misc::boxesOverlap(grid.boxMin, grid.boxMax, boxes.updateMin, boxes.updateMax);
            if (!overlapWithChunk && !overlapWithUpdate) {
                continue;
            }
            updateGridInfo_(i);
            if (grid.shouldPartition) {
                allCellsToPartition.push_back(i);
                grid.isActive = false;
            }
            if (overlapWithUpdate && !grid.wasRelevant && grid.isRelevantNow && level_ > 1) {
                rediscoveredCells.push_back(i);
            }
        }
        return rediscoveredCells;
    }
    void refreshPendingActivation_() {
        for (int id : pendingActivation_) {
            if (id < 0 || id >= static_cast<int>(grids_.size()) || grids_[id].updatedThisPass) {
                continue;
            }
            updateGridInfo_(id);
        }
        pendingActivation_.clear();
    }
    void rebuildRelevantSet_() {
        relevantIds_.clear();
        relevantIdSet_.clear();
        for (size_t i = 0; i < grids_.size(); ++ i) {
            if (isRelevant_(grids_[i])) {
                relevantIds_.push_back(static_cast<int>(i));
                relevantIdSet_.insert(static_cast<int>(i));
            }
        }
    }
    bool initializeIfNeeded_(int droneId, GridIdList& assignedIds) {
        if (initialized_) {
            return false;
        }
        if (droneId == 1 && level_ == 1) {
            assignedIds.assign(relevantIds_);
        }
        initialized_ = true;
        return true;
    }
    void pruneAssignedIds_(GridIdList& assignedIds, std::vector<int>& cellsToPartition) {
        for (auto iter = assignedIds.begin(); iter != assignedIds.end();) {
            if (relevantIdSet_.find(*iter) == relevantIdSet_.end()) {
                iter = assignedIds.erase(iter);
                continue;
            }
            if (grids_[*iter].shouldPartition) {
                cellsToPartition.push_back(*iter);
                iter = assignedIds.erase(iter);
                continue;
            }
            ++ iter;
        }
    }
    void updateGridInfo_(int adr) {
        GridInfo& grid = grids_[adr];
        if (grid.updatedThisPass) {
            return;
        }
        grid.updatedThisPass = true;
        grid.wasRelevant = grid.isRelevantNow;

        const Eigen::Vector3i index = adrToIndex_(adr);
        Eigen::Vector3d geometricCenter;
        indexToPos_(index, 0.5, geometricCenter);
        grid.center.setZero();
        grid.unknownCount = 0;
        int freeNum = 0;

        const bool intersectsMap = (grid.boxMax.array() >= mp_.origin.array()).all() && (grid.boxMin.array() <= mp_.endpoint.array()).all();
        if (!intersectsMap) {
            grid.center = geometricCenter;
            grid.isRelevantNow = isRelevant_(grid);
            return;
        }
        Eigen::Vector3i indexMin = mp_.p2i(grid.boxMin);
        Eigen::Vector3i indexMax = mp_.p2i(grid.boxMax - Eigen::Vector3d::Constant(1e-4));
        indexMin = indexMin.cwiseMax(0).cwiseMin(mp_.voxelShape - Eigen::Vector3i::Ones());
        indexMax = indexMax.cwiseMax(0).cwiseMin(mp_.voxelShape - Eigen::Vector3i::Ones());

        for (int x = indexMin(0); x <= indexMax(0); ++ x) {
            for (int y = indexMin(1); y <= indexMax(1); ++ y) {
                const size_t base = mp_.getVoxelIdx(x, y, indexMin(2));
                for (int z = indexMin(2); z <= indexMax(2); ++ z) {
                    const size_t voxelIdx = base + static_cast<size_t>(z - indexMin(2));
                    const double logit = md_.occupancyLogits[voxelIdx];
                    if (logit < mp_.lmin - 1e-3) {
                        grid.center += mp_.i2p(Eigen::Vector3i(x, y, z));
                        grid.unknownCount += 1;
                    } else if (logit <= mp_.locc) {
                        freeNum += 1;
                    }
                }
            }
        }
        if (grid.unknownCount > 0) {
            grid.center /= static_cast<double>(grid.unknownCount);
        } else {
            grid.center = geometricCenter;
        }
        grid.isRelevantNow = isRelevant_(grid);
        if (level_ == 1 && grid.isActive && freeNum > params_.minFree) {
            grid.shouldPartition = true;
        }
    }
    bool isRelevant_(const GridInfo& grid) const {
        return grid.unknownCount >= params_.minUnknown || !grid.frontierIndices.empty();
    }
    int toAddress_(const Eigen::Vector3i& index) const {
        return index(0) * gridCounts_(1) * gridCounts_(2) + index(1) * gridCounts_(2) + index(2);
    }
    Eigen::Vector3i adrToIndex_(int adr) const {
        Eigen::Vector3i index;
        const int strideX = gridCounts_(1) * gridCounts_(2);
        const int strideY = gridCounts_(2);
        index(0) = adr / strideX;
        adr %= strideX;
        index(1) = adr / strideY;
        index(2) = adr % strideY;
        return index;
    }
    Eigen::Vector3i posToIndex_(const Eigen::Vector3d& pos) const {
        return ((pos - boundsMin_).array() / resolution_.array()).floor().cast<int>();
    }
    void indexToPos_(const Eigen::Vector3i& index, double inc, Eigen::Vector3d& pos) const {
        pos = (index.array().cast<double>() + inc) * resolution_.array() + boundsMin_.array();
    }
    bool insideGrid_(const Eigen::Vector3i& index) const {
        return (index.array() >= 0).all() && (index.array() < gridCounts_.array()).all();
    }

private:
    SdfMap& map_;
    const MapParams& mp_;
    const MapData& md_;
    GridParams params_;
    MultiMapManager* multiMap_{nullptr};
    int level_{1};
    Eigen::Vector3d boundsMin_{Eigen::Vector3d::Zero()};
    Eigen::Vector3d boundsMax_{Eigen::Vector3d::Zero()};
    Eigen::Vector3d resolution_{Eigen::Vector3d::Zero()};
    Eigen::Vector3i gridCounts_{Eigen::Vector3i::Zero()};
    std::vector<GridInfo> grids_;
    std::vector<int> relevantIds_;
    std::unordered_set<int> relevantIdSet_;
    bool initialized_{false};
    std::vector<int> pendingActivation_;

    friend class HGrid;
    friend class UniformGridTestPeer;
    friend class HGridTestPeer;
};

} // arc
