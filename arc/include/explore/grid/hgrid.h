#pragma once

#include <base/iarc.h>
#include <env/multimapmanager.h>
#include <env/sdfmap.h>
#include <explore/grid/griddef.h>
#include <explore/grid/grididlist.h>
#include <explore/grid/uniformgrid.h>
#include <explore/tsp/tspmatrix.h>
#include <search/astar.h>

namespace arc {

class HGrid {
public:
    HGrid(SdfMap& map, const GridParams& params, const AstarOptions& astarOptions={})
        : params_(params)
        , astar_(map, astarOptions)
        , coarse_(std::make_unique<UniformGrid>(map, params, 1))
        , fine_(std::make_unique<UniformGrid>(map, params, 2))
    {
        coarse_->initializeGrids();
        fine_->initializeGrids();
        updateBaseCoor();
    }
    void setMultiMapManager(MultiMapManager* multiMap) {
        multiMap_ = multiMap;
        coarse_->setMultiMapManager(multiMap);
        fine_->setMultiMapManager(multiMap);
    }
    void updateBaseCoor() {
        coarse_->updateBaseCoor();
        fine_->updateBaseCoor();
    }
    void updateGridData(
        int droneId,
        GridIdList& assignedIds,
        const std::vector<int>& previousIds,
        std::vector<int>& consistentFirst,
        std::vector<int>& consistentSecond
    ) {
        auto [coarseIds, fineIds] = splitIdsByLevel(assignedIds);

        std::vector<int> coarseToPartition;
        std::vector<int> allCoarseToPartition;
        coarse_->updateGridData(droneId, coarseIds, coarseToPartition, allCoarseToPartition);
        appendFineChildren_(fineIds, coarseToPartition);

        fine_->activateGrids(expandCoarseToFine(allCoarseToPartition));
        std::vector<int> ignoredFinePartitions;
        std::vector<int> ignoredFineAll;
        fine_->updateGridData(droneId, fineIds, ignoredFinePartitions, ignoredFineAll);

        assignedIds.assign(mergeLevelIds(coarseIds, fineIds));
        buildConsistentSets(previousIds, assignedIds.values(), consistentFirst, consistentSecond);
    }
    void inputFrontiers(const std::vector<Eigen::Vector3d>& averages) {
        coarse_->inputFrontiers(averages);
        fine_->inputFrontiers(averages);
    }
    void buildCostMatrix(
        const std::vector<Eigen::Vector3d>& positions,
        const std::vector<Eigen::Vector3d>& velocities,
        const std::vector<std::vector<int>>& consistentFirst,
        const std::vector<std::vector<int>>& consistentSecond,
        const std::vector<int>& cellIds,
        Eigen::MatrixXd& mat
    ) {
        (void)velocities;
        const int droneCount = static_cast<int>(positions.size());
        const int cellCount = static_cast<int>(cellIds.size());
        const int gridOffset = 1 + droneCount;
        const int dimension = gridOffset + cellCount;
        mat = Eigen::MatrixXd::Zero(dimension, dimension);

        for (int i = 0; i < droneCount; ++ i) {
            mat(0, 1 + i) = TspMatrixCosts::DEPOT_TO_DRONE;
            mat(1 + i, 0) = TspMatrixCosts::DRONE_TO_DEPOT;
            for (int j = 0; j < droneCount; ++ j) {
                mat(1 + i, 1 + j) = TspMatrixCosts::DRONE_TO_DRONE;
            }
        }
        for (int i = 0; i < cellCount; ++ i) {
            mat(0, gridOffset + i) = TspMatrixCosts::DRONE_TO_DEPOT;
        }
        const std::vector<int> emptyFirstRegion;
        for (int i = 0; i < droneCount; ++ i) {
            const std::vector<int>& firstRegion = i < static_cast<int>(consistentFirst.size())
                ? consistentFirst[i]
                : emptyFirstRegion;
            for (int j = 0; j < cellCount; ++ j) {
                mat(1 + i, gridOffset + j) = costDroneToGrid(positions[i], cellIds[j], firstRegion);
            }
        }
        for (int i = 0; i < cellCount; ++ i) {
            for (int j = i + 1; j < cellCount; ++ j) {
                const double cost = costGridToGrid(
                    cellIds[i], cellIds[j], consistentFirst, consistentSecond, droneCount);
                mat(gridOffset + i, gridOffset + j) = cost;
                mat(gridOffset + j, gridOffset + i) = cost;
            }
        }
        for (int i = 0; i < dimension; ++ i) {
            mat(i, i) = TspMatrixCosts::SELF_LOOP;
        }
    }
    double costDroneToGrid(
        const Eigen::Vector3d& pos,
        int gridId,
        const std::vector<int>& consistentFirst
    ) {
        const GridInfo& grid = gridAt_(gridId);
        const double dist = (pos - grid.center).norm();
        double cost = 0.0;
        if (dist < NEAR_GRID_DISTANCE) {
            if (astar_.search(pos, grid.center, true) == AstarSearchStatus::DONE) {
                cost = misc::pathLength(astar_.getPath());
            } else {
                cost = dist + params_.consistentCost2;
            }
        } else {
            cost = FAR_GRID_COST_FACTOR * dist + params_.consistentCost2;
        }
        for (int firstId : consistentFirst) {
            if (gridId == firstId) {
                cost += params_.consistentCost;
                break;
            }
        }
        return cost;
    }
    double costGridToGrid(
        int firstId,
        int secondId,
        const std::vector<std::vector<int>>& consistentFirst,
        const std::vector<std::vector<int>>& consistentSecond,
        int droneCount
    ) {
        const GridInfo& first = gridAt_(firstId);
        const GridInfo& second = gridAt_(secondId);
        double cost = 0.0;
        if (areCoarseNeighbors(firstId, secondId)) {
            if (astar_.search(first.center, second.center, true) == AstarSearchStatus::DONE) {
                cost = misc::pathLength(astar_.getPath());
            } else {
                cost = (first.center - second.center).norm() + params_.consistentCost2;
            }
            if (droneCount <= 1 && sameFineParent(firstId, secondId)) {
                cost += params_.consistentCost;
            }
            for (int k = 0; k < droneCount; ++ k) {
                if (containsGrid_(consistentFirst, k, firstId) &&
                    containsGrid_(consistentSecond, k, secondId)) {
                    cost += params_.consistentCost;
                    break;
                }
            }
        } else {
            cost = FAR_GRID_COST_FACTOR * (first.center - second.center).norm() + params_.consistentCost2;
        }
        return cost;
    }
    void buildGridCenters(
        const std::vector<int>& ids,
        const Eigen::Vector3d& pos,
        std::vector<Eigen::Vector3d>& centers
    ) const {
        centers.clear();
        centers.reserve(ids.size() + 1);
        centers.push_back(pos);
        for (int id : ids) {
            centers.push_back(gridAt_(id).center);
        }
    }
    void frontiersInGrid(const std::vector<int>& gridIds, std::vector<int>& frontierIndices) {
        frontierIndices.clear();
        if (gridIds.empty()) {
            return;
        }
        const int coarseCount = static_cast<int>(coarse_->grids_.size());
        const int firstId = gridIds.front();
        if (firstId < coarseCount) {
            for (int id : coarse_->grids_[firstId].frontierIndices) {
                frontierIndices.push_back(id);
            }
            return;
        }
        const int coarse = coarseIdOf_(firstId);
        const auto fines = fineChildrenOf_(coarse);
        for (int fine : fines) {
            if (std::find(gridIds.begin(), gridIds.end(), fine + coarseCount) == gridIds.end()) {
                continue;
            }
            for (int id : fine_->grids_[fine].frontierIndices) {
                frontierIndices.push_back(id);
            }
        }
    }
    bool nextGrid(const std::vector<int>& assignedIds, Eigen::Vector3d& gridPos, double& gridYaw) const {
        if (assignedIds.empty()) {
            return false;
        }
        const int offset = coarseCount_();
        const int firstCoarse = coarseIdOf_(assignedIds.front());
        int nextCellId = -1;
        for (size_t i = 1; i < assignedIds.size(); ++ i) {
            const int coarse = coarseIdOf_(assignedIds[i]);
            if (assignedIds[i] < offset || coarse != firstCoarse) {
                nextCellId = assignedIds[i];
                break;
            }
        }
        if (nextCellId == -1) {
            return false;
        }
        const GridInfo& current = gridAt_(firstCoarse);
        const GridInfo& next = gridAt_(nextCellId);
        gridPos = next.center;
        const Eigen::Vector3d direction = next.center - current.center;
        gridYaw = std::atan2(direction(1), direction(0));
        return true;
    }
    bool nextGrid(const std::vector<int>& assignedIds, Eigen::Vector3d& gridPos) const {
        double gridYaw = 0.0;
        return nextGrid(assignedIds, gridPos, gridYaw);
    }
    void buildConsistentSets(
        const std::vector<int>& previousIds,
        const std::vector<int>& currentIds,
        std::vector<int>& consistentFirst,
        std::vector<int>& consistentSecond
    ) {
        consistentFirst.clear();
        consistentSecond.clear();
        if (previousIds.empty()) {
            return;
        }
        const int firstCoarse = coarseIdOf_(previousIds.front());
        const int secondCoarse = findSecondCoarse_(previousIds, firstCoarse);
        consistentFirst = collectIdsUnderCoarse_(currentIds, firstCoarse);
        if (secondCoarse < 0) {
            return;
        }
        const std::vector<int> second = collectIdsUnderCoarse_(currentIds, secondCoarse);
        if (consistentFirst.empty()) {
            consistentFirst = second;
        } else {
            consistentSecond = second;
        }
    }
    void activeGrids(std::vector<int>& cellIds) const {
        cellIds.clear();
        const int offset = coarseCount_();
        for (int i = 0; i < offset; ++ i) {
            if (coarse_->grids_[i].isActive && coarse_->grids_[i].isRelevantNow) {
                cellIds.push_back(i);
            }
        }
        for (int i = 0; i < static_cast<int>(fine_->grids_.size()); ++ i) {
            if (fine_->grids_[i].isActive && fine_->grids_[i].isRelevantNow) {
                cellIds.push_back(i + offset);
            }
        }
    }
    int unknownCellsOf(int cellId) const {
        return gridAt_(cellId).unknownCount;
    }
    Eigen::Vector3d centerOf(int cellId) const {
        return gridAt_(cellId).center;
    }
    bool sameCoarseParent(int firstId, int secondId) const {
        return coarseIdOf_(firstId) == coarseIdOf_(secondId);
    }
    void buildWireframe(std::vector<Eigen::Vector3d>& edgeStarts, std::vector<Eigen::Vector3d>& edgeEnds) const {
        edgeStarts.clear();
        edgeEnds.clear();
        auto append = [&](const std::vector<GridInfo>& grids) {
            for (const GridInfo& grid : grids) {
                if (!grid.isActive) {
                    continue;
                }
                for (int i = 0; i < 4; ++ i) {
                    edgeStarts.push_back(grid.vertices[i]);
                    edgeEnds.push_back(grid.vertices[(i + 1) % 4]);
                }
            }
        };
        append(coarse_->grids_);
        append(fine_->grids_);
        for (auto& point : edgeStarts) {
            point(2) = 0.5;
        }
        for (auto& point : edgeEnds) {
            point(2) = 0.5;
        }
    }
    void buildLabels(std::vector<Eigen::Vector3d>& labelPositions, std::vector<std::string>& labels) const {
        labelPositions.clear();
        labels.clear();
        for (int i = 0; i < static_cast<int>(coarse_->grids_.size()); ++ i) {
            const GridInfo& grid = coarse_->grids_[i];
            if (!grid.isActive || !grid.isRelevantNow) {
                continue;
            }
            labelPositions.push_back(grid.center);
            labels.push_back(std::to_string(i));
        }
        const int coarseCount = static_cast<int>(coarse_->grids_.size());
        for (int i = 0; i < static_cast<int>(fine_->grids_.size()); ++ i) {
            const GridInfo& grid = fine_->grids_[i];
            if (!grid.isActive || !grid.isRelevantNow) {
                continue;
            }
            labelPositions.push_back(grid.center);
            labels.push_back(std::to_string(i + coarseCount));
        }
    }

private:
    friend class HGridTestPeer;

    int coarseCount_() const {
        return static_cast<int>(coarse_->grids_.size());
    }
    std::pair<GridIdList, GridIdList> splitIdsByLevel(const GridIdList& ids) const {
        GridIdList coarseIds;
        GridIdList fineIds;
        const int offset = coarseCount_();
        for (int id : ids.values()) {
            if (id < offset) {
                coarseIds.insert(id);
            } else {
                fineIds.insert(id - offset);
            }
        }
        return { coarseIds, fineIds };
    }
    void appendFineChildren_(GridIdList& fineIds, const std::vector<int>& coarseIds) const {
        for (int id : coarseIds) {
            for (int fine : fineChildrenOf_(id)) {
                fineIds.insert(fine);
            }
        }
    }
    std::vector<int> expandCoarseToFine(const std::vector<int>& coarseIds) const {
        std::vector<int> fineIds;
        for (int id : coarseIds) {
            const auto children = fineChildrenOf_(id);
            fineIds.insert(fineIds.end(), children.begin(), children.end());
        }
        return fineIds;
    }
    std::vector<int> mergeLevelIds(const GridIdList& coarseIds, const GridIdList& fineIds) const {
        std::vector<int> merged;
        const int offset = coarseCount_();
        merged.reserve(coarseIds.size() + fineIds.size());
        for (int id : coarseIds.values()) {
            merged.push_back(id);
        }
        for (int id : fineIds.values()) {
            merged.push_back(id + offset);
        }
        return merged;
    }
    std::array<int, 4> fineChildrenOf_(int coarseId) const {
        const Eigen::Vector3i coarseIndex = coarse_->adrToIndex_(coarseId);
        const Eigen::Vector3i fineIndexes[4] = {
            Eigen::Vector3i(coarseIndex(0) * 2, coarseIndex(1) * 2, coarseIndex(2)),
            Eigen::Vector3i(coarseIndex(0) * 2 + 1, coarseIndex(1) * 2, coarseIndex(2)),
            Eigen::Vector3i(coarseIndex(0) * 2, coarseIndex(1) * 2 + 1, coarseIndex(2)),
            Eigen::Vector3i(coarseIndex(0) * 2 + 1, coarseIndex(1) * 2 + 1, coarseIndex(2))
        };
        std::array<int, 4> fines{};
        for (size_t i = 0; i < 4; ++ i) {
            fines[i] = fine_->toAddress_(fineIndexes[i]);
        }
        return fines;
    }
    int coarseParentOfFine_(int fineId) const {
        const Eigen::Vector3i fineIndex = fine_->adrToIndex_(fineId);
        const Eigen::Vector3i coarseIndex(fineIndex(0) / 2, fineIndex(1) / 2, fineIndex(2));
        return coarse_->toAddress_(coarseIndex);
    }
    int coarseIdOf_(int cellId) const {
        const int offset = coarseCount_();
        return cellId < offset ? cellId : coarseParentOfFine_(cellId - offset);
    }
    int findSecondCoarse_(const std::vector<int>& ids, int firstCoarse) const {
        for (size_t i = 1; i < ids.size(); ++ i) {
            const int coarse = coarseIdOf_(ids[i]);
            if (coarse != firstCoarse) {
                return coarse;
            }
        }
        return -1;
    }
    std::vector<int> collectIdsUnderCoarse_(
        const std::vector<int>& ids,
        int coarseId
    ) const {
        for (int id : ids) {
            if (id == coarseId) {
                return { id };
            }
        }
        std::vector<int> collected;
        const int offset = coarseCount_();
        for (int id : ids) {
            if (id >= offset && coarseParentOfFine_(id - offset) == coarseId) {
                collected.push_back(id);
            }
        }
        return collected;
    }
    bool containsGrid_(
        const std::vector<std::vector<int>>& regions,
        int droneIndex,
        int gridId
    ) const {
        if (droneIndex >= static_cast<int>(regions.size())) {
            return false;
        }
        const std::vector<int>& region = regions[droneIndex];
        return std::find(region.begin(), region.end(), gridId) != region.end();
    }
    GridInfo& gridAt_(int id) {
        const int offset = coarseCount_();
        if (id < offset) {
            return coarse_->grids_[id];
        }
        return fine_->grids_[id - offset];
    }
    const GridInfo& gridAt_(int id) const {
        const int offset = coarseCount_();
        if (id < offset) {
            return coarse_->grids_[id];
        }
        return fine_->grids_[id - offset];
    }
    bool areCoarseNeighbors(int firstId, int secondId) const {
        const Eigen::Vector3i index1 = coarse_->adrToIndex_(coarseIdOf_(firstId));
        const Eigen::Vector3i index2 = coarse_->adrToIndex_(coarseIdOf_(secondId));
        return (index1 - index2).cwiseAbs().maxCoeff() <= 1;
    }
    bool sameFineParent(int firstId, int secondId) const {
        const int offset = coarseCount_();
        if (firstId < offset || secondId < offset) {
            return false;
        }
        return coarseIdOf_(firstId) == coarseIdOf_(secondId);
    }

private:
    static constexpr double NEAR_GRID_DISTANCE{5.0};
    static constexpr double FAR_GRID_COST_FACTOR{1.5};
    GridParams params_;
    Astar astar_;
    std::unique_ptr<UniformGrid> coarse_;
    std::unique_ptr<UniformGrid> fine_;
    MultiMapManager* multiMap_{nullptr};
};

} // arc
