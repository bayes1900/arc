#pragma once

#include <array>

#include <base/iarc.h>
#include <env/sdfmap.h>
#include <explore/frontier/frontierdef.h>

namespace arc {

class FrontierDetector {
public:
    FrontierDetector(SdfMap& map, const FrontierParams& params)
        : mp_(map.params())
        , md_(map.data())
        , params_(params)
    {
        frontierCellFlags_.resize(md_.numVoxels, 0);
        seedVisitStamps_.resize(md_.numVoxels, 0);
        pendingSeedCells_.reserve(1024);
        buildNeighborOffsets_();
    }
    void detect(
        std::vector<Frontier>& frontiers,
        std::vector<Frontier>& dormant,
        std::vector<Frontier>& detected,
        std::vector<int>& removedIds
    ) {
        detected.clear();
        Eigen::Vector3d updateMin = md_.updateMin;
        Eigen::Vector3d updateMax = md_.updateMax;
        accumulateSeedBox_(updateMin, updateMax);
        const Eigen::Vector3i updateMinIndex = mp_.p2i(updateMin).cwiseMax(0);
        const Eigen::Vector3i updateMaxIndex = mp_.p2i(updateMax).cwiseMin(mp_.voxelShape - Eigen::Vector3i::Ones());

        removedIds.clear();
        for (size_t i = 0; i < frontiers.size();) {
            if (!misc::boxesOverlap(frontiers[i].boxMin, frontiers[i].boxMax, updateMin, updateMax) ||
                !changed_(frontiers[i], updateMinIndex, updateMaxIndex)) {
                i += 1;
                continue;
            }
            clearFlags_(frontiers[i]);
            queueFrontierSeeds_(frontiers[i]);
            removedIds.push_back(static_cast<int>(i));
            frontiers.erase(frontiers.begin() + i);
        }
        for (size_t i = 0; i < dormant.size();) {
            if (!misc::boxesOverlap(dormant[i].boxMin, dormant[i].boxMax, updateMin, updateMax) ||
                !changed_(dormant[i], updateMinIndex, updateMaxIndex)) {
                i += 1;
                continue;
            }
            clearFlags_(dormant[i]);
            queueFrontierSeeds_(dormant[i]);
            dormant.erase(dormant.begin() + i);
        }

        const bool fullScan = md_.needsFullScan || md_.updateRevision == 0;
        if (fullScan) {
            const Eigen::Vector3d searchMin = (updateMin - Eigen::Vector3d(1.0, 1.0, 0.2)).cwiseMax(mp_.boxMin);
            const Eigen::Vector3d searchMax = (updateMax + Eigen::Vector3d(1.0, 1.0, 0.2)).cwiseMin(mp_.boxMax);
            if ((searchMax.array() <= searchMin.array()).any()) {
                pendingSeedCells_.clear();
                md_.dirtyIndexes.clear();
                splitLargeFrontiers_(detected);
                return;
            }
            const Eigen::Vector3i minIndex = mp_.p2i(searchMin).cwiseMax(0);
            const Eigen::Vector3i maxIndex = mp_.p2i(searchMax).cwiseMin(mp_.voxelShape - Eigen::Vector3i::Ones());
            const int strideX = mp_.voxelShape(1) * mp_.voxelShape(2);
            const int strideY = mp_.voxelShape(2);
            for (int x = minIndex(0); x <= maxIndex(0); ++ x) {
                for (int y = minIndex(1); y <= maxIndex(1); ++ y) {
                    size_t address = static_cast<size_t>(x) * static_cast<size_t>(strideX) +
                        static_cast<size_t>(y) * static_cast<size_t>(strideY) + static_cast<size_t>(minIndex(2));
                    for (int z = minIndex(2); z <= maxIndex(2); ++ z, ++ address) {
                        if (frontierCellFlags_[address] != 0) {
                            continue;
                        }
                        const Eigen::Vector3i index(x, y, z);
                        if (!md_.isFrontierCell(index)) {
                            continue;
                        }
                        expandFrontier_(index, detected);
                    }
                }
            }
            if (md_.updateRevision != 0) {
                md_.needsFullScan = false;
            }
            pendingSeedCells_.clear();
            md_.dirtyIndexes.clear();
        } else {
            beginSeedGeneration_();
            for (size_t address : pendingSeedCells_) {
                addSeedCandidates_(address, detected);
            }
            for (size_t address : md_.dirtyIndexes) {
                addSeedCandidates_(address, detected);
            }
            pendingSeedCells_.clear();
            md_.dirtyIndexes.clear();
        }
        splitLargeFrontiers_(detected);
    }
    bool isCovered(const std::vector<Frontier>& frontiers, const std::vector<Frontier>& dormant) const {
        Eigen::Vector3d updateMin = md_.updateMin;
        Eigen::Vector3d updateMax = md_.updateMax;
        accumulateSeedBox_(updateMin, updateMax);
        const Eigen::Vector3i updateMinIndex = mp_.p2i(updateMin).cwiseMax(0);
        const Eigen::Vector3i updateMaxIndex = mp_.p2i(updateMax).cwiseMin(mp_.voxelShape - Eigen::Vector3i::Ones());
        auto check = [&](const std::vector<Frontier>& list) {
            for (const auto& frontier : list) {
                if (!misc::boxesOverlap(frontier.boxMin, frontier.boxMax, updateMin, updateMax)) {
                    continue;
                }
                const int threshold = static_cast<int>(params_.minViewFinishFraction * static_cast<double>(frontier.cells.size()));
                int changedNum = 0;
                for (const auto& cell : frontier.cells) {
                    const Eigen::Vector3i index = mp_.p2i(cell);
                    if ((index.array() < updateMinIndex.array()).any() ||
                        (index.array() > updateMaxIndex.array()).any()) {
                        continue;
                    }
                    if (!md_.isFrontierCell(index)) {
                        if (++ changedNum >= threshold) {
                            return true;
                        }
                    }
                }
            }
            return false;
        };
        return check(frontiers) || check(dormant);
    }
    void releaseFrontier(const Frontier& frontier) {
        clearFlags_(frontier);
        queueFrontierSeeds_(frontier);
    }
    bool hasPendingSeedCells() const {
        return !pendingSeedCells_.empty();
    }
    size_t mapRevision() const {
        return md_.updateRevision;
    }
    bool requiresFullScan() const {
        return md_.needsFullScan;
    }

private:
    void accumulateSeedBox_(Eigen::Vector3d& min, Eigen::Vector3d& max) const {
        auto extend = [&](size_t address) {
            const Eigen::Vector3d position = mp_.i2p(mp_.voxelIdxToIndex(address));
            min = min.cwiseMin(position);
            max = max.cwiseMax(position);
        };
        for (size_t address : md_.dirtyIndexes) {
            extend(address);
        }
        for (size_t address : pendingSeedCells_) {
            extend(address);
        }
    }
    void buildNeighborOffsets_() {
        const int strideX = mp_.voxelShape(1) * mp_.voxelShape(2);
        const int strideY = mp_.voxelShape(2);
        size_t index = 0;
        for (int dx = -1; dx <= 1; ++ dx) {
            for (int dy = -1; dy <= 1; ++ dy) {
                for (int dz = -1; dz <= 1; ++ dz) {
                    if (dx == 0 && dy == 0 && dz == 0) {
                        continue;
                    }
                    neighborDeltas_[index] = Eigen::Vector3i(dx, dy, dz);
                    neighborOffsets_[index] = dx * strideX + dy * strideY + dz;
                    index += 1;
                }
            }
        }
    }
    void queueFrontierSeeds_(const Frontier& frontier) {
        for (const auto& cell : frontier.cells) {
            const Eigen::Vector3i index = mp_.p2i(cell);
            if (mp_.isInMap(index)) {
                pendingSeedCells_.push_back(mp_.getVoxelIdx(index));
            }
        }
    }
    void addSeedCandidates_(size_t address, std::vector<Frontier>& detected) {
        const Eigen::Vector3i index = mp_.voxelIdxToIndex(address);
        addSeedCandidate_(index, detected);
        for (int axis = 0; axis < 3; ++ axis) {
            for (int sign = -1; sign <= 1; sign += 2) {
                Eigen::Vector3i neighbor = index;
                neighbor(axis) += sign;
                addSeedCandidate_(neighbor, detected);
            }
        }
    }
    void addSeedCandidate_(const Eigen::Vector3i& index, std::vector<Frontier>& detected) {
        if (!mp_.isInMap(index) || !mp_.isInBox(index)) {
            return;
        }
        const size_t address = mp_.getVoxelIdx(index);
        if (!markSeed_(address)) {
            return;
        }
        if (frontierCellFlags_[address] != 0) {
            return;
        }
        if (!md_.isFrontierCell(index)) {
            return;
        }
        expandFrontier_(index, detected);
    }
    void beginSeedGeneration_() {
        ++ seedGeneration_;
        if (seedGeneration_ == 0) {
            std::fill(seedVisitStamps_.begin(), seedVisitStamps_.end(), 0);
            seedGeneration_ = 1;
        }
    }
    bool markSeed_(size_t address) {
        if (seedVisitStamps_[address] == seedGeneration_) {
            return false;
        }
        seedVisitStamps_[address] = seedGeneration_;
        return true;
    }
    void expandFrontier_(const Eigen::Vector3i& first, std::vector<Frontier>& detected) {
        frontierCellQueue_.clear();
        std::vector<Eigen::Vector3d> cells;
        cells.reserve(static_cast<size_t>(std::max(64, params_.clusterMin)));
        cells.push_back(mp_.i2p(first));
        frontierCellQueue_.push_back(first);
        frontierCellFlags_[mp_.getVoxelIdx(first)] = 1;

        size_t head = 0;
        while (head < frontierCellQueue_.size()) {
            const Eigen::Vector3i current = frontierCellQueue_[head];
            const size_t currentAddress = mp_.getVoxelIdx(current);
            head += 1;
            for (size_t i = 0; i < neighborDeltas_.size(); ++ i) {
                const Eigen::Vector3i neighbor = current + neighborDeltas_[i];
                if (!mp_.isInMap(neighbor) || !mp_.isInBox(neighbor)) {
                    continue;
                }
                const size_t address = currentAddress + neighborOffsets_[i];
                if (frontierCellFlags_[address] != 0) {
                    continue;
                }
                const double zCoord = (static_cast<double>(neighbor(2)) + 0.5) * mp_.resolution + mp_.origin(2);
                if (zCoord < params_.minFrontierZ) {
                    continue;
                }
                if (!md_.isFrontierCell(neighbor)) {
                    continue;
                }
                cells.push_back(mp_.i2p(neighbor));
                frontierCellQueue_.push_back(neighbor);
                frontierCellFlags_[address] = 1;
            }
        }
        if (static_cast<int>(cells.size()) <= params_.clusterMin) {
            return;
        }
        Frontier frontier;
        frontier.cells = std::move(cells);
        frontier.finalize(mp_.origin, mp_.resolution, params_.downSample);
        detected.push_back(std::move(frontier));
    }
    void splitLargeFrontiers_(std::vector<Frontier>& frontiers) {
        std::vector<Frontier> result;
        result.reserve(frontiers.size());
        std::vector<Frontier> splits;
        for (auto& frontier : frontiers) {
            if (splitHorizontally_(frontier, splits)) {
                result.insert(result.end(), std::make_move_iterator(splits.begin()), std::make_move_iterator(splits.end()));
                splits.clear();
            } else {
                result.push_back(std::move(frontier));
            }
        }
        frontiers = std::move(result);
    }
    bool splitHorizontally_(const Frontier& frontier, std::vector<Frontier>& splits) {
        const Eigen::Vector2d mean = frontier.center.head<2>();
        bool needSplit = false;
        for (const auto& cell : frontier.downsampledCells) {
            if ((cell.head<2>() - mean).norm() > params_.clusterSizeXY) {
                needSplit = true;
                break;
            }
        }
        if (!needSplit) {
            return false;
        }
        Eigen::Matrix2d covariance = Eigen::Matrix2d::Zero();
        for (const auto& cell : frontier.downsampledCells) {
            const Eigen::Vector2d diff = cell.head<2>() - mean;
            covariance += diff * diff.transpose();
        }
        covariance /= static_cast<double>(frontier.downsampledCells.size());

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(covariance);
        const Eigen::Vector2d direction = solver.eigenvectors().col(1);

        Frontier first;
        Frontier second;
        const size_t halfSize = frontier.cells.size() / 2 + 1;
        first.cells.reserve(halfSize);
        second.cells.reserve(halfSize);
        Eigen::Vector3d firstSum = Eigen::Vector3d::Zero();
        Eigen::Vector3d secondSum = Eigen::Vector3d::Zero();
        Eigen::Vector3d firstMin = Eigen::Vector3d::Zero();
        Eigen::Vector3d firstMax = Eigen::Vector3d::Zero();
        Eigen::Vector3d secondMin = Eigen::Vector3d::Zero();
        Eigen::Vector3d secondMax = Eigen::Vector3d::Zero();
        bool firstStarted = false;
        bool secondStarted = false;
        for (const auto& cell : frontier.cells) {
            if ((cell.head<2>() - mean).dot(direction) >= 0.0) {
                if (!firstStarted) {
                    firstSum = cell;
                    firstMin = cell;
                    firstMax = cell;
                    firstStarted = true;
                } else {
                    firstSum += cell;
                    firstMin = firstMin.cwiseMin(cell);
                    firstMax = firstMax.cwiseMax(cell);
                }
                first.cells.push_back(cell);
            } else {
                if (!secondStarted) {
                    secondSum = cell;
                    secondMin = cell;
                    secondMax = cell;
                    secondStarted = true;
                } else {
                    secondSum += cell;
                    secondMin = secondMin.cwiseMin(cell);
                    secondMax = secondMax.cwiseMax(cell);
                }
                second.cells.push_back(cell);
            }
        }
        const double leafSize = mp_.resolution * static_cast<double>(std::max(1, params_.downSample));
        if (!first.cells.empty()) {
            first.center = firstSum / static_cast<double>(first.cells.size());
            first.boxMin = firstMin;
            first.boxMax = firstMax;
            first.downsample(mp_.origin, leafSize);
        }
        if (!second.cells.empty()) {
            second.center = secondSum / static_cast<double>(second.cells.size());
            second.boxMin = secondMin;
            second.boxMax = secondMax;
            second.downsample(mp_.origin, leafSize);
        }
        std::vector<Frontier> subSplits;
        subSplits.reserve(2);
        if (!first.cells.empty()) {
            if (splitHorizontally_(first, subSplits)) {
                splits.insert(splits.end(), std::make_move_iterator(subSplits.begin()), std::make_move_iterator(subSplits.end()));
                subSplits.clear();
            } else {
                splits.push_back(std::move(first));
            }
        }
        if (!second.cells.empty()) {
            if (splitHorizontally_(second, subSplits)) {
                splits.insert(splits.end(), std::make_move_iterator(subSplits.begin()), std::make_move_iterator(subSplits.end()));
            } else {
                splits.push_back(std::move(second));
            }
        }
        return true;
    }
    bool changed_(const Frontier& frontier, const Eigen::Vector3i& updateMinIndex, const Eigen::Vector3i& updateMaxIndex) const {
        for (const auto& cell : frontier.cells) {
            const Eigen::Vector3i index = mp_.p2i(cell);
            if ((index.array() < updateMinIndex.array()).any() ||
                (index.array() > updateMaxIndex.array()).any()) {
                continue;
            }
            if (!md_.isFrontierCell(index)) {
                return true;
            }
        }
        return false;
    }
    void clearFlags_(const Frontier& frontier) {
        for (const auto& cell : frontier.cells) {
            const Eigen::Vector3i index = mp_.p2i(cell);
            if (mp_.isInMap(index)) {
                frontierCellFlags_[mp_.getVoxelIdx(index)] = 0;
            }
        }
    }

private:
    const MapParams& mp_;
    MapData& md_;
    FrontierParams params_;
    std::vector<uint8_t> frontierCellFlags_;
    std::vector<Eigen::Vector3i> frontierCellQueue_;
    std::vector<uint32_t> seedVisitStamps_;
    std::vector<size_t> pendingSeedCells_;
    std::array<Eigen::Vector3i, 26> neighborDeltas_;
    std::array<int, 26> neighborOffsets_;
    uint32_t seedGeneration_{0};
};

} // arc
