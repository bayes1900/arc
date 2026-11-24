#pragma once

#include <base/iarc.h>

namespace arc {

struct Viewpoint {
    Eigen::Vector3d pos{Eigen::Vector3d::Zero()};
    double yaw{0.0};
    int visibleCellCount{0};
};

struct Frontier {
    std::vector<Eigen::Vector3d> cells;
    std::vector<Eigen::Vector3d> downsampledCells;
    Eigen::Vector3d center{Eigen::Vector3d::Zero()};
    Eigen::Vector3d boxMin{Eigen::Vector3d::Zero()};
    Eigen::Vector3d boxMax{Eigen::Vector3d::Zero()};
    std::vector<Viewpoint> viewpoints;
    std::vector<std::vector<Eigen::Vector3d>> edgePaths;
    std::vector<double> edgeCosts;
    bool needsCostUpdate{true};

    void finalize(const Eigen::Vector3d& origin, double resolution, int downSample) {
        computeGeometry();
        downsample(origin, resolution * static_cast<double>(std::max(1, downSample)));
    }
    void computeGeometry() {
        if (cells.empty()) {
            return;
        }
        center.setZero();
        boxMin = cells.front();
        boxMax = cells.front();
        for (const auto& cell : cells) {
            center += cell;
            boxMin = boxMin.cwiseMin(cell);
            boxMax = boxMax.cwiseMax(cell);
        }
        center /= static_cast<double>(cells.size());
    }
    void downsample(const Eigen::Vector3d& origin, double leafSize) {
        if (cells.empty() || leafSize <= 0.0) {
            downsampledCells.clear();
            return;
        }
        const Eigen::Vector3i firstKey = ((cells.front() - origin) / leafSize).array().floor().cast<int>();
        Eigen::Vector3i minKey = firstKey;
        Eigen::Vector3i maxKey = firstKey;
        for (const auto& cell : cells) {
            const Eigen::Vector3i key = ((cell - origin) / leafSize).array().floor().cast<int>();
            minKey = minKey.cwiseMin(key);
            maxKey = maxKey.cwiseMax(key);
        }
        const Eigen::Vector3i dims = maxKey - minKey + Eigen::Vector3i::Ones();
        const size_t volume = static_cast<size_t>(dims(0)) * static_cast<size_t>(dims(1)) * static_cast<size_t>(dims(2));
        const size_t maxDense = std::min<size_t>(std::max<size_t>(4096, cells.size() * 8), 1u << 20);
        if (volume > maxDense) {
            downsampleHash_(origin, leafSize);
            return;
        }

        std::vector<Eigen::Vector3d> sums(volume, Eigen::Vector3d::Zero());
        std::vector<int> counts(volume, 0);
        const size_t strideX = static_cast<size_t>(dims(1)) * static_cast<size_t>(dims(2));
        const size_t strideY = static_cast<size_t>(dims(2));
        for (const auto& cell : cells) {
            const Eigen::Vector3i key = ((cell - origin) / leafSize).array().floor().cast<int>();
            const Eigen::Vector3i local = key - minKey;
            const size_t offset = static_cast<size_t>(local(0)) * strideX +
                static_cast<size_t>(local(1)) * strideY + static_cast<size_t>(local(2));
            sums[offset] += cell;
            counts[offset] += 1;
        }
        downsampledCells.clear();
        downsampledCells.reserve(cells.size());
        for (int x = 0; x < dims(0); ++ x) {
            for (int y = 0; y < dims(1); ++ y) {
                size_t offset = static_cast<size_t>(x) * strideX + static_cast<size_t>(y) * strideY;
                for (int z = 0; z < dims(2); ++ z, ++ offset) {
                    if (counts[offset] == 0) {
                        continue;
                    }
                    downsampledCells.push_back(sums[offset] / static_cast<double>(counts[offset]));
                }
            }
        }
    }

private:
    void downsampleHash_(const Eigen::Vector3d& origin, double leafSize) {
        std::unordered_map<Eigen::Vector3i, std::pair<Eigen::Vector3d, int>, misc::Vector3iHash> grid;
        grid.reserve(cells.size());
        for (const auto& cell : cells) {
            const Eigen::Vector3i key = ((cell - origin) / leafSize).array().floor().cast<int>();
            auto iter = grid.find(key);
            if (iter == grid.end()) {
                grid.emplace(key, std::make_pair(cell, 1));
            } else {
                iter->second.first += cell;
                iter->second.second += 1;
            }
        }
        downsampledCells.clear();
        downsampledCells.reserve(grid.size());
        for (const auto& entry : grid) {
            downsampledCells.push_back(entry.second.first / static_cast<double>(entry.second.second));
        }
    }
};

struct FrontierParams {
    int clusterMin{100};
    double clusterSizeXY{2.0};

    double minCandidateDist{0.5};
    double minCandidateClearance{0.21};
    double candidateDphi{15.0 * __pi__ / 180.0};
    int candidateRnum{3};
    double candidateRmin{1.0};
    double candidateRmax{1.5};

    int downSample{3};
    int minVisibleNum{30};
    double minViewFinishFraction{0.2};
    double minFrontierZ{0.2};
};

} // arc
