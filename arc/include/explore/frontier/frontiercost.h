#pragma once

#include <base/iarc.h>
#include <explore/frontier/frontierdef.h>
#include <explore/frontier/viewnode.h>
#include <explore/tsp/tspmatrix.h>

namespace arc {

class FrontierCost {
public:
    explicit FrontierCost(ViewNodeContext& viewNode)
        : viewNode_(viewNode)
    {}
    void refresh(std::vector<Frontier>& frontiers, const std::vector<int>& removedFrontierIndices) {
        if (!removedFrontierIndices.empty()) {
            for (auto& frontier : frontiers) {
                for (auto iter = removedFrontierIndices.rbegin(); iter != removedFrontierIndices.rend(); ++ iter) {
                    const size_t index = static_cast<size_t>(*iter);
                    if (index >= frontier.edgeCosts.size()) {
                        continue;
                    }
                    frontier.edgeCosts.erase(frontier.edgeCosts.begin() + index);
                    frontier.edgePaths.erase(frontier.edgePaths.begin() + index);
                }
            }
        }
        for (size_t i = 0; i < frontiers.size(); ++ i) {
            if (!frontiers[i].needsCostUpdate) {
                continue;
            }
            for (size_t j = 0; j < i; ++ j) {
                appendCostPair_(frontiers, i, j);
            }
            frontiers[i].edgeCosts.push_back(0.0);
            frontiers[i].edgePaths.emplace_back();
            frontiers[i].needsCostUpdate = false;
        }
    }
    void buildFullMatrix(
        const std::vector<Frontier>& frontiers,
        const Eigen::Vector3d& curPos,
        const Eigen::Vector3d& curVel,
        double curYaw,
        Eigen::MatrixXd& mat
    ) {
        const int count = static_cast<int>(frontiers.size());
        mat = Eigen::MatrixXd::Zero(count + 1, count + 1);
        for (int i = 0; i < count; ++ i) {
            const int costCount = std::min(static_cast<int>(frontiers[i].edgeCosts.size()), count);
            for (int j = 0; j < costCount; ++ j) {
                mat(i + 1, j + 1) = frontiers[i].edgeCosts[j];
            }
        }
        for (int i = 0; i < count; ++ i) {
            if (frontiers[i].viewpoints.empty()) {
                continue;
            }
            const Viewpoint& viewpoint = frontiers[i].viewpoints.front();
            pathScratch_.clear();
            mat(0, i + 1) = viewNode_.costBetween(curPos, viewpoint.pos, curYaw, viewpoint.yaw, curVel, 0.0, pathScratch_);
        }
    }
    void buildSwarmMatrix(
        const std::vector<Frontier>& frontiers,
        const std::vector<Eigen::Vector3d>& positions,
        const std::vector<Eigen::Vector3d>& velocities,
        const std::vector<double>& yaws,
        Eigen::MatrixXd& mat
    ) {
        std::vector<int> frontierIndices(frontiers.size());
        for (size_t i = 0; i < frontiers.size(); ++ i) {
            frontierIndices[i] = static_cast<int>(i);
        }
        buildSwarmMatrix(frontiers, positions, velocities, yaws, frontierIndices, {}, mat);
    }
    void buildSwarmMatrix(
        const std::vector<Frontier>& frontiers,
        const std::vector<Eigen::Vector3d>& positions,
        const std::vector<Eigen::Vector3d>& velocities,
        const std::vector<double>& yaws,
        const std::vector<int>& frontierIndices,
        const std::vector<Eigen::Vector3d>& gridPos,
        Eigen::MatrixXd& mat
    ) {
        const int droneNum = static_cast<int>(positions.size());
        const int frontierNum = static_cast<int>(frontierIndices.size());
        const bool hasGrid = !gridPos.empty();
        int dim = 1 + droneNum + frontierNum;
        if (hasGrid) {
            ++ dim;
        }
        mat = Eigen::MatrixXd::Zero(dim, dim);

        for (int i = 0; i < droneNum; ++ i) {
            mat(0, 1 + i) = TspMatrixCosts::DEPOT_TO_DRONE;
            mat(1 + i, 0) = TspMatrixCosts::DRONE_TO_DEPOT;
            for (int j = 0; j < droneNum; ++ j) {
                mat(1 + i, 1 + j) = TspMatrixCosts::DRONE_TO_DRONE;
            }
        }
        for (int i = 0; i < frontierNum; ++ i) {
            mat(0, 1 + droneNum + i) = TspMatrixCosts::GRID_LINK;
            mat(1 + droneNum + i, 0) = 0.0;
        }
        for (int i = 0; i < droneNum; ++ i) {
            const double yaw = i < static_cast<int>(yaws.size()) ? yaws[i] : 0.0;
            for (int j = 0; j < frontierNum; ++ j) {
                const int id = frontierIndices[j];
                if (id < 0 || id >= static_cast<int>(frontiers.size()) || frontiers[id].viewpoints.empty()) {
                    continue;
                }
                const Viewpoint& viewpoint = frontiers[id].viewpoints.front();
                pathScratch_.clear();
                mat(1 + i, 1 + droneNum + j) = viewNode_.costBetween(positions[i], viewpoint.pos, yaw, viewpoint.yaw, velocities[i], 0.0, pathScratch_);
                mat(1 + droneNum + j, 1 + i) = 0.0;
            }
        }
        for (int i = 0; i < frontierNum; ++ i) {
            const int id = frontierIndices[i];
            if (id < 0 || id >= static_cast<int>(frontiers.size())) {
                continue;
            }
            for (int j = 0; j < frontierNum; ++ j) {
                const int next = frontierIndices[j];
                if (next < 0 || next >= static_cast<int>(frontiers[id].edgeCosts.size())) {
                    continue;
                }
                mat(1 + droneNum + i, 1 + droneNum + j) = frontiers[id].edgeCosts[next];
            }
        }
        for (int i = 0; i < dim; ++ i) {
            mat(i, i) = TspMatrixCosts::SELF_LOOP;
        }
        if (hasGrid) {
            const int gridIndex = 1 + droneNum + frontierNum;
            mat(0, gridIndex) = TspMatrixCosts::GRID_LINK;
            mat(gridIndex, 0) = TspMatrixCosts::DEPOT_TO_DRONE;
            for (int i = 0; i < droneNum; ++ i) {
                mat(1 + i, gridIndex) = TspMatrixCosts::GRID_LINK;
                mat(gridIndex, 1 + i) = TspMatrixCosts::GRID_LINK;
            }
            for (int i = 0; i < frontierNum; ++ i) {
                const int id = frontierIndices[i];
                if (id < 0 || id >= static_cast<int>(frontiers.size()) || frontiers[id].viewpoints.empty()) {
                    continue;
                }
                const Viewpoint& viewpoint = frontiers[id].viewpoints.front();
                pathScratch_.clear();
                const double cost = viewNode_.costBetween(gridPos.front(), viewpoint.pos, 0.0, viewpoint.yaw, Eigen::Vector3d::Zero(), 0.0, pathScratch_);
                mat(1 + droneNum + i, gridIndex) = cost;
                mat(gridIndex, 1 + droneNum + i) = cost;
            }
        }
    }
    void buildPathForTour(
        const std::vector<Frontier>& frontiers,
        const Eigen::Vector3d& pos,
        const std::vector<int>& frontierIndices,
        std::vector<Eigen::Vector3d>& path
    ) {
        if (frontierIndices.empty()) {
            return;
        }
        for (int id : frontierIndices) {
            if (id < 0 || id >= static_cast<int>(frontiers.size())) {
                return;
            }
        }
        segmentScratch_.clear();
        if (!frontiers[frontierIndices.front()].viewpoints.empty()) {
            viewNode_.findPath(pos, frontiers[frontierIndices.front()].viewpoints.front().pos, segmentScratch_);
            path.insert(path.end(), segmentScratch_.begin(), segmentScratch_.end());
        }
        for (size_t i = 0; i + 1 < frontierIndices.size(); ++ i) {
            const Frontier& frontier = frontiers[frontierIndices[i]];
            const int next = frontierIndices[i + 1];
            if (next < static_cast<int>(frontier.edgePaths.size())) {
                path.insert(path.end(), frontier.edgePaths[next].begin(), frontier.edgePaths[next].end());
            }
        }
    }

private:
    void appendCostPair_(std::vector<Frontier>& frontiers, size_t i, size_t j) {
        const Viewpoint& first = frontiers[i].viewpoints.front();
        const Viewpoint& second = frontiers[j].viewpoints.front();
        pathScratch_.clear();
        const double cost = viewNode_.costBetween(first.pos, second.pos, first.yaw, second.yaw, Eigen::Vector3d::Zero(), 0.0, pathScratch_);
        frontiers[i].edgeCosts.push_back(cost);
        frontiers[i].edgePaths.push_back(pathScratch_);
        std::reverse(pathScratch_.begin(), pathScratch_.end());
        frontiers[j].edgeCosts.push_back(cost);
        frontiers[j].edgePaths.push_back(pathScratch_);
    }

private:
    ViewNodeContext& viewNode_;
    std::vector<Eigen::Vector3d> pathScratch_;
    std::vector<Eigen::Vector3d> segmentScratch_;
};

} // arc
