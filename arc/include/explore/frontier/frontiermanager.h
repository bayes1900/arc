#pragma once

#include <base/iarc.h>
#include <env/sdfmap.h>
#include <explore/frontier/frontiercost.h>
#include <explore/frontier/frontierdef.h>
#include <explore/frontier/frontierdetector.h>
#include <explore/frontier/viewnode.h>
#include <explore/frontier/viewpointsampler.h>

namespace arc {

class FrontierManager {
public:
    FrontierManager(SdfMap& map, ViewNodeContext& viewNode, const FrontierParams& params)
        : params_(params)
        , detector_(map, params_)
        , sampler_(map, params_)
        , cost_(viewNode)
    {}
    void detectFrontiers() {
        const size_t revision = detector_.mapRevision();
        if (hasRunSearch_ && revision != 0 && revision == lastSearchRevision_ && !detector_.hasPendingSeedCells() && !detector_.requiresFullScan()) {
            if (!needsViewpointSampling_) {
                pendingFrontiers_.clear();
            }
            return;
        }
        detector_.detect(activeFrontiers_, dormantFrontiers_, pendingFrontiers_, removedFrontierIndices_);
        hasRunSearch_ = true;
        lastSearchRevision_ = revision;
        needsViewpointSampling_ = !pendingFrontiers_.empty();
    }
    void sampleNewFrontierViewpoints() {
        for (auto& frontier : pendingFrontiers_) {
            sampler_.sampleViewpoints(frontier);
            if (frontier.viewpoints.empty()) {
                dormantFrontiers_.push_back(frontier);
                continue;
            }
            std::sort(frontier.viewpoints.begin(), frontier.viewpoints.end(), [](const Viewpoint& lhs, const Viewpoint& rhs) {
                return lhs.visibleCellCount > rhs.visibleCellCount;
            });
            activeFrontiers_.push_back(frontier);
        }
        needsViewpointSampling_ = false;
    }
    void refreshCostMatrix() {
        cost_.refresh(activeFrontiers_, removedFrontierIndices_);
        removedFrontierIndices_.clear();
    }
    void getFrontiers(std::vector<std::vector<Eigen::Vector3d>>& clusters) const {
        clusters.clear();
        clusters.reserve(activeFrontiers_.size());
        for (const auto& frontier : activeFrontiers_) {
            clusters.push_back(frontier.cells);
        }
    }
    size_t activeFrontierCount() const {
        return activeFrontiers_.size();
    }
    size_t dormantFrontierCount() const {
        return dormantFrontiers_.size();
    }
    const std::vector<Frontier>& frontiers() const {
        return activeFrontiers_;
    }
    void getDormantFrontiers(std::vector<std::vector<Eigen::Vector3d>>& clusters) const {
        clusters.clear();
        clusters.reserve(dormantFrontiers_.size());
        for (const auto& frontier : dormantFrontiers_) {
            clusters.push_back(frontier.cells);
        }
    }
    void getFrontierBoxes(std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& boxes) const {
        boxes.clear();
        boxes.reserve(activeFrontiers_.size());
        for (const auto& frontier : activeFrontiers_) {
            boxes.emplace_back((frontier.boxMin + frontier.boxMax) * 0.5, frontier.boxMax - frontier.boxMin);
        }
    }
    void selectTopViewpoints(
        const Eigen::Vector3d& curPos,
        std::vector<Eigen::Vector3d>& points,
        std::vector<double>& yaws,
        std::vector<Eigen::Vector3d>& averages
    ) const {
        points.clear();
        yaws.clear();
        averages.clear();
        for (const auto& frontier : activeFrontiers_) {
            if (frontier.viewpoints.empty()) {
                continue;
            }
            const Viewpoint* selected = &frontier.viewpoints.front();
            for (const auto& viewpoint : frontier.viewpoints) {
                if ((viewpoint.pos - curPos).norm() < params_.minCandidateDist) {
                    continue;
                }
                selected = &viewpoint;
                break;
            }
            points.push_back(selected->pos);
            yaws.push_back(selected->yaw);
            averages.push_back(frontier.center);
        }
    }
    void selectViewpoints(
        const Eigen::Vector3d& curPos,
        const std::vector<int>& ids,
        int viewNum,
        double maxDecay,
        std::vector<std::vector<Eigen::Vector3d>>& points,
        std::vector<std::vector<double>>& yaws
    ) const {
        points.clear();
        yaws.clear();
        for (int id : ids) {
            if (id < 0 || id >= static_cast<int>(activeFrontiers_.size())) {
                continue;
            }
            const Frontier& frontier = activeFrontiers_[id];
            if (frontier.viewpoints.empty()) {
                points.emplace_back();
                yaws.emplace_back();
                continue;
            }

            std::vector<Eigen::Vector3d> selectedPoints;
            std::vector<double> selectedYaws;
            const int threshold = static_cast<int>(frontier.viewpoints.front().visibleCellCount * maxDecay);
            for (const auto& viewpoint : frontier.viewpoints) {
                if (static_cast<int>(selectedPoints.size()) >= viewNum || viewpoint.visibleCellCount <= threshold) {
                    break;
                }
                if ((viewpoint.pos - curPos).norm() < params_.minCandidateDist) {
                    continue;
                }
                selectedPoints.push_back(viewpoint.pos);
                selectedYaws.push_back(viewpoint.yaw);
            }
            if (selectedPoints.empty()) {
                for (const auto& viewpoint : frontier.viewpoints) {
                    if (static_cast<int>(selectedPoints.size()) >= viewNum || viewpoint.visibleCellCount <= threshold) {
                        break;
                    }
                    selectedPoints.push_back(viewpoint.pos);
                    selectedYaws.push_back(viewpoint.yaw);
                }
            }
            points.push_back(std::move(selectedPoints));
            yaws.push_back(std::move(selectedYaws));
        }
    }
    void buildFullCostMatrix(const Eigen::Vector3d& curPos, const Eigen::Vector3d& curVel, double curYaw, Eigen::MatrixXd& mat) {
        cost_.buildFullMatrix(activeFrontiers_, curPos, curVel, curYaw, mat);
    }
    void buildSwarmCostMatrix(
        const std::vector<Eigen::Vector3d>& positions,
        const std::vector<Eigen::Vector3d>& velocities,
        const std::vector<double>& yaws,
        Eigen::MatrixXd& mat
    ) {
        cost_.buildSwarmMatrix(activeFrontiers_, positions, velocities, yaws, mat);
    }
    void buildSwarmCostMatrix(
        const std::vector<Eigen::Vector3d>& positions,
        const std::vector<Eigen::Vector3d>& velocities,
        const std::vector<double>& yaws,
        const std::vector<int>& frontierIndices,
        const std::vector<Eigen::Vector3d>& gridPos,
        Eigen::MatrixXd& mat
    ) {
        cost_.buildSwarmMatrix(activeFrontiers_, positions, velocities, yaws, frontierIndices, gridPos, mat);
    }
    void buildPathForTour(const Eigen::Vector3d& pos, const std::vector<int>& frontierIndices, std::vector<Eigen::Vector3d>& path) {
        cost_.buildPathForTour(activeFrontiers_, pos, frontierIndices, path);
    }
    void removeFrontiers(const std::vector<int>& ids) {
        std::vector<bool> remove(activeFrontiers_.size(), false);
        for (int id : ids) {
            if (id >= 0 && id < static_cast<int>(activeFrontiers_.size())) {
                remove[id] = true;
            }
        }

        removedFrontierIndices_.clear();
        std::vector<Frontier> kept;
        kept.reserve(activeFrontiers_.size());
        for (size_t i = 0; i < activeFrontiers_.size(); ++ i) {
            if (!remove[i]) {
                kept.push_back(std::move(activeFrontiers_[i]));
                continue;
            }
            detector_.releaseFrontier(activeFrontiers_[i]);
            removedFrontierIndices_.push_back(static_cast<int>(i));
        }
        activeFrontiers_ = std::move(kept);
    }
    void addFrontiersFromViews(const std::vector<std::pair<Eigen::Vector3d, double>>& views) {
        for (const auto& view : views) {
            Frontier frontier;
            Viewpoint viewpoint;
            viewpoint.pos = view.first;
            viewpoint.yaw = view.second;
            frontier.viewpoints.push_back(viewpoint);
            activeFrontiers_.push_back(std::move(frontier));
        }
    }
    bool isAnyFrontierCovered() const {
        return detector_.isCovered(activeFrontiers_, dormantFrontiers_);
    }
    int viewGain(const Eigen::Vector3d& pos, double yaw) {
        return sampler_.viewGain(pos, yaw);
    }

private:
    friend class FrontierManagerTestPeer;
    FrontierParams params_;
    FrontierDetector detector_;
    ViewpointSampler sampler_;
    FrontierCost cost_;
    std::vector<Frontier> activeFrontiers_;
    std::vector<Frontier> dormantFrontiers_;
    std::vector<Frontier> pendingFrontiers_;
    std::vector<int> removedFrontierIndices_;
    size_t lastSearchRevision_{0};
    bool hasRunSearch_{false};
    bool needsViewpointSampling_{false};
};

} // arc
