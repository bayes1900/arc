#pragma once

#include <base/iarc.h>
#include <env/sdfmap.h>

namespace arc {

struct AstarOptions {
    double resolution{0};
    double lambdaHeu{1.0};
    double maxSearchTime{1.0};
    double collisionCheckStep{0.1};
    size_t maxNodes{1000000};
};

struct AstarSearchStatus {
    constexpr static uint8_t DONE    = 0;
    constexpr static uint8_t NOPATH  = 1;
    constexpr static uint8_t TIMEOUT = 2;
    constexpr static uint8_t OUTSIDE = 3;
    constexpr static uint8_t NOMEM   = 4;
};

struct AstarNodeState {
    constexpr static uint8_t INIT  = 0;
    constexpr static uint8_t OPEN  = 1;
    constexpr static uint8_t CLOSE = 2;
};

struct AstarNode {
    Eigen::Vector3i index;
    Eigen::Vector3d position;
    double gScore{0.0};
    double fScore{0.0};
    AstarNode* parent{nullptr};
    uint8_t state{0};
};

struct AstarNodeCompare {
    bool operator()(const AstarNode* lhs, const AstarNode* rhs) const {
        return lhs->fScore > rhs->fScore;
    }
};

class Astar {
public:
    Astar(SdfMap& sdfmap, const AstarOptions& options)
        : sdfmap_(sdfmap)
        , mp_(sdfmap.params())
        , options_(options)
        , boxMin_(sdfmap.params().boxMin)
        , boxMax_(sdfmap.params().boxMax)
    {
        if (options_.resolution <= 0) {
            options_.resolution = mp_.resolution;
        }
        resolutionInv_ = 1.0 / options_.resolution;
    }
    ~Astar() {
        for (AstarNode* node : nodePool_) {
            delete node;
        }
    }
    Astar(const Astar&) = delete;
    Astar& operator=(const Astar&) = delete;
    void reset() {
        path_.clear();
        openSetMap_.clear();
        closeSetMap_.clear();
        std::priority_queue<AstarNode*, std::vector<AstarNode*>, AstarNodeCompare> emptyOpenSet;
        openSet_.swap(emptyOpenSet);
        for (size_t i = 0; i < nodeNum_; ++ i) {
            nodePool_[i]->parent = nullptr;
            nodePool_[i]->state = 0;
        }
        nodeNum_ = 0;
    }
    void setBox(const Eigen::Vector3d& boxMin, const Eigen::Vector3d& boxMax) {
        if ((boxMax.array() <= boxMin.array()).any()) {
            return;
        }
        boxMin_ = boxMin;
        boxMax_ = boxMax;
    }
    const std::vector<Eigen::Vector3d>& getPath() const {
        return path_;
    }
    bool isLineFree(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2) const {
        return isInBox_(p1) && isInBox_(p2) && sdfmap_.isLineFree(p1, p2);
    }

public:
    uint8_t search(const Eigen::Vector3d& start, const Eigen::Vector3d& end, bool optimistic=true) {
        if (!isInBox_(end) ||
            !sdfmap_.params().isInMap(start) || !sdfmap_.params().isInMap(end)) {
            return AstarSearchStatus::OUTSIDE;
        }
        reset();
        const auto searchStart = std::chrono::steady_clock::now();
        AstarNode* currentNode = releaseNode_();
        if (currentNode == nullptr) {
            return AstarSearchStatus::NOMEM;
        }
        currentNode->parent = nullptr;
        currentNode->position = start;
        currentNode->index = mp_.p2i(start, resolutionInv_);
        currentNode->gScore = 0.0;
        currentNode->fScore = options_.lambdaHeu * getDiagHeu_(start, end);
        currentNode->state = AstarNodeState::OPEN;

        const Eigen::Vector3i endIndex = mp_.p2i(end, resolutionInv_);
        openSet_.push(currentNode);
        openSetMap_[currentNode->index] = currentNode;

        while (!openSet_.empty()) {
            currentNode = openSet_.top();
            if ((currentNode->index - endIndex).cwiseAbs().maxCoeff() <= 1) {
                retrievePath_(currentNode, end);
                return AstarSearchStatus::DONE;
            }
            if (options_.maxSearchTime > 0.0) {
                const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - searchStart).count();
                if (elapsed > options_.maxSearchTime) {
                    return AstarSearchStatus::TIMEOUT;
                }
            }
            openSet_.pop();
            openSetMap_.erase(currentNode->index);
            closeSetMap_[currentNode->index] = 1;
            currentNode->state = AstarNodeState::CLOSE;

            for (int dx = -1; dx <= 1; ++ dx) {
                for (int dy = -1; dy <= 1; ++ dy) {
                    for (int dz = -1; dz <= 1; ++ dz) {
                        if (dx == 0 && dy == 0 && dz == 0) {
                            continue;
                        }
                        const Eigen::Vector3d step(dx * options_.resolution, dy * options_.resolution, dz * options_.resolution);
                        const Eigen::Vector3d neighborPosition = currentNode->position + step;
                        const Eigen::Vector3i neighborIndex = mp_.p2i(neighborPosition, resolutionInv_);
                        if (closeSetMap_.find(neighborIndex) != closeSetMap_.end()) {
                            continue;
                        }
                        if (!isTransitionSafe_(currentNode->position, neighborPosition, optimistic)) {
                            continue;
                        }
                        const double newGScore = currentNode->gScore + step.norm();
                        auto openIter = openSetMap_.find(neighborIndex);
                        AstarNode* neighbor = nullptr;
                        if (openIter == openSetMap_.end()) {
                            neighbor = releaseNode_();
                            if (neighbor == nullptr) {
                                return AstarSearchStatus::NOMEM;
                            }
                            neighbor->index = neighborIndex;
                            neighbor->position = neighborPosition;
                        } else {
                            neighbor = openIter->second;
                            if (newGScore >= neighbor->gScore) {
                                continue;
                            }
                        }
                        neighbor->parent = currentNode;
                        neighbor->gScore = newGScore;
                        neighbor->fScore = newGScore + options_.lambdaHeu * getDiagHeu_(neighborPosition, end);
                        neighbor->state = AstarNodeState::OPEN;
                        openSet_.push(neighbor);
                        openSetMap_[neighborIndex] = neighbor;
                    }
                }
            }
        }
        return AstarSearchStatus::NOPATH;
    }

private:
    AstarNode* releaseNode_() {
        if (nodeNum_ >= options_.maxNodes) {
            return nullptr;
        }
        if (nodeNum_ >= nodePool_.size()) {
            nodePool_.push_back(new AstarNode);
        }
        AstarNode* node = nodePool_[nodeNum_ ++];
        node->parent = nullptr;
        node->state = AstarNodeState::INIT;
        return node;
    }
    bool isInBox_(const Eigen::Vector3d& pos) const {
        return (pos.array() > boxMin_.array()).all() && (pos.array() < boxMax_.array()).all();
    }
    bool isVoxelFree_(const Eigen::Vector3d& pos, bool optimistic) const {
        const MapParams& mp = sdfmap_.params();
        const MapData& md = sdfmap_.data();
        if (!isInBox_(pos) || !mp.isInMap(pos)) {
            return false;
        }
        if (md.getInflateOccupancy(pos) != VoxelInflateOccupancyCode::FREE) {
            return false;
        }
        if (!optimistic) {
            const uint8_t occupancy = md.getOccupancy(pos);
            if (occupancy == VoxelOccupancyCode::UNKNOWN || occupancy == VoxelOccupancyCode::OUTSIDE) {
                return false;
            }
        }
        return true;
    }
    bool isTransitionSafe_(const Eigen::Vector3d& from, const Eigen::Vector3d& to, bool optimistic) const {
        if (!isVoxelFree_(to, optimistic)) {
            return false;
        }
        const Eigen::Vector3d direction = to - from;
        const double length = direction.norm();
        if (length < 1e-6) {
            return true;
        }
        const Eigen::Vector3d unit = direction / length;
        for (double distance = options_.collisionCheckStep; distance < length - 1e-6; distance += options_.collisionCheckStep) {
            if (!isVoxelFree_(from + distance * unit, optimistic)) {
                return false;
            }
        }
        return true;
    }
    void retrievePath_(AstarNode* endNode, const Eigen::Vector3d& end) {
        path_.clear();
        path_.push_back(end);
        path_.push_back(endNode->position);
        AstarNode* node = endNode;
        while (node->parent != nullptr) {
            node = node->parent;
            path_.push_back(node->position);
        }
        std::reverse(path_.begin(), path_.end());
    }
    double getDiagHeu_(const Eigen::Vector3d& x1, const Eigen::Vector3d& x2) const {
        double dx = std::fabs(x1(0) - x2(0));
        double dy = std::fabs(x1(1) - x2(1));
        double dz = std::fabs(x1(2) - x2(2));
        const double diag = std::min(std::min(dx, dy), dz);
        dx -= diag;
        dy -= diag;
        dz -= diag;
        double heuristic = 0.0;
        if (dx < 1e-4) {
            heuristic = std::sqrt(3.0) * diag + std::sqrt(2.0) * std::min(dy, dz) + std::fabs(dy - dz);
        } else if (dy < 1e-4) {
            heuristic = std::sqrt(3.0) * diag + std::sqrt(2.0) * std::min(dx, dz) + std::fabs(dx - dz);
        } else if (dz < 1e-4) {
            heuristic = std::sqrt(3.0) * diag + std::sqrt(2.0) * std::min(dx, dy) + std::fabs(dx - dy);
        } else {
            heuristic = std::sqrt(3.0) * diag;
        }
        return tieBreaker_ * heuristic;
    }

private:
    SdfMap& sdfmap_;
    const MapParams& mp_;
    AstarOptions options_;
    Eigen::Vector3d boxMin_;
    Eigen::Vector3d boxMax_;

private:
    double resolutionInv_{1.0 / 0.3};
    double tieBreaker_{1.0 + 1.0 / 1000.0};
    size_t nodeNum_{0};
    std::vector<AstarNode*> nodePool_;
    std::priority_queue<AstarNode*, std::vector<AstarNode*>, AstarNodeCompare> openSet_;
    std::unordered_map<Eigen::Vector3i, AstarNode*, misc::Vector3iHash> openSetMap_;
    std::unordered_map<Eigen::Vector3i, uint8_t, misc::Vector3iHash> closeSetMap_;
    std::vector<Eigen::Vector3d> path_;
};

} // arc
