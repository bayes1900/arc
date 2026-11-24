#pragma once

#include <base/base.h>
#include <base/ikdtree.h>
#include <lio/lioconfig.h>
#include <lio/liotypes.h>

namespace arc {

class IncrementalMap {
public:
    using Tree = KdTree<LioPoint>;

    explicit IncrementalMap(const LioConfig& config)
        : config_(config)
        , tree_(std::make_unique<Tree>(0.5f, 0.6f, 0.2f))
    {
        tree_->setDownsampleParam(static_cast<float>(config_.filterSizeMap));
    }
    bool empty() const {
        return tree_->root_ == nullptr;
    }
    void setDownsampleParam(double size) {
        tree_->setDownsampleParam(static_cast<float>(size));
    }
    void build(const LioPointVector& points) {
        tree_->build(points);
    }
    void segmentFov(const Vec3d& positionLidar) {
        cubNeedRemove_.clear();

        if (!localMapInitialized_) {
            for (int i = 0; i < 3; ++ i) {
                localMapPoints_.vertex_min[i] = static_cast<float>(
                    positionLidar(i) - config_.cubeSideLength / 2.0);
                localMapPoints_.vertex_max[i] = static_cast<float>(
                    positionLidar(i) + config_.cubeSideLength / 2.0);
            }
            localMapInitialized_ = true;
            return;
        }
        const double detectRange = config_.detRange;
        bool needMove = false;
        double distToEdge[3][2];
        for (int i = 0; i < 3; ++ i) {
            distToEdge[i][0] = std::fabs(positionLidar(i) - localMapPoints_.vertex_min[i]);
            distToEdge[i][1] = std::fabs(positionLidar(i) - localMapPoints_.vertex_max[i]);
            if (distToEdge[i][0] <= MOV_THRESHOLD * detectRange
                || distToEdge[i][1] <= MOV_THRESHOLD * detectRange) {
                needMove = true;
            }
        }
        if (!needMove) {
            return;
        }
        BoxPointType newLocalMap = localMapPoints_;
        const double moveDist = std::max(
            (config_.cubeSideLength - 2.0 * MOV_THRESHOLD * detectRange) * 0.5 * 0.9,
            detectRange * (MOV_THRESHOLD - 1.0)
        );
        for (int i = 0; i < 3; ++ i) {
            BoxPointType removedBox = localMapPoints_;
            if (distToEdge[i][0] <= MOV_THRESHOLD * detectRange) {
                newLocalMap.vertex_max[i] -= static_cast<float>(moveDist);
                newLocalMap.vertex_min[i] -= static_cast<float>(moveDist);
                removedBox.vertex_min[i] = localMapPoints_.vertex_max[i] - static_cast<float>(moveDist);
                cubNeedRemove_.push_back(removedBox);
            } else if (distToEdge[i][1] <= MOV_THRESHOLD * detectRange) {
                newLocalMap.vertex_max[i] += static_cast<float>(moveDist);
                newLocalMap.vertex_min[i] += static_cast<float>(moveDist);
                removedBox.vertex_max[i] = localMapPoints_.vertex_min[i] + static_cast<float>(moveDist);
                cubNeedRemove_.push_back(removedBox);
            }
        }
        localMapPoints_ = newLocalMap;

        if (!cubNeedRemove_.empty()) {
            tree_->deletePointBoxes(cubNeedRemove_);
        }
    }
    int nearestSearch(
        const LioPoint& point,
        int k,
        LioPointVector& nearest,
        std::vector<float>& squaredDistances
    ) {
        tree_->nearestSearch(point, k, nearest, squaredDistances);
        return static_cast<int>(nearest.size());
    }
    int addPoints(LioPointVector& points, bool downsampleOn) {
        return tree_->addPoints(points, downsampleOn);
    }
    int deletePointBoxes(std::vector<BoxPointType>& boxes) {
        return tree_->deletePointBoxes(boxes);
    }
    int size() {
        return tree_->size();
    }
    int validNum() {
        return tree_->validNum();
    }
    void flatten(LioPointVector& storage) {
        tree_->flatten(tree_->root_, storage, DeleteStorage::NOT_RECORD);
    }
    void acquireRemovedPoints(LioPointVector& points) {
        tree_->acquireRemovedPoints(points);
    }

private:
    static constexpr double MOV_THRESHOLD = 1.5;

    LioConfig config_;
    std::unique_ptr<Tree> tree_;
    BoxPointType localMapPoints_{};
    bool localMapInitialized_{false};
    std::vector<BoxPointType> cubNeedRemove_;
};

} // arc
