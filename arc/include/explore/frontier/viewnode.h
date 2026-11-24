#pragma once

#include <base/iarc.h>
#include <env/sdfmap.h>
#include <search/astar.h>

namespace arc {

struct ViewNodeParams {
    double vm{1.5};
    double yd{80.0 * 3.14159265358979323846 / 180.0};
    double wDir{1.5};

    double noPathPenaltyFactor{10.0};
    double minNoPathPenaltyLength{100.0};
    double noPathPenaltyLength{100.0};
};

class ViewNodeContext {
public:
    ViewNodeContext(SdfMap& map, const ViewNodeParams& params, const AstarOptions& options)
        : params_(params)
        , astar_(map, options)
    {
        const double mapDiagonal = map.params().size.norm();
        params_.noPathPenaltyLength = std::max(
            params_.minNoPathPenaltyLength,
            params_.noPathPenaltyFactor * mapDiagonal
        );
    }
    const ViewNodeParams& params() const {
        return params_;
    }
    double costBetween(
        const Eigen::Vector3d& p1,
        const Eigen::Vector3d& p2,
        double y1,
        double y2,
        const Eigen::Vector3d& v1,
        double,
        std::vector<Eigen::Vector3d>& path
    ) {
        double posCost = findPath(p1, p2, path) / std::max(params_.vm, 1e-6);
        if (v1.norm() > 1e-3) {
            const Eigen::Vector3d delta = p2 - p1;
            if (delta.norm() > 1e-6) {
                const Eigen::Vector3d direction = delta.normalized();
                const Eigen::Vector3d velocityDirection = v1.normalized();
                const double dot = std::max(-1.0, std::min(1.0, velocityDirection.dot(direction)));
                posCost += params_.wDir * std::acos(dot);
            }
        }
        const double yawDiff = std::fabs(misc::normalizeAngle(y2) - misc::normalizeAngle(y1));
        const double yawCost = std::min(yawDiff, 2.0 * __pi__ - yawDiff) / std::max(params_.yd, 1e-6);
        return std::max(posCost, yawCost);
    }
    double findPath(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2, std::vector<Eigen::Vector3d>& path) {
        if (astar_.isLineFree(p1, p2)) {
            path = {p1, p2};
            return (p1 - p2).norm();
        }
        astar_.reset();
        if (astar_.search(p1, p2, true) == AstarSearchStatus::DONE) {
            path = astar_.getPath();
            return misc::pathLength(path);
        }
        path = {p1, p2};
        return params_.noPathPenaltyLength;
    }

private:
    ViewNodeParams params_;
    Astar astar_;
};

struct ViewNode {
    ViewNodeContext& context;
    std::vector<ViewNode*> neighbors;
    ViewNode* parent{nullptr};
    Eigen::Vector3d pos{Eigen::Vector3d::Zero()};
    Eigen::Vector3d vel{Eigen::Vector3d::Zero()};
    double yaw{0.0};
    double yawDot{0.0};

    ViewNode(ViewNodeContext& nodeContext, const Eigen::Vector3d& position, double nodeYaw)
        : context(nodeContext)
        , pos(position)
        , yaw(misc::normalizeAngle(nodeYaw))
    {}
    double costTo(const ViewNode& node) const {
        std::vector<Eigen::Vector3d> path;
        return context.costBetween(pos, node.pos, yaw, node.yaw, vel, yawDot, path);
    }
};

} // arc
