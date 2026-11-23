#pragma once

#include <env/sdfmap.h>
#include <planning/bsplineoptdef.h>

namespace arc {

class BSplineCost {
public:
    BSplineCost(const BSplineOptParams& params, const SdfMap& map)
        : params_(params)
        , map_(map)
    {}
    double evaluate(
        const BSplineOptRequest& request,
        const Eigen::MatrixXd& points,
        double interval,
        double ptAvgDist,
        Eigen::MatrixXd& gradient,
        double& timeGradient
    ) const {
        const int numPoints = points.rows();
        gradient.setZero(numPoints, 3);
        timeGradient = 0.0;
        double cost = 0.0;

        if (request.cost.SMOOTHNESS) {
            cost += addSmoothnessCost_(points, ptAvgDist, gradient);
        }
        if (request.cost.DISTANCE) {
            cost += addDistanceCost_(points, gradient);
        }
        if (request.cost.FEASIBILITY) {
            cost += addFeasibilityCost_(points, interval, gradient, timeGradient);
        }
        if (request.cost.START) {
            cost += addStartCost_(request, points, interval, gradient, timeGradient);
        }
        if (request.cost.END) {
            cost += addEndCost_(request, points, interval, gradient, timeGradient);
        }
        if (request.cost.GUIDE) {
            cost += addGuideCost_(request, points, gradient);
        }
        if (request.cost.WAYPOINTS) {
            cost += addWaypointsCost_(request, points, gradient);
        }
        if (request.cost.VIEWCONS) {
            cost += addViewCost_(request, points, gradient);
        }
        if (request.cost.MINTIME) {
            cost += addTimeCost_(numPoints, interval, request.timeLowerBound, timeGradient);
        }
        if (request.cost.SWARM) {
            cost += addSwarmCost_(request, points, interval, gradient);
        }
        return cost;
    }

private:
    double addSmoothnessCost_(const Eigen::MatrixXd& points, double ptAvgDist, Eigen::MatrixXd& gradient) const {
        double cost = 0.0;
        for (int i = 0; i + 3 < points.rows(); ++ i) {
            const Eigen::Vector3d diff = (points.row(i + 3) - 3.0 * points.row(i + 2) + 3.0 * points.row(i + 1) - points.row(i)) .transpose() / ptAvgDist;
            cost += params_.smoothWeight * diff.squaredNorm();
            const Eigen::Vector3d grad = 2.0 * params_.smoothWeight * diff / ptAvgDist;
            gradient.row(i) -= grad.transpose();
            gradient.row(i + 1) += 3.0 * grad.transpose();
            gradient.row(i + 2) -= 3.0 * grad.transpose();
            gradient.row(i + 3) += grad.transpose();
        }
        return cost;
    }
    double addDistanceCost_(const Eigen::MatrixXd& points, Eigen::MatrixXd& gradient) const {
        double cost = 0.0;
        for (int i = 0; i < points.rows(); ++ i) {
            const Eigen::Vector3d position = points.row(i).transpose();
            if (!map_.params().isInMap(position)) {
                continue;
            }
            Eigen::Vector3d grad = Eigen::Vector3d::Zero();
            const double distance = map_.getDistWithGrad(position, grad);
            if (!std::isfinite(distance) || !grad.allFinite()) {
                continue;
            }
            const double error = distance - params_.distanceThreshold;
            if (error < 0.0) {
                cost += params_.distanceWeight * error * error;
                gradient.row(i) += (2.0 * params_.distanceWeight * error * grad).transpose();
            }
        }
        return cost;
    }
    double addFeasibilityCost_(const Eigen::MatrixXd& points, double interval, Eigen::MatrixXd& gradient, double& timeGradient) const {
        if (interval < 1e-6 || params_.maxVel <= 0.0 || params_.maxAcc <= 0.0) {
            return 0.0;
        }
        double cost = 0.0;
        const double invInterval = 1.0 / interval;
        const double invInterval2 = invInterval * invInterval;
        for (int i = 0; i + 1 < points.rows(); ++ i) {
            const Eigen::Vector3d velocity = (points.row(i + 1) - points.row(i)).transpose() * invInterval;
            for (int axis = 0; axis < 3; ++ axis) {
                const double excess = std::abs(velocity(axis)) - params_.maxVel;
                if (excess <= 0.0) {
                    continue;
                }
                cost += params_.feasibilityWeight * excess * excess;
                const double sign = velocity(axis) > 0.0 ? 1.0 : -1.0;
                const double grad = 2.0 * params_.feasibilityWeight * excess * sign * invInterval;
                gradient(i, axis) -= grad;
                gradient(i + 1, axis) += grad;
                timeGradient -= grad * velocity(axis);
            }
        }
        for (int i = 0; i + 2 < points.rows(); ++ i) {
            const Eigen::Vector3d acceleration = (points.row(i + 2) - 2.0 * points.row(i + 1) + points.row(i)).transpose() * invInterval2;
            for (int axis = 0; axis < 3; ++ axis) {
                const double excess = std::abs(acceleration(axis)) - params_.maxAcc;
                if (excess <= 0.0) {
                    continue;
                }
                cost += params_.feasibilityWeight * excess * excess;
                const double sign = acceleration(axis) > 0.0 ? 1.0 : -1.0;
                const double grad = 2.0 * params_.feasibilityWeight * excess * sign * invInterval2;
                gradient(i, axis) += grad;
                gradient(i + 1, axis) -= 2.0 * grad;
                gradient(i + 2, axis) += grad;
                timeGradient -= 2.0 * grad * acceleration(axis) * interval;
            }
        }
        return cost;
    }
    double addStartCost_(const BSplineOptRequest& request, const Eigen::MatrixXd& points, double interval, Eigen::MatrixXd& gradient, double& timeGradient) const {
        if (request.startState.empty() || points.rows() < 3) {
            return 0.0;
        }
        double cost = 0.0;
        const Eigen::Vector3d q0 = points.row(0).transpose();
        const Eigen::Vector3d q1 = points.row(1).transpose();
        const Eigen::Vector3d q2 = points.row(2).transpose();

        Eigen::Vector3d error = (q0 + 4.0 * q1 + q2) / 6.0 - request.startState[0];
        cost += params_.startWeight * params_.startPositionWeight * error.squaredNorm();
        gradient.row(0) += (params_.startWeight * params_.startPositionWeight * 2.0 * error / 6.0).transpose();
        gradient.row(1) += (params_.startWeight * params_.startPositionWeight * 8.0 * error / 6.0).transpose();
        gradient.row(2) += (params_.startWeight * params_.startPositionWeight * 2.0 * error / 6.0).transpose();

        if (request.startState.size() >= 2) {
            error = (q2 - q0) / (2.0 * interval) - request.startState[1];
            cost += params_.startWeight * error.squaredNorm();
            gradient.row(0) -= (params_.startWeight * error / interval).transpose();
            gradient.row(2) += (params_.startWeight * error / interval).transpose();
            timeGradient -= params_.startWeight * error.dot(q2 - q0) / (interval * interval);
        }
        if (request.startState.size() >= 3) {
            error = (q0 - 2.0 * q1 + q2) / (interval * interval) - request.startState[2];
            cost += params_.startWeight * error.squaredNorm();
            gradient.row(0) += (params_.startWeight * 2.0 * error / (interval * interval)).transpose();
            gradient.row(1) -= (params_.startWeight * 4.0 * error / (interval * interval)).transpose();
            gradient.row(2) += (params_.startWeight * 2.0 * error / (interval * interval)).transpose();
            timeGradient -= params_.startWeight * error.dot(q0 - 2.0 * q1 + q2) / (interval * interval * interval);
        }
        return cost;
    }
    double addEndCost_(const BSplineOptRequest& request, const Eigen::MatrixXd& points, double interval, Eigen::MatrixXd& gradient, double& timeGradient) const {
        if (request.endState.empty() || points.rows() < 3) {
            return 0.0;
        }
        double cost = 0.0;
        const int n = points.rows();
        const Eigen::Vector3d q0 = points.row(n - 3).transpose();
        const Eigen::Vector3d q1 = points.row(n - 2).transpose();
        const Eigen::Vector3d q2 = points.row(n - 1).transpose();

        Eigen::Vector3d error = (q2 + 4.0 * q1 + q0) / 6.0 - request.endState[0];
        cost += params_.endWeight * error.squaredNorm();
        gradient.row(n - 3) += (params_.endWeight * 2.0 * error / 6.0).transpose();
        gradient.row(n - 2) += (params_.endWeight * 8.0 * error / 6.0).transpose();
        gradient.row(n - 1) += (params_.endWeight * 2.0 * error / 6.0).transpose();

        if (request.endState.size() >= 2) {
            error = (q2 - q0) / (2.0 * interval) - request.endState[1];
            cost += params_.endWeight * error.squaredNorm();
            gradient.row(n - 1) += (params_.endWeight * error / interval).transpose();
            gradient.row(n - 3) -= (params_.endWeight * error / interval).transpose();
            timeGradient -= params_.endWeight * error.dot(q2 - q0) / (interval * interval);
        }
        if (request.endState.size() >= 3) {
            error = (q2 - 2.0 * q1 + q0) / (interval * interval) - request.endState[2];
            cost += params_.endWeight * error.squaredNorm();
            gradient.row(n - 1) += (params_.endWeight * 2.0 * error / (interval * interval)).transpose();
            gradient.row(n - 2) -= (params_.endWeight * 4.0 * error / (interval * interval)).transpose();
            gradient.row(n - 3) += (params_.endWeight * 2.0 * error / (interval * interval)).transpose();
            timeGradient -= params_.endWeight * error.dot(q2 - 2.0 * q1 + q0) / (interval * interval * interval);
        }
        return cost;
    }
    double addGuideCost_(const BSplineOptRequest& request, const Eigen::MatrixXd& points, Eigen::MatrixXd& gradient) const {
        if (request.guidePoints.empty()) {
            return 0.0;
        }
        double cost = 0.0;
        for (int i = 3; i + 3 < points.rows(); ++ i) {
            const int guideIndex = i - 3;
            if (guideIndex >= static_cast<int>(request.guidePoints.size())) {
                break;
            }
            const Eigen::Vector3d error = points.row(i).transpose() - request.guidePoints[guideIndex];
            cost += params_.guideWeight * error.squaredNorm();
            gradient.row(i) += (2.0 * params_.guideWeight * error).transpose();
        }
        return cost;
    }
    double addWaypointsCost_(const BSplineOptRequest& request, const Eigen::MatrixXd& points, Eigen::MatrixXd& gradient) const {
        double cost = 0.0;
        const int count = std::min(request.waypoints.size(), request.waypointIndices.size());
        for (int i = 0; i < count; ++ i) {
            const int index = request.waypointIndices[i];
            if (index < 0 || index + 2 >= points.rows()) {
                continue;
            }
            const Eigen::Vector3d position = (points.row(index) + 4.0 * points.row(index + 1) + points.row(index + 2)).transpose() / 6.0;
            const Eigen::Vector3d error = position - request.waypoints[i];
            cost += params_.waypointWeight * error.squaredNorm();
            gradient.row(index) += (params_.waypointWeight * 2.0 * error / 6.0).transpose();
            gradient.row(index + 1) += (params_.waypointWeight * 8.0 * error / 6.0).transpose();
            gradient.row(index + 2) += (params_.waypointWeight * 2.0 * error / 6.0).transpose();
        }
        return cost;
    }
    double addViewCost_(const BSplineOptRequest& request, const Eigen::MatrixXd& points, Eigen::MatrixXd& gradient) const {
        if (!request.view || request.view->index < 0 || request.view->index >= points.rows() || request.view->direction.norm() < 1e-6) {
            return 0.0;
        }
        double cost = 0.0;
        const int index = request.view->index;
        const Eigen::Vector3d direction = request.view->direction.normalized();
        const Eigen::Matrix3d projector = Eigen::Matrix3d::Identity() - direction * direction.transpose();
        const Eigen::Vector3d offset = points.row(index).transpose() - request.view->point;
        const Eigen::Vector3d perpendicular = projector * offset;
        cost += params_.viewWeight * perpendicular.squaredNorm();
        gradient.row(index) += (2.0 * params_.viewWeight * perpendicular).transpose();
        const double parallel = offset.dot(direction);
        const double safeDistance = request.view->direction.norm();
        if (std::abs(parallel) < safeDistance) {
            const double error = std::abs(parallel) - safeDistance;
            cost += params_.viewParallelWeight * error * error;
            if (std::abs(parallel) > 1e-6) {
                gradient.row(index) += (params_.viewParallelWeight * 2.0 * error * (parallel > 0.0 ? direction : -direction)).transpose();
            }
        }
        return cost;
    }
    double addTimeCost_(int numPoints, double interval, double timeLowerBound, double& timeGradient) const {
        double cost = 0.0;
        const double duration = static_cast<double>(numPoints - 3) * interval;
        cost += params_.timeWeight * duration;
        timeGradient += params_.timeWeight * static_cast<double>(numPoints - 3);
        if (timeLowerBound > 0.0 && duration < timeLowerBound) {
            const double error = duration - timeLowerBound;
            cost += params_.timeLowerBoundWeight * error * error;
            timeGradient += params_.timeLowerBoundWeight * 2.0 * error * static_cast<double>(numPoints - 3);
        }
        return cost;
    }
    double addSwarmCost_(const BSplineOptRequest& request, const Eigen::MatrixXd& points, double interval, Eigen::MatrixXd& gradient) const {
        if (request.swarm.empty()) {
            return 0.0;
        }
        double cost = 0.0;
        constexpr double a = 2.5;
        constexpr double b = 1.0;
        constexpr double invA2 = 1.0 / (a * a);
        constexpr double invB2 = 1.0 / (b * b);

        for (const auto& other : request.swarm) {
            for (int i = 3; i + 3 < points.rows(); ++ i) {
                const double time = request.planStartTime + static_cast<double>(i - 3 + 2) * interval;
                const double relativeTime = time - other.startTime;
                const Eigen::VectorXd otherPosition = other.trajectory.evalt(relativeTime);
                if (otherPosition.size() < 3) {
                    continue;
                }
                const Eigen::Vector3d delta = points.row(i).transpose() - otherPosition.head<3>();
                const double ellipticalDistance = std::sqrt(delta(2) * delta(2) * invA2 + (delta(0) * delta(0) + delta(1) * delta(1)) * invB2);
                if (ellipticalDistance >= params_.swarmSafeDistance || ellipticalDistance < 1e-6) {
                    continue;
                }
                const double error = ellipticalDistance - params_.swarmSafeDistance;
                cost += params_.swarmWeight * error * error;
                const double scale = 2.0 * params_.swarmWeight * (1.0 - params_.swarmSafeDistance / ellipticalDistance);
                gradient(i, 0) += scale * delta(0) * invB2;
                gradient(i, 1) += scale * delta(1) * invB2;
                gradient(i, 2) += scale * delta(2) * invA2;
            }
        }
        return cost;
    }

private:
    const BSplineOptParams& params_;
    const SdfMap& map_;
};

} // arc
