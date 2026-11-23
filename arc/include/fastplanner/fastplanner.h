#pragma once

#include <base/ros.h>

#include <base/iarc.h>
#include <env/sdfmap.h>
#include <fastplanner/fastplanneroptions.h>
#include <planning/bsplineopt.h>
#include <planning/controlpointfitter.h>
#include <planning/trajectory.h>
#include <planning/polytraj.h>
#include <planning/yawopt.h>
#include <search/kinoastar.h>

namespace arc {

class FastPlanner {
public:
    FastPlanner(rclcpp::Node::SharedPtr node, SdfMap& map, const FastPlannerOptions& plannerOptions)
        : options_(plannerOptions)
        , node_(node)
        , map_(map)
        , kinoAstar_(map, plannerOptions.kino)
        , bsplineOptimizer_(node, map)
    {
        swarm_.init(1, 1);
        bsplineOptions_.maxVel = options_.kino.maxVel;
        bsplineOptions_.maxAcc = options_.kino.maxAcc;
        bsplineOptions_.maxExpandRatio = 1.1;

        yawBsplineOptions_.maxVel = options_.maxYawRate;
        yawBsplineOptions_.maxAcc = 1.0;
        yawBsplineOptions_.maxExpandRatio = 1.1;
    }

public:
    PlannerResult kinodynamicReplan(
        const Eigen::Vector3d& startPoint,
        const Eigen::Vector3d& startVelocity,
        const Eigen::Vector3d& startAcceleration,
        const Eigen::Vector3d& endPoint,
        const Eigen::Vector3d& endVelocity,
        double timeLowerBound=-1.0
    ) {
        PlannerResult result;
        if ((startPoint - endPoint).norm() < 1e-2) {
            result.code = PlannerResult::INPUTINVALID;
            return result;
        }
        stats_.searchTime = 0.0;
        stats_.optimizeTime = 0.0;
        const auto searchStart = std::chrono::steady_clock::now();

        kinoAstar_.reset();
        uint8_t status = kinoAstar_.search(
            startPoint,
            startVelocity,
            startAcceleration,
            endPoint,
            endVelocity,
            true,
            options_.optimistic
        );
        if (status == KinoAstarSearchStatus::NOPATH) {
            kinoAstar_.reset();
            status = kinoAstar_.search(
                startPoint,
                startVelocity,
                startAcceleration,
                endPoint,
                endVelocity,
                false,
                options_.optimistic
            );
        }
        result.searchStatus = status;
        result.searchTime = std::chrono::duration<double>(std::chrono::steady_clock::now() - searchStart).count();
        if (status != KinoAstarSearchStatus::DONE && status != KinoAstarSearchStatus::HORIZON && status != KinoAstarSearchStatus::NEAR_END) {
            result.code = PlannerResult::SEARCHFAILED;
            RCLCPP_DEBUG(
                node_->get_logger(),
                "kinodynamicReplan search failed status=%d goal=(%.2f,%.2f,%.2f)",
                static_cast<int>(result.searchStatus),
                endPoint(0), endPoint(1), endPoint(2)
            );
            return result;
        }
        planningData_.kinoPath = kinoAstar_.getKinoTraj(0.01);
        stats_.searchTime = result.searchTime;

        double interval = options_.controlPointDistance / options_.kino.maxVel;
        std::vector<Eigen::Vector3d> pointSet;
        std::vector<Eigen::Vector3d> boundaryDerivatives;
        kinoAstar_.getSamples(interval, pointSet, boundaryDerivatives);
        if (pointSet.size() < 2 || boundaryDerivatives.size() != 4) {
            result.code = PlannerResult::SAMPLEFAILED;
            return result;
        }
        Eigen::MatrixXd controlPoints = ControlPointFitter::fitControlPoininterval(pointSet, boundaryDerivatives, 3, interval);
        if (controlPoints.rows() < 4) {
            result.code = PlannerResult::PARAMETERIZEFAILED;
            return result;
        }
        BSpline initialTrajectory(bsplineOptions_);
        initialTrajectory.setUniformBSpline(controlPoints, 3, interval);
        planningData_.initialLocalSegment = initialTrajectory;
        planningData_.localStartEndDerivative = boundaryDerivatives;

        std::vector<Eigen::Vector3d> startState;
        std::vector<Eigen::Vector3d> endState;
        initialTrajectory.getBoundaryStates(2, startState, endState);
        endState.resize(1);
        uint8_t optimizeStatus = 255;
        const bool optimized = optimizeTrajectory_(controlPoints, interval, startState, endState, timeLowerBound, optimizeStatus);
        result.optimizeStatus = optimizeStatus;
        result.optimizeTime = stats_.optimizeTime;
        if (!optimized) {
            result.code = PlannerResult::OPTIMIZEFAILED;
            return result;
        }

        localTrajectory_.position = BSpline(bsplineOptions_);
        localTrajectory_.position.setUniformBSpline(controlPoints, 3, interval);
        if (!localTrajectory_.position.valid()) {
            result.code = PlannerResult::TRAJECTORYINVALID;
            return result;
        }
        planningData_.controlPoints = controlPoints;
        updateTrajectoryInfo_();
        result.code = PlannerResult::SUCCESS;
        return result;
    }
    PlannerResult planExploreTrajectory(
        const std::vector<Eigen::Vector3d>& tour,
        const Eigen::Vector3d& currentVelocity,
        const Eigen::Vector3d& currentAcceleration,
        double timeLowerBound=-1.0
    ) {
        PlannerResult result;
        if (tour.size() < 2) {
            result.code = PlannerResult::INPUTINVALID;
            return result;
        }
        stats_.searchTime = 0.0;
        stats_.optimizeTime = 0.0;

        const int pointCount = static_cast<int>(tour.size());
        Eigen::MatrixXd positions(pointCount, 3);
        for (int i = 0; i < pointCount; ++ i) {
            positions.row(i) = tour[i].transpose();
        }
        Eigen::VectorXd times(pointCount - 1);
        for (int i = 0; i < pointCount - 1; ++ i) {
            times(i) = std::max(1e-3, (positions.row(i + 1) - positions.row(i)).norm() / (options_.kino.maxVel * 0.5));
        }
        PolynomialTrajectory polynomial;
        if (!PolynomialTrajectory::waypointsTraj(positions, currentVelocity, Eigen::Vector3d::Zero(), currentAcceleration, Eigen::Vector3d::Zero(), times, polynomial)) {
            result.code = PlannerResult::PARAMETERIZEFAILED;
            return result;
        }
        const double duration = polynomial.getTotalTime();
        int segmentCount = std::max(8, static_cast<int>(polynomial.getLength() / options_.controlPointDistance));
        double interval = duration / static_cast<double>(segmentCount);
        if (interval <= 1e-6) {
            result.code = PlannerResult::PARAMETERIZEFAILED;
            return result;
        }
        std::vector<Eigen::Vector3d> pointSet(segmentCount + 1);
        for (int i = 0; i <= segmentCount; ++ i) {
            pointSet[i] = polynomial.evaluate(interval * static_cast<double>(i), 0);
        }
        std::vector<Eigen::Vector3d> boundaryDerivatives = {
            polynomial.evaluate(0.0, 1),
            polynomial.evaluate(duration, 1),
            polynomial.evaluate(0.0, 2),
            polynomial.evaluate(duration, 2)
        };
        Eigen::MatrixXd controlPoints = ControlPointFitter::fitControlPoininterval(pointSet, boundaryDerivatives, 3, interval);
        if (controlPoints.rows() < 4) {
            result.code = PlannerResult::PARAMETERIZEFAILED;
            return result;
        }
        BSpline initialTrajectory(bsplineOptions_);
        initialTrajectory.setUniformBSpline(controlPoints, 3, interval);
        planningData_.initialLocalSegment = initialTrajectory;
        planningData_.localStartEndDerivative = boundaryDerivatives;

        std::vector<Eigen::Vector3d> startState;
        std::vector<Eigen::Vector3d> endState;
        initialTrajectory.getBoundaryStates(2, startState, endState);
        startState[0] = tour.front();
        startState[1] = currentVelocity;
        startState[2] = currentAcceleration;
        endState.resize(1);
        uint8_t optimizeStatus = 255;
        const bool optimized = optimizeTrajectory_(controlPoints, interval, startState, endState, timeLowerBound, optimizeStatus);
        result.optimizeStatus = optimizeStatus;
        result.optimizeTime = stats_.optimizeTime;
        if (!optimized) {
            result.code = PlannerResult::OPTIMIZEFAILED;
            return result;
        }
        localTrajectory_.position = BSpline(bsplineOptions_);
        localTrajectory_.position.setUniformBSpline(controlPoints, 3, interval);
        if (!localTrajectory_.position.valid()) {
            result.code = PlannerResult::TRAJECTORYINVALID;
            return result;
        }
        planningData_.controlPoints = controlPoints;
        updateTrajectoryInfo_();
        result.code = PlannerResult::SUCCESS;
        return result;
    }
    PlannerResult planYawExplore(
        const Eigen::Vector3d& startYaw,
        double endYaw,
        bool lookForward,
        double relaxTime = -1.0
    ) {
        PlannerResult result;
        if (!localTrajectory_.position.valid() || localTrajectory_.duration <= 1e-6) {
            result.code = PlannerResult::INPUTINVALID;
            return result;
        }
        const int segmentCount = 12;
        const double interval = localTrajectory_.duration / static_cast<double>(segmentCount);

        Eigen::Vector3d start = startYaw;
        start(0) = std::remainder(start(0), 2.0 * __pi__);
        double lastYaw = start(0);
        int lastIndex = 0;

        Eigen::MatrixXd yawPoints(segmentCount + 3, 1);
        yawPoints.setZero();
        Eigen::Matrix3d statesToPoints;
        statesToPoints << 1.0, -interval, interval * interval / 3.0,
            1.0, 0.0, -interval * interval / 6.0,
            1.0, interval, interval * interval / 3.0;
        yawPoints.block<3, 1>(0, 0) = statesToPoints * start;

        std::vector<double> waypoints;
        std::vector<int> waypointIndices;
        if (lookForward) {
            constexpr double forwardTime = 2.0;
            const double tailRelaxTime = relaxTime > 0.0 ? relaxTime : options_.relaxTime2;
            const int relaxCount = std::max(0, static_cast<int>(tailRelaxTime / interval));
            const int waypointEnd = std::max(1, segmentCount - relaxCount);
            for (int i = 1; i < waypointEnd; ++ i) {
                const double tc = static_cast<double>(i) * interval;
                const Eigen::Vector3d currentPoint = localTrajectory_.position.evalt(tc);
                const Eigen::Vector3d futurePoint = localTrajectory_.position.evalt(std::min(localTrajectory_.duration, tc + forwardTime));
                const Eigen::Vector3d direction = futurePoint - currentPoint;
                if (direction.head<2>().norm() < 1e-6) {
                    continue;
                }
                double waypointYaw = std::atan2(direction(1), direction(0));
                calcNextYaw(lastYaw, waypointYaw);

                const double dt1 = static_cast<double>(i - lastIndex) * interval;
                const double dt2 = static_cast<double>(segmentCount + 1 - i) * interval;
                const double diff1 = std::abs(waypointYaw - lastYaw);
                const double rate1 = std::min(diff1, 2.0 * __pi__ - diff1) / dt1;

                double expandedEndYaw = std::remainder(endYaw, 2.0 * __pi__);
                calcNextYaw(waypointYaw, expandedEndYaw);
                const double diff2 = std::abs(waypointYaw - expandedEndYaw);
                const double rate2 = std::min(diff2, 2.0 * __pi__ - diff2) / dt2;

                if (rate1 < options_.maxYawRate && rate2 < options_.maxYawRate) {
                    lastYaw = waypointYaw;
                    lastIndex = i;
                    waypoints.push_back(waypointYaw);
                    waypointIndices.push_back(i);
                }
            }
        }
        Eigen::Vector3d end(std::remainder(endYaw, 2.0 * __pi__), 0.0, 0.0);
        calcNextYaw(lastYaw, end(0));
        yawPoints.block<3, 1>(segmentCount, 0) = statesToPoints * end;

        YawOptRequest request;
        request.points = yawPoints;
        request.interval = interval;
        request.startState = {start(0), start(1), start(2)};
        request.endState = {end(0), 0.0};
        request.waypoints = waypoints;
        request.waypointIndices = waypointIndices;

        YawOptimizer optimizer(options_.yaw);
        const auto optimizeStart = std::chrono::steady_clock::now();
        const YawOptResult yawResult = optimizer.optimize(request);
        result.optimizeTime = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - optimizeStart).count();
        result.optimizeStatus = yawResult.code;
        stats_.optimizeTime = result.optimizeTime;
        if (yawResult.code != YawOptResult::SUCCEEDED) {
            result.code = PlannerResult::OPTIMIZEFAILED;
            return result;
        }
        yawPoints = yawResult.points;

        if (yawPoints.rows() < 4 || !yawPoints.allFinite()) {
            result.code = PlannerResult::TRAJECTORYINVALID;
            return result;
        }
        localTrajectory_.yaw = BSpline(yawBsplineOptions_);
        localTrajectory_.yaw.setUniformBSpline(yawPoints, 3, interval);
        localTrajectory_.yawRate = localTrajectory_.yaw.getDerivativeBSpline();
        localTrajectory_.yawAcceleration = localTrajectory_.yawRate.getDerivativeBSpline();
        planningData_.dtYaw = interval;
        planningData_.pathYaw = waypoints;
        result.code = PlannerResult::SUCCESS;
        return result;
    }
    static void calcNextYaw(double lastYaw, double& yaw) {
        const double roundLast = std::remainder(lastYaw, 2.0 * __pi__);
        const double diff = yaw - roundLast;
        if (std::abs(diff) <= __pi__) {
            yaw = lastYaw + diff;
        } else if (diff > __pi__) {
            yaw = lastYaw + diff - 2.0 * __pi__;
        } else {
            yaw = lastYaw + diff + 2.0 * __pi__;
        }
    }
    bool checkTrajectoryCollision(double& distance) const {
        distance = 0.0;
        if (!localTrajectory_.valid()) {
            return true;
        }
        const double currentTime = std::max(0.0, node_->now().seconds() - localTrajectory_.startTime);
        const Eigen::Vector3d currentPoint = Eigen::Vector3d(localTrajectory_.position.evalt(currentTime).head<3>());
        double radius = 0.0;
        double futureTime = 0.02;
        while (radius < options_.collisionCheckRadius && currentTime + futureTime < localTrajectory_.duration) {
            const Eigen::Vector3d futurePoint = Eigen::Vector3d(localTrajectory_.position.evalt(currentTime + futureTime).head<3>());
            if (map_.data().getInflateOccupancy(futurePoint) == VoxelInflateOccupancyCode::OCCUPIED) {
                distance = radius;
                return false;
            }
            radius = (futurePoint - currentPoint).norm();
            futureTime += options_.collisionCheckStep;
        }
        return true;
    }
    void initSwarm(int droneId, int droneNum) {
        swarm_.init(droneId, droneNum);
    }
    void updateSwarmTrajectory(int droneId, const LocalTrajectory& trajectory) {
        if (droneId <= 0 || droneId > swarm_.droneNum) {
            return;
        }
        swarm_.trajectories[static_cast<size_t>(droneId - 1)] = trajectory;
        swarm_.receiveFlags[static_cast<size_t>(droneId - 1)] = trajectory.valid();
    }
    void clearSwarmTrajectory(int droneId) {
        if (droneId <= 0 || droneId > swarm_.droneNum) {
            return;
        }
        swarm_.trajectories[static_cast<size_t>(droneId - 1)] = LocalTrajectory();
        swarm_.receiveFlags[static_cast<size_t>(droneId - 1)] = false;
    }
    const SwarmTrajectories& swarmTrajectories() const {
        return swarm_;
    }
    bool checkSwarmCollision(int droneId, double& distance) const {
        distance = std::numeric_limits<double>::max();
        if (droneId <= 0 || droneId > swarm_.droneNum) {
            return true;
        }
        const int index = droneId - 1;
        if (!swarm_.receiveFlags[index] || !swarm_.trajectories[index].valid() || !localTrajectory_.valid()) {
            return true;
        }
        const double currentTime = node_->now().seconds();
        double selfTime = 0.02 + currentTime - localTrajectory_.startTime;
        double otherTime = 0.02 + currentTime - swarm_.trajectories[index].startTime;
        double minDistance = std::numeric_limits<double>::max();
        while (selfTime < localTrajectory_.duration && selfTime > 0.0 && otherTime < swarm_.trajectories[index].duration) {
            const Eigen::Vector3d selfPosition = Eigen::Vector3d(localTrajectory_.position.evalt(selfTime).head<3>());
            const Eigen::Vector3d otherPosition = Eigen::Vector3d(swarm_.trajectories[index].position.evalt(otherTime).head<3>());
            const double currentDistance = (selfPosition - otherPosition).head<2>().norm();
            minDistance = std::min(minDistance, currentDistance);
            if (currentDistance < options_.swarmSafeDistance) {
                distance = currentDistance;
                return false;
            }
            selfTime += 0.02;
            otherTime += 0.02;
        }
        distance = minDistance;
        return true;
    }
    bool checkSwarmCollision(int droneId) const {
        double distance = 0.0;
        return checkSwarmCollision(droneId, distance);
    }
    bool checkAnySwarmCollision(double& distance, int& otherDroneId) const {
        distance = std::numeric_limits<double>::max();
        otherDroneId = -1;
        for (int id = 1; id <= swarm_.droneNum; ++ id) {
            if (id == swarm_.droneId) {
                continue;
            }
            double currentDistance = 0.0;
            if (!checkSwarmCollision(id, currentDistance)) {
                if (otherDroneId < 0 || currentDistance < distance) {
                    distance = currentDistance;
                    otherDroneId = id;
                }
            }
        }
        return otherDroneId < 0;
    }
    bool checkAnySwarmCollision() const {
        double distance = 0.0;
        int otherDroneId = -1;
        return checkAnySwarmCollision(distance, otherDroneId);
    }

public:
    const FastPlannerOptions& options() const { return options_; }
    const PlannerStats& stats() const { return stats_; }
    const LocalTrajectory& localTrajectory() const { return localTrajectory_; }
    const PlanningData& planningData() const { return planningData_; }
    void setLocalTrajectoryStartTime(double startTime) { localTrajectory_.startTime = startTime; }

private:
    bool optimizeTrajectory_(
        Eigen::MatrixXd& controlPoints,
        double& interval,
        const std::vector<Eigen::Vector3d>& startState,
        const std::vector<Eigen::Vector3d>& endState,
        double timeLowerBound,
        uint8_t& optimizeStatus
    ) {
        const auto optimizeStart = std::chrono::steady_clock::now();

        BSplineOptRequest request;
        request.points = controlPoints;
        request.interval = interval;
        request.cost.SMOOTHNESS = 1;
        request.cost.DISTANCE = 1;
        request.cost.FEASIBILITY = 1;
        request.cost.START = 1;
        request.cost.END = 1;
        request.cost.SWARM = 1;
        request.cost.MINTIME = options_.minTime;
        request.startState = startState;
        request.endState = endState;
        request.timeLowerBound = std::max(0.0, timeLowerBound);
        request.planStartTime = node_->now().seconds();
        swarm_.getValid(request.swarm);

        auto acceptable = [](const BSplineOptResult& result) {
            return result.code == BSplineOptResult::SUCCEEDED ||
                result.code == BSplineOptResult::MAXEVALREACHED ||
                result.code == BSplineOptResult::MAXTIMEREACHED;
        };
        BSplineOptResult result = bsplineOptimizer_.optimize(request);
        if (!acceptable(result) && request.cost.MINTIME) {
            request.cost.MINTIME = 0;
            result = bsplineOptimizer_.optimize(request);
        }
        optimizeStatus = result.code;
        if (!acceptable(result)) {
            stats_.optimizeTime = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - optimizeStart).count();
            return false;
        }
        controlPoints = result.points;
        interval = result.interval;
        stats_.optimizeTime = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - optimizeStart).count();
        return true;
    }
    void updateTrajectoryInfo_() {
        localTrajectory_.velocity = localTrajectory_.position.getDerivativeBSpline();
        localTrajectory_.acceleration = localTrajectory_.velocity.getDerivativeBSpline();
        localTrajectory_.startPosition = Eigen::Vector3d(localTrajectory_.position.evalt(0.0).head<3>());
        localTrajectory_.duration = localTrajectory_.position.duration();
        localTrajectory_.id += 1;
    }

private:
    FastPlannerOptions options_;
    BSplineOptions bsplineOptions_;
    BSplineOptions yawBsplineOptions_;
    PlannerStats stats_;
    LocalTrajectory localTrajectory_;
    GlobalTrajectory globalTrajectory_;
    PlanningData planningData_;
    SwarmTrajectories swarm_;
    rclcpp::Node::SharedPtr node_;
    SdfMap& map_;
    KinoAstar kinoAstar_;
    BSplineOptimizer bsplineOptimizer_;
};

} // arc
