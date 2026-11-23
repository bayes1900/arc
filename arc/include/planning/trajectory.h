#pragma once

#include <base/iarc.h>
#include <planning/bspline.h>
#include <planning/bsplineoptdef.h>
#include <planning/polytraj.h>

namespace arc {

struct LocalTrajectory {
    int id{0};
    double duration{0.0};
    double startTime{0.0};
    Eigen::Vector3d startPosition{Eigen::Vector3d::Zero()};
    BSpline position;
    BSpline velocity;
    BSpline acceleration;
    BSpline yaw;
    BSpline yawRate;
    BSpline yawAcceleration;

    bool valid() const {
        return position.valid();
    }
};

struct LocalTrajectoryState {
    Eigen::Vector3d position{Eigen::Vector3d::Zero()};
    Eigen::Vector3d velocity{Eigen::Vector3d::Zero()};
    Eigen::Vector3d acceleration{Eigen::Vector3d::Zero()};
    double yaw{0.0};
    double yawRate{0.0};
    int id{0};
    bool finished{false};
};

class LocalTrajectoryServer {
public:
    const LocalTrajectory& current() const {
        return current_;
    }
    const LocalTrajectory& pending() const {
        return pending_;
    }
    void clear() {
        current_ = LocalTrajectory();
        pending_ = LocalTrajectory();
    }
    void addTrajectory(const LocalTrajectory& trajectory) {
        if (!trajectory.valid()) {
            return;
        }
        if (current_.id == 0) {
            current_ = trajectory;
        } else {
            pending_ = trajectory;
        }
    }
    bool evaluate(double time, LocalTrajectoryState& state) {
        if (current_.id == 0 || !current_.valid()) {
            return false;
        }
        if (pending_.id != 0 && time >= pending_.startTime) {
            current_ = pending_;
            pending_ = LocalTrajectory();
        }
        const double t = time - current_.startTime;
        if (t < 0.0) {
            return false;
        }
        state.id = current_.id;
        if (t < current_.duration) {
            state.position = Eigen::Vector3d(current_.position.evalt(t).head<3>());
            state.velocity = Eigen::Vector3d(current_.velocity.evalt(t).head<3>());
            state.acceleration = Eigen::Vector3d(current_.acceleration.evalt(t).head<3>());
            state.yaw = current_.yaw.valid() ? current_.yaw.evalt(t)(0) : 0.0;
            state.yawRate = current_.yawRate.valid() ? current_.yawRate.evalt(t)(0) : 0.0;
            state.finished = false;
            return true;
        }
        const double endTime = std::min(current_.duration, current_.position.duration());
        state.position = Eigen::Vector3d(current_.position.evalt(endTime).head<3>());
        state.velocity.setZero();
        state.acceleration.setZero();
        state.yaw = current_.yaw.valid() ? current_.yaw.evalt(endTime)(0) : 0.0;
        state.yawRate = 0.0;
        state.finished = true;
        return true;
    }
    void resetDuration(double time, double forward=0.0) {
        if (current_.id == 0) {
            return;
        }
        const double stopTime = time - current_.startTime + forward;
        current_.duration = std::min(current_.duration, std::max(0.0, stopTime));
    }

private:
    LocalTrajectory current_;
    LocalTrajectory pending_;
};

class GlobalTrajectory {
public:
    void setGlobalTrajectory(const PolynomialTrajectory& trajectory, double startTime) {
        globalTrajectory_ = trajectory;
        globalDuration_ = globalTrajectory_.getTotalTime();
        globalStartTime_ = startTime;
        localTrajectory_.clear();
        localStartTime_ = -1.0;
        localEndTime_ = -1.0;
        timeChange_ = 0.0;
        lastTimeInc_ = 0.0;
    }
    void setLocalTrajectory(const BSpline& trajectory, double localStartTime, double localEndTime, double timeChange) {
        localTrajectory_.resize(3);
        localTrajectory_[0] = trajectory;
        localTrajectory_[1] = localTrajectory_[0].getDerivativeBSpline();
        localTrajectory_[2] = localTrajectory_[1].getDerivativeBSpline();
        localStartTime_ = localStartTime;
        localEndTime_ = localEndTime;
        globalDuration_ += timeChange;
        timeChange_ += timeChange;
        lastTimeInc_ = timeChange;
    }
    Eigen::Vector3d state(double t, int k) const {
        if (t >= -1e-3 && t <= localStartTime_) {
            return globalTrajectory_.evaluate(t - timeChange_ + lastTimeInc_, k);
        }
        if (t >= localEndTime_ && t <= globalDuration_ + 1e-3) {
            return globalTrajectory_.evaluate(t - timeChange_, k);
        }
        return Eigen::Vector3d(localTrajectory_[k].evalt(t - localStartTime_).head<3>());
    }
    bool localTrajectoryReachedTarget() const {
        return std::fabs(localEndTime_ - globalDuration_) < 1e-3;
    }
    void getTrajectoryInfoInDuration(
        double startTime,
        double duration,
        double dt,
        std::vector<Eigen::Vector3d>& pointSet,
        std::vector<Eigen::Vector3d>& startEndDerivative
    ) const {
        for (double t = 0.0; t <= duration + 1e-4; t += dt) {
            pointSet.push_back(state(startTime + t, 0));
        }
        startEndDerivative.push_back(state(startTime, 1));
        startEndDerivative.push_back(state(startTime + duration, 1));
        startEndDerivative.push_back(state(startTime, 2));
        startEndDerivative.push_back(state(startTime + duration, 2));
    }
    void getTrajectoryInfoInSphere(
        double startTime,
        double radius,
        double pointDistance,
        std::vector<Eigen::Vector3d>& pointSet,
        std::vector<Eigen::Vector3d>& startEndDerivative,
        double& dt,
        double& duration
    ) const {
        double segmentLength = 0.0;
        double segmentTime = 0.0;
        const Eigen::Vector3d firstPoint = state(startTime, 0);
        Eigen::Vector3d previousPoint = firstPoint;
        Eigen::Vector3d currentPoint = firstPoint;
        constexpr double deltaTime = 0.2;
        while ((currentPoint - firstPoint).norm() < radius && startTime + segmentTime < globalDuration_ - 1e-3) {
            segmentTime = std::min(segmentTime + deltaTime, globalDuration_ - startTime);
            currentPoint = state(startTime + segmentTime, 0);
            segmentLength += (currentPoint - previousPoint).norm();
            previousPoint = currentPoint;
        }
        int segmentCount = static_cast<int>(std::floor(segmentLength / pointDistance));
        segmentCount = std::max(6, segmentCount);
        duration = segmentTime;
        dt = duration / static_cast<double>(segmentCount);
        getTrajectoryInfoInDuration(startTime, duration, dt, pointSet, startEndDerivative);
    }

private:
    PolynomialTrajectory globalTrajectory_;
    std::vector<BSpline> localTrajectory_;
    double globalDuration_{0.0};
    double globalStartTime_{0.0};
    double localStartTime_{-1.0};
    double localEndTime_{-1.0};
    double timeChange_{0.0};
    double lastTimeInc_{0.0};
};

struct PlanningData {
    std::vector<Eigen::Vector3d> globalWaypoints;
    BSpline initialLocalSegment;
    std::vector<Eigen::Vector3d> localStartEndDerivative;
    std::vector<Eigen::Vector3d> kinoPath;
    std::vector<double> pathYaw;
    double dtYaw{0.0};
    Eigen::MatrixXd controlPoints;
};

struct SwarmTrajectories {
    int droneId{0};
    int droneNum{1};
    std::vector<LocalTrajectory> trajectories;
    std::vector<bool> receiveFlags;

    void init(int id, int num) {
        droneId = id;
        droneNum = num;
        trajectories.assign(num, LocalTrajectory());
        receiveFlags.assign(num, false);
    }
    void getValid(std::vector<BSplineOptSwarm>& validTrajectories) const {
        validTrajectories.clear();
        for (size_t i = 0; i < trajectories.size(); ++ i) {
            if (receiveFlags[i] && trajectories[i].valid()) {
                validTrajectories.push_back({trajectories[i].position, trajectories[i].startTime});
            }
        }
    }
    void resetReceiveFlags() {
        std::fill(receiveFlags.begin(), receiveFlags.end(), false);
    }
};

} // arc
