#pragma once

#include <base/iarc.h>
#include <planning/bspline.h>
#include <planning/bsplineoptdef.h>

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
};

} // arc
