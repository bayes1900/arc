#pragma once

#include <base/iarc.h>

namespace arc {

struct TrajectoryStatsSnapshot {
    uint32_t commandCount{0};
    double commandRateHz{0.0};
    double commandMaxGap{0.0};
    uint32_t commandGapsOver50ms{0};
    uint32_t commandGapsOver200ms{0};
    uint32_t trajectoryReceived{0};
    uint32_t trajectoryDroppedOld{0};
    uint32_t trajectoryCompleted{0};
    double trackingErrorPosition{0.0};
    double trackingErrorYaw{0.0};
};

class TrajectoryStats {
public:
    void onCommand(double now, const Eigen::Vector3d& position, double yaw) {
        if (lastCommandTime_ >= 0.0) {
            const double gap = std::max(0.0, now - lastCommandTime_);
            commandMaxGap_ = std::max(commandMaxGap_, gap);
            if (gap > 0.05) {
                gapOver50ms_ += 1;
            }
            if (gap > 0.2) {
                gapOver200ms_ += 1;
            }
        } else {
            firstCommandTime_ = now;
        }
        lastCommandTime_ = now;
        commandCount_ += 1;
        commandPosition_ = position;
        commandYaw_ = yaw;
        hasCommand_ = true;
    }
    void onOdometry(const Eigen::Vector3d& position, double yaw) {
        odomPosition_ = position;
        odomYaw_ = yaw;
    }
    void onTrajectoryReceived() {
        receivedTrajectoryCount_ += 1;
    }
    void onTrajectoryDroppedOld() {
        droppedOldCount_ += 1;
    }
    void onTrajectoryCompleted() {
        completedCount_ += 1;
    }
    TrajectoryStatsSnapshot snapshot() const {
        TrajectoryStatsSnapshot result;
        result.commandCount = commandCount_;
        if (commandCount_ > 1 && lastCommandTime_ > firstCommandTime_) {
            result.commandRateHz =
                static_cast<double>(commandCount_ - 1) / (lastCommandTime_ - firstCommandTime_);
        }
        result.commandMaxGap = commandMaxGap_;
        result.commandGapsOver50ms = gapOver50ms_;
        result.commandGapsOver200ms = gapOver200ms_;
        result.trajectoryReceived = receivedTrajectoryCount_;
        result.trajectoryDroppedOld = droppedOldCount_;
        result.trajectoryCompleted = completedCount_;
        if (hasCommand_) {
            result.trackingErrorPosition = (commandPosition_ - odomPosition_).norm();
            result.trackingErrorYaw = std::remainder(commandYaw_ - odomYaw_, 2.0 * __pi__);
        }
        return result;
    }

private:
    bool hasCommand_{false};
    double firstCommandTime_{-1.0};
    double lastCommandTime_{-1.0};
    double commandMaxGap_{0.0};
    uint32_t commandCount_{0};
    uint32_t gapOver50ms_{0};
    uint32_t gapOver200ms_{0};
    uint32_t receivedTrajectoryCount_{0};
    uint32_t droppedOldCount_{0};
    uint32_t completedCount_{0};
    Eigen::Vector3d odomPosition_{Eigen::Vector3d::Zero()};
    Eigen::Vector3d commandPosition_{Eigen::Vector3d::Zero()};
    double odomYaw_{0.0};
    double commandYaw_{0.0};
};

} // arc
