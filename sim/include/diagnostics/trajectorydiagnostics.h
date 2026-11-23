#pragma once

#include <base/iarc.h>
#include <diagnostics/format.h>
#include <diagnostics/trajectorystats.h>
#include <arcmsgs/msg/exploration_event.hpp>
#include <arcmsgs/msg/trajectory_metrics.hpp>
#include <rclcpp/rclcpp.hpp>

namespace arc {

class TrajectoryDiagnostics {
public:
    TrajectoryDiagnostics(rclcpp::Node::SharedPtr node, int droneId)
        : node_(std::move(node))
        , droneId_(droneId)
    {
        enabled_ = node_->declare_parameter<bool>("diagnostics.enabled", true);
        eventPub_ = node_->create_publisher<arcmsgs::msg::ExplorationEvent>(
            "/traj_server/events", rclcpp::QoS(200)
        );
        metricsPub_ = node_->create_publisher<arcmsgs::msg::TrajectoryMetrics>(
            "/traj_server/metrics", rclcpp::QoS(10)
        );
    }
    void onCommand(double now, const Eigen::Vector3d& position, double yaw) {
        stats_.onCommand(now, position, yaw);
    }
    void onOdometry(const Eigen::Vector3d& position, double yaw) {
        stats_.onOdometry(position, yaw);
    }
    void onTrajectoryReceived() {
        stats_.onTrajectoryReceived();
    }
    void onTrajectoryInvalid(int64_t id) {
        publishEvent_("traj_invalid", "INVALID", id, "fromMessage failed");
    }
    void onTrajectoryDroppedOld(int64_t id, int64_t currentId) {
        stats_.onTrajectoryDroppedOld();
        RCLCPP_WARN(
            node_->get_logger(),
            "trajectory dropped old id=%ld current=%ld",
            static_cast<long>(id),
            static_cast<long>(currentId)
        );
        publishEvent_(
            "traj_dropped_old",
            "OLD",
            id,
            "current=" + std::to_string(currentId)
        );
    }
    void onTrajectoryCompleted(int64_t id, double duration) {
        stats_.onTrajectoryCompleted();
        publishEvent_(
            "traj_completed",
            "COMPLETED",
            id,
            "duration=" + DiagFormat::number(duration)
        );
    }
    void publishMetrics() {
        if (!enabled_) {
            return;
        }
        const TrajectoryStatsSnapshot snapshot = stats_.snapshot();
        arcmsgs::msg::TrajectoryMetrics message;
        message.header.stamp = node_->now();
        message.header.frame_id = "world";
        message.drone_id = droneId_;
        message.command_count = snapshot.commandCount;
        message.command_rate_hz = snapshot.commandRateHz;
        message.command_max_gap_ms = snapshot.commandMaxGap * 1000.0;
        message.command_gaps_over_50ms = snapshot.commandGapsOver50ms;
        message.command_gaps_over_200ms = snapshot.commandGapsOver200ms;
        message.trajectory_received = snapshot.trajectoryReceived;
        message.trajectory_dropped_old = snapshot.trajectoryDroppedOld;
        message.trajectory_completed = snapshot.trajectoryCompleted;
        message.tracking_error_position_m = snapshot.trackingErrorPosition;
        message.tracking_error_yaw_rad = snapshot.trackingErrorYaw;
        metricsPub_->publish(message);
    }

private:
    void publishEvent_(
        const std::string& reason,
        const std::string& code,
        int64_t id,
        const std::string& detail
    ) {
        if (!enabled_) {
            return;
        }
        arcmsgs::msg::ExplorationEvent message;
        message.header.stamp = node_->now();
        message.header.frame_id = "world";
        message.source = "traj_server";
        message.event_type = "trajectory_event";
        message.stage = "traj_publish";
        message.reason = reason;
        message.code = code;
        message.drone_id = droneId_;
        message.value = static_cast<double>(id);
        message.detail = detail;
        eventPub_->publish(message);
    }

private:
    rclcpp::Node::SharedPtr node_;
    int droneId_{0};
    bool enabled_{true};
    TrajectoryStats stats_;
    rclcpp::Publisher<arcmsgs::msg::ExplorationEvent>::SharedPtr eventPub_;
    rclcpp::Publisher<arcmsgs::msg::TrajectoryMetrics>::SharedPtr metricsPub_;
};

} // arc
