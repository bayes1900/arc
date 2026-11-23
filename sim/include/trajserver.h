#pragma once

#include <base/ros.h>

#include <base/iarc.h>
#include <diagnostics/trajectorydiagnostics.h>
#include <explore/frontier/fovmodel.h>
#include <swarmrosbridge.h>
#include <vis/planningvisualization.h>

#include <arcmsgs/msg/position_command.hpp>
#include <arcmsgs/msg/swarm_trajectory.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace arc {

class TrajectoryServer {
public:
    explicit TrajectoryServer(rclcpp::Node::SharedPtr node)
        : node_(std::move(node))
    {
        droneId_ = node_->declare_parameter<int>("drone_id", 1);
        droneNum_ = std::max(
            1,
            static_cast<int>(node_->declare_parameter<int>("drone_num", 1))
        );

        trajectorySub_ = node_->create_subscription<arcmsgs::msg::SwarmTrajectory>(
            "trajectory",
            rclcpp::QoS(100),
            [this](const arcmsgs::msg::SwarmTrajectory::SharedPtr msg) {
                onTrajectory_(msg);
            }
        );
        odomSub_ = node_->create_subscription<nav_msgs::msg::Odometry>(
            "odometry",
            50,
            [this](const nav_msgs::msg::Odometry::SharedPtr msg) {
                onOdometry_(msg);
            }
        );
        commandPub_ = node_->create_publisher<arcmsgs::msg::PositionCommand>("position_cmd", 50);
        positionCmdVisPub_ = node_->create_publisher<visualization_msgs::msg::Marker>(
            "position_cmd_vis", 10
        );
        travelTrajPub_ = node_->create_publisher<visualization_msgs::msg::Marker>(
            "travel_traj", 10
        );
        diagnostics_ = std::make_unique<TrajectoryDiagnostics>(node_, droneId_);

        commandTimer_ = node_->create_wall_timer(
            std::chrono::milliseconds(10),
            [this]() { publishCommand_(); }
        );
        visualizationTimer_ = node_->create_wall_timer(
            std::chrono::milliseconds(200),
            [this]() { publishVisualization_(); }
        );
        metricsTimer_ = node_->create_wall_timer(
            std::chrono::milliseconds(1000),
            [this]() { diagnostics_->publishMetrics(); }
        );
    }

private:
    void onTrajectory_(const arcmsgs::msg::SwarmTrajectory::SharedPtr msg) {
        if (msg->drone_id != droneId_) {
            return;
        }
        LocalTrajectory trajectory;
        if (!SwarmRosConverter::fromMessage(*msg, trajectory)) {
            diagnostics_->onTrajectoryInvalid(msg->traj_id);
            return;
        }
        diagnostics_->onTrajectoryReceived();
        if (trajectory.id <= trajectoryId_) {
            diagnostics_->onTrajectoryDroppedOld(trajectory.id, trajectoryId_);
            return;
        }
        trajectory_ = trajectory;
        trajectoryId_ = trajectory.id;
        hasTrajectory_ = true;
        completedEventSent_ = false;
    }

    void onOdometry_(const nav_msgs::msg::Odometry::SharedPtr msg) {
        const auto& orientation = msg->pose.pose.orientation;
        const Eigen::Quaterniond quaternion(
            orientation.w,
            orientation.x,
            orientation.y,
            orientation.z
        );
        const Eigen::Matrix3d rotation = quaternion.toRotationMatrix();
        odomYaw_ = std::atan2(rotation(1, 0), rotation(0, 0));
        diagnostics_->onOdometry(
            Eigen::Vector3d(
                msg->pose.pose.position.x,
                msg->pose.pose.position.y,
                msg->pose.pose.position.z
            ),
            odomYaw_
        );
    }

    void publishCommand_() {
        if (!hasTrajectory_) {
            return;
        }
        const double now = node_->now().seconds();
        const double tc = std::max(0.0, now - trajectory_.startTime);
        const double te = std::min(tc, trajectory_.duration);

        const Eigen::Vector3d position(trajectory_.position.evalt(te).head<3>());
        Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
        Eigen::Vector3d acceleration = Eigen::Vector3d::Zero();
        double yaw = odomYaw_;
        double yawDot = 0.0;
        if (tc < trajectory_.duration) {
            if (trajectory_.velocity.valid()) {
                velocity = Eigen::Vector3d(trajectory_.velocity.evalt(te).head<3>());
            }
            if (trajectory_.acceleration.valid()) {
                acceleration = Eigen::Vector3d(trajectory_.acceleration.evalt(te).head<3>());
            }
            if (trajectory_.yaw.valid()) {
                yaw = trajectory_.yaw.evalt(te)(0);
            }
            if (trajectory_.yawRate.valid()) {
                yawDot = trajectory_.yawRate.evalt(te)(0);
            }
        } else if (trajectory_.yaw.valid()) {
            yaw = trajectory_.yaw.evalt(trajectory_.duration)(0);
        }

        arcmsgs::msg::PositionCommand command;
        command.header.stamp = node_->now();
        command.header.frame_id = "world";
        command.position.x = position(0);
        command.position.y = position(1);
        command.position.z = position(2);
        command.velocity.x = velocity(0);
        command.velocity.y = velocity(1);
        command.velocity.z = velocity(2);
        command.acceleration.x = acceleration(0);
        command.acceleration.y = acceleration(1);
        command.acceleration.z = acceleration(2);
        command.yaw = yaw;
        command.yaw_dot = yawDot;
        command.trajectory_id = static_cast<uint32_t>(trajectory_.id);
        command.trajectory_flag = tc >= trajectory_.duration
            ? arcmsgs::msg::PositionCommand::TRAJECTORY_STATUS_COMPLETED
            : arcmsgs::msg::PositionCommand::TRAJECTORY_STATUS_READY;
        commandPub_->publish(command);
        diagnostics_->onCommand(now, position, yaw);

        if (tc >= trajectory_.duration && !completedEventSent_) {
            completedEventSent_ = true;
            diagnostics_->onTrajectoryCompleted(trajectory_.id, trajectory_.duration);
        }

        commandPosition_ = position;
        commandYaw_ = yaw;
        hasCommand_ = true;

        if (lastTravelSampleTime_ < 0.0 || now - lastTravelSampleTime_ > 0.1) {
            lastTravelSampleTime_ = now;
            if (travelTrajectory_.empty() ||
                (travelTrajectory_.back() - position).norm() > 0.03) {
                travelTrajectory_.push_back(position);
                travelDirty_ = true;
                if (travelTrajectory_.size() > 20000) {
                    travelTrajectory_.erase(travelTrajectory_.begin());
                }
            }
        }
    }

    void publishVisualization_() {
        publishFovMarker_();
        publishTravelTrajectory_();
    }

    void publishFovMarker_() {
        if (!hasCommand_) {
            return;
        }
        fov_.setPose(commandPosition_, commandYaw_);
        std::vector<Eigen::Vector3d> list1;
        std::vector<Eigen::Vector3d> list2;
        fov_.getFOV(list1, list2);

        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "world";
        marker.header.stamp = node_->now();
        marker.ns = "current_pose";
        marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::LINE_LIST;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.color.r = 0.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;
        marker.color.a = 1.0;
        marker.scale.x = 0.04;
        marker.scale.y = 0.04;
        marker.scale.z = 0.04;
        for (size_t i = 0; i < list1.size(); ++ i) {
            geometry_msgs::msg::Point point;
            point.x = list1[i](0);
            point.y = list1[i](1);
            point.z = list1[i](2);
            marker.points.push_back(point);
            point.x = list2[i](0);
            point.y = list2[i](1);
            point.z = list2[i](2);
            marker.points.push_back(point);
        }
        positionCmdVisPub_->publish(marker);
    }

    void publishTravelTrajectory_() {
        if (travelTrajectory_.empty() || !travelDirty_) {
            return;
        }
        travelDirty_ = false;

        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "world";
        marker.header.stamp = node_->now();
        marker.ns = "travel_traj";
        marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
        marker.action = visualization_msgs::msg::Marker::DELETE;
        travelTrajPub_->publish(marker);

        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        const double hue = static_cast<double>(std::max(droneId_ - 1, 0)) /
            static_cast<double>(std::max(droneNum_, 1));
        const Eigen::Vector4d color = PlanningVisualization::getColor(hue);
        marker.color.r = static_cast<float>(color(0));
        marker.color.g = static_cast<float>(color(1));
        marker.color.b = static_cast<float>(color(2));
        marker.color.a = static_cast<float>(color(3));
        marker.scale.x = 0.05;
        marker.scale.y = 0.05;
        marker.scale.z = 0.05;
        marker.points.reserve(travelTrajectory_.size());
        for (const auto& point : travelTrajectory_) {
            geometry_msgs::msg::Point markerPoint;
            markerPoint.x = point(0);
            markerPoint.y = point(1);
            markerPoint.z = point(2);
            marker.points.push_back(markerPoint);
        }
        travelTrajPub_->publish(marker);
    }

private:
    rclcpp::Node::SharedPtr node_;
    int droneId_{1};
    int droneNum_{1};
    int trajectoryId_{0};
    bool hasTrajectory_{false};
    bool hasCommand_{false};
    bool travelDirty_{false};
    bool completedEventSent_{false};
    double odomYaw_{0.0};
    double commandYaw_{0.0};
    double lastTravelSampleTime_{-1.0};
    Eigen::Vector3d commandPosition_{Eigen::Vector3d::Zero()};
    LocalTrajectory trajectory_;
    FovModel fov_;
    std::vector<Eigen::Vector3d> travelTrajectory_;

    rclcpp::Subscription<arcmsgs::msg::SwarmTrajectory>::SharedPtr trajectorySub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odomSub_;
    rclcpp::Publisher<arcmsgs::msg::PositionCommand>::SharedPtr commandPub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr positionCmdVisPub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr travelTrajPub_;
    std::unique_ptr<TrajectoryDiagnostics> diagnostics_;
    rclcpp::TimerBase::SharedPtr commandTimer_;
    rclcpp::TimerBase::SharedPtr visualizationTimer_;
    rclcpp::TimerBase::SharedPtr metricsTimer_;
};

} // arc
