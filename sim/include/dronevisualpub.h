#pragma once

#include <base/ros.h>

#include <base/iarc.h>
#include <vis/planningvisualization.h>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <visualization_msgs/msg/marker.hpp>

namespace arc {

class DroneVisualPub {
public:
    DroneVisualPub(rclcpp::Node::SharedPtr node)
        : node_(node)
    {
        droneId_ = static_cast<int>(node->declare_parameter<int>("drone_id", 1));
        const int droneNum = static_cast<int>(node->declare_parameter<int>("drone_num", 1));
        droneNum_ = std::max(1, droneNum);
        robotScale_ = node->declare_parameter<double>("robot_scale", 1.0);
        meshResource_ = node->declare_parameter<std::string>(
            "mesh_resource", "package://arc_sim/resource/meshes/hummingbird.mesh"
        );
        odomSub_ = node->create_subscription<nav_msgs::msg::Odometry>(
            "odometry",
            100,
            [this](const nav_msgs::msg::Odometry::SharedPtr msg) { onOdometryMsg(msg); }
        );
        robotPub_ = node->create_publisher<visualization_msgs::msg::Marker>(
            "/" + std::string(node->get_name()) + "/robot",
            rclcpp::QoS(rclcpp::KeepLast(100)).reliable().transient_local()
        );
    }

public:
    void onOdometryMsg(const nav_msgs::msg::Odometry::SharedPtr msg) {
        visualization_msgs::msg::Marker robot;
        robot.header.frame_id = msg->header.frame_id;
        robot.header.stamp = msg->header.stamp;
        robot.ns = "robot";
        robot.id = 0;
        robot.action = visualization_msgs::msg::Marker::ADD;
        robot.pose = msg->pose.pose;
        if (!meshResource_.empty()) {
            robot.type = visualization_msgs::msg::Marker::MESH_RESOURCE;
            robot.mesh_resource = meshResource_;
            robot.mesh_use_embedded_materials = false;
            robot.scale.x = robotScale_;
            robot.scale.y = robotScale_;
            robot.scale.z = robotScale_;
        } else {
            robot.type = visualization_msgs::msg::Marker::CUBE;
            robot.scale.x = robotScale_;
            robot.scale.y = robotScale_;
            robot.scale.z = robotScale_ * 0.4;
        }
        const double hue = static_cast<double>(std::max(droneId_ - 1, 0)) /
            static_cast<double>(std::max(droneNum_ - 1, 1));
        const Eigen::Vector4d color = PlanningVisualization::getColor(hue, 1.0);
        robot.color.r = static_cast<float>(color(0));
        robot.color.g = static_cast<float>(color(1));
        robot.color.b = static_cast<float>(color(2));
        robot.color.a = static_cast<float>(color(3));
        robotPub_->publish(robot);
    }

private:
    rclcpp::Node::SharedPtr node_;
    int droneId_{1};
    int droneNum_{1};
    double robotScale_{1.0};
    std::string meshResource_;

private:
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odomSub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr robotPub_;
};

} // arc
