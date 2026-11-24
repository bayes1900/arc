#pragma once

#include <base/ros.h>

#include <base/iarc.h>
#include <geometry_msgs/msg/point.hpp>
#include <visualization_msgs/msg/marker.hpp>

namespace arc {

class PlanningVisualization {
public:
    enum PublisherId {
        TRAJECTORY = 0,
        TOPO_PATH = 1,
        FRONTIER = 2,
        YAW = 3,
        VIEWPOINT = 4
    };
    explicit PlanningVisualization(rclcpp::Node::SharedPtr node)
        : node_(std::move(node))
    {
        pubs_.push_back(node_->create_publisher<visualization_msgs::msg::Marker>(
            "/planning_vis/trajectory", rclcpp::QoS(100)
        ));
        pubs_.push_back(node_->create_publisher<visualization_msgs::msg::Marker>(
            "/planning_vis/topo_path", rclcpp::QoS(100)
        ));
        pubs_.push_back(node_->create_publisher<visualization_msgs::msg::Marker>(
            "/planning_vis/frontier", rclcpp::QoS(10000)
        ));
        pubs_.push_back(node_->create_publisher<visualization_msgs::msg::Marker>(
            "/planning_vis/yaw", rclcpp::QoS(100)
        ));
        pubs_.push_back(node_->create_publisher<visualization_msgs::msg::Marker>(
            "/planning_vis/viewpoints", rclcpp::QoS(1000)
        ));
    }
    void fillBasicInfo(
        visualization_msgs::msg::Marker& marker,
        const Eigen::Vector3d& scale,
        const Eigen::Vector4d& color,
        const std::string& ns,
        int id,
        int shape
    ) {
        marker.header.frame_id = "world";
        marker.header.stamp = node_->now();
        marker.id = id;
        marker.ns = ns;
        marker.type = shape;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;
        marker.pose.orientation.w = 1.0;
        marker.color.r = static_cast<float>(color(0));
        marker.color.g = static_cast<float>(color(1));
        marker.color.b = static_cast<float>(color(2));
        marker.color.a = static_cast<float>(color(3));
        marker.scale.x = scale(0);
        marker.scale.y = scale(1);
        marker.scale.z = scale(2);
    }
    void fillGeometryInfo(visualization_msgs::msg::Marker& marker, const std::vector<Eigen::Vector3d>& list) {
        marker.points.reserve(marker.points.size() + list.size());
        for (const auto& point : list) {
            geometry_msgs::msg::Point p;
            p.x = point(0);
            p.y = point(1);
            p.z = point(2);
            marker.points.push_back(p);
        }
    }
    void fillGeometryInfo(
        visualization_msgs::msg::Marker& marker,
        const std::vector<Eigen::Vector3d>& list1,
        const std::vector<Eigen::Vector3d>& list2
    ) {
        const size_t size = std::min(list1.size(), list2.size());
        marker.points.reserve(marker.points.size() + 2 * size);
        for (size_t i = 0; i < size; ++ i) {
            geometry_msgs::msg::Point p;
            p.x = list1[i](0);
            p.y = list1[i](1);
            p.z = list1[i](2);
            marker.points.push_back(p);
            p.x = list2[i](0);
            p.y = list2[i](1);
            p.z = list2[i](2);
            marker.points.push_back(p);
        }
    }
    void drawSpheres(
        const std::vector<Eigen::Vector3d>& list,
        double scale,
        const Eigen::Vector4d& color,
        const std::string& ns,
        int id,
        int pubId
    ) {
        visualization_msgs::msg::Marker marker;
        fillBasicInfo(marker, Eigen::Vector3d(scale, scale, scale), color, ns, id, visualization_msgs::msg::Marker::SPHERE_LIST);
        marker.action = visualization_msgs::msg::Marker::DELETE;
        publishMarker(marker, pubId);
        if (list.empty()) {
            return;
        }
        marker.action = visualization_msgs::msg::Marker::ADD;
        fillGeometryInfo(marker, list);
        publishMarker(marker, pubId);
    }
    void drawCubes(
        const std::vector<Eigen::Vector3d>& list,
        double scale,
        const Eigen::Vector4d& color,
        const std::string& ns,
        int id,
        int pubId
    ) {
        visualization_msgs::msg::Marker marker;
        fillBasicInfo(marker, Eigen::Vector3d(scale, scale, scale), color, ns, id, visualization_msgs::msg::Marker::CUBE_LIST);
        marker.action = visualization_msgs::msg::Marker::DELETE;
        publishMarker(marker, pubId);
        if (list.empty()) {
            return;
        }
        marker.action = visualization_msgs::msg::Marker::ADD;
        fillGeometryInfo(marker, list);
        publishMarker(marker, pubId);
    }
    void drawLines(
        const std::vector<Eigen::Vector3d>& list1,
        const std::vector<Eigen::Vector3d>& list2,
        double scale,
        const Eigen::Vector4d& color,
        const std::string& ns,
        int id,
        int pubId
    ) {
        visualization_msgs::msg::Marker marker;
        fillBasicInfo(marker, Eigen::Vector3d(scale, scale, scale), color, ns, id, visualization_msgs::msg::Marker::LINE_LIST);
        marker.action = visualization_msgs::msg::Marker::DELETE;
        publishMarker(marker, pubId);
        if (list1.empty() || list2.empty()) {
            return;
        }
        marker.action = visualization_msgs::msg::Marker::ADD;
        fillGeometryInfo(marker, list1, list2);
        publishMarker(marker, pubId);
    }
    void drawLines(
        const std::vector<Eigen::Vector3d>& list,
        double scale,
        const Eigen::Vector4d& color,
        const std::string& ns,
        int id,
        int pubId
    ) {
        if (list.size() < 2) {
            std::vector<Eigen::Vector3d> empty;
            drawLines(empty, empty, scale, color, ns, id, pubId);
            return;
        }
        std::vector<Eigen::Vector3d> list1;
        std::vector<Eigen::Vector3d> list2;
        list1.reserve(list.size() - 1);
        list2.reserve(list.size() - 1);
        for (size_t i = 0; i + 1 < list.size(); ++ i) {
            list1.push_back(list[i]);
            list2.push_back(list[i + 1]);
        }
        drawLines(list1, list2, scale, color, ns, id, pubId);
    }
    void drawBox(
        const Eigen::Vector3d& center,
        const Eigen::Vector3d& scale,
        const Eigen::Vector4d& color,
        const std::string& ns,
        int id,
        int pubId
    ) {
        visualization_msgs::msg::Marker marker;
        fillBasicInfo(marker, scale, color, ns, id, visualization_msgs::msg::Marker::CUBE);
        marker.action = visualization_msgs::msg::Marker::DELETE;
        publishMarker(marker, pubId);
        marker.pose.position.x = center(0);
        marker.pose.position.y = center(1);
        marker.pose.position.z = center(2);
        marker.action = visualization_msgs::msg::Marker::ADD;
        publishMarker(marker, pubId);
    }
    void drawText(
        const Eigen::Vector3d& pos,
        const std::string& text,
        double scale,
        const Eigen::Vector4d& color,
        const std::string& ns,
        int id,
        int pubId
    ) {
        visualization_msgs::msg::Marker marker;
        fillBasicInfo(marker, Eigen::Vector3d(scale, scale, scale), color, ns, id, visualization_msgs::msg::Marker::TEXT_VIEW_FACING);
        marker.action = visualization_msgs::msg::Marker::DELETE;
        publishMarker(marker, pubId);
        marker.text = text;
        marker.pose.position.x = pos(0);
        marker.pose.position.y = pos(1);
        marker.pose.position.z = pos(2);
        marker.action = visualization_msgs::msg::Marker::ADD;
        publishMarker(marker, pubId);
    }
    static Eigen::Vector4d getColor(double h, double alpha = 1.0) {
        double h1 = h;
        if (h1 < 0.0 || h1 > 1.0) {
            h1 = 0.0;
        }
        double lambda = 0.0;
        Eigen::Vector4d color1(1, 0, 0, 1);
        Eigen::Vector4d color2(1, 0, 0, 1);
        if (h1 < 1.0 / 6.0) {
            lambda = h1 * 6.0;
            color1 = Eigen::Vector4d(1, 0, 0, 1);
            color2 = Eigen::Vector4d(1, 0, 1, 1);
        } else if (h1 < 2.0 / 6.0) {
            lambda = (h1 - 1.0 / 6.0) * 6.0;
            color1 = Eigen::Vector4d(1, 0, 1, 1);
            color2 = Eigen::Vector4d(0, 0, 1, 1);
        } else if (h1 < 3.0 / 6.0) {
            lambda = (h1 - 2.0 / 6.0) * 6.0;
            color1 = Eigen::Vector4d(0, 0, 1, 1);
            color2 = Eigen::Vector4d(0, 1, 1, 1);
        } else if (h1 < 4.0 / 6.0) {
            lambda = (h1 - 3.0 / 6.0) * 6.0;
            color1 = Eigen::Vector4d(0, 1, 1, 1);
            color2 = Eigen::Vector4d(0, 1, 0, 1);
        } else if (h1 < 5.0 / 6.0) {
            lambda = (h1 - 4.0 / 6.0) * 6.0;
            color1 = Eigen::Vector4d(0, 1, 0, 1);
            color2 = Eigen::Vector4d(1, 1, 0, 1);
        } else {
            lambda = (h1 - 5.0 / 6.0) * 6.0;
            color1 = Eigen::Vector4d(1, 1, 0, 1);
            color2 = Eigen::Vector4d(1, 0, 0, 1);
        }
        Eigen::Vector4d color = (1.0 - lambda) * color1 + lambda * color2;
        color(3) = alpha;
        return color;
    }

private:
    void publishMarker(visualization_msgs::msg::Marker& marker, int pubId) {
        if (pubId < 0 || pubId >= static_cast<int>(pubs_.size())) {
            return;
        }
        pubs_[pubId]->publish(marker);
    }

private:
    rclcpp::Node::SharedPtr node_;
    std::vector<rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr> pubs_;
};

} // arc
