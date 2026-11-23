#pragma once

#include <base/ros.h>

#include <base/iarc.h>
#include <sensor_msgs/msg/point_cloud2.hpp>

namespace arc {

class MapPub {
public:
    MapPub(rclcpp::Node::SharedPtr node);
    bool initialize();

private:
    bool loadMap_();

private:
    rclcpp::Node::SharedPtr node_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr mapPub_;
    sensor_msgs::msg::PointCloud2 mapMsg_;
    std::string pcdPath_;
    std::string frameId_;
    bool addGround_{true};
};

} // arc
