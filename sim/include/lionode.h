#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <builtin_interfaces/msg/time.hpp>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <lio/lio.h>

namespace arc {

class LioNode {
public:
    explicit LioNode(rclcpp::Node::SharedPtr node);
    ~LioNode();

    LioNode(const LioNode&) = delete;
    LioNode& operator=(const LioNode&) = delete;

private:
    void loadParams_();
    void initRos_();
    void lidarCallback_(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    // Livox CustomMsg extension point: add an overload that fills LidarFrame
    // from offset_time/line/tag, then feed it to pipeline_->addLidar().
    void imuCallback_(const sensor_msgs::msg::Imu::SharedPtr msg);
    void publishState_();
    LidarFrame convertCloud_(const sensor_msgs::msg::PointCloud2& msg) const;
    Mat4d initialTransform_() const;

    static double stampToSeconds_(const builtin_interfaces::msg::Time& stamp);
    static bool readFloat_(const uint8_t* data, size_t offset, uint8_t datatype, float& value);
    static bool readUInt16_(const uint8_t* data, size_t offset, uint8_t datatype, uint16_t& value);
    static bool readUInt32_(const uint8_t* data, size_t offset, uint8_t datatype, uint32_t& value);

private:
    rclcpp::Node::SharedPtr node_;
    LioConfig config_;
    std::unique_ptr<LioPipeline> pipeline_;

    std::string configFile_;
    std::string lidarTopic_;
    std::string imuTopic_;
    std::string worldFrame_;
    std::string bodyFrame_;
    std::vector<double> initialPose_;
    bool publishTf_{true};
    bool publishCloud_{true};
    double lidarScanPeriod_{0.1};

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidarSub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imuSub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odomPub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloudPub_;
    rclcpp::TimerBase::SharedPtr publishTimer_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tfBroadcaster_;

    double lastPublishedStamp_{-1.0};
};

} // arc
