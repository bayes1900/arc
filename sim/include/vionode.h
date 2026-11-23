#pragma once

#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <vio/vio.h>
#include <vio/pipeline.h>
#include <vio/io/config.h>

namespace arc {

class VioNode {
public:
    explicit VioNode(rclcpp::Node::SharedPtr node);
    ~VioNode();

    VioNode(const VioNode&) = delete;
    VioNode& operator=(const VioNode&) = delete;

private:
    void loadParams_();
    void loadCalibration_();
    void initRos_();
    void imageCallback_(const sensor_msgs::msg::Image::SharedPtr msg, size_t cameraIndex);
    void imuCallback_(const sensor_msgs::msg::Imu::SharedPtr msg);
    void tryDispatchImage_();
    void publishState_();
    static ManagedImage<uint16_t>::Ptr convertImage_(
        const sensor_msgs::msg::Image& msg);

private:
    rclcpp::Node::SharedPtr node_;
    VioConfig config_;
    Calibration<double> calibration_;
    std::unique_ptr<VioPipeline> pipeline_;

    std::string configFile_;
    std::string calibFile_;
    std::string imuTopic_;
    std::vector<std::string> imageTopics_;
    std::string worldFrame_;
    std::string bodyFrame_;
    std::vector<double> initialPose_;
    bool publishTf_{true};
    bool hasInitialPose_{false};
    double imageSyncTolerance_{0.005};
    std::vector<sensor_msgs::msg::Image::SharedPtr> pendingImages_;
    std::vector<int64_t> pendingImageNs_;

    std::vector<rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr> imageSubs_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imuSub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odomPub_;
    rclcpp::TimerBase::SharedPtr stateTimer_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tfBroadcaster_;

    PoseVelBiasState::Ptr lastState_;
    bool started_{false};
    int64_t lastImageNs_{0};
};

} // arc
