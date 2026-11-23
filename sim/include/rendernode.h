#pragma once

#include <opencv2/core.hpp>

#include <pcl/point_types.h>

#include <pcl/point_cloud.h>

#include <base/ros.h>

#include <base/iarc.h>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <pcl_conversions/pcl_conversions.h>

#if __has_include(<cv_bridge/cv_bridge.hpp>)
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif

namespace arc {

class RenderNode {
public:
    explicit RenderNode(rclcpp::Node::SharedPtr node)
        : node_(std::move(node))
    {
        width_ = node_->declare_parameter<int>("cam_width", 640);
        height_ = node_->declare_parameter<int>("cam_height", 480);
        fx_ = node_->declare_parameter<double>("cam_fx", 387.229);
        fy_ = node_->declare_parameter<double>("cam_fy", 387.229);
        cx_ = node_->declare_parameter<double>("cam_cx", 321.046);
        cy_ = node_->declare_parameter<double>("cam_cy", 243.449);
        sensingHorizon_ = node_->declare_parameter<double>("sensing_horizon", 5.0);
        sensingRate_ = node_->declare_parameter<double>("sensing_rate", 10.0);
        imageRate_ = node_->declare_parameter<double>("image_rate", 20.0);
        estimationRate_ = node_->declare_parameter<double>("estimation_rate", 10.0);
        textureScale_ = node_->declare_parameter<double>("texture_scale", 1.0);

        cam02body_ << 0.0, 0.0, 1.0, 0.0,
            -1.0, 0.0, 0.0, 0.0,
            0.0, -1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0;
        cam2world_ = Eigen::Matrix4d::Identity();
        cam2worldQuat_.setIdentity();
        lastOdomStamp_ = rclcpp::Time(0, 0, RCL_SYSTEM_TIME);

        globalMapSub_ = node_->create_subscription<sensor_msgs::msg::PointCloud2>(
            "global_map",
            rclcpp::QoS(1).reliable().transient_local(),
            [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
                onGlobalMap_(msg);
            }
        );
        odomSub_ = node_->create_subscription<nav_msgs::msg::Odometry>(
            "odometry",
            50,
            [this](const nav_msgs::msg::Odometry::SharedPtr msg) {
                onOdometry_(msg);
            }
        );
        depthPub_ = node_->create_publisher<sensor_msgs::msg::Image>("depth", 1000);
        imagePub_ = node_->create_publisher<sensor_msgs::msg::Image>("image", 1000);
        sensorPosePub_ = node_->create_publisher<geometry_msgs::msg::PoseStamped>("sensor_pose", 1000);

        sensingTimer_ = node_->create_wall_timer(
            std::chrono::duration<double>(1.0 / std::max(sensingRate_, 1e-3)),
            [this]() { renderFrame_(true, false); }
        );
        if (imageRate_ > 0.0) {
            imageTimer_ = node_->create_wall_timer(
                std::chrono::duration<double>(1.0 / imageRate_),
                [this]() { renderFrame_(false, true); }
            );
        }
        poseTimer_ = node_->create_wall_timer(
            std::chrono::duration<double>(1.0 / std::max(estimationRate_, 1e-3)),
            [this]() { publishSensorPose_(); }
        );
    }

private:
    void onGlobalMap_(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        if (hasGlobalMap_) {
            return;
        }
        pcl::fromROSMsg(*msg, cloudIn_);
        if (cloudIn_.empty()) {
            return;
        }
        hasGlobalMap_ = true;
        RCLCPP_INFO(node_->get_logger(), "global map received: %zu points", cloudIn_.size());
    }

    void onOdometry_(const nav_msgs::msg::Odometry::SharedPtr msg) {
        hasOdom_ = true;
        lastOdomStamp_ = rclcpp::Time(msg->header.stamp);

        Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
        Eigen::Vector3d position(
            msg->pose.pose.position.x,
            msg->pose.pose.position.y,
            msg->pose.pose.position.z
        );
        Eigen::Quaterniond orientation(
            msg->pose.pose.orientation.w,
            msg->pose.pose.orientation.x,
            msg->pose.pose.orientation.y,
            msg->pose.pose.orientation.z
        );
        pose.block<3, 3>(0, 0) = orientation.toRotationMatrix();
        pose.block<3, 1>(0, 3) = position;

        cam2world_ = pose * cam02body_;
        cam2worldQuat_ = Eigen::Quaterniond(cam2world_.block<3, 3>(0, 0));
    }

    void renderFrame_(bool publishDepth, bool publishImage) {
        if (!hasGlobalMap_ || !hasOdom_) {
            return;
        }
        depthMat_ = cv::Mat::zeros(height_, width_, CV_32FC1);
        intensityMat_ = cv::Mat::zeros(height_, width_, CV_8UC1);

        const Eigen::Matrix4d Tcw = cam2world_.inverse();
        const Eigen::Matrix3d Rcw = Tcw.block<3, 3>(0, 0);
        const Eigen::Vector3d tcw = Tcw.block<3, 1>(0, 3);
        const Eigen::Vector3d cameraPos = cam2world_.block<3, 1>(0, 3);

        for (const auto& point : cloudIn_.points) {
            const Eigen::Vector3d pw(point.x, point.y, point.z);
            if ((cameraPos - pw).norm() > sensingHorizon_) {
                continue;
            }
            const Eigen::Vector3d pc = Rcw * pw + tcw;
            if (pc(2) <= 0.0) {
                continue;
            }
            const double projectedX = pc(0) / pc(2) * fx_ + cx_;
            const double projectedY = pc(1) / pc(2) * fy_ + cy_;
            if (projectedX < 0.0 || projectedX >= width_ || projectedY < 0.0 || projectedY >= height_) {
                continue;
            }

            const double depth = pc(2);
            const double waveA = std::sin(
                textureScale_ * (3.1 * point.x + 1.7 * point.y + 2.3 * point.z));
            const double waveB = std::cos(
                textureScale_ * (5.3 * point.x - 2.9 * point.y + 1.1 * point.z));
            const uint32_t hash = static_cast<uint32_t>(
                static_cast<int32_t>(point.x * 1000.0f) * 73856093u ^
                static_cast<int32_t>(point.y * 1000.0f) * 19349663u ^
                static_cast<int32_t>(point.z * 1000.0f) * 83492791u);
            const double noise = static_cast<double>(hash & 0xffu) / 255.0 - 0.5;
            const int intensity = static_cast<int>(
                std::clamp(128.0 + 70.0 * waveA * waveB + 110.0 * noise, 0.0, 255.0));

            const int radius = static_cast<int>(0.0573 * fx_ / depth + 0.5);
            const int minX = std::max(static_cast<int>(projectedX) - radius, 0);
            const int maxX = std::min(static_cast<int>(projectedX) + radius, width_ - 1);
            const int minY = std::max(static_cast<int>(projectedY) - radius, 0);
            const int maxY = std::min(static_cast<int>(projectedY) + radius, height_ - 1);
            for (int x = minX; x <= maxX; ++ x) {
                for (int y = minY; y <= maxY; ++ y) {
                    const float oldDepth = depthMat_.at<float>(y, x);
                    const float newDepth = static_cast<float>(depth);
                    if (oldDepth < 1e-3f || newDepth < oldDepth) {
                        depthMat_.at<float>(y, x) = newDepth;
                        intensityMat_.at<uint8_t>(y, x) = static_cast<uint8_t>(intensity);
                    }
                }
            }
        }

        if (publishDepth) {
            cv_bridge::CvImage image;
            image.header.stamp = lastOdomStamp_;
            image.header.frame_id = "world";
            image.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
            image.image = depthMat_.clone();
            depthPub_->publish(*image.toImageMsg());
        }

        if (publishImage) {
            cv_bridge::CvImage image;
            image.header.stamp = lastOdomStamp_;
            image.header.frame_id = "camera";
            image.encoding = sensor_msgs::image_encodings::MONO8;
            image.image = intensityMat_.clone();
            imagePub_->publish(*image.toImageMsg());
        }
    }

    void publishSensorPose_() {
        if (!hasOdom_) {
            return;
        }
        geometry_msgs::msg::PoseStamped pose;
        pose.header.stamp = lastOdomStamp_;
        pose.header.frame_id = "world";
        pose.pose.position.x = cam2world_(0, 3);
        pose.pose.position.y = cam2world_(1, 3);
        pose.pose.position.z = cam2world_(2, 3);
        pose.pose.orientation.w = cam2worldQuat_.w();
        pose.pose.orientation.x = cam2worldQuat_.x();
        pose.pose.orientation.y = cam2worldQuat_.y();
        pose.pose.orientation.z = cam2worldQuat_.z();
        sensorPosePub_->publish(pose);
    }

private:
    rclcpp::Node::SharedPtr node_;
    int width_{640};
    int height_{480};
    double fx_{387.229};
    double fy_{387.229};
    double cx_{321.046};
    double cy_{243.449};
    double sensingHorizon_{5.0};
    double sensingRate_{10.0};
    double imageRate_{20.0};
    double estimationRate_{10.0};
    double textureScale_{1.0};

    Eigen::Matrix4d cam02body_{Eigen::Matrix4d::Identity()};
    Eigen::Matrix4d cam2world_{Eigen::Matrix4d::Identity()};
    Eigen::Quaterniond cam2worldQuat_{Eigen::Quaterniond::Identity()};
    rclcpp::Time lastOdomStamp_;
    pcl::PointCloud<pcl::PointXYZ> cloudIn_;
    cv::Mat depthMat_;
    cv::Mat intensityMat_;
    bool hasGlobalMap_{false};
    bool hasOdom_{false};

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr globalMapSub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odomSub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depthPub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr imagePub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr sensorPosePub_;
    rclcpp::TimerBase::SharedPtr sensingTimer_;
    rclcpp::TimerBase::SharedPtr imageTimer_;
    rclcpp::TimerBase::SharedPtr poseTimer_;
};

} // arc
