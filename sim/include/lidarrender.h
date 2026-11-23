#pragma once

#include <base/base.h>
#include <base/ros.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <deque>
#include <random>

namespace arc {

struct LidarRenderPoint {
    PCL_ADD_POINT4D;
    float intensity{0.0f};
    float time{0.0f};
    uint16_t ring{0};
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

} // arc

POINT_CLOUD_REGISTER_POINT_STRUCT(arc::LidarRenderPoint,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (float, time, time)
    (uint16_t, ring, ring)
)

namespace arc {

class LidarRenderNode {
public:
    explicit LidarRenderNode(rclcpp::Node::SharedPtr node)
        : node_(std::move(node))
    {
        lidarRate_ = node_->declare_parameter<double>("lidar_rate", 10.0);
        rings_ = node_->declare_parameter<int>("lidar_rings", 16);
        azimuthSamples_ = node_->declare_parameter<int>("lidar_azimuth_samples", 900);
        verticalMinDeg_ = node_->declare_parameter<double>("lidar_vertical_min_deg", -15.0);
        verticalMaxDeg_ = node_->declare_parameter<double>("lidar_vertical_max_deg", 15.0);
        minRange_ = node_->declare_parameter<double>("lidar_min_range", 0.2);
        maxRange_ = node_->declare_parameter<double>("lidar_max_range", 15.0);
        rangeNoiseStd_ = node_->declare_parameter<double>("range_noise_std", 0.01);
        motionDistortion_ = node_->declare_parameter<bool>("motion_distortion", true);
        lidarFrame_ = node_->declare_parameter<std::string>("lidar_frame", "lidar");
        rng_.seed(static_cast<uint32_t>(node_->declare_parameter<int>("drone_id", 1)));

        mapSub_ = node_->create_subscription<sensor_msgs::msg::PointCloud2>(
            "global_map",
            rclcpp::QoS(1).reliable().transient_local(),
            [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
                onMap_(msg);
            }
        );
        odomSub_ = node_->create_subscription<nav_msgs::msg::Odometry>(
            "odometry",
            200,
            [this](const nav_msgs::msg::Odometry::SharedPtr msg) {
                onOdom_(msg);
            }
        );
        scanPub_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>("scan", 10);
        timer_ = node_->create_wall_timer(
            std::chrono::duration<double>(1.0 / std::max(lidarRate_, 1e-3)),
            [this]() { render_(); }
        );
    }

private:
    void onMap_(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        if (hasMap_) {
            return;
        }
        pcl::PointCloud<pcl::PointXYZ> cloud;
        pcl::fromROSMsg(*msg, cloud);
        if (cloud.empty()) {
            return;
        }
        mapPoints_.reserve(cloud.size());
        for (const auto& point : cloud.points) {
            if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
                continue;
            }
            mapPoints_.emplace_back(point.x, point.y, point.z);
        }
        hasMap_ = true;
        RCLCPP_INFO(node_->get_logger(), "LiDAR render map: %zu points", mapPoints_.size());
    }

    void onOdom_(const nav_msgs::msg::Odometry::SharedPtr msg) {
        Mat4d pose = Mat4d::Identity();
        const auto& position = msg->pose.pose.position;
        const auto& orientationMsg = msg->pose.pose.orientation;
        const Eigen::Quaterniond orientation(
            orientationMsg.w, orientationMsg.x, orientationMsg.y, orientationMsg.z);
        pose.block<3, 3>(0, 0) = orientation.toRotationMatrix();
        pose.block<3, 1>(0, 3) = Eigen::Vector3d(position.x, position.y, position.z);

        const double stamp = static_cast<double>(msg->header.stamp.sec)
            + static_cast<double>(msg->header.stamp.nanosec) * 1e-9;
        if (!history_.empty() && stamp < history_.back().first) {
            history_.clear();
        }
        history_.emplace_back(stamp, pose);
        while (!history_.empty() && history_.front().first < stamp - 2.0) {
            history_.pop_front();
        }
        lastOdomTime_ = stamp;
        lastOdomPose_ = pose;
        hasOdom_ = true;
    }

    Mat4d poseAt_(double time) const {
        if (history_.empty()) {
            return lastOdomPose_;
        }
        if (time <= history_.front().first) {
            return history_.front().second;
        }
        if (time >= history_.back().first) {
            return history_.back().second;
        }
        for (size_t i = 1; i < history_.size(); ++ i) {
            if (history_[i].first >= time) {
                const double t0 = history_[i - 1].first;
                const double t1 = history_[i].first;
                const double ratio = t1 > t0 ? (time - t0) / (t1 - t0) : 0.0;
                const Mat4d& p0 = history_[i - 1].second;
                const Mat4d& p1 = history_[i].second;
                Mat4d pose = Mat4d::Identity();
                pose.block<3, 1>(0, 3) = p0.block<3, 1>(0, 3)
                    + ratio * (p1.block<3, 1>(0, 3) - p0.block<3, 1>(0, 3));
                const Eigen::Quaterniond q0(p0.block<3, 3>(0, 0));
                const Eigen::Quaterniond q1(p1.block<3, 3>(0, 0));
                pose.block<3, 3>(0, 0) = q0.slerp(ratio, q1).toRotationMatrix();
                return pose;
            }
        }
        return history_.back().second;
    }

    void render_() {
        if (!hasMap_ || !hasOdom_ || mapPoints_.empty() || history_.empty()) {
            return;
        }

        const double scanPeriod = 1.0 / std::max(lidarRate_, 1e-3);
        const double scanStart = std::max(history_.front().first, lastOdomTime_ - scanPeriod);
        if (scanStart >= lastOdomTime_) {
            return;
        }
        const Mat4d startPose = poseAt_(scanStart);
        const Eigen::Vector3d sensorPosition = startPose.block<3, 1>(0, 3);
        const Mat3d startRotation = startPose.block<3, 3>(0, 0).transpose();

        const int ringCount = std::max(rings_, 1);
        const int azimuthCount = std::max(azimuthSamples_, 1);
        const double verticalMin = verticalMinDeg_ * __pi__ / 180.0;
        const double verticalMax = verticalMaxDeg_ * __pi__ / 180.0;
        const double verticalSpan = std::max(verticalMax - verticalMin, 1e-6);
        const size_t pixelCount = static_cast<size_t>(ringCount) * azimuthCount;

        std::vector<float> rangeImage(pixelCount, static_cast<float>(maxRange_ + 1.0));
        std::vector<int> pointImage(pixelCount, -1);

        for (size_t i = 0; i < mapPoints_.size(); ++ i) {
            const Eigen::Vector3d local = startRotation * (mapPoints_[i] - sensorPosition);
            const double range = local.norm();
            if (range < minRange_ || range > maxRange_) {
                continue;
            }
            const double azimuth = std::atan2(local.y(), local.x());
            const double normalizedAzimuth = azimuth < 0.0
                ? (azimuth + 2.0 * __pi__) / (2.0 * __pi__)
                : azimuth / (2.0 * __pi__);
            const double elevation = std::asin(std::clamp(local.z() / range, -1.0, 1.0));
            if (elevation < verticalMin || elevation >= verticalMax) {
                continue;
            }
            const int azimuthIndex = std::min(
                static_cast<int>(normalizedAzimuth * azimuthCount), azimuthCount - 1);
            const int ringIndex = std::min(
                static_cast<int>((elevation - verticalMin) / verticalSpan * ringCount),
                ringCount - 1);
            const size_t pixel = static_cast<size_t>(ringIndex) * azimuthCount
                + static_cast<size_t>(azimuthIndex);
            if (static_cast<float>(range) < rangeImage[pixel]) {
                rangeImage[pixel] = static_cast<float>(range);
                pointImage[pixel] = static_cast<int>(i);
            }
        }

        pcl::PointCloud<LidarRenderPoint> cloud;
        cloud.reserve(pixelCount);
        for (int ring = 0; ring < ringCount; ++ ring) {
            for (int azimuthIndex = 0; azimuthIndex < azimuthCount; ++ azimuthIndex) {
                const size_t pixel = static_cast<size_t>(ring) * azimuthCount
                    + static_cast<size_t>(azimuthIndex);
                if (pointImage[pixel] < 0) {
                    continue;
                }
                const double pointTime = static_cast<double>(azimuthIndex) / azimuthCount * scanPeriod;
                const Eigen::Vector3d& worldPoint = mapPoints_[static_cast<size_t>(pointImage[pixel])];

                Eigen::Vector3d local;
                if (motionDistortion_) {
                    const Mat4d pointPose = poseAt_(scanStart + pointTime);
                    local = pointPose.block<3, 3>(0, 0).transpose()
                        * (worldPoint - pointPose.block<3, 1>(0, 3));
                } else {
                    local = startRotation * (worldPoint - sensorPosition);
                }

                const double range = local.norm();
                if (range < minRange_ || range > maxRange_) {
                    continue;
                }
                const double noise = rangeNoiseStd_ > 0.0 ? normal_(rng_) * rangeNoiseStd_ : 0.0;
                const double noisyRange = range + noise;
                if (noisyRange < minRange_ || noisyRange > maxRange_) {
                    continue;
                }
                const Eigen::Vector3d noisy = local * (noisyRange / range);

                LidarRenderPoint point;
                point.x = static_cast<float>(noisy.x());
                point.y = static_cast<float>(noisy.y());
                point.z = static_cast<float>(noisy.z());
                point.intensity = static_cast<float>(
                    std::clamp(255.0 - noisyRange / maxRange_ * 200.0, 1.0, 255.0));
                point.time = static_cast<float>(pointTime);
                point.ring = static_cast<uint16_t>(ring);
                cloud.push_back(point);
            }
        }

        sensor_msgs::msg::PointCloud2 message;
        pcl::toROSMsg(cloud, message);
        message.header.stamp = rclcpp::Time(static_cast<int64_t>(scanStart * 1e9));
        message.header.frame_id = lidarFrame_;
        message.is_dense = false;
        scanPub_->publish(message);
    }

private:
    rclcpp::Node::SharedPtr node_;
    double lidarRate_{10.0};
    int rings_{16};
    int azimuthSamples_{900};
    double verticalMinDeg_{-15.0};
    double verticalMaxDeg_{15.0};
    double minRange_{0.2};
    double maxRange_{15.0};
    double rangeNoiseStd_{0.01};
    bool motionDistortion_{true};
    std::string lidarFrame_{"lidar"};

    std::vector<Eigen::Vector3d> mapPoints_;
    std::deque<std::pair<double, Mat4d>> history_;
    Mat4d lastOdomPose_{Mat4d::Identity()};
    double lastOdomTime_{-1.0};
    bool hasMap_{false};
    bool hasOdom_{false};
    std::mt19937 rng_;
    std::normal_distribution<double> normal_{0.0, 1.0};

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr mapSub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odomSub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr scanPub_;
    rclcpp::TimerBase::SharedPtr timer_;
};

} // arc
