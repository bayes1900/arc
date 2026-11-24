#include "lionode.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

namespace arc {

LioNode::LioNode(rclcpp::Node::SharedPtr node)
    : node_(std::move(node))
{
    loadParams_();
    initRos_();
}

LioNode::~LioNode() {
    if (pipeline_) {
        pipeline_->stop();
    }
}

void LioNode::loadParams_() {
    configFile_ = node_->declare_parameter<std::string>("config_file", "");
    lidarTopic_ = node_->declare_parameter<std::string>("lidar_topic", "/sensor/lidar_1");
    imuTopic_ = node_->declare_parameter<std::string>("imu_topic", "/sensor/imu_1");
    worldFrame_ = node_->declare_parameter<std::string>("world_frame", "world");
    bodyFrame_ = node_->declare_parameter<std::string>("body_frame", "imu");
    publishTf_ = node_->declare_parameter<bool>("publish_tf", true);
    publishCloud_ = node_->declare_parameter<bool>("publish_cloud", true);
    initialPose_ = node_->declare_parameter<std::vector<double>>(
        "initial_pose", std::vector<double>{});

    if (!configFile_.empty()) {
        config_ = LioConfig::loadConfig(configFile_);
    } else {
        RCLCPP_WARN(node_->get_logger(), "config_file is empty; using LIO defaults");
    }
    lidarScanPeriod_ = 1.0 / std::max(config_.scanRate, 1);
}

void LioNode::initRos_() {
    lidarSub_ = node_->create_subscription<sensor_msgs::msg::PointCloud2>(
        lidarTopic_,
        rclcpp::QoS(10).reliable(),
        [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
            lidarCallback_(msg);
        }
    );
    imuSub_ = node_->create_subscription<sensor_msgs::msg::Imu>(
        imuTopic_,
        rclcpp::QoS(500).reliable(),
        [this](const sensor_msgs::msg::Imu::SharedPtr msg) {
            imuCallback_(msg);
        }
    );

    odomPub_ = node_->create_publisher<nav_msgs::msg::Odometry>("odom", 100);
    cloudPub_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>("cloud_registered", 10);
    if (publishTf_) {
        tfBroadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(node_);
    }

    pipeline_ = std::make_unique<LioPipeline>(config_, publishCloud_);
    pipeline_->start();

    publishTimer_ = node_->create_wall_timer(
        std::chrono::milliseconds(10),
        [this]() { publishState_(); }
    );
}

void LioNode::lidarCallback_(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    try {
        pipeline_->addLidar(convertCloud_(*msg));
    } catch (const std::exception& error) {
        RCLCPP_ERROR(node_->get_logger(), "LIO lidar callback failed: %s", error.what());
    }
}

void LioNode::imuCallback_(const sensor_msgs::msg::Imu::SharedPtr msg) {
    ImuSample imu;
    imu.time = stampToSeconds_(msg->header.stamp) - config_.timeOffsetLidarToImu;
    imu.accel = Vec3d(
        msg->linear_acceleration.x,
        msg->linear_acceleration.y,
        msg->linear_acceleration.z);
    imu.gyro = Vec3d(
        msg->angular_velocity.x,
        msg->angular_velocity.y,
        msg->angular_velocity.z);
    pipeline_->addImu(imu);
}

void LioNode::publishState_() {
    if (!pipeline_) {
        return;
    }
    LioOutput output;
    if (!pipeline_->getOutput(output) || !output.valid || output.stamp <= lastPublishedStamp_) {
        return;
    }

    const Mat4d initial = initialTransform_();
    Mat4d body = Mat4d::Identity();
    body.block<3, 3>(0, 0) = output.rotation;
    body.block<3, 1>(0, 3) = output.position;
    const Mat4d world = initial * body;

    const rclcpp::Time stamp(static_cast<int64_t>(output.stamp * 1e9));
    const Eigen::Quaterniond orientation(world.block<3, 3>(0, 0));
    const Eigen::Vector3d velocity = initial.block<3, 3>(0, 0) * output.velocity;

    nav_msgs::msg::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = worldFrame_;
    odom.child_frame_id = bodyFrame_;
    odom.pose.pose.position.x = world(0, 3);
    odom.pose.pose.position.y = world(1, 3);
    odom.pose.pose.position.z = world(2, 3);
    odom.pose.pose.orientation.w = orientation.w();
    odom.pose.pose.orientation.x = orientation.x();
    odom.pose.pose.orientation.y = orientation.y();
    odom.pose.pose.orientation.z = orientation.z();
    odom.twist.twist.linear.x = velocity.x();
    odom.twist.twist.linear.y = velocity.y();
    odom.twist.twist.linear.z = velocity.z();
    odomPub_->publish(odom);

    if (tfBroadcaster_) {
        geometry_msgs::msg::TransformStamped transform;
        transform.header.stamp = stamp;
        transform.header.frame_id = worldFrame_;
        transform.child_frame_id = bodyFrame_;
        transform.transform.translation.x = world(0, 3);
        transform.transform.translation.y = world(1, 3);
        transform.transform.translation.z = world(2, 3);
        transform.transform.rotation.w = orientation.w();
        transform.transform.rotation.x = orientation.x();
        transform.transform.rotation.y = orientation.y();
        transform.transform.rotation.z = orientation.z();
        tfBroadcaster_->sendTransform(transform);
    }

    if (publishCloud_ && output.cloud && !output.cloud->empty()) {
        pcl::PointCloud<pcl::PointXYZI> cloud;
        cloud.reserve(output.cloud->size());
        for (const LioPoint& point : output.cloud->points) {
            const Eigen::Vector4d local(point.x, point.y, point.z, 1.0);
            const Eigen::Vector4d worldPoint = initial * local;
            pcl::PointXYZI converted;
            converted.x = static_cast<float>(worldPoint.x());
            converted.y = static_cast<float>(worldPoint.y());
            converted.z = static_cast<float>(worldPoint.z());
            converted.intensity = point.intensity;
            cloud.push_back(converted);
        }
        sensor_msgs::msg::PointCloud2 message;
        pcl::toROSMsg(cloud, message);
        message.header.stamp = stamp;
        message.header.frame_id = worldFrame_;
        cloudPub_->publish(message);
    }

    lastPublishedStamp_ = output.stamp;
}

LidarFrame LioNode::convertCloud_(const sensor_msgs::msg::PointCloud2& msg) const {
    const auto findField = [&msg](const std::string& name) -> const sensor_msgs::msg::PointField* {
        for (const auto& field : msg.fields) {
            if (field.name == name) {
                return &field;
            }
        }
        return nullptr;
    };

    const auto* xField = findField("x");
    const auto* yField = findField("y");
    const auto* zField = findField("z");
    if (xField == nullptr || yField == nullptr || zField == nullptr) {
        throw std::runtime_error("PointCloud2 is missing x/y/z fields");
    }
    const auto* intensityField = findField("intensity");
    const auto* timeField = findField("time");
    const auto* tField = findField("t");
    const auto* ringField = findField("ring");
    if (ringField == nullptr) {
        ringField = findField("line");
    }

    LidarFrame frame;
    frame.stamp = msg.header.stamp.sec == 0 && msg.header.stamp.nanosec == 0
        ? node_->now().seconds()
        : stampToSeconds_(msg.header.stamp);
    frame.hasTime = timeField != nullptr || tField != nullptr;

    const size_t pointCount = static_cast<size_t>(msg.width) * msg.height;
    frame.points.reserve(pointCount);

    double unitScale = 1.0;
    switch (config_.timestampUnit) {
        case 0: unitScale = 1.0; break;
        case 1: unitScale = 1e-3; break;
        case 2: unitScale = 1e-6; break;
        case 3: unitScale = 1e-9; break;
        default: unitScale = 1e-6; break;
    }

    double maxTime = 0.0;
    for (size_t i = 0; i < pointCount; ++ i) {
        const uint8_t* data = msg.data.data() + i * msg.point_step;
        LidarPoint point;
        float value = 0.0f;
        if (!readFloat_(data, xField->offset, xField->datatype, value)) {
            continue;
        }
        point.x = value;
        if (!readFloat_(data, yField->offset, yField->datatype, value)) {
            continue;
        }
        point.y = value;
        if (!readFloat_(data, zField->offset, zField->datatype, value)) {
            continue;
        }
        point.z = value;
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
            continue;
        }

        if (intensityField != nullptr) {
            readFloat_(data, intensityField->offset, intensityField->datatype, point.intensity);
        }
        if (timeField != nullptr) {
            float timeValue = 0.0f;
            if (readFloat_(data, timeField->offset, timeField->datatype, timeValue)) {
                point.time = timeValue;
                maxTime = std::max(maxTime, static_cast<double>(timeValue));
            }
        } else if (tField != nullptr) {
            uint32_t rawTime = 0;
            if (readUInt32_(data, tField->offset, tField->datatype, rawTime)) {
                point.time = static_cast<float>(rawTime * unitScale);
                maxTime = std::max(maxTime, static_cast<double>(point.time));
            }
        }
        if (ringField != nullptr) {
            uint16_t ring = 0;
            if (readUInt16_(data, ringField->offset, ringField->datatype, ring)) {
                point.ring = ring;
            }
        }
        frame.points.push_back(point);
    }

    if (frame.hasTime && maxTime > 0.0) {
        frame.endTime = frame.stamp + maxTime;
    } else {
        frame.endTime = frame.stamp + lidarScanPeriod_;
    }
    return frame;
}

Mat4d LioNode::initialTransform_() const {
    Mat4d transform = Mat4d::Identity();
    if (initialPose_.size() < 3) {
        return transform;
    }
    transform(0, 3) = initialPose_[0];
    transform(1, 3) = initialPose_[1];
    transform(2, 3) = initialPose_[2];

    double yaw = 0.0;
    if (initialPose_.size() >= 4) {
        yaw = initialPose_[3];
    }
    if (initialPose_.size() >= 6) {
        const double roll = initialPose_[3];
        const double pitch = initialPose_[4];
        yaw = initialPose_[5];
        const Mat3d rotation =
            (Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ())
                * Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY())
                * Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX())).toRotationMatrix();
        transform.block<3, 3>(0, 0) = rotation;
    } else {
        transform.block<3, 3>(0, 0) =
            Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    }
    return transform;
}

double LioNode::stampToSeconds_(const builtin_interfaces::msg::Time& stamp) {
    return static_cast<double>(stamp.sec) + static_cast<double>(stamp.nanosec) * 1e-9;
}

bool LioNode::readFloat_(const uint8_t* data, size_t offset, uint8_t datatype, float& value) {
    if (datatype == sensor_msgs::msg::PointField::FLOAT32) {
        std::memcpy(&value, data + offset, sizeof(float));
        return true;
    }
    if (datatype == sensor_msgs::msg::PointField::FLOAT64) {
        double doubleValue = 0.0;
        std::memcpy(&doubleValue, data + offset, sizeof(double));
        value = static_cast<float>(doubleValue);
        return true;
    }
    return false;
}

bool LioNode::readUInt16_(const uint8_t* data, size_t offset, uint8_t datatype, uint16_t& value) {
    if (datatype == sensor_msgs::msg::PointField::UINT16
        || datatype == sensor_msgs::msg::PointField::INT16) {
        std::memcpy(&value, data + offset, sizeof(uint16_t));
        return true;
    }
    if (datatype == sensor_msgs::msg::PointField::UINT8
        || datatype == sensor_msgs::msg::PointField::INT8) {
        uint8_t byteValue = 0;
        std::memcpy(&byteValue, data + offset, sizeof(uint8_t));
        value = byteValue;
        return true;
    }
    return false;
}

bool LioNode::readUInt32_(const uint8_t* data, size_t offset, uint8_t datatype, uint32_t& value) {
    if (datatype == sensor_msgs::msg::PointField::UINT32
        || datatype == sensor_msgs::msg::PointField::INT32) {
        std::memcpy(&value, data + offset, sizeof(uint32_t));
        return true;
    }
    if (datatype == sensor_msgs::msg::PointField::UINT16
        || datatype == sensor_msgs::msg::PointField::INT16) {
        uint16_t shortValue = 0;
        std::memcpy(&shortValue, data + offset, sizeof(uint16_t));
        value = shortValue;
        return true;
    }
    return false;
}

} // arc

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("lio_node");
    arc::LioNode lioNode(node);
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
