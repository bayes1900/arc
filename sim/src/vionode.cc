#include "vionode.h"

#include <algorithm>
#include <limits>
#include <cmath>
#include <stdexcept>

#include <opencv2/imgproc.hpp>
#include <sensor_msgs/image_encodings.hpp>

#if __has_include(<cv_bridge/cv_bridge.hpp>)
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif

namespace arc {

VioNode::VioNode(rclcpp::Node::SharedPtr node)
    : node_(std::move(node))
{
    loadParams_();
    loadCalibration_();
    pipeline_ = std::make_unique<VioPipeline>(config_, calibration_);
    initRos_();
}

VioNode::~VioNode() {
    if (pipeline_) {
        pipeline_->stop();
    }
}

void VioNode::loadParams_() {
    configFile_ = node_->declare_parameter<std::string>("config_file", "");
    calibFile_ = node_->declare_parameter<std::string>("calib_file", "");
    imageTopics_ = node_->declare_parameter<std::vector<std::string>>(
        "image_topics", std::vector<std::string>{"/sensor/image_1"});
    imuTopic_ = node_->declare_parameter<std::string>("imu_topic", "/sensor/imu_1");
    worldFrame_ = node_->declare_parameter<std::string>("world_frame", "world");
    bodyFrame_ = node_->declare_parameter<std::string>("body_frame", "imu");
    publishTf_ = node_->declare_parameter<bool>("publish_tf", true);
    imageSyncTolerance_ = node_->declare_parameter<double>(
        "image_sync_tolerance", 0.005);
    initialPose_ = node_->declare_parameter<std::vector<double>>(
        "initial_pose", std::vector<double>{});

    pendingImages_.resize(imageTopics_.size());
    pendingImageNs_.resize(imageTopics_.size(), 0);
    hasInitialPose_ = initialPose_.size() == 6;
    if (!configFile_.empty()) {
        config_ = VioIo::loadConfig(configFile_);
    }
}

void VioNode::loadCalibration_() {
    if (!calibFile_.empty()) {
        calibration_ = VioIo::loadCalibration(calibFile_);
        return;
    }

    RCLCPP_WARN(
        node_->get_logger(),
        "calib_file is empty; using default pinhole calibration without extrinsics");
    auto camera = GenericCamera<double>::fromString("pinhole");
    camera.setFromInit(Eigen::Vector4d(387.229, 387.229, 321.046, 243.449));
    calibration_.intrinsics.push_back(camera);
    calibration_.resolution.emplace_back(640, 480);
    calibration_.tIC.emplace_back(Sophus::SE3d());
}

void VioNode::initRos_() {
    if (imageTopics_.size() < calibration_.intrinsics.size()) {
        RCLCPP_WARN(
            node_->get_logger(),
            "fewer image topics (%zu) than calibrated cameras (%zu); VIO will not start",
            imageTopics_.size(), calibration_.intrinsics.size());
    }

    if (imageTopics_.size() > calibration_.intrinsics.size()) {
        RCLCPP_WARN(
            node_->get_logger(),
            "more image topics (%zu) than calibrated cameras (%zu); ignoring extras",
            imageTopics_.size(), calibration_.intrinsics.size());
        imageTopics_.resize(calibration_.intrinsics.size());
        pendingImages_.resize(calibration_.intrinsics.size());
        pendingImageNs_.resize(calibration_.intrinsics.size(), 0);
    }

    const auto qos = rclcpp::QoS(50);
    for (size_t i = 0; i < imageTopics_.size(); ++ i) {
        imageSubs_.push_back(node_->create_subscription<sensor_msgs::msg::Image>(
            imageTopics_[i], qos,
            [this, i](const sensor_msgs::msg::Image::SharedPtr msg) {
                imageCallback_(msg, i);
            }
        ));
    }

    imuSub_ = node_->create_subscription<sensor_msgs::msg::Imu>(
        imuTopic_, rclcpp::QoS(500),
        [this](const sensor_msgs::msg::Imu::SharedPtr msg) {
            imuCallback_(msg);
        }
    );

    odomPub_ = node_->create_publisher<nav_msgs::msg::Odometry>("odom", 100);
    if (publishTf_) {
        tfBroadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(node_);
    }
    stateTimer_ = node_->create_wall_timer(
        std::chrono::milliseconds(10),
        [this]() { publishState_(); }
    );
}

ManagedImage<uint16_t>::Ptr VioNode::convertImage_(
    const sensor_msgs::msg::Image& msg) {
    cv_bridge::CvImageConstPtr bridge;
    try {
        bridge = cv_bridge::toCvCopy(msg, msg.encoding);
    } catch (const cv_bridge::Exception& error) {
        throw std::runtime_error(std::string("cv_bridge: ") + error.what());
    }

    cv::Mat gray;
    if (bridge->image.channels() == 3) {
        const int code = msg.encoding == sensor_msgs::image_encodings::RGB8
            ? cv::COLOR_RGB2GRAY
            : cv::COLOR_BGR2GRAY;
        cv::cvtColor(bridge->image, gray, code);
    } else {
        gray = bridge->image;
    }

    auto image = std::make_shared<ManagedImage<uint16_t>>(
        static_cast<size_t>(gray.cols), static_cast<size_t>(gray.rows));
    if (gray.type() == CV_16UC1) {
        for (int y = 0; y < gray.rows; ++ y) {
            const uint16_t* src = gray.ptr<uint16_t>(y);
            uint16_t* dst = image->rowPtr(static_cast<size_t>(y));
            std::copy(src, src + gray.cols, dst);
        }
    } else if (gray.type() == CV_8UC1) {
        for (int y = 0; y < gray.rows; ++ y) {
            const uint8_t* src = gray.ptr<uint8_t>(y);
            uint16_t* dst = image->rowPtr(static_cast<size_t>(y));
            for (int x = 0; x < gray.cols; ++ x) {
                dst[x] = static_cast<uint16_t>(src[x]) << 8;
            }
        }
    } else {
        throw std::runtime_error("unsupported VIO image type");
    }
    return image;
}

void VioNode::imageCallback_(
    const sensor_msgs::msg::Image::SharedPtr msg, size_t cameraIndex) {
    if (cameraIndex >= calibration_.intrinsics.size() ||
        cameraIndex >= pendingImages_.size()) {
        return;
    }

    const int64_t tNs = static_cast<int64_t>(msg->header.stamp.sec) * 1000000000LL +
        static_cast<int64_t>(msg->header.stamp.nanosec);
    if (tNs <= 0) {
        return;
    }
    pendingImages_[cameraIndex] = msg;
    pendingImageNs_[cameraIndex] = tNs;
    try {
        tryDispatchImage_();
    } catch (const std::exception& error) {
        RCLCPP_ERROR(node_->get_logger(), "VIO image callback failed: %s", error.what());
    }
}

void VioNode::tryDispatchImage_() {
    if (pendingImages_.empty()) {
        return;
    }
    int64_t minNs = std::numeric_limits<int64_t>::max();
    int64_t maxNs = std::numeric_limits<int64_t>::min();
    for (size_t i = 0; i < pendingImages_.size(); ++ i) {
        if (!pendingImages_[i]) {
            return;
        }
        minNs = std::min(minNs, pendingImageNs_[i]);
        maxNs = std::max(maxNs, pendingImageNs_[i]);
    }

    const int64_t toleranceNs = static_cast<int64_t>(
        imageSyncTolerance_ * 1e9);
    if (maxNs - minNs > toleranceNs) {
        if (maxNs - minNs > 100000000LL) {
            for (auto& image : pendingImages_) {
                image.reset();
            }
        }
        return;
    }

    const int64_t tNs = pendingImageNs_[0];
    if (tNs <= lastImageNs_) {
        for (auto& image : pendingImages_) {
            image.reset();
        }
        return;
    }
    lastImageNs_ = tNs;
    if (!started_) {
        if (hasInitialPose_) {
            const Eigen::Vector3d translation(
                initialPose_[0], initialPose_[1], initialPose_[2]);
            const Eigen::Matrix3d rotation =
                (Eigen::AngleAxisd(initialPose_[5], Eigen::Vector3d::UnitZ()) *
                 Eigen::AngleAxisd(initialPose_[4], Eigen::Vector3d::UnitY()) *
                 Eigen::AngleAxisd(initialPose_[3], Eigen::Vector3d::UnitX()))
                    .toRotationMatrix();
            pipeline_->start(
                tNs, Sophus::SE3d(rotation, translation),
                Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
                Eigen::Vector3d::Zero());
        } else {
            pipeline_->startAutoAt(
                tNs, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
        }
        started_ = true;
    }

    auto input = std::make_shared<OpticalFlowInput>();
    input->tNs = tNs;
    input->imgData.resize(calibration_.intrinsics.size());
    try {
        for (size_t i = 0; i < pendingImages_.size(); ++ i) {
            input->imgData[i].img = convertImage_(*pendingImages_[i]);
            pendingImages_[i].reset();
        }
    } catch (const std::exception& error) {
        RCLCPP_ERROR(node_->get_logger(), "%s", error.what());
        return;
    }
    try {
        pipeline_->addImage(input);
    } catch (const std::exception& error) {
        RCLCPP_ERROR(node_->get_logger(), "VIO addImage failed: %s", error.what());
    }
}

void VioNode::imuCallback_(const sensor_msgs::msg::Imu::SharedPtr msg) {
    const int64_t tNs = static_cast<int64_t>(msg->header.stamp.sec) * 1000000000LL +
        static_cast<int64_t>(msg->header.stamp.nanosec);
    if (tNs <= 0) {
        return;
    }
    auto data = std::make_shared<ImuData>();
    data->tNs = tNs;
    data->accel = Eigen::Vector3d(
        msg->linear_acceleration.x,
        msg->linear_acceleration.y,
        msg->linear_acceleration.z);
    data->gyro = Eigen::Vector3d(
        msg->angular_velocity.x,
        msg->angular_velocity.y,
        msg->angular_velocity.z);
    try {
        pipeline_->addImu(data);
    } catch (const std::exception& error) {
        RCLCPP_ERROR(node_->get_logger(), "VIO addImu failed: %s", error.what());
    }
}

void VioNode::publishState_() {
    PoseVelBiasState::Ptr state;
    while (pipeline_->tryPopState(state)) {
        lastState_ = state;
    }
    if (!lastState_) {
        return;
    }

    const rclcpp::Time stamp(lastState_->tNs);
    const Eigen::Quaterniond orientation(lastState_->tWI.so3().matrix());

    nav_msgs::msg::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = worldFrame_;
    odom.child_frame_id = bodyFrame_;
    odom.pose.pose.position.x = lastState_->tWI.translation().x();
    odom.pose.pose.position.y = lastState_->tWI.translation().y();
    odom.pose.pose.position.z = lastState_->tWI.translation().z();
    odom.pose.pose.orientation.w = orientation.w();
    odom.pose.pose.orientation.x = orientation.x();
    odom.pose.pose.orientation.y = orientation.y();
    odom.pose.pose.orientation.z = orientation.z();
    odom.twist.twist.linear.x = lastState_->velWI.x();
    odom.twist.twist.linear.y = lastState_->velWI.y();
    odom.twist.twist.linear.z = lastState_->velWI.z();
    for (size_t i = 0; i < 6; ++ i) {
        odom.pose.covariance[i * 6 + i] = 0.01;
        odom.twist.covariance[i * 6 + i] = 0.01;
    }
    odomPub_->publish(odom);

    if (tfBroadcaster_) {
        geometry_msgs::msg::TransformStamped transform;
        transform.header = odom.header;
        transform.child_frame_id = bodyFrame_;
        transform.transform.translation.x = odom.pose.pose.position.x;
        transform.transform.translation.y = odom.pose.pose.position.y;
        transform.transform.translation.z = odom.pose.pose.position.z;
        transform.transform.rotation = odom.pose.pose.orientation;
        tfBroadcaster_->sendTransform(transform);
    }
}

} // arc

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("vio_node");
    arc::VioNode vioNode(node);
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
