#pragma once

#include <base/ros.h>

#include <base/iarc.h>
#include <arcmsgs/msg/position_command.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include <random>

namespace arc {

class OdometerPub {
public:
    OdometerPub(rclcpp::Node::SharedPtr node)
        : node_(std::move(node))
    {
        droneId_ = node_->declare_parameter<int>("drone_id", 1);
        initX_ = node_->declare_parameter<double>("init_x", 0.0);
        initY_ = node_->declare_parameter<double>("init_y", 0.0);
        initZ_ = node_->declare_parameter<double>("init_z", 1.0);
        publishRate_ = node_->declare_parameter<double>("publish_rate", 100.0);
        imuRate_ = node_->declare_parameter<double>("imu_rate", 200.0);
        imuAccelNoiseStd_ = node_->declare_parameter<std::vector<double>>(
            "imu_accel_noise_std", std::vector<double>{0.0, 0.0, 0.0});
        imuGyroNoiseStd_ = node_->declare_parameter<std::vector<double>>(
            "imu_gyro_noise_std", std::vector<double>{0.0, 0.0, 0.0});
        imuAccelBias_ = node_->declare_parameter<std::vector<double>>(
            "imu_accel_bias", std::vector<double>{0.0, 0.0, 0.0});
        imuGyroBias_ = node_->declare_parameter<std::vector<double>>(
            "imu_gyro_bias", std::vector<double>{0.0, 0.0, 0.0});
        rng_.seed(static_cast<uint32_t>(droneId_));

        cmdSub_ = node_->create_subscription<arcmsgs::msg::PositionCommand>(
            "command", 10,
            [this](const arcmsgs::msg::PositionCommand::SharedPtr msg) {
                onPositionCommand(*msg);
            }
        );
        odomPub_ = node_->create_publisher<nav_msgs::msg::Odometry>("odometry", 10);
        imuPub_ = node_->create_publisher<sensor_msgs::msg::Imu>("imu", 500);
        timer_ = node_->create_wall_timer(
            std::chrono::duration<double>(1.0 / std::max(publishRate_, 1e-3)),
            [this]() { publishOdom(); }
        );
        imuTimer_ = node_->create_wall_timer(
            std::chrono::duration<double>(1.0 / std::max(imuRate_, 1e-3)),
            [this]() { publishImu(); }
        );
    }

public:
    void onPositionCommand(const arcmsgs::msg::PositionCommand& cmd) {
        commandReceived_ = true;
        command_ = cmd;
    }

    void publishOdom() {
        nav_msgs::msg::Odometry odom;
        odom.header.stamp = node_->now();
        odom.header.frame_id = "world";
        odom.child_frame_id = std::to_string(droneId_);

        Eigen::Vector3d accelerationWorld(0.0, 0.0, 9.8);
        if (commandReceived_) {
            odom.pose.pose.position.x = command_.position.x;
            odom.pose.pose.position.y = command_.position.y;
            odom.pose.pose.position.z = command_.position.z;

            accelerationWorld << command_.acceleration.x,
                command_.acceleration.y,
                command_.acceleration.z + 9.8;
            Eigen::Vector3d xC(std::cos(command_.yaw), std::sin(command_.yaw), 0.0);
            Eigen::Vector3d yC(-std::sin(command_.yaw), std::cos(command_.yaw), 0.0);
            Eigen::Vector3d xB = yC.cross(accelerationWorld).normalized();
            Eigen::Vector3d yB = accelerationWorld.cross(xB).normalized();
            Eigen::Vector3d zB = xB.cross(yB);
            rotation_.col(0) = xB;
            rotation_.col(1) = yB;
            rotation_.col(2) = zB;

            const Eigen::Quaterniond q(rotation_);
            odom.pose.pose.orientation.w = q.w();
            odom.pose.pose.orientation.x = q.x();
            odom.pose.pose.orientation.y = q.y();
            odom.pose.pose.orientation.z = q.z();

            odom.twist.twist.linear.x = command_.velocity.x;
            odom.twist.twist.linear.y = command_.velocity.y;
            odom.twist.twist.linear.z = command_.velocity.z;
        } else {
            rotation_.setIdentity();
            odom.pose.pose.position.x = initX_;
            odom.pose.pose.position.y = initY_;
            odom.pose.pose.position.z = initZ_;
            odom.pose.pose.orientation.w = 1.0;
            odom.pose.pose.orientation.x = 0.0;
            odom.pose.pose.orientation.y = 0.0;
            odom.pose.pose.orientation.z = 0.0;
        }

        accelBody_ = rotation_.transpose() * accelerationWorld;
        odomPub_->publish(odom);
    }

    void publishImu() {
        const rclcpp::Time now = node_->now();
        Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
        if (hasLastImuTime_) {
            const double dt = (now - lastImuTime_).seconds();
            if (dt > 1e-9) {
                const Eigen::Matrix3d delta = lastImuRotation_.transpose() * rotation_;
                const Eigen::AngleAxisd angleAxis(delta);
                if (std::abs(angleAxis.angle()) > 1e-12) {
                    gyro = angleAxis.axis() * angleAxis.angle() / dt;
                }
            }
        }

        const Eigen::Vector3d accel = accelBody_ + vectorParam_(imuAccelBias_, 0.0) +
            Eigen::Vector3d(
                imuAccelNoiseStd_[0] * normal_(rng_),
                imuAccelNoiseStd_[1] * normal_(rng_),
                imuAccelNoiseStd_[2] * normal_(rng_));
        const Eigen::Vector3d gyroMeasured = gyro + vectorParam_(imuGyroBias_, 0.0) +
            Eigen::Vector3d(
                imuGyroNoiseStd_[0] * normal_(rng_),
                imuGyroNoiseStd_[1] * normal_(rng_),
                imuGyroNoiseStd_[2] * normal_(rng_));
        const Eigen::Quaterniond orientation(rotation_);

        sensor_msgs::msg::Imu imu;
        imu.header.stamp = now;
        imu.header.frame_id = "imu_" + std::to_string(droneId_);
        imu.orientation.w = orientation.w();
        imu.orientation.x = orientation.x();
        imu.orientation.y = orientation.y();
        imu.orientation.z = orientation.z();
        imu.linear_acceleration.x = accel.x();
        imu.linear_acceleration.y = accel.y();
        imu.linear_acceleration.z = accel.z();
        imu.angular_velocity.x = gyroMeasured.x();
        imu.angular_velocity.y = gyroMeasured.y();
        imu.angular_velocity.z = gyroMeasured.z();
        imuPub_->publish(imu);

        lastImuRotation_ = rotation_;
        lastImuTime_ = now;
        hasLastImuTime_ = true;
    }

private:
    static Eigen::Vector3d vectorParam_(
        const std::vector<double>& values, double defaultValue) {
        if (values.size() != 3) {
            return Eigen::Vector3d::Constant(defaultValue);
        }
        return Eigen::Vector3d(values[0], values[1], values[2]);
    }

private:
    rclcpp::Node::SharedPtr node_;
    rclcpp::Subscription<arcmsgs::msg::PositionCommand>::SharedPtr cmdSub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odomPub_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imuPub_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::TimerBase::SharedPtr imuTimer_;
    arcmsgs::msg::PositionCommand command_;
    bool commandReceived_{false};
    int droneId_{1};
    double initX_{0.0};
    double initY_{0.0};
    double initZ_{1.0};
    double publishRate_{100.0};
    double imuRate_{200.0};
    std::vector<double> imuAccelNoiseStd_{0.0, 0.0, 0.0};
    std::vector<double> imuGyroNoiseStd_{0.0, 0.0, 0.0};
    std::vector<double> imuAccelBias_{0.0, 0.0, 0.0};
    std::vector<double> imuGyroBias_{0.0, 0.0, 0.0};
    Eigen::Matrix3d rotation_{Eigen::Matrix3d::Identity()};
    Eigen::Matrix3d lastImuRotation_{Eigen::Matrix3d::Identity()};
    Eigen::Vector3d accelBody_{0.0, 0.0, 9.8};
    rclcpp::Time lastImuTime_{0, 0, RCL_ROS_TIME};
    bool hasLastImuTime_{false};
    std::mt19937 rng_;
    std::normal_distribution<double> normal_{0.0, 1.0};
};

} // namespace arc
