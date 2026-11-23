#pragma once

#include <base/iarc.h>

namespace arc {

struct FovParams {
    double topAngle{0.56125};
    double leftAngle{0.69222};
    double rightAngle{0.68901};
    double maxDist{4.5};
    double visDist{1.0};
};

class FovModel {
public:
    explicit FovModel(const FovParams& params = FovParams())
        : params_(params)
    {
        const double halfPi = 0.5 * __pi__;
        nTop_ << 0.0, std::sin(halfPi - params_.topAngle), std::cos(halfPi - params_.topAngle);
        nBottom_ << 0.0, -std::sin(halfPi - params_.topAngle), std::cos(halfPi - params_.topAngle);
        nLeft_ << std::sin(halfPi - params_.leftAngle), 0.0, std::cos(halfPi - params_.leftAngle);
        nRight_ << -std::sin(halfPi - params_.rightAngle), 0.0, std::cos(halfPi - params_.rightAngle);

        T_cb_ << 0.0, -1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0;
        T_bc_ = T_cb_.inverse();

        const double horizontal = params_.visDist * std::tan(params_.leftAngle);
        const double vertical = params_.visDist * std::tan(params_.topAngle);
        const Eigen::Vector3d origin = Eigen::Vector3d::Zero();
        const Eigen::Vector3d leftUp(params_.visDist, horizontal, vertical);
        const Eigen::Vector3d leftDown(params_.visDist, horizontal, -vertical);
        const Eigen::Vector3d rightUp(params_.visDist, -horizontal, vertical);
        const Eigen::Vector3d rightDown(params_.visDist, -horizontal, -vertical);

        camVertices1_ = {origin, origin, origin, origin, leftUp, rightUp, rightDown, leftDown};
        camVertices2_ = {leftUp, leftDown, rightUp, rightDown, rightUp, rightDown, leftDown, leftUp};
    }
    void setPose(const Eigen::Vector3d& pos, double yaw) {
        pos_ = pos;
        yaw_ = yaw;
        Eigen::Matrix3d rotation;
        rotation << std::cos(yaw), -std::sin(yaw), 0.0,
            std::sin(yaw), std::cos(yaw), 0.0,
            0.0, 0.0, 1.0;
        const Eigen::Matrix3d cameraRotation = rotation * T_bc_.block<3, 3>(0, 0);
        worldTop_ = cameraRotation * nTop_;
        worldBottom_ = cameraRotation * nBottom_;
        worldLeft_ = cameraRotation * nLeft_;
        worldRight_ = cameraRotation * nRight_;
    }
    bool insideFOV(const Eigen::Vector3d& point) const {
        const Eigen::Vector3d direction = point - pos_;
        const double distance = direction.norm();
        if (distance > params_.maxDist || distance < 1e-9) {
            return false;
        }
        const Eigen::Vector3d unit = direction / distance;
        return unit.dot(worldTop_) >= 0.0 &&
            unit.dot(worldBottom_) >= 0.0 &&
            unit.dot(worldLeft_) >= 0.0 &&
            unit.dot(worldRight_) >= 0.0;
    }
    void getFOV(std::vector<Eigen::Vector3d>& list1, std::vector<Eigen::Vector3d>& list2) const {
        list1.clear();
        list2.clear();
        list1.reserve(camVertices1_.size());
        list2.reserve(camVertices2_.size());
        Eigen::Matrix3d rotation;
        rotation << std::cos(yaw_), -std::sin(yaw_), 0.0,
            std::sin(yaw_), std::cos(yaw_), 0.0,
            0.0, 0.0, 1.0;
        for (size_t i = 0; i < camVertices1_.size(); ++ i) {
            list1.push_back(rotation * camVertices1_[i] + pos_);
            list2.push_back(rotation * camVertices2_[i] + pos_);
        }
    }
    void getFOVBoundingBox(Eigen::Vector3d& bmin, Eigen::Vector3d& bmax) const {
        const double left = yaw_ + params_.leftAngle;
        const double right = yaw_ - params_.rightAngle;
        const Eigen::Vector3d leftPoint = pos_ + params_.maxDist * Eigen::Vector3d(std::cos(left), std::sin(left), 0.0);
        const Eigen::Vector3d rightPoint = pos_ + params_.maxDist * Eigen::Vector3d(std::cos(right), std::sin(right), 0.0);

        std::vector<Eigen::Vector3d> points = {leftPoint, rightPoint};
        if (left > 0.0 && right < 0.0) {
            points.push_back(pos_ + params_.maxDist * Eigen::Vector3d(1.0, 0.0, 0.0));
        } else if (left > 0.5 * __pi__ && right < 0.5 * __pi__) {
            points.push_back(pos_ + params_.maxDist * Eigen::Vector3d(0.0, 1.0, 0.0));
        } else if (left > -0.5 * __pi__ && right < -0.5 * __pi__) {
            points.push_back(pos_ + params_.maxDist * Eigen::Vector3d(0.0, -1.0, 0.0));
        } else if ((left > __pi__ && right < __pi__) || (left > -__pi__ && right < -__pi__)) {
            points.push_back(pos_ + params_.maxDist * Eigen::Vector3d(-1.0, 0.0, 0.0));
        }
        bmin = pos_;
        bmax = pos_;
        for (const auto& point : points) {
            bmin = bmin.cwiseMin(point);
            bmax = bmax.cwiseMax(point);
        }
    }

private:
    FovParams params_;
    Eigen::Vector3d pos_{Eigen::Vector3d::Zero()};
    double yaw_{0.0};

    Eigen::Matrix4d T_cb_{Eigen::Matrix4d::Identity()};
    Eigen::Matrix4d T_bc_{Eigen::Matrix4d::Identity()};

    Eigen::Vector3d nTop_{Eigen::Vector3d::Zero()};
    Eigen::Vector3d nBottom_{Eigen::Vector3d::Zero()};
    Eigen::Vector3d nLeft_{Eigen::Vector3d::Zero()};
    Eigen::Vector3d nRight_{Eigen::Vector3d::Zero()};

    Eigen::Vector3d worldTop_{Eigen::Vector3d::Zero()};
    Eigen::Vector3d worldBottom_{Eigen::Vector3d::Zero()};
    Eigen::Vector3d worldLeft_{Eigen::Vector3d::Zero()};
    Eigen::Vector3d worldRight_{Eigen::Vector3d::Zero()};

    std::vector<Eigen::Vector3d> camVertices1_;
    std::vector<Eigen::Vector3d> camVertices2_;
};

} // arc
