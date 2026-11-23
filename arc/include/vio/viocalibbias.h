#pragma once

#include <Eigen/Dense>

namespace arc {

template <typename Scalar>
class CalibAccelBias {
public:
    using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
    using Mat33 = Eigen::Matrix<Scalar, 3, 3>;
    CalibAccelBias() { accelBiasFull_.setZero(); }
    void setRandom() {
        accelBiasFull_.setRandom();
        accelBiasFull_.template head<3>() /= 10;
        accelBiasFull_.template tail<6>() /= 100;
    }
    const Eigen::Matrix<Scalar, 9, 1>& getParam() const { return accelBiasFull_; }
    Eigen::Matrix<Scalar, 9, 1>& getParam() { return accelBiasFull_; }
    void operator+=(const Eigen::Matrix<Scalar, 9, 1>& inc) { accelBiasFull_ += inc; }
    void getBiasAndScale(Vec3& accelBias, Mat33& accelScale) const {
        accelBias = accelBiasFull_.template head<3>();
        accelScale.setZero();
        accelScale.col(0) = accelBiasFull_.template segment<3>(3);
        accelScale(1, 1) = accelBiasFull_(6);
        accelScale(2, 1) = accelBiasFull_(7);
        accelScale(2, 2) = accelBiasFull_(8);
    }
    Vec3 getCalibrated(const Vec3& rawMeasurement) const {
        Vec3 accelBias;
        Mat33 accelScale;
        getBiasAndScale(accelBias, accelScale);
        return (rawMeasurement + accelScale * rawMeasurement - accelBias);
    }
    Vec3 invertCalibration(const Vec3& calibratedMeasurement) const {
        Vec3 accelBias;
        Mat33 accelScale;
        getBiasAndScale(accelBias, accelScale);
        Mat33 accelScaleInv = (Eigen::Matrix3d::Identity() + accelScale).inverse();
        return accelScaleInv * (calibratedMeasurement + accelBias);
    }

private:
    Eigen::Matrix<Scalar, 9, 1> accelBiasFull_;
};

template <typename Scalar>
class CalibGyroBias {
public:
    using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
    using Mat33 = Eigen::Matrix<Scalar, 3, 3>;
    CalibGyroBias() { gyroBiasFull_.setZero(); }
    void setRandom() {
        gyroBiasFull_.setRandom();
        gyroBiasFull_.template head<3>() /= 10;
        gyroBiasFull_.template tail<9>() /= 100;
    }
    const Eigen::Matrix<Scalar, 12, 1>& getParam() const { return gyroBiasFull_; }
    Eigen::Matrix<Scalar, 12, 1>& getParam() { return gyroBiasFull_; }
    void operator+=(const Eigen::Matrix<Scalar, 12, 1>& inc) { gyroBiasFull_ += inc; }
    void getBiasAndScale(Vec3& gyroBias, Mat33& gyroScale) const {
        gyroBias = gyroBiasFull_.template head<3>();
        gyroScale.col(0) = gyroBiasFull_.template segment<3>(3);
        gyroScale.col(1) = gyroBiasFull_.template segment<3>(6);
        gyroScale.col(2) = gyroBiasFull_.template segment<3>(9);
    }
    Vec3 getCalibrated(const Vec3& rawMeasurement) const {
        Vec3 gyroBias;
        Mat33 gyroScale;
        getBiasAndScale(gyroBias, gyroScale);
        return (rawMeasurement + gyroScale * rawMeasurement - gyroBias);
    }
    Vec3 invertCalibration(const Vec3& calibratedMeasurement) const {
        Vec3 gyroBias;
        Mat33 gyroScale;
        getBiasAndScale(gyroBias, gyroScale);
        Mat33 gyroScaleInv = (Eigen::Matrix3d::Identity() + gyroScale).inverse();
        return gyroScaleInv * (calibratedMeasurement + gyroBias);
    }

private:
    Eigen::Matrix<Scalar, 12, 1> gyroBiasFull_;
};

} // arc
