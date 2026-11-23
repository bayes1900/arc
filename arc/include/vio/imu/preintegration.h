#pragma once

#include <vio/imu/state.h>
#include <base/base.h>
#include <base/liegroup.h>

namespace arc {

class IntegratedImuMeasurement {
public:
    using Ptr = std::shared_ptr<IntegratedImuMeasurement>;
    using MatNN = Eigen::Matrix<double, 9, 9>;
    using MatN3 = Eigen::Matrix<double, 9, 3>;
    using MatN6 = Eigen::Matrix<double, 9, 6>;

    static void propagateState(
        const PoseVelState& currState,
        const ImuData& data,
        PoseVelState& nextState,
        MatNN* dNextDCurr = nullptr,
        MatN3* dNextDAccel = nullptr,
        MatN3* dNextDGyro = nullptr
    ) {
        ensure(data.tNs > currState.tNs, "IMU timestamps must increase");
        int64_t dtNs = data.tNs - currState.tNs;
        double dt = dtNs * double(1e-9);
        Sophus::SO3d R_w_i_new_2 = currState.tWI.so3() * Sophus::SO3d::exp(double(0.5) * dt * data.gyro);
        Mat3d RR_w_i_new_2 = R_w_i_new_2.matrix();
        Vec3d accelWorld = RR_w_i_new_2 * data.accel;
        nextState.tNs = data.tNs;
        nextState.tWI.so3() = currState.tWI.so3() * Sophus::SO3d::exp(dt * data.gyro);
        nextState.velWI = currState.velWI + accelWorld * dt;
        nextState.tWI.translation() = currState.tWI.translation() + currState.velWI * dt + 0.5 * accelWorld * dt * dt;
        if (dNextDCurr) {
            dNextDCurr->setIdentity();
            dNextDCurr->template block<3, 3>(0, 6).diagonal().setConstant(dt);
            dNextDCurr->template block<3, 3>(6, 3) = Sophus::SO3d::hat(-accelWorld * dt);
            dNextDCurr->template block<3, 3>(0, 3) = dNextDCurr->template block<3, 3>(6, 3) * dt * double(0.5);
        }
        if (dNextDAccel) {
            dNextDAccel->setZero();
            dNextDAccel->template block<3, 3>(0, 0) = double(0.5) * RR_w_i_new_2 * dt * dt;
            dNextDAccel->template block<3, 3>(6, 0) = RR_w_i_new_2 * dt;
        }
        if (dNextDGyro) {
            dNextDGyro->setZero();
            Mat3d Jr;
            rightJacobianSO3(dt * data.gyro, Jr);
            Mat3d Jr2;
            rightJacobianSO3(double(0.5) * dt * data.gyro, Jr2);
            dNextDGyro->template block<3, 3>(3, 0) = nextState.tWI.so3().matrix() * Jr * dt;
            dNextDGyro->template block<3, 3>(6, 0) = Sophus::SO3d::hat(-accelWorld * dt) * RR_w_i_new_2 * Jr2 * double(0.5) * dt;
            dNextDGyro->template block<3, 3>(0, 0) = double(0.5) * dt * dNextDGyro->template block<3, 3>(6, 0);
        }
    }
    IntegratedImuMeasurement() noexcept {
        cov_.setZero();
        dStateDBa_.setZero();
        dStateDBg_.setZero();
        biasGyroLin_.setZero();
        biasAccelLin_.setZero();
        sqrtCovInv_.setZero();
    }
    IntegratedImuMeasurement(int64_t startTNs, const Vec3d& biasGyroLin, const Vec3d& biasAccelLin) noexcept
        : startTNs_(startTNs)
        , biasGyroLin_(biasGyroLin)
        , biasAccelLin_(biasAccelLin)
    {
        cov_.setZero();
        dStateDBa_.setZero();
        dStateDBg_.setZero();
        sqrtCovInv_.setZero();
    }
    void integrate(const ImuData& data, const Vec3d& accelCov, const Vec3d& gyroCov) {
        ImuData dataCorrected = data;
        dataCorrected.tNs -= startTNs_;
        dataCorrected.accel -= biasAccelLin_;
        dataCorrected.gyro -= biasGyroLin_;
        PoseVelState newState;
        MatNN F;
        MatN3 A;
        MatN3 G;
        propagateState(deltaState_, dataCorrected, newState, &F, &A, &G);
        deltaState_ = newState;
        cov_ = F * cov_ * F.transpose() + A * accelCov.asDiagonal() * A.transpose() + G * gyroCov.asDiagonal() * G.transpose();
        sqrtCovInvComputed_ = false;
        dStateDBa_ = -A + F * dStateDBa_;
        dStateDBg_ = -G + F * dStateDBg_;
    }
    void predictState(const PoseVelState& state0, const Vec3d& g, PoseVelState& state1) const {
        double dt = deltaState_.tNs * double(1e-9);
        state1.tWI.so3() = state0.tWI.so3() * deltaState_.tWI.so3();
        state1.velWI = state0.velWI + g * dt + state0.tWI.so3() * deltaState_.velWI;
        state1.tWI.translation() = state0.tWI.translation() + state0.velWI * dt + double(0.5) * g * dt * dt + state0.tWI.so3() * deltaState_.tWI.translation();
    }
    Vec9d residual(
        const PoseVelState& state0,
        const Vec3d& g,
        const PoseVelState& state1,
        const Vec3d& currBg,
        const Vec3d& currBa,
        MatNN* dResDState0 = nullptr,
        MatNN* dResDState1 = nullptr,
        MatN3* dResDBg = nullptr,
        MatN3* dResDBa = nullptr
    ) const {
        double dt = deltaState_.tNs * double(1e-9);
        Vec9d res;
        Vec9d bgDiff;
        Vec9d baDiff;
        bgDiff = dStateDBg_ * (currBg - biasGyroLin_);
        baDiff = dStateDBa_ * (currBa - biasAccelLin_);
        ensure(baDiff.segment<3>(3).isApproxToConstant(0), "VIO check failed");
        Mat3d R0_inv = state0.tWI.so3().inverse().matrix();
        Vec3d tmp = R0_inv * (state1.tWI.translation() - state0.tWI.translation() - state0.velWI * dt - double(0.5) * g * dt * dt);
        res.segment<3>(0) = tmp - (deltaState_.tWI.translation() + bgDiff.segment<3>(0) + baDiff.segment<3>(0));
        res.segment<3>(3) = (Sophus::SO3d::exp(bgDiff.segment<3>(3)) * deltaState_.tWI.so3() * state1.tWI.so3().inverse() * state0.tWI.so3()).log();
        Vec3d tmp2 = R0_inv * (state1.velWI - state0.velWI - g * dt);
        res.segment<3>(6) = tmp2 - (deltaState_.velWI + bgDiff.segment<3>(6) + baDiff.segment<3>(6));
        if (dResDState0 || dResDState1) {
            Mat3d J;
            rightJacobianInvSO3(res.segment<3>(3), J);
            if (dResDState0) {
                dResDState0->setZero();
                dResDState0->template block<3, 3>(0, 0) = -R0_inv;
                dResDState0->template block<3, 3>(0, 3) = Sophus::SO3d::hat(tmp) * R0_inv;
                dResDState0->template block<3, 3>(3, 3) = J * R0_inv;
                dResDState0->template block<3, 3>(6, 3) = Sophus::SO3d::hat(tmp2) * R0_inv;
                dResDState0->template block<3, 3>(0, 6) = -R0_inv * dt;
                dResDState0->template block<3, 3>(6, 6) = -R0_inv;
            }
            if (dResDState1) {
                dResDState1->setZero();
                dResDState1->template block<3, 3>(0, 0) = R0_inv;
                dResDState1->template block<3, 3>(3, 3) = -J * R0_inv;
                dResDState1->template block<3, 3>(6, 6) = R0_inv;
            }
        }
        if (dResDBa) {
            *dResDBa = -dStateDBa_;
        }
        if (dResDBg) {
            dResDBg->setZero();
            *dResDBg = -dStateDBg_;
            Mat3d J;
            leftJacobianInvSO3(res.segment<3>(3), J);
            dResDBg->template block<3, 3>(3, 0) = J * dStateDBg_.block<3, 3>(3, 0);
        }
        return res;
    }
    int64_t getDtNs() const { return deltaState_.tNs; }
    int64_t getStartTNs() const { return startTNs_; }
    MatNN getCovInv() const {
        if (!sqrtCovInvComputed_) {
            computeSqrtCovInv_();
            sqrtCovInvComputed_ = true;
        }
        return sqrtCovInv_.transpose() * sqrtCovInv_;
    }
    const MatNN& getSqrtCovInv() const {
        if (!sqrtCovInvComputed_) {
            computeSqrtCovInv_();
            sqrtCovInvComputed_ = true;
        }
        return sqrtCovInv_;
    }
    const MatNN& getCov() const { return cov_; }
    const PoseVelState& getDeltaState() const { return deltaState_; }
    const MatN3& getDStateDBa() const { return dStateDBa_; }
    const MatN3& getDStateDBg() const { return dStateDBg_; }

private:
    void computeSqrtCovInv_() const {
        sqrtCovInv_.setIdentity();
        auto ldlt = cov_.ldlt();
        sqrtCovInv_ = ldlt.transpositionsP() * sqrtCovInv_;
        ldlt.matrixL().solveInPlace(sqrtCovInv_);
        Vec9d D_inv_sqrt;
        for (size_t i = 0; i < 9; ++ i) {
            if (ldlt.vectorD()[i] < std::numeric_limits<double>::min()) {
                D_inv_sqrt[i] = 0;
            } else {
                D_inv_sqrt[i] = double(1.0) / sqrt(ldlt.vectorD()[i]);
            }
        }
        sqrtCovInv_ = D_inv_sqrt.asDiagonal() * sqrtCovInv_;
    }

private:
    int64_t startTNs_{0};
    PoseVelState deltaState_;
    MatNN cov_;
    mutable MatNN sqrtCovInv_;
    mutable bool sqrtCovInvComputed_{false};
    MatN3 dStateDBa_, dStateDBg_;
    Vec3d biasGyroLin_, biasAccelLin_;
};

} // arc
