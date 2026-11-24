#pragma once

#include <vio/imu/preintegration.h>
#include <vio/estimator/accumulator.h>
#include <vio/estimator/state.h>

namespace arc {

class ImuBlock {
public:
    ImuBlock(
        const IntegratedImuMeasurement* meas, const ImuLinData* imuLinData, const AbsOrderMap& aom)
        : imuMeas_(meas)
, imuLinData_(imuLinData)
, aom_(aom)
    {
        jp_.resize(15, 2 * 15);
        r_.resize(15);
    }
    double linearizeImu(const std::map<int64_t, PoseVelBiasStateWithLin>& frameStates) {
        jp_.setZero();
        r_.setZero();
        const int64_t startT = imuMeas_->getStartTNs();
        const int64_t endT = imuMeas_->getStartTNs() + imuMeas_->getDtNs();
        const size_t startIdx = 0;
        const size_t endIdx = 15;
        PoseVelBiasStateWithLin startState = frameStates.at(startT);
        PoseVelBiasStateWithLin endState = frameStates.at(endT);
        IntegratedImuMeasurement::MatNN dResDStart, dResDEnd;
        IntegratedImuMeasurement::MatN3 dResDBg, dResDBa;
        Vec9d res = imuMeas_->residual(startState.getStateLin(),
            imuLinData_->g,
            endState.getStateLin(),
            startState.getStateLin().biasGyro,
            startState.getStateLin().biasAccel,
            &dResDStart,
            &dResDEnd,
            &dResDBg,
            &dResDBa);
        if (startState.isLinearized() || endState.isLinearized()) {
            res = imuMeas_->residual(startState.getState(),
                imuLinData_->g,
                endState.getState(),
                startState.getState().biasGyro,
                startState.getState().biasAccel);
        }
        double imuError = double(0.5) * (imuMeas_->getSqrtCovInv() * res).squaredNorm();
        jp_.block<9, 9>(0, startIdx) = imuMeas_->getSqrtCovInv() * dResDStart;
        jp_.block<9, 9>(0, endIdx) = imuMeas_->getSqrtCovInv() * dResDEnd;
        jp_.block<9, 3>(0, startIdx + 9) = imuMeas_->getSqrtCovInv() * dResDBg;
        jp_.block<9, 3>(0, startIdx + 12) = imuMeas_->getSqrtCovInv() * dResDBa;
        r_.segment<9>(0) = imuMeas_->getSqrtCovInv() * res;
        double dt = imuMeas_->getDtNs() * double(1e-9);
        Vec3d gyroBiasWeightDt = imuLinData_->gyroBiasWeightSqrt / std::sqrt(dt);
        Vec3d resBg = startState.getState().biasGyro - endState.getState().biasGyro;
        jp_.block<3, 3>(9, startIdx + 9) = gyroBiasWeightDt.asDiagonal();
        jp_.block<3, 3>(9, endIdx + 9) = (-gyroBiasWeightDt).asDiagonal();
        r_.segment<3>(9) += gyroBiasWeightDt.asDiagonal() * resBg;
        double bgError = double(0.5) * (gyroBiasWeightDt.asDiagonal() * resBg).squaredNorm();
        Vec3d accelBiasWeightDt = imuLinData_->accelBiasWeightSqrt / std::sqrt(dt);
        Vec3d resBa = startState.getState().biasAccel - endState.getState().biasAccel;
        jp_.block<3, 3>(12, startIdx + 12) = accelBiasWeightDt.asDiagonal();
        jp_.block<3, 3>(12, endIdx + 12) = (-accelBiasWeightDt).asDiagonal();
        r_.segment<3>(12) += accelBiasWeightDt.asDiagonal() * resBa;
        double baError = double(0.5) * (accelBiasWeightDt.asDiagonal() * resBa).squaredNorm();
        return imuError + bgError + baError;
    }
    void addDenseQ2JpQ2r(MatXd& Q2Jp, VecXd& Q2r, size_t rowStartIdx) const {
        int64_t startT = imuMeas_->getStartTNs();
        int64_t endT = imuMeas_->getStartTNs() + imuMeas_->getDtNs();
        const size_t startIdx = aom_.absOrderMap.at(startT).first;
        const size_t endIdx = aom_.absOrderMap.at(endT).first;
        Q2Jp.block<15, 15>(rowStartIdx, startIdx) += jp_.topLeftCorner<15, 15>();
        Q2Jp.block<15, 15>(rowStartIdx, endIdx) += jp_.topRightCorner<15, 15>();
        Q2r.segment<15>(rowStartIdx) += r_;
    }
    void addDenseHB(DenseAccumulator& accum) const {
        int64_t startT = imuMeas_->getStartTNs();
        int64_t endT = imuMeas_->getStartTNs() + imuMeas_->getDtNs();
        const size_t startIdx = aom_.absOrderMap.at(startT).first;
        const size_t endIdx = aom_.absOrderMap.at(endT).first;
        const MatXd H = jp_.transpose() * jp_;
        const VecXd b = jp_.transpose() * r_;
        accum.addH<15, 15>(startIdx, startIdx, H.block<15, 15>(0, 0));
        accum.addH<15, 15>(endIdx, startIdx, H.block<15, 15>(15, 0));
        accum.addH<15, 15>(startIdx, endIdx, H.block<15, 15>(0, 15));
        accum.addH<15, 15>(endIdx, endIdx, H.block<15, 15>(15, 15));
        accum.addB<15>(startIdx, b.segment<15>(0));
        accum.addB<15>(endIdx, b.segment<15>(15));
    }
    void scaleJpCols(const VecXd& jacobianScaling) {
        int64_t startT = imuMeas_->getStartTNs();
        int64_t endT = imuMeas_->getStartTNs() + imuMeas_->getDtNs();
        const size_t startIdx = aom_.absOrderMap.at(startT).first;
        const size_t endIdx = aom_.absOrderMap.at(endT).first;
        jp_.topLeftCorner<15, 15>() *= jacobianScaling.segment<15>(startIdx).asDiagonal();
        jp_.topRightCorner<15, 15>() *= jacobianScaling.segment<15>(endIdx).asDiagonal();
    }
    void backSubstitute(const VecXd& poseInc, double& lDiff) {
        int64_t startT = imuMeas_->getStartTNs();
        int64_t endT = imuMeas_->getStartTNs() + imuMeas_->getDtNs();
        const size_t startIdx = aom_.absOrderMap.at(startT).first;
        const size_t endIdx = aom_.absOrderMap.at(endT).first;
        VecXd poseIncReduced(2 * 15);
        poseIncReduced.head<15>() = poseInc.segment<15>(startIdx);
        poseIncReduced.tail<15>() = poseInc.segment<15>(endIdx);
        VecXd Jinc = jp_ * poseIncReduced;
        lDiff -= Jinc.transpose() * (double(0.5) * Jinc + r_);
    }

protected:
    MatXd jp_;
    VecXd r_;
    const IntegratedImuMeasurement* imuMeas_;
    const ImuLinData* imuLinData_;
    const AbsOrderMap& aom_;
};

} // arc
