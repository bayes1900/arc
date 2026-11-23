#pragma once

#include <base/base.h>

#include <atomic>

#include <sophus/se3.hpp>

#include <vio/imu/state.h>
#include <vio/opticalflow/base.h>
#include <vio/viodef.h>
#include <base/liegroup.h>

namespace arc {

class IntegratedImuMeasurement;
struct PoseStateWithLin;

namespace constants {

static const Eigen::Vector3d G(0, 0, -9.81);
static const Eigen::Vector3d G_DIR(0, 0, -1);
}

struct PoseVelBiasStateWithLin {
    PoseVelBiasStateWithLin() {
        linearized_ = false;
        delta_.setZero();
        backupDelta_.setZero();
    };
    PoseVelBiasStateWithLin(
        int64_t tNs,
        const Sophus::SE3d& tWI,
        const Vec3d& velWI,
        const Vec3d& biasGyro,
        const Vec3d& biasAccel,
        bool linearized
    )
        : linearized_(linearized)
        , stateLinearized_(tNs, tWI, velWI, biasGyro, biasAccel)
    {
        delta_.setZero();
        backupDelta_.setZero();
        stateCurrent_ = stateLinearized_;
    }

    explicit PoseVelBiasStateWithLin(const PoseVelBiasState& other)
        : linearized_(false)
        , stateLinearized_(other)
    {
        delta_.setZero();
        backupDelta_.setZero();
        stateCurrent_ = other;
    }
    void setLinFalse() {
        linearized_ = false;
        delta_.setZero();
    }
    void setLinTrue() {
        linearized_ = true;
        ensure(delta_.isApproxToConstant(0), "VIO check failed");
        stateCurrent_ = stateLinearized_;
    }
    void applyInc(const Vec15d& inc) {
        if (!linearized_) {
            stateLinearized_.applyInc(inc);
        } else {
            delta_ += inc;
            stateCurrent_ = stateLinearized_;
            stateCurrent_.applyInc(delta_);
        }
    }
    const PoseVelBiasState& getState() const {
        if (!linearized_) {
            return stateLinearized_;
        } else {
            return stateCurrent_;
        }
    }
    const PoseVelBiasState& getStateLin() const { return stateLinearized_; }
    bool isLinearized() const { return linearized_; }
    const Vec15d& getDelta() const { return delta_; }
    int64_t getTNs() const { return stateLinearized_.tNs; }
    void backup() {
        backupDelta_ = delta_;
        backupStateLinearized_ = stateLinearized_;
        backupStateCurrent_ = stateCurrent_;
    }
    void restore() {
        delta_ = backupDelta_;
        stateLinearized_ = backupStateLinearized_;
        stateCurrent_ = backupStateCurrent_;
    }

private:
    bool linearized_;
    Vec15d delta_;
    PoseVelBiasState stateLinearized_, stateCurrent_;
    Vec15d backupDelta_;
    PoseVelBiasState backupStateLinearized_, backupStateCurrent_;
    friend struct PoseStateWithLin;
};

struct PoseStateWithLin {
    PoseStateWithLin() {
        linearized_ = false;
        delta_.setZero();
    };

    PoseStateWithLin(int64_t tNs, const Sophus::SE3d& tWI, bool linearized = false)
        : linearized_(linearized)
        , poseLinearized_(tNs, tWI)
    {
        delta_.setZero();
        backupDelta_.setZero();
        tWICurrent_ = tWI;
    }

    explicit PoseStateWithLin(const PoseVelBiasStateWithLin& other)
        : linearized_(other.linearized_)
        , delta_(other.delta_.head<6>())
        , poseLinearized_(other.stateLinearized_.tNs, other.stateLinearized_.tWI)
    {
        tWICurrent_ = poseLinearized_.tWI;
        PoseState::incPose(delta_, tWICurrent_);
    }
    void setLinTrue() {
        linearized_ = true;
        ensure(delta_.isApproxToConstant(0), "VIO check failed");
        tWICurrent_ = poseLinearized_.tWI;
    }
    const Sophus::SE3d& getPose() const {
        if (!linearized_) {
            return poseLinearized_.tWI;
        } else {
            return tWICurrent_;
        }
    }
    const Sophus::SE3d& getPoseLin() const { return poseLinearized_.tWI; }
    void applyInc(const Vec6d& inc) {
        if (!linearized_) {
            PoseState::incPose(inc, poseLinearized_.tWI);
        } else {
            delta_ += inc;
            tWICurrent_ = poseLinearized_.tWI;
            PoseState::incPose(delta_, tWICurrent_);
        }
    }
    bool isLinearized() const { return linearized_; }
    const Vec6d& getDelta() const { return delta_; }
    int64_t getTNs() const { return poseLinearized_.tNs; }
    void backup() {
        backupDelta_ = delta_;
        backupPoseLinearized_ = poseLinearized_;
        backupTWICurrent_ = tWICurrent_;
    }
    void restore() {
        delta_ = backupDelta_;
        poseLinearized_ = backupPoseLinearized_;
        tWICurrent_ = backupTWICurrent_;
    }

private:
    bool linearized_;
    Vec6d delta_;
    PoseState poseLinearized_;
    Sophus::SE3d tWICurrent_;
    Vec6d backupDelta_;
    PoseState backupPoseLinearized_;
    Sophus::SE3d backupTWICurrent_;
};

struct AbsOrderMap {
    std::map<int64_t, std::pair<int, int>> absOrderMap;
    size_t items = 0;
    size_t totalSize = 0;
    void printOrder() {
        for (const auto& kv : absOrderMap) {
            std::cout << kv.first << " (" << kv.second.first << "," << kv.second.second << ")" << std::endl;
        }
        std::cout << std::endl;
    }
};

struct ImuLinData {
    const Eigen::Matrix<double, 3, 1>& g;
    const Eigen::Matrix<double, 3, 1>& gyroBiasWeightSqrt;
    const Eigen::Matrix<double, 3, 1>& accelBiasWeightSqrt;
    std::map<int64_t, const IntegratedImuMeasurement*> imuMeas;
};

struct MargLinData {
    bool isSqrt = true;
    AbsOrderMap order;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> h;
    Eigen::Matrix<double, Eigen::Dynamic, 1> b;
};

struct MargData {
    using Ptr = std::shared_ptr<MargData>;
    AbsOrderMap aom;
    Eigen::MatrixXd absH;
    Eigen::VectorXd absB;
    std::map<int64_t, PoseVelBiasStateWithLin> frameStates;
    std::map<int64_t, PoseStateWithLin> framePoses;
    std::set<int64_t> kfsAll;
    std::set<int64_t> kfsToMarg;
    bool useImu;
    std::vector<OpticalFlowResult::Ptr> optFlowRes;
};

struct RelPoseFactor {
    int64_t tINs, tJNs;
    Sophus::SE3d tIJ;
    Sophus::Matrix6d covInv;
};

struct RollPitchFactor {
    int64_t tNs;
    Sophus::SO3d rWIMeas;
    Eigen::Matrix2d covInv;
};
} // arc
