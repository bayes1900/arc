#pragma once

#include <base/base.h>
#include <base/liegroup.h>
#include <vio/viodef.h>

namespace arc {

struct PoseState {
    int64_t tNs;
    Sophus::SE3d tWI;
    PoseState() { tNs = 0; }
    PoseState(int64_t tNs, const Sophus::SE3d& tWI)
        : tNs(tNs)
        , tWI(tWI)
    {}
    void applyInc(const Vec6d& inc) {
        incPose(inc, tWI);
    }
    static void incPose(const Vec6d& inc, Sophus::SE3d& T) {
        T.translation() += inc.head<3>();
        T.so3() = Sophus::SO3d::exp(inc.tail<3>()) * T.so3();
    }
};

struct PoseVelState : public PoseState {
    Vec3d velWI;
    PoseVelState() { velWI.setZero(); }
    PoseVelState(int64_t tNs, const Sophus::SE3d& tWI, const Vec3d& velWI)
        : PoseState(tNs, tWI)
        , velWI(velWI)
    {}
    void applyInc(const Vec9d& inc) {
        PoseState::applyInc(inc.head<6>());
        velWI += inc.tail<3>();
    }
    Vec9d diff(const PoseVelState& other) const {
        Vec9d res;
        res.segment<3>(0) = other.tWI.translation() - this->tWI.translation();
        res.segment<3>(3) = (other.tWI.so3() * this->tWI.so3().inverse()).log();
        res.tail<3>() = other.velWI - velWI;
        return res;
    }
};

struct PoseVelBiasState : public PoseVelState {
    using Ptr = std::shared_ptr<PoseVelBiasState>;
    Vec3d biasGyro;
    Vec3d biasAccel;
    PoseVelBiasState() {
        biasGyro.setZero();
        biasAccel.setZero();
    }
    PoseVelBiasState(int64_t tNs, const Sophus::SE3d& tWI, const Vec3d& velWI, const Vec3d& biasGyro, const Vec3d& biasAccel)
        : PoseVelState(tNs, tWI, velWI)
        , biasGyro(biasGyro)
        , biasAccel(biasAccel)
    {}
    void applyInc(const Vec15d& inc) {
        PoseVelState::applyInc(inc.head<9>());
        biasGyro += inc.segment<3>(9);
        biasAccel += inc.segment<3>(12);
    }
    Vec15d diff(const PoseVelBiasState& other) const {
        Vec15d res;
        res.segment<9>(0) = PoseVelState::diff(other);
        res.segment<3>(9) = other.biasGyro - biasGyro;
        res.segment<3>(12) = other.biasAccel - biasAccel;
        return res;
    }
};

struct ImuData {
    using Ptr = std::shared_ptr<ImuData>;
    int64_t tNs;
    Vec3d accel;
    Vec3d gyro;
    ImuData() {
        tNs = 0;
        accel.setZero();
        gyro.setZero();
    }
};

} // arc
