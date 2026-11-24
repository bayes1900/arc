#pragma once

#include <base/base.h>
#include <base/manifold/s2.h>
#include <base/manifold/so3.h>
#include <base/manifold/vector.h>

namespace arc {

using LioVect3 = manifold::ManifoldVector<3, double>;
using LioSo3 = manifold::ManifoldSo3<double>;
using LioS2 = manifold::ManifoldS2<double, 98090, 10000, 1>;

class LioState {
public:
    static constexpr int DOF = 23;
    static constexpr int DIM = 24;

    struct VectorState {
        int index;
        int dimension;
        int dof;
    };

    struct MatrixState {
        int index;
        int dimension;
    };

    LioVect3 pos;
    LioSo3 rot;
    LioSo3 offsetRLI;
    LioVect3 offsetTLI;
    LioVect3 vel;
    LioVect3 bg;
    LioVect3 ba;
    LioS2 grav;

    LioState() {
        buildStateIndex();
    }
    void buildStateIndex() {
        vectorState_.clear();
        so3State_.clear();
        s2State_.clear();
        vectorState_.push_back({0, 0, 3});
        vectorState_.push_back({9, 9, 3});
        vectorState_.push_back({12, 12, 3});
        vectorState_.push_back({15, 15, 3});
        vectorState_.push_back({18, 18, 3});
        so3State_.push_back({3, 3});
        so3State_.push_back({6, 6});
        s2State_.push_back({21, 21});
    }
    const std::vector<VectorState>& vectorState() const {
        return vectorState_;
    }
    const std::vector<MatrixState>& so3State() const {
        return so3State_;
    }
    const std::vector<MatrixState>& s2State() const {
        return s2State_;
    }
    void boxPlus(const Eigen::Matrix<double, DOF, 1>& delta, double scale = 1.0) {
        pos.boxPlus(delta.segment<3>(0), scale);
        rot.boxPlus(delta.segment<3>(3), scale);
        offsetRLI.boxPlus(delta.segment<3>(6), scale);
        offsetTLI.boxPlus(delta.segment<3>(9), scale);
        vel.boxPlus(delta.segment<3>(12), scale);
        bg.boxPlus(delta.segment<3>(15), scale);
        ba.boxPlus(delta.segment<3>(18), scale);
        grav.boxPlus(delta.segment<2>(21), scale);
    }
    void boxMinus(Eigen::Matrix<double, DOF, 1>& result, const LioState& other) const {
        Eigen::Matrix<double, 3, 1> vectorDelta;
        Eigen::Matrix<double, 2, 1> sphereDelta;
        pos.boxMinus(vectorDelta, other.pos);
        result.segment<3>(0) = vectorDelta;
        rot.boxMinus(vectorDelta, other.rot);
        result.segment<3>(3) = vectorDelta;
        offsetRLI.boxMinus(vectorDelta, other.offsetRLI);
        result.segment<3>(6) = vectorDelta;
        offsetTLI.boxMinus(vectorDelta, other.offsetTLI);
        result.segment<3>(9) = vectorDelta;
        vel.boxMinus(vectorDelta, other.vel);
        result.segment<3>(12) = vectorDelta;
        bg.boxMinus(vectorDelta, other.bg);
        result.segment<3>(15) = vectorDelta;
        ba.boxMinus(vectorDelta, other.ba);
        result.segment<3>(18) = vectorDelta;
        grav.boxMinus(sphereDelta, other.grav);
        result.segment<2>(21) = sphereDelta;
    }
    void oPlus(const Eigen::Matrix<double, DIM, 1>& delta, double scale = 1.0) {
        pos.oPlus(delta.segment<3>(0), scale);
        rot.oPlus(delta.segment<3>(3), scale);
        offsetRLI.oPlus(delta.segment<3>(6), scale);
        offsetTLI.oPlus(delta.segment<3>(9), scale);
        vel.oPlus(delta.segment<3>(12), scale);
        bg.oPlus(delta.segment<3>(15), scale);
        ba.oPlus(delta.segment<3>(18), scale);
        grav.oPlus(delta.segment<3>(21), scale);
    }
    void s2Hat(Eigen::Matrix<double, 3, 3>& result, int index) const {
        (void)index;
        grav.s2Hat(result);
    }
    void s2NxYy(Eigen::Matrix<double, 2, 3>& result, int index) const {
        (void)index;
        grav.s2NxYy(result);
    }
    void s2Mx(Eigen::Matrix<double, 3, 2>& result, const Eigen::Matrix<double, 2, 1>& delta, int index) const {
        (void)index;
        grav.s2Mx(result, delta);
    }

private:
    std::vector<VectorState> vectorState_;
    std::vector<MatrixState> so3State_;
    std::vector<MatrixState> s2State_;
};

class LioInput {
public:
    static constexpr int DOF = 6;
    static constexpr int DIM = 6;

    LioVect3 acc;
    LioVect3 gyro;
};

using LioProcessNoiseCov = Eigen::Matrix<double, 12, 12>;

inline LioProcessNoiseCov processNoiseCov() {
    LioProcessNoiseCov covariance = LioProcessNoiseCov::Zero();
    covariance.block<3, 3>(0, 0).diagonal() = Eigen::Vector3d::Constant(0.0001);
    covariance.block<3, 3>(3, 3).diagonal() = Eigen::Vector3d::Constant(0.0001);
    covariance.block<3, 3>(6, 6).diagonal() = Eigen::Vector3d::Constant(0.00001);
    covariance.block<3, 3>(9, 9).diagonal() = Eigen::Vector3d::Constant(0.00001);
    return covariance;
}

inline Eigen::Matrix<double, 24, 1> lioGetF(LioState& state, const LioInput& input) {
    Eigen::Matrix<double, 24, 1> result = Eigen::Matrix<double, 24, 1>::Zero();
    const LioVect3 omega = input.gyro - state.bg;
    const LioVect3 accInertial = state.rot * (input.acc - state.ba);
    for (int i = 0; i < 3; ++ i) {
        result(i) = state.vel[i];
        result(i + 3) = omega[i];
        result(i + 12) = accInertial[i] + state.grav[i];
    }
    return result;
}

inline Eigen::Matrix<double, 24, 23> lioDfDx(LioState& state, const LioInput& input) {
    Eigen::Matrix<double, 24, 23> covariance = Eigen::Matrix<double, 24, 23>::Zero();
    covariance.block<3, 3>(0, 12) = Eigen::Matrix3d::Identity();
    const LioVect3 acc = input.acc - state.ba;
    covariance.block<3, 3>(12, 3) = -state.rot.toRotationMatrix() * manifold::hat<double>(acc);
    covariance.block<3, 3>(12, 18) = -state.rot.toRotationMatrix();
    const Eigen::Matrix<double, 2, 1> zero = Eigen::Matrix<double, 2, 1>::Zero();
    Eigen::Matrix<double, 3, 2> gravityMatrix;
    state.s2Mx(gravityMatrix, zero, 21);
    covariance.block<3, 2>(12, 21) = gravityMatrix;
    covariance.block<3, 3>(3, 15) = -Eigen::Matrix3d::Identity();
    return covariance;
}

inline Eigen::Matrix<double, 24, 12> lioDfDw(LioState& state, const LioInput& input) {
    (void)input;
    Eigen::Matrix<double, 24, 12> covariance = Eigen::Matrix<double, 24, 12>::Zero();
    covariance.block<3, 3>(12, 3) = -state.rot.toRotationMatrix();
    covariance.block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity();
    covariance.block<3, 3>(15, 6) = Eigen::Matrix3d::Identity();
    covariance.block<3, 3>(18, 9) = Eigen::Matrix3d::Identity();
    return covariance;
}

} // arc
