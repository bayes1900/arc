#pragma once

#include <base/base.h>
#include <tbb/parallel_for.h>
#include <vio/estimator/ba.h>
#include <vio/estimator/scba.h>

namespace arc {

class SqrtBundleAdjustmentBase : public BundleAdjustmentBase {
public:
    using RelLinDataBase = typename ScBundleAdjustmentBase::RelLinDataBase;
    using FrameRelLinData = typename ScBundleAdjustmentBase::FrameRelLinData;
    using RelLinData = typename ScBundleAdjustmentBase::RelLinData;
    using FrameAbsLinData = typename ScBundleAdjustmentBase::FrameAbsLinData;
    using AbsLinData = typename ScBundleAdjustmentBase::AbsLinData;
    using BundleAdjustmentBase::computeDelta;
    void linearizeHelper(
        std::vector<RelLinData>& rldVec,
        const std::unordered_map<TimeCamId, std::map<TimeCamId, std::set<KeypointId>>>& obsToLin,
        double& error
    ) const {
        ScBundleAdjustmentBase::linearizeHelperStatic(rldVec, obsToLin, this, error);
    }
    void linearizeAbsHelper(
        std::vector<AbsLinData>& aldVec,
        const std::unordered_map<TimeCamId, std::map<TimeCamId, std::set<KeypointId>>>& obsToLin,
        double& error
    ) const {
        ScBundleAdjustmentBase::linearizeHelperAbsStatic(aldVec, obsToLin, this, error);
    }
    static void linearizeRel(const RelLinData& rld, MatXd& H, VecXd& b) {
        ScBundleAdjustmentBase::linearizeRel(rld, H, b);
    }
    static void updatePoints(
        const AbsOrderMap& aom,
        const RelLinData& rld,
        const VecXd& inc,
        LandmarkDatabase& lmdb,
        double* lDiff = nullptr
    ) {
        ScBundleAdjustmentBase::updatePoints(aom, rld, inc, lmdb, lDiff);
    }
    static void updatePointsAbs(
        const AbsOrderMap& aom,
        const AbsLinData& ald,
        const VecXd& inc,
        LandmarkDatabase& lmdb,
        double* lDiff = nullptr
    ) {
        ScBundleAdjustmentBase::updatePointsAbs(aom, ald, inc, lmdb, lDiff);
    }
    static Eigen::VectorXd checkNullspace(
        const MargLinData& mld,
        const std::map<int64_t, PoseVelBiasStateWithLin>& frameStates,
        const std::map<int64_t, PoseStateWithLin>& framePoses,
        bool verbose = true
    ) {
        ensure(size_t(mld.h.cols()) == mld.order.totalSize, "VIO check failed");
        size_t margSize = mld.order.totalSize;
        VecXd incX, incY, incZ, incRoll, incPitch, incYaw;
        incX.setZero(margSize);
        incY.setZero(margSize);
        incZ.setZero(margSize);
        incRoll.setZero(margSize);
        incPitch.setZero(margSize);
        incYaw.setZero(margSize);
        int numTrans = 0;
        Vec3d meanTrans;
        meanTrans.setZero();
        for (const auto& kv : mld.order.absOrderMap) {
            Vec3d trans;
            if (kv.second.second == 6) {
                meanTrans += framePoses.at(kv.first).getPoseLin().translation().cast<double>();
                ++ numTrans;
            } else if (kv.second.second == 15) {
                meanTrans += frameStates.at(kv.first).getStateLin().tWI.translation().cast<double>();
                ++ numTrans;
            } else {
                std::cerr << "Unknown size of the state: " << kv.second.second << std::endl;
                std::abort();
            }
        }
        meanTrans /= numTrans;
        double eps = 0.01;
        for (const auto& kv : mld.order.absOrderMap) {
            incX(kv.second.first + 0) = eps;
            incY(kv.second.first + 1) = eps;
            incZ(kv.second.first + 2) = eps;
            incRoll(kv.second.first + 3) = eps;
            incPitch(kv.second.first + 4) = eps;
            incYaw(kv.second.first + 5) = eps;
            Vec3d trans;
            if (kv.second.second == 6) {
                trans = framePoses.at(kv.first).getPoseLin().translation().cast<double>();
            } else if (kv.second.second == 15) {
                trans = frameStates.at(kv.first).getStateLin().tWI.translation().cast<double>();
            } else {
                throw std::logic_error("unreachable");
            }
            trans -= meanTrans;
            Mat3d J = -Sophus::SO3d::hat(trans);
            J *= eps;
            incRoll.segment<3>(kv.second.first) = J.col(0);
            incPitch.segment<3>(kv.second.first) = J.col(1);
            incYaw.segment<3>(kv.second.first) = J.col(2);
            if (kv.second.second == 15) {
                Vec3d vel = frameStates.at(kv.first).getStateLin().velWI.cast<double>();
                Mat3d J_vel = -Sophus::SO3d::hat(vel);
                J_vel *= eps;
                incRoll.segment<3>(kv.second.first + 6) = J_vel.col(0);
                incPitch.segment<3>(kv.second.first + 6) = J_vel.col(1);
                incYaw.segment<3>(kv.second.first + 6) = J_vel.col(2);
            }
        }
        incX.normalize();
        incY.normalize();
        incZ.normalize();
        incRoll.normalize();
        incPitch.normalize();
        incYaw.normalize();
        VecXd incRandom;
        incRandom.setRandom(margSize);
        incRandom.normalize();
        MatXd H_d;
        VecXd bD;
        if (mld.isSqrt) {
            MatXd H_sqrt_d = mld.h.cast<double>();
            VecXd bSqrtD = mld.b.cast<double>();
            H_d = H_sqrt_d.transpose() * H_sqrt_d;
            bD = H_sqrt_d.transpose() * bSqrtD;
        } else {
            H_d = mld.h.cast<double>();
            bD = mld.b.cast<double>();
        }
        VecXd xHx(7);
        VecXd xb(7);
        xHx[0] = incX.transpose() * H_d * incX;
        xHx[1] = incY.transpose() * H_d * incY;
        xHx[2] = incZ.transpose() * H_d * incZ;
        xHx[3] = incRoll.transpose() * H_d * incRoll;
        xHx[4] = incPitch.transpose() * H_d * incPitch;
        xHx[5] = incYaw.transpose() * H_d * incYaw;
        xHx[6] = incRandom.transpose() * H_d * incRandom;
        xb[0] = incX.transpose() * bD;
        xb[1] = incY.transpose() * bD;
        xb[2] = incZ.transpose() * bD;
        xb[3] = incRoll.transpose() * bD;
        xb[4] = incPitch.transpose() * bD;
        xb[5] = incYaw.transpose() * bD;
        xb[6] = incRandom.transpose() * bD;
        if (verbose) {
            std::cout << "nullspace x_trans: " << xHx[0] << " + " << xb[0] << std::endl;
            std::cout << "nullspace y_trans: " << xHx[1] << " + " << xb[1] << std::endl;
            std::cout << "nullspace z_trans: " << xHx[2] << " + " << xb[2] << std::endl;
            std::cout << "nullspace roll   : " << xHx[3] << " + " << xb[3] << std::endl;
            std::cout << "nullspace pitch  : " << xHx[4] << " + " << xb[4] << std::endl;
            std::cout << "nullspace yaw    : " << xHx[5] << " + " << xb[5] << std::endl;
            std::cout << "nullspace random : " << xHx[6] << " + " << xb[6] << std::endl;
        }
        return xHx + xb;
    }
    static Eigen::VectorXd checkEigenvalues(const MargLinData& mld, bool verbose = true) {
        Eigen::MatrixXd H;
        if (mld.isSqrt) {
            Eigen::MatrixXd sqrt_H_double = mld.h.cast<double>();
            H = sqrt_H_double.transpose() * sqrt_H_double;
        } else {
            H = mld.h.cast<double>();
        }
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(H);
        if (eigensolver.info() != Eigen::Success) {
            throw std::runtime_error("eigen solver failed");
        }
        if (verbose) {
            std::cout << "EV:\n" << eigensolver.eigenvalues() << std::endl;
        }
        return eigensolver.eigenvalues();
    }
    static void linearizeAbs(
        const MatXd& rel_H,
        const VecXd& relB,
        const RelLinDataBase& rld,
        const AbsOrderMap& aom,
        DenseAccumulator& accum
    ) {
        ScBundleAdjustmentBase::linearizeAbs(rel_H, relB, rld, aom, accum);
    }
    static void linearizeAbs(const AbsLinData& ald, const AbsOrderMap& aom, DenseAccumulator& accum) {
        ScBundleAdjustmentBase::linearizeAbs(ald, aom, accum);
    }
};

} // arc
