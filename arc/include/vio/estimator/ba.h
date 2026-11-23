#pragma once

#include <base/base.h>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <vio/estimator/landmarks.h>
#include <vio/estimator/projection.h>

namespace arc {

class BundleAdjustmentBase {
public:
    void computeError(
        double& error,
        std::map<int, std::vector<std::pair<TimeCamId, double>>>* outliers = nullptr,
        double outlierThreshold = 0
    ) const;
    void filterOutliers(double outlierThreshold, int minNumObs);
    void optimizeSingleFramePose(
        PoseStateWithLin& stateT, const std::vector<std::vector<int>>& connectedObs) const;
    void getCurrentPoints(std::vector<Eigen::Vector3d>& points, std::vector<int>& ids) const;
    void computeDelta(const AbsOrderMap& margOrder, VecXd& delta) const;
    void linearizeMargPrior(
        const MargLinData& mld,
        const AbsOrderMap& aom,
        MatXd& abs_H,
        VecXd& absB,
        double& margPriorError
    ) const;
    void computeMargPriorError(const MargLinData& mld, double& margPriorError) const;
    double computeMargPriorModelCostChange(
        const MargLinData& mld, const VecXd& margScaling, const VecXd& margPoseInc) const;
    double computeModelCostChange(const MatXd& H, const VecXd& b, const VecXd& inc) const;
    void computeProjections(std::vector<std::vector<Eigen::Vector4d>>& data, int64_t lastStateTNs) const;

    template <class Derived>
    static Eigen::Matrix<typename Derived::Scalar, 4, 1> triangulate(
        const Eigen::MatrixBase<Derived>& f0,
        const Eigen::MatrixBase<Derived>& f1,
        const Sophus::SE3<typename Derived::Scalar>& T_0_1
    ) {
        static_assert(Derived::IsVectorAtCompileTime && Derived::SizeAtCompileTime == 3,
            "Derived must be a vector of size 3");
        using Scalar2 = typename Derived::Scalar;
        using Vec42 = Eigen::Matrix<Scalar2, 4, 1>;
        Eigen::Matrix<Scalar2, 3, 4> P1, P2;
        P1.setIdentity();
        P2 = T_0_1.inverse().matrix3x4();
        Eigen::Matrix<Scalar2, 4, 4> A(4, 4);
        A.row(0) = f0[0] * P1.row(2) - f0[2] * P1.row(0);
        A.row(1) = f0[1] * P1.row(2) - f0[2] * P1.row(1);
        A.row(2) = f1[0] * P2.row(2) - f1[2] * P2.row(0);
        A.row(3) = f1[1] * P2.row(2) - f1[2] * P2.row(1);
        Eigen::JacobiSVD<Eigen::Matrix<Scalar2, 4, 4>> mySVD(A, Eigen::ComputeFullV);
        Vec42 worldPoint = mySVD.matrixV().col(3);
        worldPoint /= worldPoint.template head<3>().norm();
        if (f0.dot(worldPoint.template head<3>()) < 0)
            worldPoint *= -1;
        return worldPoint;
    }
    void backup() {
        for (auto& kv : frameStates)
            kv.second.backup();
        for (auto& kv : framePoses)
            kv.second.backup();
        lmdb.backup();
    }
    void restore() {
        for (auto& kv : frameStates)
            kv.second.restore();
        for (auto& kv : framePoses)
            kv.second.restore();
        lmdb.restore();
    }
    PoseStateWithLin getPoseStateWithLin(int64_t tNs) const {
        auto it = framePoses.find(tNs);
        if (it != framePoses.end())
            return it->second;
        auto it2 = frameStates.find(tNs);
        if (it2 == frameStates.end()) {
            std::cerr << "Could not find pose " << tNs << std::endl;
            std::abort();
        }
        return PoseStateWithLin(it2->second);
    }

public:
    std::map<int64_t, PoseVelBiasStateWithLin> frameStates;
    std::map<int64_t, PoseStateWithLin> framePoses;
    LandmarkDatabase lmdb;
    double obsStdDev;
    double huberThresh;
    arc::Calibration<double> calib;
};
inline void BundleAdjustmentBase::optimizeSingleFramePose(
    PoseStateWithLin& stateT, const std::vector<std::vector<int>>& connectedObs) const {
    const int numIter = 2;
    struct AbsLinData {
        Mat4d tTH;
        Mat6d dRelDT;
    };
    for (int iter = 0; iter < numIter; ++ iter) {
        double error = 0;
        Mat6d Ht;
        Vec6d bt;
        Ht.setZero();
        bt.setZero();
        std::unordered_map<std::pair<TimeCamId, TimeCamId>, AbsLinData> absLinData;
        for (size_t camId = 0; camId < connectedObs.size(); ++ camId) {
            TimeCamId tcidT(stateT.getTNs(), camId);
            for (const auto& lmId : connectedObs[camId]) {
                const Keypoint& kptPos = lmdb.getLandmark(lmId);
                std::pair<TimeCamId, TimeCamId> mapKey(kptPos.hostKfId, tcidT);
                if (absLinData.count(mapKey) == 0) {
                    const PoseStateWithLin& stateH = framePoses.at(kptPos.hostKfId.frameId);
                    ensure(kptPos.hostKfId.frameId != stateT.getTNs(), "VIO check failed");
                    AbsLinData& ald = absLinData[mapKey];
                    Sophus::SE3d T_t_h_sophus = computeRelPose(stateH.getPose(),
                        calib.tIC[kptPos.hostKfId.camId],
                        stateT.getPose(),
                        calib.tIC[camId],
                        nullptr,
                        &ald.dRelDT);
                    ald.tTH = T_t_h_sophus.matrix();
                }
            }
        }
        for (size_t camId = 0; camId < connectedObs.size(); ++ camId) {
            std::visit(
                [&](const auto& cam) {
                    for (const auto& lmId : connectedObs[camId]) {
                        TimeCamId tcidT(stateT.getTNs(), camId);
                        const Keypoint& kptPos = lmdb.getLandmark(lmId);
                        const Vec2d& kptObs = kptPos.obs.at(tcidT);
                        const AbsLinData& ald = absLinData.at(std::make_pair(kptPos.hostKfId, tcidT));
                        Vec2d res;
                        Eigen::Matrix<double, 2, 6> dResDXi;
                        bool valid = linearizePoint(kptObs, kptPos, ald.tTH, cam, res, &dResDXi);
                        if (valid) {
                            double e = res.norm();
                            double huberWeight = e < huberThresh ? double(1.0) : huberThresh / e;
                            double obsWeight = huberWeight / (obsStdDev * obsStdDev);
                            error += double(0.5) * (2 - huberWeight) * obsWeight * res.transpose() * res;
                            dResDXi *= ald.dRelDT;
                            Ht.noalias() += dResDXi.transpose() * dResDXi;
                            bt.noalias() += dResDXi.transpose() * res;
                        }
                    }
                },
                calib.intrinsics[camId].variant);
        }
        constexpr double LAMBDA = 1e-6;
        Vec6d diag = Ht.diagonal();
        diag *= LAMBDA;
        Ht.diagonal().array() += diag.array().max(LAMBDA);
        Vec6d inc = -Ht.ldlt().solve(bt);
        stateT.applyInc(inc);
    }
}
inline void BundleAdjustmentBase::computeError(
    double& error, std::map<int, std::vector<std::pair<TimeCamId, double>>>* outliers, double outlierThreshold) const {
    std::vector<TimeCamId> hostFrames;
    for (const auto& [tcid, _] : lmdb.getObservations()) {
        hostFrames.push_back(tcid);
    }
    tbb::concurrent_unordered_map<int, std::vector<std::pair<TimeCamId, double>>> outliersConcurrent;
    auto body = [&](const tbb::blocked_range<size_t>& range, double localError) {
        for (size_t r = range.begin(); r != range.end(); ++ r) {
            const TimeCamId& tcidH = hostFrames[r];
            for (const auto& obsKv : lmdb.getObservations().at(tcidH)) {
                const TimeCamId& tcidT = obsKv.first;
                Mat4d T_t_h;
                if (tcidH != tcidT) {
                    PoseStateWithLin stateH = getPoseStateWithLin(tcidH.frameId);
                    PoseStateWithLin stateT = getPoseStateWithLin(tcidT.frameId);
                    Sophus::SE3d T_t_h_sophus = computeRelPose(
                        stateH.getPose(), calib.tIC[tcidH.camId], stateT.getPose(), calib.tIC[tcidT.camId]);
                    T_t_h = T_t_h_sophus.matrix();
                } else {
                    T_t_h.setIdentity();
                }
                std::visit(
                    [&](const auto& cam) {
                        for (KeypointId kptId : obsKv.second) {
                            const Keypoint& kptPos = lmdb.getLandmark(kptId);
                            const Vec2d& kptObs = kptPos.obs.at(tcidT);
                            Vec2d res;
                            bool valid = linearizePoint(kptObs, kptPos, T_t_h, cam, res);
                            if (valid) {
                                double e = res.norm();
                                if (outliers && e > outlierThreshold) {
                                    outliersConcurrent[kptId].emplace_back(tcidT, tcidH != tcidT ? e : -2);
                                }
                                double huberWeight = e < huberThresh ? double(1.0) : huberThresh / e;
                                double obsWeight = huberWeight / (obsStdDev * obsStdDev);
                                localError += double(0.5) * (2 - huberWeight) * obsWeight * res.transpose() * res;
                            } else {
                                if (outliers) {
                                    outliersConcurrent[kptId].emplace_back(tcidT, tcidH != tcidT ? -1 : -2);
                                }
                            }
                        }
                    },
                    calib.intrinsics[tcidT.camId].variant);
            }
        }
        return localError;
    };
    tbb::blocked_range<size_t> range(0, hostFrames.size());
    double init = 0;
    auto join = std::plus<double>();
    error = tbb::parallel_reduce(range, init, body, join);
    if (outliers) {
        outliers->clear();
        for (auto& [k, v] : outliersConcurrent) {
            outliers->emplace(k, std::move(v));
        }
    }
}
inline void BundleAdjustmentBase::getCurrentPoints(
    std::vector<Eigen::Vector3d>& points, std::vector<int>& ids) const {
    points.clear();
    ids.clear();
    for (const auto& tcidHost : lmdb.getHostKfs()) {
        Sophus::SE3d tWI;
        int64_t id = tcidHost.frameId;
        if (frameStates.count(id) > 0) {
            PoseVelBiasStateWithLin state = frameStates.at(id);
            tWI = state.getState().tWI;
        } else if (framePoses.count(id) > 0) {
            PoseStateWithLin state = framePoses.at(id);
            tWI = state.getPose();
        } else {
            std::cout << "Unknown frame id: " << id << std::endl;
            std::abort();
        }
        const Sophus::SE3d& tIC = calib.tIC[tcidHost.camId];
        Mat4d T_w_c = (tWI * tIC).matrix();
        for (const Keypoint* kptPos : lmdb.getLandmarksForHost(tcidHost)) {
            Vec4d ptCam = StereographicParam::unproject(kptPos->direction);
            ptCam[3] = kptPos->invDist;
            Vec4d ptW = T_w_c * ptCam;
            points.emplace_back(ptW.head<3>() / ptW[3]);
            ids.emplace_back(1);
        }
    }
}
inline void BundleAdjustmentBase::filterOutliers(double outlierThreshold, int minNumObs) {
    double error;
    std::map<int, std::vector<std::pair<TimeCamId, double>>> outliers;
    computeError(error, &outliers, outlierThreshold);
    for (const auto& kv : outliers) {
        int numObs = lmdb.numObservations(kv.first);
        int numOutliers = kv.second.size();
        bool remove = false;
        if (numObs - numOutliers < minNumObs)
            remove = true;
        for (const auto& kv2 : kv.second) {
            if (kv2.second == -2)
                remove = true;
        }
        if (remove) {
            lmdb.removeLandmark(kv.first);
        } else {
            std::set<TimeCamId> outliers;
            for (const auto& kv2 : kv.second)
                outliers.emplace(kv2.first);
            lmdb.removeObservations(kv.first, outliers);
        }
    }
}
inline void BundleAdjustmentBase::computeDelta(const AbsOrderMap& margOrder, VecXd& delta) const {
    size_t margSize = margOrder.totalSize;
    delta.setZero(margSize);
    for (const auto& kv : margOrder.absOrderMap) {
        if (kv.second.second == 6) {
            ensure(framePoses.at(kv.first).isLinearized(), "VIO check failed");
            delta.segment<6>(kv.second.first) = framePoses.at(kv.first).getDelta();
        } else if (kv.second.second == 15) {
            ensure(frameStates.at(kv.first).isLinearized(), "VIO check failed");
            delta.segment<15>(kv.second.first) = frameStates.at(kv.first).getDelta();
        } else {
            throw std::logic_error("unreachable");
        }
    }
}
inline double BundleAdjustmentBase::computeModelCostChange(const MatXd& H, const VecXd& b, const VecXd& inc) const {
    double lDiff = -inc.dot(double(0.5) * (H * inc) + b);
    return lDiff;
}
inline void BundleAdjustmentBase::computeProjections(
    std::vector<std::vector<Eigen::Vector4d>>& data, int64_t lastStateTNs) const {
    for (const auto& kv : lmdb.getObservations()) {
        const TimeCamId& tcidH = kv.first;
        for (const auto& obsKv : kv.second) {
            const TimeCamId& tcidT = obsKv.first;
            if (tcidT.frameId != lastStateTNs)
                continue;
            Mat4d T_t_h;
            if (tcidH != tcidT) {
                PoseStateWithLin stateH = getPoseStateWithLin(tcidH.frameId);
                PoseStateWithLin stateT = getPoseStateWithLin(tcidT.frameId);
                Sophus::SE3d T_t_h_sophus = computeRelPose(
                    stateH.getPose(), calib.tIC[tcidH.camId], stateT.getPose(), calib.tIC[tcidT.camId]);
                T_t_h = T_t_h_sophus.matrix();
            } else {
                T_t_h.setIdentity();
            }
            std::visit(
                [&](const auto& cam) {
                    for (KeypointId kptId : obsKv.second) {
                        const Keypoint& kptPos = lmdb.getLandmark(kptId);
                        Vec2d res;
                        Vec4d proj;
                        using CamT = std::decay_t<decltype(cam)>;
                        linearizePoint<CamT>(Vec2d::Zero(), kptPos, T_t_h, cam, res, nullptr, nullptr, &proj);
                        proj[3] = kptId;
                        data[tcidT.camId].emplace_back(proj);
                    }
                },
                calib.intrinsics[tcidT.camId].variant);
        }
    }
}
inline void BundleAdjustmentBase::linearizeMargPrior(
    const MargLinData& mld, const AbsOrderMap& aom, MatXd& abs_H, VecXd& absB, double& margPriorError) const {
    ensure(size_t(mld.h.cols()) == mld.order.totalSize, "VIO check failed");
    for (const auto& kv : mld.order.absOrderMap) {
        (void)(aom);
        (void)(kv);
        ensure(aom.absOrderMap.at(kv.first) == kv.second, "VIO check failed");
        ensure(kv.second.first < int(mld.order.totalSize), "VIO check failed");
    }
    const size_t margSize = mld.order.totalSize;
    VecXd delta;
    computeDelta(mld.order, delta);
    if (mld.isSqrt) {
        abs_H.topLeftCorner(margSize, margSize) += mld.h.transpose() * mld.h;
        absB.head(margSize) += mld.h.transpose() * (mld.b + mld.h * delta);
        margPriorError = delta.transpose() * mld.h.transpose() * (double(0.5) * mld.h * delta + mld.b);
    } else {
        abs_H.topLeftCorner(margSize, margSize) += mld.h;
        absB.head(margSize) += mld.h * delta + mld.b;
        margPriorError = delta.transpose() * (double(0.5) * mld.h * delta + mld.b);
    }
}
inline void BundleAdjustmentBase::computeMargPriorError(
    const MargLinData& mld, double& margPriorError) const {
    ensure(size_t(mld.h.cols()) == mld.order.totalSize, "VIO check failed");
    VecXd delta;
    computeDelta(mld.order, delta);
    if (mld.isSqrt) {
        margPriorError = delta.transpose() * mld.h.transpose() * (double(0.5) * mld.h * delta + mld.b);
    } else {
        margPriorError = delta.transpose() * (double(0.5) * mld.h * delta + mld.b);
    }
}
inline double BundleAdjustmentBase::computeMargPriorModelCostChange(
    const MargLinData& mld, const VecXd& margScaling, const VecXd& margPoseInc) const {
    VecXd delta;
    computeDelta(mld.order, delta);
    VecXd J_inc = margPoseInc;
    if (margScaling.rows() > 0)
        J_inc = margScaling.asDiagonal() * J_inc;
    double lDiff;
    if (mld.isSqrt) {
        const VecXd b_Jdelta = mld.h * delta + mld.b;
        J_inc = mld.h * J_inc;
        lDiff = -J_inc.transpose() * (b_Jdelta + double(0.5) * J_inc);
    } else {
        lDiff = -J_inc.dot(mld.h * (delta + double(0.5) * J_inc) + mld.b);
    }
    return lDiff;
}

} // arc
