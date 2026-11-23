#pragma once

#include <base/base.h>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <vio/estimator/accumulator.h>
#include <vio/estimator/ba.h>
#include <vio/estimator/projection.h>
#include <vio/imu/preintegration.h>

namespace arc {

class ScBundleAdjustmentBase : public BundleAdjustmentBase {
public:
    struct RelLinDataBase {
        std::vector<std::pair<TimeCamId, TimeCamId>> order;
        std::vector<Mat6d> dRelDH;
        std::vector<Mat6d> dRelDT;
    };

    struct FrameRelLinData {
        Mat6d hpp;
        Vec6d bp;
        std::vector<int> lmId;
        std::vector<Mat63d> hpl;
        FrameRelLinData() {
            hpp.setZero();
            bp.setZero();
        }

    };

    struct RelLinData : public RelLinDataBase {
        RelLinData(size_t numKeypoints, size_t numRelPoses) {
            hll.reserve(numKeypoints);
            hllinv.reserve(numKeypoints);
            bl.reserve(numKeypoints);
            lmToObs.reserve(numKeypoints);
            hpppl.reserve(numRelPoses);
            order.reserve(numRelPoses);
            dRelDH.reserve(numRelPoses);
            dRelDT.reserve(numRelPoses);
            error = 0;
        }
        void invertKeypointHessians() {
            for (const auto& [kptIdx, hll] : hll) {
                Mat3d Hll_inv;
                Hll_inv.setIdentity();
                hll.ldlt().solveInPlace(Hll_inv);
                hllinv[kptIdx] = Hll_inv;
            }
        }

        using RelLinDataBase::dRelDH;
        using RelLinDataBase::dRelDT;
        using RelLinDataBase::order;
        std::unordered_map<int, Mat3d> hll;
        std::unordered_map<int, Mat3d> hllinv;
        std::unordered_map<int, Vec3d> bl;
        std::unordered_map<int, std::vector<std::pair<size_t, size_t>>> lmToObs;
        std::vector<FrameRelLinData> hpppl;
        double error;
    };

    struct FrameAbsLinData {
        Mat6d hphph;
        Vec6d bph;
        Mat6d hptpt;
        Vec6d bpt;
        Mat6d hphpt;
        std::vector<int> lmId;
        std::vector<Mat63d> hphl;
        std::vector<Mat63d> hptl;
        FrameAbsLinData() {
            hphph.setZero();
            hptpt.setZero();
            hphpt.setZero();
            bph.setZero();
            bpt.setZero();
        }

    };

    struct AbsLinData {
        AbsLinData(size_t numKeypoints, size_t numRelPoses) {
            hll.reserve(numKeypoints);
            hllinv.reserve(numKeypoints);
            bl.reserve(numKeypoints);
            lmToObs.reserve(numKeypoints);
            hpppl.reserve(numRelPoses);
            order.reserve(numRelPoses);
            error = 0;
        }
        void invertKeypointHessians() {
            for (const auto& [kptIdx, hll] : hll) {
                Mat3d Hll_inv;
                Hll_inv.setIdentity();
                hll.ldlt().solveInPlace(Hll_inv);
                hllinv[kptIdx] = Hll_inv;
            }
        }

        std::vector<std::pair<TimeCamId, TimeCamId>> order;
        std::unordered_map<int, Mat3d> hll;
        std::unordered_map<int, Mat3d> hllinv;
        std::unordered_map<int, Vec3d> bl;
        std::unordered_map<int, std::vector<std::pair<size_t, size_t>>> lmToObs;
        std::vector<FrameAbsLinData> hpppl;
        double error;
    };
    using BundleAdjustmentBase::computeDelta;
    void linearizeHelper(
        std::vector<RelLinData>& rldVec,
        const std::unordered_map<TimeCamId, std::map<TimeCamId, std::set<KeypointId>>>& obsToLin,
        double& error
   ) const {
        linearizeHelperStatic(rldVec, obsToLin, this, error);
    }
    void linearizeAbsHelper(
        std::vector<AbsLinData>& aldVec,
        const std::unordered_map<TimeCamId, std::map<TimeCamId, std::set<KeypointId>>>& obsToLin,
        double& error
   ) const {
        linearizeHelperAbsStatic(aldVec, obsToLin, this, error);
    }
    static void linearizeHelperStatic(
        std::vector<RelLinData>& rldVec,
        const std::unordered_map<TimeCamId, std::map<TimeCamId, std::set<KeypointId>>>& obsToLin,
        const BundleAdjustmentBase* baBase,
        double& error
   );
    void linearizeHelperAbs(
        std::vector<AbsLinData>& aldVec,
        const std::unordered_map<TimeCamId, std::map<TimeCamId, std::set<KeypointId>>>& obsToLin,
        double& error
   ) const {
        linearizeHelperAbsStatic(aldVec, obsToLin, this, error);
    }
    static void linearizeHelperAbsStatic(
        std::vector<AbsLinData>& aldVec,
        const std::unordered_map<TimeCamId, std::map<TimeCamId, std::set<KeypointId>>>& obsToLin,
        const BundleAdjustmentBase* baBase,
        double& error
   );
    static void linearizeRel(const RelLinData& rld, MatXd& H, VecXd& b);
    static void updatePoints(
        const AbsOrderMap& aom,
        const RelLinData& rld,
        const VecXd& inc,
        LandmarkDatabase& lmdb,
        double* lDiff = nullptr
   );
    static void updatePointsAbs(
        const AbsOrderMap& aom,
        const AbsLinData& ald,
        const VecXd& inc,
        LandmarkDatabase& lmdb,
        double* lDiff = nullptr
   );
    static Eigen::VectorXd checkNullspace(
        const MatXd& H,
        const VecXd& b,
        const AbsOrderMap& margOrder,
        const std::map<int64_t, PoseVelBiasStateWithLin>& frameStates,
        const std::map<int64_t, PoseStateWithLin>& framePoses,
        bool verbose = true
   );
    static Eigen::VectorXd checkEigenvalues(const MatXd& H, bool verbose = true);
    static void computeImuError(
        const AbsOrderMap& aom,
        double& imuError,
        double& bgError,
        double& baError,
        const std::map<int64_t, PoseVelBiasStateWithLin>& states,
        const std::map<int64_t, IntegratedImuMeasurement>& imuMeas,
        const Vec3d& gyroBiasWeight,
        const Vec3d& accelBiasWeight,
        const Vec3d& g
   );
    static void linearizeAbs(
        const MatXd& rel_H,
        const VecXd& relB,
        const RelLinDataBase& rld,
        const AbsOrderMap& aom,
        DenseAccumulator& accum
   ) {
        for (size_t i = 0; i < rld.order.size(); ++ i) {
            const TimeCamId& tcidH = rld.order[i].first;
            const TimeCamId& tcidTi = rld.order[i].second;
            int absHIdx = aom.absOrderMap.at(tcidH.frameId).first;
            int absTiIdx = aom.absOrderMap.at(tcidTi.frameId).first;
            accum.addB<6>(
                absHIdx, rld.dRelDH[i].transpose() * relB.segment<6>(i * 6));
            accum.addB<6>(
                absTiIdx, rld.dRelDT[i].transpose() * relB.segment<6>(i * 6));
            for (size_t j = 0; j < rld.order.size(); ++ j) {
                ensure(rld.order[i].first == rld.order[j].first, "VIO check failed");
                const TimeCamId& tcidTj = rld.order[j].second;
                int absTjIdx = aom.absOrderMap.at(tcidTj.frameId).first;
                if (tcidH.frameId == tcidTi.frameId || tcidH.frameId == tcidTj.frameId)
                    continue;
                accum.addH<6, 6>(absHIdx,
                    absHIdx,
                    rld.dRelDH[i].transpose() *
                        rel_H.block<6, 6>(6 * i, 6 * j) * rld.dRelDH[j]);
                accum.addH<6, 6>(absTiIdx,
                    absHIdx,
                    rld.dRelDT[i].transpose() *
                        rel_H.block<6, 6>(6 * i, 6 * j) * rld.dRelDH[j]);
                accum.addH<6, 6>(absHIdx,
                    absTjIdx,
                    rld.dRelDH[i].transpose() *
                        rel_H.block<6, 6>(6 * i, 6 * j) * rld.dRelDT[j]);
                accum.addH<6, 6>(absTiIdx,
                    absTjIdx,
                    rld.dRelDT[i].transpose() *
                        rel_H.block<6, 6>(6 * i, 6 * j) * rld.dRelDT[j]);
            }
        }
    }
    static void linearizeAbs(const AbsLinData& ald, const AbsOrderMap& aom, DenseAccumulator& accum) {
        for (size_t i = 0; i < ald.order.size(); ++ i) {
            const TimeCamId& tcidH = ald.order[i].first;
            const TimeCamId& tcidTi = ald.order[i].second;
            int absHIdx = aom.absOrderMap.at(tcidH.frameId).first;
            int absTiIdx = aom.absOrderMap.at(tcidTi.frameId).first;
            const FrameAbsLinData& fald = ald.hpppl.at(i);
            accum.addH<6, 6>(absHIdx, absHIdx, fald.hphph);
            accum.addH<6, 6>(absTiIdx, absTiIdx, fald.hptpt);
            accum.addH<6, 6>(absHIdx, absTiIdx, fald.hphpt);
            accum.addH<6, 6>(absTiIdx, absHIdx, fald.hphpt.transpose());
            accum.addB<6>(absHIdx, fald.bph);
            accum.addB<6>(absTiIdx, fald.bpt);
            for (size_t j = 0; j < fald.lmId.size(); ++ j) {
                Eigen::Matrix<double, 6, 3> H_phl_H_ll_inv, H_ptl_H_ll_inv;
                int lmId = fald.lmId.at(j);
                H_phl_H_ll_inv = fald.hphl[j] * ald.hllinv.at(lmId);
                H_ptl_H_ll_inv = fald.hptl[j] * ald.hllinv.at(lmId);
                accum.addB<6>(absHIdx, -H_phl_H_ll_inv * ald.bl.at(lmId));
                accum.addB<6>(absTiIdx, -H_ptl_H_ll_inv * ald.bl.at(lmId));
                const auto& otherObs = ald.lmToObs.at(lmId);
                for (size_t k = 0; k < otherObs.size(); ++ k) {
                    int otherFrameIdx = otherObs[k].first;
                    int otherLmIdx = otherObs[k].second;
                    const FrameAbsLinData& faldOther = ald.hpppl.at(otherFrameIdx);
                    const TimeCamId& tcidHk = ald.order.at(otherFrameIdx).first;
                    const TimeCamId& tcidTk = ald.order.at(otherFrameIdx).second;
                    ensure(tcidHk.frameId == tcidH.frameId && tcidHk.camId == tcidH.camId, "VIO check failed");
                    int absTkIdx = aom.absOrderMap.at(tcidTk.frameId).first;
                    Eigen::Matrix<double, 3, 6> H_l_ph_other = faldOther.hphl[otherLmIdx].transpose();
                    Eigen::Matrix<double, 3, 6> H_l_pt_other = faldOther.hptl[otherLmIdx].transpose();
                    accum.addH<6, 6>(absHIdx, absHIdx, -H_phl_H_ll_inv * H_l_ph_other);
                    accum.addH<6, 6>(absTiIdx, absHIdx, -H_ptl_H_ll_inv * H_l_ph_other);
                    accum.addH<6, 6>(absHIdx, absTkIdx, -H_phl_H_ll_inv * H_l_pt_other);
                    accum.addH<6, 6>(absTiIdx, absTkIdx, -H_ptl_H_ll_inv * H_l_pt_other);
                }
            }
        }
    }

    struct LinearizeAbsReduce {
        using RelLinConstDataIter = typename std::vector<RelLinData>::const_iterator;

        LinearizeAbsReduce(const AbsOrderMap& aom)
            : aom(aom)
        {
            accum.reset(aom.totalSize);
        }

        LinearizeAbsReduce(const LinearizeAbsReduce& other, tbb::split)
            : aom(other.aom)
        {
            accum.reset(aom.totalSize);
        }
        void operator()(const tbb::blocked_range<RelLinConstDataIter>& range) {
            for (const RelLinData& rld : range) {
                MatXd rel_H;
                VecXd relB;
                linearizeRel(rld, rel_H, relB);
                linearizeAbs(rel_H, relB, rld, aom, accum);
            }
        }
        void join(LinearizeAbsReduce& rhs) { accum.join(rhs.accum); }

        const AbsOrderMap& aom;
        DenseAccumulator accum;
    };

    struct LinearizeAbsReduce2 {
        using AbsLinDataConstIter = typename std::vector<AbsLinData>::const_iterator;

        LinearizeAbsReduce2(const AbsOrderMap& aom)
            : aom(aom)
        {
            accum.reset(aom.totalSize);
        }

        LinearizeAbsReduce2(const LinearizeAbsReduce2& other, tbb::split)
            : aom(other.aom)
        {
            accum.reset(aom.totalSize);
        }
        void operator()(const tbb::blocked_range<AbsLinDataConstIter>& range) {
            for (const AbsLinData& ald : range) {
                linearizeAbs(ald, aom, accum);
            }
        }
        void join(LinearizeAbsReduce2& rhs) { accum.join(rhs.accum); }

        const AbsOrderMap& aom;
        DenseAccumulator accum;
    };
};
inline void ScBundleAdjustmentBase::updatePoints(
    const AbsOrderMap& aom, const RelLinData& rld, const VecXd& inc, LandmarkDatabase& lmdb, double* lDiff) {
    VecXd relInc;
    relInc.setZero(rld.order.size() * 6);
    for (size_t i = 0; i < rld.order.size(); ++ i) {
        const TimeCamId& tcidH = rld.order[i].first;
        const TimeCamId& tcidT = rld.order[i].second;
        if (tcidH.frameId != tcidT.frameId) {
            int absHIdx = aom.absOrderMap.at(tcidH.frameId).first;
            int absTIdx = aom.absOrderMap.at(tcidT.frameId).first;
            Vec6d incP = rld.dRelDH[i] * inc.segment<6>(absHIdx) +
                rld.dRelDT[i] * inc.segment<6>(absTIdx);
            relInc.segment<6>(i * 6) = incP;
            if (lDiff) {
                const FrameRelLinData& frld = rld.hpppl[i];
                *lDiff -= -incP.dot((frld.bp - double(0.5) * (frld.hpp * incP)));
            }
        }
    }
    for (const auto& kv : rld.lmToObs) {
        int lmIdx = kv.first;
        const auto& lmObs = kv.second;
        Vec3d H_l_p_x;
        H_l_p_x.setZero();
        for (size_t k = 0; k < lmObs.size(); ++ k) {
            int relIdx = lmObs[k].first;
            const FrameRelLinData& frld = rld.hpppl.at(relIdx);
            const Eigen::Matrix<double, 6, 3>& H_p_l_other = frld.hpl.at(lmObs[k].second);
            H_l_p_x += H_p_l_other.transpose() * relInc.segment<6>(relIdx * 6);
        }
        Vec3d incL = -(rld.hllinv.at(lmIdx) * (rld.bl.at(lmIdx) - H_l_p_x));
        Keypoint& kpt = lmdb.getLandmark(lmIdx);
        kpt.direction += incL.head<2>();
        kpt.invDist += incL[2];
        kpt.invDist = std::max(double(0), kpt.invDist);
        if (lDiff) {
            *lDiff -= incL.transpose() * (rld.bl.at(lmIdx) + double(0.5) * (rld.hll.at(lmIdx) * incL) - H_l_p_x);
        }
    }
}
inline void ScBundleAdjustmentBase::updatePointsAbs(
    const AbsOrderMap& aom, const AbsLinData& ald, const VecXd& inc, LandmarkDatabase& lmdb, double* lDiff) {
    if (lDiff) {
        for (size_t i = 0; i < ald.order.size(); ++ i) {
            const TimeCamId& tcidH = ald.order[i].first;
            const TimeCamId& tcidT = ald.order[i].second;
            int absHIdx = aom.absOrderMap.at(tcidH.frameId).first;
            int absTIdx = aom.absOrderMap.at(tcidT.frameId).first;
            auto incH = inc.segment<6>(absHIdx);
            auto incT = inc.segment<6>(absTIdx);
            const FrameAbsLinData& fald = ald.hpppl[i];
            *lDiff -= -incH.dot((fald.bph - double(0.5) * (fald.hphph * incH)) - fald.hphpt * incT) -
                incT.dot((fald.bpt - double(0.5) * (fald.hptpt * incT)));
        }
    }
    for (const auto& kv : ald.lmToObs) {
        int lmIdx = kv.first;
        const auto& lmObs = kv.second;
        Vec3d H_l_p_x;
        H_l_p_x.setZero();
        for (size_t k = 0; k < lmObs.size(); ++ k) {
            int relIdx = lmObs[k].first;
            const TimeCamId& tcidH = ald.order.at(relIdx).first;
            const TimeCamId& tcidT = ald.order.at(relIdx).second;
            int absHIdx = aom.absOrderMap.at(tcidH.frameId).first;
            int absTIdx = aom.absOrderMap.at(tcidT.frameId).first;
            auto incH = inc.segment<6>(absHIdx);
            auto incT = inc.segment<6>(absTIdx);
            const FrameAbsLinData& fald = ald.hpppl.at(relIdx);
            const Eigen::Matrix<double, 6, 3>& H_ph_l_other = fald.hphl.at(lmObs[k].second);
            const Eigen::Matrix<double, 6, 3>& H_pt_l_other = fald.hptl.at(lmObs[k].second);
            H_l_p_x += H_ph_l_other.transpose() * incH;
            H_l_p_x += H_pt_l_other.transpose() * incT;
        }
        Vec3d incL = -(ald.hllinv.at(lmIdx) * (ald.bl.at(lmIdx) - H_l_p_x));
        Keypoint& kpt = lmdb.getLandmark(lmIdx);
        kpt.direction += incL.head<2>();
        kpt.invDist += incL[2];
        kpt.invDist = std::max(double(0), kpt.invDist);
        if (lDiff) {
            *lDiff -= incL.transpose() * (ald.bl.at(lmIdx) + double(0.5) * (ald.hll.at(lmIdx) * incL) - H_l_p_x);
        }
    }
}
inline void ScBundleAdjustmentBase::linearizeHelperStatic(
    std::vector<RelLinData>& rldVec,
    const std::unordered_map<TimeCamId, std::map<TimeCamId, std::set<KeypointId>>>& obsToLin,
    const BundleAdjustmentBase* baBase,
    double& error
) {
    error = 0;
    rldVec.clear();
    std::vector<TimeCamId> obsTcidVec;
    for (const auto& kv : obsToLin) {
        obsTcidVec.emplace_back(kv.first);
        rldVec.emplace_back(baBase->lmdb.numLandmarks(), kv.second.size());
    }
    tbb::parallel_for(tbb::blocked_range<size_t>(0, obsTcidVec.size()), [&](const tbb::blocked_range<size_t>& range) {
        for (size_t r = range.begin(); r != range.end(); ++ r) {
            auto kv = obsToLin.find(obsTcidVec[r]);
            RelLinData& rld = rldVec[r];
            rld.error = double(0);
            const TimeCamId& tcidH = kv->first;
            for (const auto& obsKv : kv->second) {
                const TimeCamId& tcidT = obsKv.first;
                rld.order.emplace_back(std::make_pair(tcidH, tcidT));
                Mat4d T_t_h;
                if (tcidH != tcidT) {
                    PoseStateWithLin stateH = baBase->getPoseStateWithLin(tcidH.frameId);
                    PoseStateWithLin stateT = baBase->getPoseStateWithLin(tcidT.frameId);
                    Mat6d dRelDH, dRelDT;
                    Sophus::SE3d T_t_h_sophus = computeRelPose(stateH.getPoseLin(),
                        baBase->calib.tIC[tcidH.camId],
                        stateT.getPoseLin(),
                        baBase->calib.tIC[tcidT.camId],
                        &dRelDH,
                        &dRelDT);
                    rld.dRelDH.emplace_back(dRelDH);
                    rld.dRelDT.emplace_back(dRelDT);
                    if (stateH.isLinearized() || stateT.isLinearized()) {
                        T_t_h_sophus = computeRelPose(stateH.getPose(),
                            baBase->calib.tIC[tcidH.camId],
                            stateT.getPose(),
                            baBase->calib.tIC[tcidT.camId]);
                    }
                    T_t_h = T_t_h_sophus.matrix();
                } else {
                    T_t_h.setIdentity();
                    rld.dRelDH.emplace_back(Mat6d::Zero());
                    rld.dRelDT.emplace_back(Mat6d::Zero());
                }
                FrameRelLinData frld;
                std::visit(
                    [&](const auto& cam) {
                        for (KeypointId kptId : obsKv.second) {
                            const Keypoint& kptPos = baBase->lmdb.getLandmark(kptId);
                            const Vec2d& kptObs = kptPos.obs.at(tcidT);
                            Vec2d res;
                            Eigen::Matrix<double, 2, 6> dResDXi;
                            Eigen::Matrix<double, 2, 3> dResDP;
                            bool valid = linearizePoint(kptObs, kptPos, T_t_h, cam, res, &dResDXi, &dResDP);
                            if (valid) {
                                double e = res.norm();
                                double huberWeight = e < baBase->huberThresh ? double(1.0) : baBase->huberThresh / e;
                                double obsWeight = huberWeight / (baBase->obsStdDev * baBase->obsStdDev);
                                rld.error += double(0.5) * (2 - huberWeight) * obsWeight * res.transpose() * res;
                                if (rld.hll.count(kptId) == 0) {
                                    rld.hll[kptId].setZero();
                                    rld.bl[kptId].setZero();
                                }
                                rld.hll[kptId] += obsWeight * dResDP.transpose() * dResDP;
                                rld.bl[kptId] += obsWeight * dResDP.transpose() * res;
                                frld.hpp += obsWeight * dResDXi.transpose() * dResDXi;
                                frld.bp += obsWeight * dResDXi.transpose() * res;
                                frld.hpl.emplace_back(obsWeight * dResDXi.transpose() * dResDP);
                                frld.lmId.emplace_back(kptId);
                                rld.lmToObs[kptId].emplace_back(rld.hpppl.size(), frld.lmId.size() - 1);
                            }
                        }
                    },
                    baBase->calib.intrinsics[tcidT.camId].variant);
                rld.hpppl.emplace_back(frld);
            }
            rld.invertKeypointHessians();
        }
    });
    for (const auto& rld : rldVec)
        error += rld.error;
}
inline void ScBundleAdjustmentBase::linearizeHelperAbsStatic(
    std::vector<AbsLinData>& aldVec,
    const std::unordered_map<TimeCamId, std::map<TimeCamId, std::set<KeypointId>>>& obsToLin,
    const BundleAdjustmentBase* baBase,
    double& error
) {
    error = 0;
    aldVec.clear();
    std::vector<TimeCamId> obsTcidVec;
    for (const auto& kv : obsToLin) {
        obsTcidVec.emplace_back(kv.first);
        aldVec.emplace_back(baBase->lmdb.numLandmarks(), kv.second.size());
    }
    tbb::parallel_for(tbb::blocked_range<size_t>(0, obsTcidVec.size()), [&](const tbb::blocked_range<size_t>& range) {
        for (size_t r = range.begin(); r != range.end(); ++ r) {
            auto kv = obsToLin.find(obsTcidVec[r]);
            AbsLinData& ald = aldVec[r];
            ald.error = double(0);
            const TimeCamId& tcidH = kv->first;
            for (const auto& obsKv : kv->second) {
                const TimeCamId& tcidT = obsKv.first;
                ald.order.emplace_back(std::make_pair(tcidH, tcidT));
                Mat4d T_t_h;
                Mat6d dRelDH, dRelDT;
                if (tcidH != tcidT) {
                    PoseStateWithLin stateH = baBase->getPoseStateWithLin(tcidH.frameId);
                    PoseStateWithLin stateT = baBase->getPoseStateWithLin(tcidT.frameId);
                    Sophus::SE3d T_t_h_sophus = computeRelPose(stateH.getPoseLin(),
                        baBase->calib.tIC[tcidH.camId],
                        stateT.getPoseLin(),
                        baBase->calib.tIC[tcidT.camId],
                        &dRelDH,
                        &dRelDT);
                    if (stateH.isLinearized() || stateT.isLinearized()) {
                        T_t_h_sophus = computeRelPose(stateH.getPose(),
                            baBase->calib.tIC[tcidH.camId],
                            stateT.getPose(),
                            baBase->calib.tIC[tcidT.camId]);
                    }
                    T_t_h = T_t_h_sophus.matrix();
                } else {
                    T_t_h.setIdentity();
                    dRelDH.setZero();
                    dRelDT.setZero();
                }
                FrameAbsLinData fald;
                std::visit(
                    [&](const auto& cam) {
                        for (KeypointId kptId : obsKv.second) {
                            const Keypoint& kptPos = baBase->lmdb.getLandmark(kptId);
                            const Vec2d& kptObs = kptPos.obs.at(tcidT);
                            Vec2d res;
                            Eigen::Matrix<double, 2, 6> dResDXiH, dResDXiT;
                            Eigen::Matrix<double, 2, 3> dResDP;
                            bool valid;
                            {
                                Eigen::Matrix<double, 2, 6> dResDXi;
                                valid = linearizePoint(kptObs, kptPos, T_t_h, cam, res, &dResDXi, &dResDP);
                                dResDXiH = dResDXi * dRelDH;
                                dResDXiT = dResDXi * dRelDT;
                            }
                            if (valid) {
                                double e = res.norm();
                                double huberWeight = e < baBase->huberThresh ? double(1.0) : baBase->huberThresh / e;
                                double obsWeight = huberWeight / (baBase->obsStdDev * baBase->obsStdDev);
                                ald.error += double(0.5) * (2 - huberWeight) * obsWeight * res.transpose() * res;
                                if (ald.hll.count(kptId) == 0) {
                                    ald.hll[kptId].setZero();
                                    ald.bl[kptId].setZero();
                                }
                                ald.hll[kptId] += obsWeight * dResDP.transpose() * dResDP;
                                ald.bl[kptId] += obsWeight * dResDP.transpose() * res;
                                fald.hphph += obsWeight * dResDXiH.transpose() * dResDXiH;
                                fald.hptpt += obsWeight * dResDXiT.transpose() * dResDXiT;
                                fald.hphpt += obsWeight * dResDXiH.transpose() * dResDXiT;
                                fald.bph += obsWeight * dResDXiH.transpose() * res;
                                fald.bpt += obsWeight * dResDXiT.transpose() * res;
                                fald.hphl.emplace_back(obsWeight * dResDXiH.transpose() * dResDP);
                                fald.hptl.emplace_back(obsWeight * dResDXiT.transpose() * dResDP);
                                fald.lmId.emplace_back(kptId);
                                ald.lmToObs[kptId].emplace_back(ald.hpppl.size(), fald.lmId.size() - 1);
                            }
                        }
                    },
                    baBase->calib.intrinsics[tcidT.camId].variant);
                ald.hpppl.emplace_back(fald);
            }
            ald.invertKeypointHessians();
        }
    });
    for (const auto& rld : aldVec)
        error += rld.error;
}
inline void ScBundleAdjustmentBase::linearizeRel(const RelLinData& rld, MatXd& H, VecXd& b) {
    size_t msize = rld.order.size();
    H.setZero(6 * msize, 6 * msize);
    b.setZero(6 * msize);
    for (size_t i = 0; i < rld.order.size(); ++ i) {
        const FrameRelLinData& frld = rld.hpppl.at(i);
        H.block<6, 6>(6 * i, 6 * i) += frld.hpp;
        b.segment<6>(6 * i) += frld.bp;
        for (size_t j = 0; j < frld.lmId.size(); ++ j) {
            Eigen::Matrix<double, 6, 3> H_pl_H_ll_inv;
            int lmId = frld.lmId[j];
            H_pl_H_ll_inv = frld.hpl[j] * rld.hllinv.at(lmId);
            b.segment<6>(6 * i) -= H_pl_H_ll_inv * rld.bl.at(lmId);
            const auto& otherObs = rld.lmToObs.at(lmId);
            for (size_t k = 0; k < otherObs.size(); ++ k) {
                const FrameRelLinData& frldOther = rld.hpppl.at(otherObs[k].first);
                int otherI = otherObs[k].first;
                Eigen::Matrix<double, 3, 6> H_l_p_other = frldOther.hpl[otherObs[k].second].transpose();
                H.block<6, 6>(6 * i, 6 * otherI) -= H_pl_H_ll_inv * H_l_p_other;
            }
        }
    }
}
inline Eigen::VectorXd ScBundleAdjustmentBase::checkNullspace(
    const MatXd& H,
    const VecXd& b,
    const AbsOrderMap& order,
    const std::map<int64_t, PoseVelBiasStateWithLin>& frameStates,
    const std::map<int64_t, PoseStateWithLin>& framePoses,
    bool verbose
) {
    ensure(size_t(H.cols()) == order.totalSize, "VIO check failed");
    size_t margSize = order.totalSize;
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
    for (const auto& kv : order.absOrderMap) {
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
    for (const auto& kv : order.absOrderMap) {
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
    MatXd H_d = H.cast<double>();
    VecXd bD = b.cast<double>();
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
inline Eigen::VectorXd ScBundleAdjustmentBase::checkEigenvalues(const MatXd& H, bool verbose) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(H.cast<double>());
    if (eigensolver.info() != Eigen::Success) {
        throw std::runtime_error("eigen solver failed");
    }
    if (verbose) {
        std::cout << "EV:\n" << eigensolver.eigenvalues() << std::endl;
    }
    return eigensolver.eigenvalues();
}
inline void ScBundleAdjustmentBase::computeImuError(
    const AbsOrderMap& aom,
    double& imuError,
    double& bgError,
    double& baError,
    const std::map<int64_t, PoseVelBiasStateWithLin>& states,
    const std::map<int64_t, IntegratedImuMeasurement>& imuMeas,
    const Vec3d& gyroBiasWeight,
    const Vec3d& accelBiasWeight,
    const Vec3d& g
) {
    imuError = 0;
    bgError = 0;
    baError = 0;
    for (const auto& kv : imuMeas) {
        if (kv.second.getDtNs() != 0) {
            int64_t startT = kv.second.getStartTNs();
            int64_t endT = kv.second.getStartTNs() + kv.second.getDtNs();
            if (aom.absOrderMap.count(startT) == 0 || aom.absOrderMap.count(endT) == 0)
                continue;
            PoseVelBiasStateWithLin startState = states.at(startT);
            PoseVelBiasStateWithLin endState = states.at(endT);
            const Vec9d res = kv.second.residual(startState.getState(),
                g,
                endState.getState(),
                startState.getState().biasGyro,
                startState.getState().biasAccel);
            imuError += double(0.5) * res.transpose() * kv.second.getCovInv() * res;
            double dt = kv.second.getDtNs() * double(1e-9);
            {
                Vec3d gyroBiasWeightDt = gyroBiasWeight / dt;
                Vec3d resBg = startState.getState().biasGyro - endState.getState().biasGyro;
                bgError += double(0.5) * resBg.transpose() * gyroBiasWeightDt.asDiagonal() * resBg;
            }
            {
                Vec3d accelBiasWeightDt = accelBiasWeight / dt;
                Vec3d resBa = startState.getState().biasAccel - endState.getState().biasAccel;
                baError += double(0.5) * resBa.transpose() * accelBiasWeightDt.asDiagonal() * resBa;
            }
        }
    }
}

} // arc
