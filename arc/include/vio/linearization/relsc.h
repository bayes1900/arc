#pragma once

#include <base/base.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <vio/estimator/accumulator.h>
#include <vio/estimator/stats.h>
#include <vio/estimator/landmarks.h>
#include <vio/estimator/projection.h>
#include <vio/estimator/scba.h>
#include <vio/estimator/sqrtba.h>
#include <vio/linearization/imublock.h>
#include <vio/linearization/landmarkblock.h>
#include <vio/linearization/base.h>
#include <vio/linearization/absqr.h>
#include <vio/linearization/abssc.h>

namespace arc {

class ImuBlock;

class LinearizationRelSC : public LinearizationBase {
public:
    using Base = LinearizationBase;
    using LandmarkBlockPtr = std::unique_ptr<LandmarkBlock>;
    using ImuBlockPtr = std::unique_ptr<ImuBlock>;
    using RelLinData = typename ScBundleAdjustmentBase::RelLinData;
    using typename Base::Options;

    LinearizationRelSC(
        BundleAdjustmentBase* estimator,
        const AbsOrderMap& aom,
        const Options& options,
        const MargLinData* margLinData,
        const ImuLinData* imuLinData,
        const std::set<int64_t>* usedFrames,
        const std::unordered_set<KeypointId>* lostLandmarks,
        int64_t lastStateToMarg
    )
        : options_(options)
        , estimator_(estimator)
        , lmdb_(estimator->lmdb)
        , calib_(estimator->calib)
        , aom_(aom)
        , usedFrames_(usedFrames)
        , margLinData_(margLinData)
        , imuLinData_(imuLinData)
        , lostLandmarks_(lostLandmarks)
        , lastStateToMarg_(lastStateToMarg)
    {
        ensure(options.lbOptions.huberParameter == estimator->huberThresh, "huber threshold mismatch");
        ensure(options.lbOptions.obsStdDev == estimator->obsStdDev, "obs std dev mismatch");
        if (imuLinData) {
            for (const auto& kv : imuLinData->imuMeas) {
                imuBlocks_.emplace_back(new ImuBlock(kv.second, imuLinData, aom));
            }
        }
    }
    ~LinearizationRelSC() = default;
    LinearizationRelSC(const LinearizationRelSC&) = default;
    LinearizationRelSC(LinearizationRelSC&&) = default;
    LinearizationRelSC& operator=(const LinearizationRelSC&) = default;
    LinearizationRelSC& operator=(LinearizationRelSC&&) = default;

    void logProblemStats(ExecutionStats& stats) const override {
        (void)(stats);
    }
    double linearizeProblem(bool* numericallyValid) override {
        margScaling_ = VecXd();
        std::unordered_map<TimeCamId, std::map<TimeCamId, std::set<KeypointId>>> obsToLin;
        if (usedFrames_) {
            const auto& obs = lmdb_.getObservations();
            if (lostLandmarks_) {
                for (auto it = obs.cbegin(); it != obs.cend(); ++ it) {
                    if (usedFrames_->count(it->first.frameId) > 0) {
                        for (auto it2 = it->second.cbegin(); it2 != it->second.cend(); ++ it2) {
                            if (it2->first.frameId <= lastStateToMarg_)
                                obsToLin[it->first].emplace(*it2);
                        }
                    } else {
                        std::map<TimeCamId, std::set<KeypointId>> lostObsMap;
                        for (auto it2 = it->second.cbegin(); it2 != it->second.cend(); ++ it2) {
                            for (const auto& lmId : it2->second) {
                                if (lostLandmarks_->count(lmId)) {
                                    if (it2->first.frameId <= lastStateToMarg_)
                                        lostObsMap[it2->first].emplace(lmId);
                                }
                            }
                        }
                        if (!lostObsMap.empty()) {
                            obsToLin[it->first] = lostObsMap;
                        }
                    }
                }
            } else {
                for (auto it = obs.cbegin(); it != obs.cend(); ++ it) {
                    if (usedFrames_->count(it->first.frameId) > 0) {
                        for (auto it2 = it->second.cbegin(); it2 != it->second.cend(); ++ it2) {
                            if (it2->first.frameId <= lastStateToMarg_)
                                obsToLin[it->first].emplace(*it2);
                        }
                    }
                }
            }
        } else {
            obsToLin = lmdb_.getObservations();
        }
        double error;
        ScBundleAdjustmentBase::linearizeHelperStatic(rldVec_, obsToLin, estimator_, error);
        if (numericallyValid)
            *numericallyValid = true;
        if (imuLinData_) {
            for (auto& imuBlock : imuBlocks_) {
                error += imuBlock->linearizeImu(estimator_->frameStates);
            }
        }
        if (margLinData_) {
            double margPriorError;
            estimator_->computeMargPriorError(*margLinData_, margPriorError);
            error += margPriorError;
        }
        return error;
    }
    void performQR() override {
    }
    double backSubstitute(const VecXd& poseInc) override {
        ensure(poseInc.size() == static_cast<Eigen::Index>(aom_.totalSize), "VIO check failed");
        tbb::blocked_range<size_t> keysRange(0, rldVec_.size());
        auto updatePointsFunc = [&](const tbb::blocked_range<size_t>& r, double lDiff) {
            for (size_t i = r.begin(); i != r.end(); ++ i) {
                const auto& rld = rldVec_[i];
                ScBundleAdjustmentBase::updatePoints(aom_, rld, -poseInc, lmdb_, &lDiff);
            }
            return lDiff;
        };
        double lDiff = tbb::parallel_reduce(keysRange, double(0), updatePointsFunc, std::plus<double>());
        if (imuLinData_) {
            for (auto& imuBlock : imuBlocks_) {
                imuBlock->backSubstitute(poseInc, lDiff);
            }
        }
        if (margLinData_) {
            size_t margSize = margLinData_->h.cols();
            VecXd poseIncMarg = poseInc.head(margSize);
            lDiff += estimator_->computeMargPriorModelCostChange(*margLinData_, margScaling_, poseIncMarg);
        }
        return lDiff;
    }
    void scaleJlCols() {
        throw std::logic_error("not implemented");
    }
    void scaleJpCols(const VecXd& jacobianScaling) {
        (void)(jacobianScaling);
        throw std::logic_error("not implemented");
    }
    void setLandmarkDamping(double lambda) {
        (void)(lambda);
        throw std::logic_error("not implemented");
    }
    void getDenseQ2JpQ2r(MatXd& Q2Jp, VecXd& Q2r) const override {
        MatXd H;
        VecXd b;
        getDenseHB(H, b);
        Eigen::LDLT<Eigen::Ref<MatXd>> ldlt(H);
        VecXd D_sqrt = ldlt.vectorD().array().max(0).sqrt().matrix();
        Q2Jp.setIdentity(H.rows(), H.cols());
        Q2Jp = ldlt.transpositionsP() * Q2Jp;
        Q2Jp = ldlt.matrixU() * Q2Jp;
        Q2Jp = D_sqrt.asDiagonal() * Q2Jp;
        Q2r = ldlt.transpositionsP() * b;
        ldlt.matrixL().solveInPlace(Q2r);
        for (int i = 0; i < Q2r.size(); ++ i) {
            if (D_sqrt(i) > std::sqrt(std::numeric_limits<double>::min())) {
                Q2r(i) /= D_sqrt(i);
            } else {
                Q2r(i) = 0;
            }
        }
    }
    void getDenseHB(MatXd& H, VecXd& b) const override {
        ScBundleAdjustmentBase::LinearizeAbsReduce loptAbs(aom_);
        tbb::blocked_range<typename std::vector<RelLinData>::const_iterator> range(rldVec_.cbegin(), rldVec_.cend());
        tbb::parallel_reduce(range, loptAbs);
        addDenseHBImu(loptAbs.accum.getH(), loptAbs.accum.getB());
        addDenseHBMargPrior(loptAbs.accum.getH(), loptAbs.accum.getB());
        H = std::move(loptAbs.accum.getH());
        b = std::move(loptAbs.accum.getB());
    }

protected:
    using PoseLinMapType = std::unordered_map<std::pair<TimeCamId, TimeCamId>, RelPoseLin>;
    using PoseLinMapTypeConstIter = typename PoseLinMapType::const_iterator;
    using HostLandmarkMapType = std::unordered_map<TimeCamId, std::vector<const LandmarkBlock*>>;

protected:
    void getDenseQ2JpQ2rMargPrior(MatXd& Q2Jp, VecXd& Q2r, size_t startIdx) const {
        if (!margLinData_)
            return;
        size_t margRows = margLinData_->h.rows();
        size_t margCols = margLinData_->h.cols();
        VecXd delta;
        estimator_->computeDelta(margLinData_->order, delta);
        if (margScaling_.rows() > 0) {
            Q2Jp.block(startIdx, 0, margRows, margCols) = margLinData_->h * margScaling_.asDiagonal();
        } else {
            Q2Jp.block(startIdx, 0, margRows, margCols) = margLinData_->h;
        }
        Q2r.segment(startIdx, margRows) = margLinData_->h * delta + margLinData_->b;
    }
    void addDenseHBMargPrior(MatXd& H, VecXd& b) const {
        if (!margLinData_)
            return;
        ensure(margScaling_.rows() == 0, "VIO check failed");
        double margPriorError;
        estimator_->linearizeMargPrior(*margLinData_, aom_, H, b, margPriorError);
    }
    void addDenseHBImu(DenseAccumulator& accum) const {
        if (!imuLinData_)
            return;
        for (const auto& imuBlock : imuBlocks_) {
            imuBlock->addDenseHB(accum);
        }
    }
    void addDenseHBImu(MatXd& H, VecXd& b) const {
        if (!imuLinData_)
            return;
        DenseAccumulator accum;
        accum.reset(b.size());
        for (const auto& imuBlock : imuBlocks_) {
            imuBlock->addDenseHB(accum);
        }
        H += accum.getH();
        b += accum.getB();
    }
    static void accumulateDenseHBRelToAbs(
        const MatXd& rel_H,
        const VecXd& relB,
        const std::vector<PoseLinMapTypeConstIter>& rpph,
        const AbsOrderMap& aom,
        DenseAccumulator& accum
    );

protected:
    Options options_;
    std::vector<ImuBlockPtr> imuBlocks_;
    const BundleAdjustmentBase* estimator_;
    LandmarkDatabase& lmdb_;
    const Calibration<double>& calib_;
    const AbsOrderMap& aom_;
    const std::set<int64_t>* usedFrames_;
    const MargLinData* margLinData_;
    const ImuLinData* imuLinData_;
    const std::unordered_set<KeypointId>* lostLandmarks_;
    int64_t lastStateToMarg_;
    std::vector<RelLinData> rldVec_;
    VecXd margScaling_;
    size_t numRowsQ2r_;
};
inline bool isLinearizationSqrt(const LinearizationType& type) {
    switch (type) {
        case LinearizationType::ABS_QR:
            return true;
        case LinearizationType::ABS_SC:
        case LinearizationType::REL_SC:
            return false;
        default:
            throw std::logic_error("not implemented");
            return false;
    }
}

} // arc
