#pragma once

#include <base/base.h>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <vio/estimator/accumulator.h>
#include <vio/estimator/stats.h>
#include <vio/estimator/landmarks.h>
#include <vio/estimator/projection.h>
#include <vio/linearization/imublock.h>
#include <vio/linearization/landmarkblock.h>
#include <vio/linearization/landmarkblockdynamic.h>
#include <vio/linearization/base.h>

namespace arc {

class ImuBlock;

class LinearizationAbsQR : public LinearizationBase {
public:
    using Base = LinearizationBase;
    using LandmarkBlockPtr = std::unique_ptr<LandmarkBlock>;
    using ImuBlockPtr = std::unique_ptr<ImuBlock>;
    using typename Base::Options;
    LinearizationAbsQR(
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
        , framePoses_(estimator->framePoses)
        , calib_(estimator->calib)
        , aom_(aom)
        , usedFrames_(usedFrames)
        , margLinData_(margLinData)
        , imuLinData_(imuLinData)
    {
        (void)(lastStateToMarg);
        ensure(options.lbOptions.huberParameter == estimator->huberThresh, "huber threshold mismatch");
        ensure(options.lbOptions.obsStdDev == estimator->obsStdDev, "obs std dev mismatch");
        for (const auto& [tcidH, targetMap] : lmdb_.getObservations()) {
            const size_t hostIdx = hostToIdx_.size();
            hostToIdx_.try_emplace(tcidH, hostIdx);
            hostToLandmarkBlock_.try_emplace(tcidH);
            ensure(!targetMap.empty(), "VIO check failed");
            for (const auto& [tcidT, obs] : targetMap) {
                ensure(!obs.empty(), "VIO check failed");
                std::pair<TimeCamId, TimeCamId> key(tcidH, tcidT);
                relativePoseLin_.emplace(key, RelPoseLin());
            }
        }
        for (const auto& [tcidH, targetMap] : lmdb_.getObservations()) {
            relativePosePerHost_.emplace_back();
            for (const auto& [tcidT, _] : targetMap) {
                std::pair<TimeCamId, TimeCamId> key(tcidH, tcidT);
                auto it = relativePoseLin_.find(key);
                ensure(it != relativePoseLin_.end(), "VIO check failed");
                relativePosePerHost_.back().emplace_back(it);
            }
        }
        numCameras_ = framePoses_.size();
        landmarkIds_.clear();
        for (const auto& [k, v] : lmdb_.getLandmarks()) {
            if (usedFrames || lostLandmarks) {
                if (usedFrames && usedFrames->count(v.hostKfId.frameId)) {
                    landmarkIds_.emplace_back(k);
                } else if (lostLandmarks && lostLandmarks->count(k)) {
                    landmarkIds_.emplace_back(k);
                }
            } else {
                landmarkIds_.emplace_back(k);
            }
        }
        size_t numLandmakrs = landmarkIds_.size();
        landmarkBlocks_.resize(numLandmakrs);
        {
            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++ r) {
                    KeypointId lmId = landmarkIds_[r];
                    auto& lb = landmarkBlocks_[r];
                    auto& landmark = lmdb_.getLandmark(lmId);
                    lb = LandmarkBlock::createLandmarkBlock();
                    lb->allocateLandmark(landmark, relativePoseLin_, calib_, aom, options.lbOptions);
                }
            };
            tbb::blocked_range<size_t> range(0, numLandmakrs);
            tbb::parallel_for(range, body);
        }
        landmarkBlockIdx_.reserve(numLandmakrs);
        numRowsQ2r_ = 0;
        for (size_t i = 0; i < numLandmakrs; ++ i) {
            landmarkBlockIdx_.emplace_back(numRowsQ2r_);
            const auto& lb = landmarkBlocks_[i];
            numRowsQ2r_ += lb->numQ2rows();
            hostToLandmarkBlock_.at(lb->getHostKf()).emplace_back(lb.get());
        }
        if (imuLinData) {
            for (const auto& kv : imuLinData->imuMeas) {
                imuBlocks_.emplace_back(new ImuBlock(kv.second, imuLinData, aom));
            }
        }
    }

    ~LinearizationAbsQR() = default;
    LinearizationAbsQR(const LinearizationAbsQR&) = default;
    LinearizationAbsQR(LinearizationAbsQR&&) = default;
    LinearizationAbsQR& operator=(const LinearizationAbsQR&) = default;
    LinearizationAbsQR& operator=(LinearizationAbsQR&&) = default;
    void logProblemStats(ExecutionStats& stats) const override {
        (void)(stats);
    }
    double linearizeProblem(bool* numericallyValid) override {
        margScaling_ = VecXd();
        for (const auto& [tcidH, targetMap] : lmdb_.getObservations()) {
            for (const auto& [tcidT, _] : targetMap) {
                std::pair<TimeCamId, TimeCamId> key(tcidH, tcidT);
                RelPoseLin& rpl = relativePoseLin_.at(key);
                if (tcidH != tcidT) {
                    const PoseStateWithLin& stateH = estimator_->getPoseStateWithLin(tcidH.frameId);
                    const PoseStateWithLin& stateT = estimator_->getPoseStateWithLin(tcidT.frameId);
                    Sophus::SE3d T_t_h_sophus = computeRelPose(stateH.getPoseLin(),
                        calib_.tIC[tcidH.camId],
                        stateT.getPoseLin(),
                        calib_.tIC[tcidT.camId],
                        &rpl.dRelDH,
                        &rpl.dRelDT);
                    if (stateH.isLinearized() || stateT.isLinearized()) {
                        T_t_h_sophus = computeRelPose(
                            stateH.getPose(), calib_.tIC[tcidH.camId], stateT.getPose(), calib_.tIC[tcidT.camId]);
                    }
                    rpl.tTH = T_t_h_sophus.matrix();
                } else {
                    rpl.tTH.setIdentity();
                    rpl.dRelDH.setZero();
                    rpl.dRelDT.setZero();
                }
            }
        }
        size_t numLandmarks = landmarkBlocks_.size();
        auto body = [&](const tbb::blocked_range<size_t>& range, std::pair<double, bool> errorValid) {
            for (size_t r = range.begin(); r != range.end(); ++ r) {
                errorValid.first += landmarkBlocks_[r]->linearizeLandmark();
                errorValid.second = errorValid.second && !landmarkBlocks_[r]->isNumericalFailure();
            }
            return errorValid;
        };
        std::pair<double, bool> initialValue = {0.0, true};
        auto join = [](auto p1, auto p2) {
            p1.first += p2.first;
            p1.second = p1.second && p2.second;
            return p1;
        };
        tbb::blocked_range<size_t> range(0, numLandmarks);
        auto reductionRes = tbb::parallel_reduce(range, initialValue, body, join);
        if (numericallyValid)
            *numericallyValid = reductionRes.second;
        if (imuLinData_) {
            for (auto& imuBlock : imuBlocks_) {
                reductionRes.first += imuBlock->linearizeImu(estimator_->frameStates);
            }
        }
        if (margLinData_) {
            double margPriorError;
            estimator_->computeMargPriorError(*margLinData_, margPriorError);
            reductionRes.first += margPriorError;
        }
        return reductionRes.first;
    }
    void performQR() override {
        auto body = [&](const tbb::blocked_range<size_t>& range) {
            for (size_t r = range.begin(); r != range.end(); ++ r) {
                landmarkBlocks_[r]->performQR();
            }
        };
        tbb::blocked_range<size_t> range(0, landmarkBlockIdx_.size());
        tbb::parallel_for(range, body);
    }
    double backSubstitute(const VecXd& poseInc) override {
        ensure(poseInc.size() == static_cast<Eigen::Index>(aom_.totalSize), "VIO check failed");
        auto body = [&](const tbb::blocked_range<size_t>& range, double lDiff) {
            for (size_t r = range.begin(); r != range.end(); ++ r) {
                landmarkBlocks_[r]->backSubstitute(poseInc, lDiff);
            }
            return lDiff;
        };
        tbb::blocked_range<size_t> range(0, landmarkBlockIdx_.size());
        double lDiff = tbb::parallel_reduce(range, double(0), body, std::plus<double>());
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
        auto body = [&](const tbb::blocked_range<size_t>& range) {
            for (size_t r = range.begin(); r != range.end(); ++ r) {
                landmarkBlocks_[r]->scaleJlCols();
            }
        };
        tbb::blocked_range<size_t> range(0, landmarkBlockIdx_.size());
        tbb::parallel_for(range, body);
    }
    void scaleJpCols(const VecXd& jacobianScaling) {
        if (true) {
            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++ r) {
                    landmarkBlocks_[r]->scaleJpCols(jacobianScaling);
                }
            };
            tbb::blocked_range<size_t> range(0, landmarkBlockIdx_.size());
            tbb::parallel_for(range, body);
        } else {
            for (auto& [k, v] : relativePoseLin_) {
                size_t hIdx = aom_.absOrderMap.at(k.first.frameId).first;
                size_t tIdx = aom_.absOrderMap.at(k.second.frameId).first;
                v.dRelDH *= jacobianScaling.segment<6>(hIdx).asDiagonal();
                v.dRelDT *= jacobianScaling.segment<6>(tIdx).asDiagonal();
            }
        }
        if (imuLinData_) {
            for (auto& imuBlock : imuBlocks_) {
                imuBlock->scaleJpCols(jacobianScaling);
            }
        }
        if (margLinData_) {
            ensure(margScaling_.size() == 0, "VIO check failed");
            size_t margSize = margLinData_->h.cols();
            margScaling_ = jacobianScaling.head(margSize);
        }
    }
    void setLandmarkDamping(double lambda) {
        auto body = [&](const tbb::blocked_range<size_t>& range) {
            for (size_t r = range.begin(); r != range.end(); ++ r) {
                landmarkBlocks_[r]->setLandmarkDamping(lambda);
            }
        };
        tbb::blocked_range<size_t> range(0, landmarkBlockIdx_.size());
        tbb::parallel_for(range, body);
    }
    void getDenseQ2JpQ2r(MatXd& Q2Jp, VecXd& Q2r) const override {
        size_t totalSize = numRowsQ2r_;
        size_t posesSize = aom_.totalSize;
        size_t lmStartIdx = 0;
        size_t imuStartIdx = totalSize;
        if (imuLinData_) {
            totalSize += imuLinData_->imuMeas.size() * 15;
        }
        size_t margStartIdx = totalSize;
        if (margLinData_)
            totalSize += margLinData_->h.rows();
        Q2Jp.setZero(totalSize, posesSize);
        Q2r.setZero(totalSize);
        auto body = [&](const tbb::blocked_range<size_t>& range) {
            for (size_t r = range.begin(); r != range.end(); ++ r) {
                const auto& lb = landmarkBlocks_[r];
                lb->getDenseQ2JpQ2r(Q2Jp, Q2r, lmStartIdx + landmarkBlockIdx_[r]);
            }
        };
        tbb::blocked_range<size_t> range(0, landmarkBlockIdx_.size());
        tbb::parallel_for(range, body);
        if (imuLinData_) {
            size_t startIdx = imuStartIdx;
            for (const auto& imuBlock : imuBlocks_) {
                imuBlock->addDenseQ2JpQ2r(Q2Jp, Q2r, startIdx);
                startIdx += 15;
            }
        }
        getDenseQ2JpQ2rMargPrior(Q2Jp, Q2r, margStartIdx);
    }
    void getDenseHB(MatXd& H, VecXd& b) const override {
        struct Reductor {
            Reductor(size_t optSize, const std::vector<LandmarkBlockPtr>& landmarkBlocks)
        : optSize(optSize)
        , landmarkBlocks(landmarkBlocks)
            {
                h.setZero(optSize, optSize);
                bVec.setZero(optSize);
            }
            void operator()(const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++ r) {
                    auto& lb = landmarkBlocks[r];
                    lb->addDenseHB(h, bVec);
                }
            }
            Reductor(Reductor& a, tbb::split)
        : optSize(a.optSize)
        , landmarkBlocks(a.landmarkBlocks)
            {
                h.setZero(optSize, optSize);
                bVec.setZero(optSize);
            };
            void join(Reductor& other) {
                h += other.h;
                bVec += other.bVec;
            }
            size_t optSize;
            const std::vector<LandmarkBlockPtr>& landmarkBlocks;
            MatXd h;
            VecXd bVec;
        };
        size_t optSize = aom_.totalSize;
        Reductor r(optSize, landmarkBlocks_);
        tbb::blocked_range<size_t> range(0, landmarkBlockIdx_.size());
        tbb::parallel_reduce(range, r);
        addDenseHBImu(r.h, r.bVec);
        addDenseHBMargPrior(r.h, r.bVec);
        H = std::move(r.h);
        b = std::move(r.bVec);
    }

protected:
    using PoseLinMapType = std::unordered_map<std::pair<TimeCamId, TimeCamId>, RelPoseLin>;
    using PoseLinMapTypeConstIter = typename PoseLinMapType::const_iterator;
    using HostLandmarkMapType = std::unordered_map<TimeCamId, std::vector<const LandmarkBlock*>>;

protected:
    void getDenseQ2JpQ2rMargPrior(MatXd& Q2Jp, VecXd& Q2r, size_t startIdx) const {
        if (!margLinData_)
            return;
        ensure(margLinData_->isSqrt, "VIO check failed");
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

protected:
    Options options_;
    std::vector<KeypointId> landmarkIds_;
    std::vector<LandmarkBlockPtr> landmarkBlocks_;
    std::vector<ImuBlockPtr> imuBlocks_;
    std::unordered_map<TimeCamId, size_t> hostToIdx_;
    HostLandmarkMapType hostToLandmarkBlock_;
    std::vector<size_t> landmarkBlockIdx_;
    const BundleAdjustmentBase* estimator_;
    LandmarkDatabase& lmdb_;
    const std::map<int64_t, PoseStateWithLin>& framePoses_;
    const Calibration<double>& calib_;
    const AbsOrderMap& aom_;
    const std::set<int64_t>* usedFrames_;
    const MargLinData* margLinData_;
    const ImuLinData* imuLinData_;
    PoseLinMapType relativePoseLin_;
    std::vector<std::vector<PoseLinMapTypeConstIter>> relativePosePerHost_;
    VecXd margScaling_;
    size_t numCameras_;
    size_t numRowsQ2r_;
};

} // arc
