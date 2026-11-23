#pragma once

#include <base/base.h>
#include <vio/estimator/projection.h>
#include <Eigen/Sparse>
#include <vio/linearization/landmarkblock.h>
#include <base/liegroup.h>

namespace arc {

class LandmarkBlockAbsDynamic : public LandmarkBlock {
public:
    using Options = LandmarkBlock::Options;
    using State = LandmarkBlock::State;
    using RowMatX = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    bool isNumericalFailure() const override { return state_ == State::NUMERICAL_FAILURE; }

    void allocateLandmark(
        Keypoint& lm,
        const std::unordered_map<std::pair<TimeCamId, TimeCamId>, RelPoseLin>& relativePoseLin,
        const Calibration<double>& calib,
        const AbsOrderMap& aom,
        const Options& options,
        const std::map<TimeCamId, size_t>* relOrder = nullptr
   ) override {
        ensure(state_ == State::UNINITIALIZED, "VIO check failed");
        (void)(relOrder);
        lmPtr_ = &lm;
        options_ = &options;
        calib_ = &calib;
        aom_ = &aom;
        poseLinVec_.clear();
        poseLinVec_.reserve(lm.obs.size());
        poseTcidVec_.clear();
        poseTcidVec_.reserve(lm.obs.size());
        ensure(aom.absOrderMap.count(lm.hostKfId.frameId) > 0, "VIO check failed");
        for (const auto& [tcidT, pos] : lm.obs) {
            size_t i = poseLinVec_.size();
            auto it = relativePoseLin.find(std::make_pair(lm.hostKfId, tcidT));
            ensure(it != relativePoseLin.end(), "VIO check failed");
            if (aom.absOrderMap.count(tcidT.frameId) > 0) {
                poseLinVec_.push_back(&it->second);
            } else {
                poseLinVec_.push_back(nullptr);
            }
            poseTcidVec_.push_back(&it->first);
            resIdxByAbsPose_[it->first.first.frameId].insert(i);
            resIdxByAbsPose_[it->first.second.frameId].insert(i);
        }
        paddingIdx_ = aom_->totalSize;
        numRows_ = poseLinVec_.size() * 2 + 3;
        size_t pad = paddingIdx_ % 4;
        if (pad != 0) {
            paddingSize_ = 4 - pad;
        }
        lmIdx_ = paddingIdx_ + paddingSize_;
        resIdx_ = lmIdx_ + 3;
        numCols_ = resIdx_ + 1;
        ensure(numCols_ % 4 == 0, "VIO check failed");
        storage_.resize(numRows_, numCols_);
        dampingRotations_.clear();
        dampingRotations_.reserve(6);
        state_ = State::ALLOCATED;
    }
    double linearizeLandmark() override {
        ensure(state_ == State::ALLOCATED || state_ == State::NUMERICAL_FAILURE || state_ == State::LINEARIZED || state_ == State::MARGINALIZED, "invalid landmark block state");
        storage_.setZero();
        dampingRotations_.clear();
        dampingRotations_.reserve(6);
        bool numericallyValid = true;
        double errorSum = 0;
        size_t i = 0;
        for (const auto& [tcidT, obs] : lmPtr_->obs) {
            std::visit(
                [&, obs = obs](const auto& cam) {
                    if (poseLinVec_[i]) {
                        size_t obsIdx = i * 2;
                        size_t absHIdx = aom_->absOrderMap.at(poseTcidVec_[i]->first.frameId).first;
                        size_t absTIdx = aom_->absOrderMap.at(poseTcidVec_[i]->second.frameId).first;
                        Vec2d res;
                        Eigen::Matrix<double, 2, 6> dResDXi;
                        Eigen::Matrix<double, 2, 3> dResDP;
                        using CamT = std::decay_t<decltype(cam)>;
                        bool valid = linearizePoint<CamT>(
                            obs, *lmPtr_, poseLinVec_[i]->tTH, cam, res, &dResDXi, &dResDP);
                        if (!options_->useValidProjectionsOnly || valid) {
                            numericallyValid = numericallyValid && dResDXi.array().isFinite().all() && dResDP.array().isFinite().all();
                            const double resSquared = res.squaredNorm();
                            const auto [weightedError, weight] = computeErrorWeight_(resSquared);
                            const double sqrtWeight = std::sqrt(weight) / options_->obsStdDev;
                            errorSum += weightedError / (options_->obsStdDev * options_->obsStdDev);
                            storage_.block<2, 3>(obsIdx, lmIdx_) = sqrtWeight * dResDP;
                            storage_.block<2, 1>(obsIdx, resIdx_) = sqrtWeight * res;
                            dResDXi *= sqrtWeight;
                            storage_.block<2, 6>(obsIdx, absHIdx) += dResDXi * poseLinVec_[i]->dRelDH;
                            storage_.block<2, 6>(obsIdx, absTIdx) += dResDXi * poseLinVec_[i]->dRelDT;
                        }
                    }
                    ++ i;
                },
                calib_->intrinsics[tcidT.camId].variant);
        }
        if (numericallyValid) {
            state_ = State::LINEARIZED;
        } else {
            state_ = State::NUMERICAL_FAILURE;
        }
        return errorSum;
    }
    void performQR() override {
        ensure(state_ == State::LINEARIZED, "VIO check failed");
        if (options_->useHouseholder) {
            performQRHouseholder_();
        } else {
            performQRGivens_();
        }
        state_ = State::MARGINALIZED;
    }
    void setLandmarkDamping(double lambda) override {
        ensure(state_ == State::MARGINALIZED, "VIO check failed");
        ensure(lambda >= 0, "VIO check failed");
        if (hasLandmarkDamping()) {
            ensure(dampingRotations_.size() == 6, "VIO check failed");
            for (int n = 2; n >= 0; n--) {
                for (int m = n; m >= 0; m--) {
                    storage_.applyOnTheLeft(numRows_ - 3 + n - m, n, dampingRotations_.back().adjoint());
                    dampingRotations_.pop_back();
                }
            }
        }
        if (lambda == 0) {
            storage_.block<3, 3>(numRows_ - 3, lmIdx_).diagonal().setZero();
        } else {
            ensure(jlColScale_.array().isFinite().all(), "VIO check failed");
            storage_.block<3, 3>(numRows_ - 3, lmIdx_).diagonal().setConstant(sqrt(lambda));
            ensure(dampingRotations_.empty(), "VIO check failed");
            for (int n = 0; n < 3; ++ n) {
                for (int m = 0; m <= n; ++ m) {
                    dampingRotations_.emplace_back();
                    dampingRotations_.back().makeGivens(
                        storage_(n, lmIdx_ + n), storage_(numRows_ - 3 + n - m, lmIdx_ + n));
                    storage_.applyOnTheLeft(numRows_ - 3 + n - m, n, dampingRotations_.back());
                }
            }
        }
    }
    void backSubstitute(const VecXd& poseInc, double& lDiff) override {
        ensure(state_ == State::MARGINALIZED, "VIO check failed");
        ensure(poseInc.size() == static_cast<Eigen::Index>(paddingIdx_), "VIO check failed");
        const auto Q1Jl = storage_.block<3, 3>(0, lmIdx_).triangularView<Eigen::Upper>();
        const auto Q1Jr = storage_.col(resIdx_).head<3>();
        const auto Q1Jp = storage_.topLeftCorner(3, paddingIdx_);
        Vec3d inc = -Q1Jl.solve(Q1Jr + Q1Jp * poseInc);
        setLandmarkDamping(0);
        VecXd QJinc = storage_.topLeftCorner(numRows_ - 3, paddingIdx_) * poseInc;
        QJinc.head<3>() += Q1Jl * inc;
        auto Qr = storage_.col(resIdx_).head(numRows_ - 3);
        lDiff -= QJinc.transpose() * (double(0.5) * QJinc + Qr);
        if (!inc.array().isFinite().all() || !lmPtr_->direction.array().isFinite().all() ||
            !std::isfinite(lmPtr_->invDist)) {
            std::cerr << "Numerical failure in backsubstitution\n";
        }
        inc.array() *= jlColScale_.array();
        lmPtr_->direction += inc.head<2>();
        lmPtr_->invDist = std::max(double(0), lmPtr_->invDist + inc[2]);
    }
    size_t numReducedCams() const override {
        throw std::runtime_error("check what we mean by numReducedCams for absolute poses");
        return poseLinVec_.size();
    }
    void addQ2JpTQ2JpMultX(VecXd& res, const VecXd& xPose) const override {
        (void)(res);
        (void)(xPose);
        throw std::runtime_error("not implemented");
    }
    void addQ2JpTQ2r(VecXd& res) const override {
        (void)(res);
        throw std::runtime_error("not implemented");
    }
    void addQ2JpTQ2JpBlockdiag(BlockDiagonalAccumulator& accu) const override {
        (void)(accu);
        throw std::runtime_error("not implemented");
    }
    void scaleJlCols() override {
        ensure(state_ == State::LINEARIZED, "VIO check failed");
        jlColScale_ = (options_->jacobiScalingEps + storage_.block(0, lmIdx_, numRows_ - 3, 3).colwise().norm().array())
                          .inverse();
        storage_.block(0, lmIdx_, numRows_ - 3, 3) *= jlColScale_.asDiagonal();
    }
    void scaleJpCols(const VecXd& jacobianScaling) override {
        ensure(state_ == State::MARGINALIZED, "VIO check failed");
        ensure(!hasLandmarkDamping(), "VIO check failed");
        storage_.topLeftCorner(numRows_ - 3, paddingIdx_) *= jacobianScaling.asDiagonal();
    }
    bool hasLandmarkDamping() const { return !dampingRotations_.empty(); }
    void printStorage(const std::string& filename) const override {
        std::ofstream f(filename);
        Eigen::IOFormat CleanFmt(4, 0, " ", "\n", "", "");
        f << "Storage (state: " << state_ << ", damping: " << (hasLandmarkDamping() ? "yes" : "no")
          << " Jl_col_scale: " << jlColScale_.transpose() << "):\n"
          << storage_.format(CleanFmt) << std::endl;
        f.close();
    }

    State getState() const override {
        return state_;
    }
    size_t numQ2rows() const override {
        return numRows_ - 3;
    }

protected:
    void performQRGivens_() {
        Eigen::JacobiRotation<double> gr;
        for (size_t n = 0; n < 3; ++ n) {
            for (size_t m = numRows_ - 4; m > n; m--) {
                gr.makeGivens(storage_(m - 1, lmIdx_ + n), storage_(m, lmIdx_ + n));
                storage_.applyOnTheLeft(m, m - 1, gr);
            }
        }
    }
    void performQRHouseholder_() {
        VecXd tempVector1(numCols_);
        VecXd tempVector2(numRows_ - 3);
        for (size_t k = 0; k < 3; ++ k) {
            size_t remainingRows = numRows_ - k - 3;
            double beta;
            double tau;
            storage_.col(lmIdx_ + k).segment(k, remainingRows).makeHouseholder(tempVector2, tau, beta);
            storage_.block(k, 0, remainingRows, numCols_)
                .applyHouseholderOnTheLeft(tempVector2, tau, tempVector1.data());
        }
    }
    std::tuple<double, double> computeErrorWeight_(double resSquared) const {
        if (options_->huberParameter > 0) {
            const double huberWeight = resSquared <= options_->huberParameter * options_->huberParameter
                ? double(1)
                : options_->huberParameter / std::sqrt(resSquared);
            const double error = double(0.5) * (2 - huberWeight) * huberWeight * resSquared;
            return {error, huberWeight};
        } else {
            return {double(0.5) * resSquared, double(1)};
        }
    }
    void getDenseQ2JpQ2r(MatXd& Q2Jp, VecXd& Q2r, size_t startIdx) const override {
        Q2r.segment(startIdx, numRows_ - 3) = storage_.col(resIdx_).tail(numRows_ - 3);
        ensure(Q2Jp.cols() == static_cast<Eigen::Index>(paddingIdx_), "VIO check failed");
        Q2Jp.block(startIdx, 0, numRows_ - 3, paddingIdx_) = storage_.block(3, 0, numRows_ - 3, paddingIdx_);
    }
    void getDenseQ2JpQ2rRel(
        MatXd& Q2Jp, VecXd& Q2r, size_t startIdx, const std::map<TimeCamId, size_t>& relOrder) const override {
        (void)(Q2Jp);
        (void)(Q2r);
        (void)(startIdx);
        (void)(relOrder);
        throw std::runtime_error("not implemented");
    }
    void addDenseHB(DenseAccumulator& accum) const override {
        (void)(accum);
        throw std::runtime_error("not implemented");
    }
    void addDenseHB(MatXd& H, VecXd& b) const override {
        const auto r = storage_.col(resIdx_).tail(numRows_ - 3);
        const auto J = storage_.block(3, 0, numRows_ - 3, paddingIdx_);
        H.noalias() += J.transpose() * J;
        b.noalias() += J.transpose() * r;
    }
    void addDenseHBRel(MatXd& H_rel, VecXd& bRel, const std::map<TimeCamId, size_t>& relOrder) const override {
        (void)(H_rel);
        (void)(bRel);
        (void)(relOrder);
        throw std::runtime_error("not implemented");
    }
    const Eigen::PermutationMatrix<Eigen::Dynamic>& getRelPermutation() const override {
        throw std::runtime_error("not implemented");
    }
    Eigen::PermutationMatrix<Eigen::Dynamic> computeRelPermutation(
        const std::map<TimeCamId, size_t>& relOrder) const override {
        (void)(relOrder);
        throw std::runtime_error("not implemented");
    }
    void addDenseHBRel2(MatXd& H_rel, VecXd& bRel) const override {
        (void)(H_rel);
        (void)(bRel);
        throw std::runtime_error("not implemented");
    }
    virtual TimeCamId getHostKf() const override {
        return lmPtr_->hostKfId;
    }

private:
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> storage_;

    Vec3d jlColScale_ = Vec3d::Ones();
    std::vector<Eigen::JacobiRotation<double>> dampingRotations_;
    std::vector<const RelPoseLin*> poseLinVec_;
    std::vector<const std::pair<TimeCamId, TimeCamId>*> poseTcidVec_;
    size_t paddingIdx_ = 0;
    size_t paddingSize_ = 0;
    size_t lmIdx_ = 0;
    size_t resIdx_ = 0;
    size_t numCols_ = 0;
    size_t numRows_ = 0;
    const Options* options_ = nullptr;
    State state_ = State::UNINITIALIZED;
    Keypoint* lmPtr_ = nullptr;
    const Calibration<double>* calib_ = nullptr;
    const AbsOrderMap* aom_ = nullptr;
    std::map<int64_t, std::set<int>> resIdxByAbsPose_;
};
inline std::unique_ptr<LandmarkBlock> LandmarkBlock::createLandmarkBlock() {
    return std::make_unique<LandmarkBlockAbsDynamic>();
}

} // arc
