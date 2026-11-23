#pragma once

#include <Eigen/Dense>

#include <vio/estimator/ba.h>
#include <vio/linearization/landmarkblock.h>
#include <vio/estimator/stats.h>

namespace arc {

class LinearizationBase {
public:
    struct Options {
        typename LandmarkBlock::Options lbOptions;
        LinearizationType linearizationType;
    };
    virtual ~LinearizationBase() = default;
    virtual void logProblemStats(ExecutionStats& stats) const = 0;
    virtual double linearizeProblem(bool* numericallyValid = nullptr) = 0;
    virtual void performQR() = 0;
    virtual double backSubstitute(const VecXd& poseInc) = 0;
    virtual void getDenseQ2JpQ2r(MatXd& Q2Jp, VecXd& Q2r) const = 0;
    virtual void getDenseHB(MatXd& H, VecXd& b) const = 0;
    static std::unique_ptr<LinearizationBase> create(
        BundleAdjustmentBase* estimator,
        const AbsOrderMap& aom,
        const Options& options,
        const MargLinData* margLinData = nullptr,
        const ImuLinData* imuLinData = nullptr,
        const std::set<int64_t>* usedFrames = nullptr,
        const std::unordered_set<KeypointId>* lostLandmarks = nullptr,
        int64_t lastStateToMarg = std::numeric_limits<int64_t>::max()
    );
};
bool isLinearizationSqrt(const LinearizationType& type);

} // arc
