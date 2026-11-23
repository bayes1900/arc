#pragma once

#include <vio/linearization/base.h>
#include <vio/linearization/absqr.h>
#include <vio/linearization/abssc.h>
#include <vio/linearization/relsc.h>

namespace arc {

inline std::unique_ptr<LinearizationBase> LinearizationBase::create(
    BundleAdjustmentBase* estimator,
    const AbsOrderMap& aom,
    const Options& options,
    const MargLinData* margLinData,
    const ImuLinData* imuLinData,
    const std::set<int64_t>* usedFrames,
    const std::unordered_set<KeypointId>* lostLandmarks,
    int64_t lastStateToMarg) {
    switch (options.linearizationType) {
        case LinearizationType::ABS_QR:
            return std::make_unique<LinearizationAbsQR>(
                estimator, aom, options, margLinData, imuLinData, usedFrames, lostLandmarks, lastStateToMarg);
        case LinearizationType::ABS_SC:
            return std::make_unique<LinearizationAbsSC>(
                estimator, aom, options, margLinData, imuLinData, usedFrames, lostLandmarks, lastStateToMarg);
        case LinearizationType::REL_SC:
            return std::make_unique<LinearizationRelSC>(
                estimator, aom, options, margLinData, imuLinData, usedFrames, lostLandmarks, lastStateToMarg);
        default:
            std::cerr << "Could not select a valid linearization." << std::endl;
            std::abort();
    }
}

} // arc
