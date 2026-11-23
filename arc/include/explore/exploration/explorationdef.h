#pragma once

#include <base/iarc.h>

namespace arc {

struct ExplorationStatus {
    static constexpr int SUCCEED = 0;
    static constexpr int FAIL = 1;
    static constexpr int NO_GRID = 2;
};

struct ExplorationParams {
    bool refineLocal{true};
    int refinedNum{7};
    double refinedRadius{5.0};
    int topViewNum{15};
    double maxDecay{0.8};

    int droneId{1};
    int droneNum{1};
    int initPlanNum{2};

    double relaxTime{1.0};
    double replanTime{0.2};

    double radiusFar{7.0};
    double radiusClose{1.5};
    double shortenDist{3.0};
    double planRetryBaseDelay{0.1};
    double planRetryMaxDelay{0.5};
    double planRetryBackoffFactor{2.0};
    int planRefreshAfterFailures{3};
    bool planTrajectory{true};
    bool finishUnreachableFrontier{false};
};

} // arc