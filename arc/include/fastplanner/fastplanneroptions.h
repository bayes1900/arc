#pragma once

#include <search/kinoastar.h>
#include <planning/yawopt.h>

namespace arc {

struct FastPlannerOptions {
    double controlPointDistance{0.5};
    double maxYawRate{2.0944};
    double relaxTime2{1.5};

    bool minTime{false};
    bool optimistic{true};

    double collisionCheckRadius{6.0};
    double collisionCheckStep{0.02};
    double swarmSafeDistance{0.5};

    KinoAstarOptions kino;
    YawOptParams yaw;
};

struct PlannerStats {
    double searchTime{0.0};
    double optimizeTime{0.0};
};

struct PlannerResult {
    static constexpr uint8_t SUCCESS = 0;
    static constexpr uint8_t INPUTINVALID = 1;
    static constexpr uint8_t SEARCHFAILED = 2;
    static constexpr uint8_t SAMPLEFAILED = 3;
    static constexpr uint8_t PARAMETERIZEFAILED = 4;
    static constexpr uint8_t OPTIMIZEFAILED = 5;
    static constexpr uint8_t TRAJECTORYINVALID = 6;

    uint8_t code{255};
    uint8_t searchStatus{255};
    uint8_t optimizeStatus{255};
    double searchTime{0.0};
    double optimizeTime{0.0};

    bool succeeded() const { return code == SUCCESS; }
};

} // arc
