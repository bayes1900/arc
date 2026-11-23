#pragma once

#include <base/iarc.h>
#include <base/misc.h>
#include <diagnostics/types.h>
#include <explore/exploration/explorationdata.h>

namespace arc {

struct ExplorationMetricValues {
    double frontierCount{0.0};
    double dormantFrontierCount{0.0};
    double assignedGridCount{0.0};
    double viewpointCount{0.0};
    double planSuccessRate{0.0};
    double lastPlanDurationMs{0.0};
    double tspLastDurationMs{0.0};
    double gridTourLength{0.0};
    double frontierTourLength{0.0};
    double droppedEvents{0.0};
};

class ExplorationStats {
public:
    bool shouldUpdateGauges(double now, double interval) const {
        return lastGaugeUpdateTime_ < 0.0 || now - lastGaugeUpdateTime_ >= interval;
    }
    void markGaugesUpdated(double now) {
        lastGaugeUpdateTime_ = now;
    }
    bool shouldEmitCollision(double now) const {
        return lastCollisionEventTime_ < 0.0 || now - lastCollisionEventTime_ >= 1.0;
    }
    void markCollision(double now) {
        lastCollisionEventTime_ = now;
    }
    ExplorationMetricValues compute(
        const ExplorationData& data,
        size_t frontierCount,
        size_t dormantFrontierCount,
        const DiagSnapshot& snapshot
    ) const {
        ExplorationMetricValues values;
        values.frontierCount = static_cast<double>(frontierCount);
        values.dormantFrontierCount = static_cast<double>(dormantFrontierCount);
        values.assignedGridCount = static_cast<double>(data.gridTour.assignedGridIds.size());
        values.viewpointCount = static_cast<double>(data.viewpoints.positions.size());
        const uint32_t attempts = snapshot.counters[
            static_cast<size_t>(DiagCounter::PLAN_ATTEMPTS)];
        const uint32_t successes = snapshot.counters[
            static_cast<size_t>(DiagCounter::PLAN_SUCCESSES)];
        values.planSuccessRate = attempts > 0
            ? static_cast<double>(successes) / static_cast<double>(attempts)
            : 0.0;
        values.lastPlanDurationMs =
            snapshot.stages[static_cast<size_t>(DiagStage::PLAN_TOTAL)].last * 1000.0;
        values.tspLastDurationMs =
            snapshot.stages[static_cast<size_t>(DiagStage::TSP)].last * 1000.0;
        values.gridTourLength = misc::pathLength(data.gridTour.centers);
        values.frontierTourLength = misc::pathLength(data.frontierTour.path);
        values.droppedEvents = static_cast<double>(snapshot.droppedEvents);
        return values;
    }

private:
    double lastCollisionEventTime_{-1.0};
    double lastGaugeUpdateTime_{-1.0};
};

} // arc
