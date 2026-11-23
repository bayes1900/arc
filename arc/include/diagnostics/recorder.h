#pragma once

#include <base/iarc.h>
#include <diagnostics/collector.h>
#include <diagnostics/format.h>
#include <diagnostics/names.h>
#include <diagnostics/stage.h>
#include <diagnostics/types.h>

#include <string>

namespace arc {

struct DiagFailure {
    DiagStage stage{DiagStage::PLAN_TOTAL};
    std::string reason;
    std::string code;
    std::string detail;
    double duration{0.0};
    int gridCount{0};
    int frontierCount{0};
    int viewpointCount{0};
    DiagCounter counter{DiagCounter::COUNT};
};

struct PlanSuccessInfo {
    double duration{0.0};
    double distance{0.0};
    int gridCount{0};
    int frontierCount{0};
    int viewpointCount{0};
    std::string detail;
};

struct PlanNoGridInfo {
    double duration{0.0};
    std::string reason;
    std::string detail;
};

struct CollisionInfo {
    bool swarm{false};
    double distance{0.0};
    int otherDroneId{-1};
    Eigen::Vector3d position{Eigen::Vector3d::Zero()};
};

struct PairOptInfo {
    std::string reason;
    std::string detail;
    int otherDroneId{-1};
    double duration{0.0};
    int selfCount{0};
    int otherCount{0};
    DiagCounter counter{DiagCounter::COUNT};
};

struct TrajectoryInfo {
    std::string reason;
    std::string code;
    std::string detail;
    int64_t id{0};
};

class DiagnosticsRecorder {
public:
    DiagnosticsRecorder() = default;
    DiagnosticsRecorder(DiagnosticsCollector* collector, int droneId)
        : collector_(collector)
        , droneId_(droneId)
    {}
    void attach(DiagnosticsCollector* collector, int droneId) {
        collector_ = collector;
        droneId_ = droneId;
    }
    void initializeState(int state, double now) {
        if (collector_) {
            collector_->initializeState(state, now);
        }
    }
    StageTimer stage(DiagStage stage) const {
        return StageTimer(collector_, stage);
    }
    void fsmTransition(int from, int to, const char* reason, double now) {
        if (!collector_) {
            return;
        }
        collector_->recordStateTransition(from, to, now, reason ? reason : "none");
    }
    void planAttempt() {
        if (collector_) {
            collector_->increment(DiagCounter::PLAN_ATTEMPTS);
        }
    }
    void planSuccess(double now, const PlanSuccessInfo& info) {
        if (!collector_) {
            return;
        }
        collector_->increment(DiagCounter::PLAN_SUCCESSES);
        emit_(
            "plan_success", "plan_total", "plan_success", "SUCCEED",
            now, info.duration, -1, -1, -1, info.distance, 0.0,
            info.gridCount, info.frontierCount, info.viewpointCount,
            info.detail
        );
    }
    void planFailure(double now, const DiagFailure& failure) {
        if (!collector_) {
            return;
        }
        if (failure.counter != DiagCounter::COUNT) {
            collector_->increment(failure.counter);
        }
        emit_(
            "plan_failure", DiagNames::stage(failure.stage), failure.reason, failure.code,
            now, failure.duration, -1, -1, -1, 0.0, 0.0,
            failure.gridCount, failure.frontierCount, failure.viewpointCount,
            failure.detail
        );
    }
    void planNoGrid(double now, const PlanNoGridInfo& info) {
        if (!collector_) {
            return;
        }
        collector_->increment(DiagCounter::PLAN_NO_GRID);
        emit_(
            "plan_failure", "plan_total", info.reason, "NO_GRID",
            now, info.duration, -1, -1, -1, 0.0, 0.0,
            0, 0, 0, info.detail
        );
    }
    void collision(double now, const CollisionInfo& info) {
        if (!collector_) {
            return;
        }
        collector_->increment(
            info.swarm ? DiagCounter::COLLISION_SWARM : DiagCounter::COLLISION_STATIC
        );
        collector_->setGauge(DiagGauge::LAST_COLLISION_DISTANCE, info.distance);
        collector_->setGauge(
            DiagGauge::LAST_COLLISION_OTHER_ID,
            static_cast<double>(info.otherDroneId)
        );
        const std::string reason = info.swarm ? "collision_swarm" : "collision_static";
        const std::string detail = DiagDetail()
            .add("pos", DiagFormat::vectorParen(info.position))
            .str();
        emit_(
            "collision", "collision", reason, "COLLISION",
            now, 0.0, info.otherDroneId, -1, -1, 0.0, info.distance,
            0, 0, 0, detail
        );
    }
    void pairOpt(double now, const PairOptInfo& info) {
        if (!collector_) {
            return;
        }
        if (info.counter != DiagCounter::COUNT) {
            collector_->increment(info.counter);
        }
        emit_(
            "pair_opt", "pair_opt", info.reason, "",
            now, info.duration, info.otherDroneId, -1, -1, 0.0, 0.0,
            info.selfCount, info.otherCount, 0, info.detail
        );
    }
    void trajectoryPublished(double now, const TrajectoryInfo& info) {
        if (!collector_) {
            return;
        }
        collector_->increment(DiagCounter::TRAJECTORY_PUBLISHED);
        emit_(
            "trajectory_event", "traj_publish", info.reason, info.code,
            now, 0.0, -1, -1, -1, static_cast<double>(info.id), 0.0,
            0, 0, 0, info.detail
        );
    }
    void recordStage(DiagStage stage, double seconds) {
        if (collector_) {
            collector_->recordStage(stage, seconds);
        }
    }
    void setGauge(DiagGauge gauge, double value) {
        if (collector_) {
            collector_->setGauge(gauge, value);
        }
    }

private:
    void emit_(
        const char* eventType,
        const char* stage,
        const std::string& reason,
        const std::string& code,
        double stamp,
        double duration,
        int otherDroneId,
        int fromState,
        int toState,
        double value,
        double distance,
        int gridCount,
        int frontierCount,
        int viewpointCount,
        const std::string& detail
    ) {
        DiagRecord record;
        record.stamp = stamp;
        record.eventType = eventType;
        record.stage = stage;
        record.reason = reason;
        record.code = code;
        record.droneId = droneId_;
        record.otherDroneId = otherDroneId;
        record.fromState = fromState;
        record.toState = toState;
        record.duration = duration;
        record.value = value;
        record.distance = distance;
        record.gridCount = gridCount;
        record.frontierCount = frontierCount;
        record.viewpointCount = viewpointCount;
        record.detail = detail;
        collector_->recordEvent(std::move(record));
    }

private:
    DiagnosticsCollector* collector_{nullptr};
    int droneId_{0};
};

} // arc
