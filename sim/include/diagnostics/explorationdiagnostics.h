#pragma once

#include <base/iarc.h>
#include <diagnostics/collector.h>
#include <diagnostics/explorationstats.h>
#include <diagnostics/format.h>
#include <diagnostics/recorder.h>
#include <diagnostics/rosbridge.h>
#include <explore/exploration/explorationdata.h>

#include <rclcpp/rclcpp.hpp>

namespace arc {

class ExplorationDiagnostics {
public:
    ExplorationDiagnostics(
        rclcpp::Node::SharedPtr node,
        DiagnosticsCollector& collector,
        DiagnosticsRecorder& recorder,
        int droneId
    )
        : node_(std::move(node))
        , collector_(collector)
        , recorder_(recorder)
        , bridge_(node_, collector_)
    {
        bridge_.setDroneId(droneId);
    }
    void tick(
        double now,
        double interval,
        const ExplorationData& data,
        size_t frontierCount,
        size_t dormantFrontierCount
    ) {
        bridge_.publishEvents();
        if (stats_.shouldUpdateGauges(now, interval)) {
            stats_.markGaugesUpdated(now);
            updateGauges_(data, frontierCount, dormantFrontierCount, now);
        }
        bridge_.publishMetrics(now, interval);
    }
    void collision(double now, const CollisionInfo& info) {
        if (!collector_.enabled() || !stats_.shouldEmitCollision(now)) {
            return;
        }
        stats_.markCollision(now);
        recorder_.collision(now, info);
        RCLCPP_WARN(
            node_->get_logger(),
            "collision type=%s distance=%.3f other=%d pos=(%s)",
            info.swarm ? "collision_swarm" : "collision_static",
            info.distance,
            info.otherDroneId,
            DiagFormat::vectorParen(info.position).c_str()
        );
    }
    void trajectoryPublished(double now, int64_t id, double duration) {
        TrajectoryInfo info;
        info.reason = "traj_published";
        info.code = "PUBLISHED";
        info.id = id;
        info.detail = DiagDetail().add("duration", duration).str();
        recorder_.trajectoryPublished(now, info);
    }

private:
    void updateGauges_(
        const ExplorationData& data,
        size_t frontierCount,
        size_t dormantFrontierCount,
        double now
    ) {
        const DiagSnapshot snapshot = collector_.snapshot(now);
        const ExplorationMetricValues values = stats_.compute(
            data, frontierCount, dormantFrontierCount, snapshot);
        collector_.setGauge(DiagGauge::FRONTIER_COUNT, values.frontierCount);
        collector_.setGauge(DiagGauge::DORMANT_FRONTIER_COUNT, values.dormantFrontierCount);
        collector_.setGauge(DiagGauge::ASSIGNED_GRID_COUNT, values.assignedGridCount);
        collector_.setGauge(DiagGauge::VIEWPOINT_COUNT, values.viewpointCount);
        collector_.setGauge(DiagGauge::PLAN_SUCCESS_RATE, values.planSuccessRate);
        collector_.setGauge(DiagGauge::LAST_PLAN_DURATION_MS, values.lastPlanDurationMs);
        collector_.setGauge(DiagGauge::TSP_LAST_DURATION_MS, values.tspLastDurationMs);
        collector_.setGauge(DiagGauge::GRID_TOUR_LENGTH, values.gridTourLength);
        collector_.setGauge(DiagGauge::FRONTIER_TOUR_LENGTH, values.frontierTourLength);
        collector_.setGauge(DiagGauge::DROPPED_EVENTS, values.droppedEvents);
    }

private:
    rclcpp::Node::SharedPtr node_;
    DiagnosticsCollector& collector_;
    DiagnosticsRecorder& recorder_;
    DiagnosticsRosBridge bridge_;
    ExplorationStats stats_;
};

} // arc
