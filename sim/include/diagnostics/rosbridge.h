#pragma once

#include <diagnostics/collector.h>
#include <diagnostics/names.h>
#include <diagnostics/types.h>
#include <arcmsgs/msg/exploration_event.hpp>
#include <arcmsgs/msg/exploration_metrics.hpp>
#include <rclcpp/rclcpp.hpp>

namespace arc {

class DiagnosticsRosBridge {
public:
    DiagnosticsRosBridge(
        rclcpp::Node::SharedPtr node,
        DiagnosticsCollector& collector
    )
        : node_(std::move(node))
        , collector_(collector)
    {
        eventPub_ = node_->create_publisher<arcmsgs::msg::ExplorationEvent>(
            "/exploration/events", rclcpp::QoS(200)
        );
        metricsPub_ = node_->create_publisher<arcmsgs::msg::ExplorationMetrics>(
            "/exploration/metrics", rclcpp::QoS(10)
        );
    }
    void publishEvents() {
        if (!collector_.enabled()) {
            return;
        }
        const std::vector<DiagRecord> events = collector_.takeEvents();
        for (const DiagRecord& event : events) {
            arcmsgs::msg::ExplorationEvent message;
            message.header.stamp = event.stamp > 0.0
                ? rclcpp::Time(static_cast<int64_t>(event.stamp * 1e9))
                : node_->now();
            message.header.frame_id = "world";
            message.source = event.source;
            message.event_type = event.eventType;
            message.stage = event.stage;
            message.reason = event.reason;
            message.code = event.code;
            message.drone_id = event.droneId;
            message.other_drone_id = event.otherDroneId;
            message.from_state = event.fromState;
            message.to_state = event.toState;
            message.duration_ms = event.duration * 1000.0;
            message.value = event.value;
            message.distance = event.distance;
            message.grid_count = event.gridCount;
            message.frontier_count = event.frontierCount;
            message.viewpoint_count = event.viewpointCount;
            message.detail = event.detail;
            eventPub_->publish(message);
        }
    }
    void publishMetrics(double now, double interval) {
        if (!collector_.enabled()) {
            return;
        }
        if (lastMetricsTime_ >= 0.0 && now - lastMetricsTime_ < interval) {
            return;
        }
        lastMetricsTime_ = now;
        const DiagSnapshot snapshot = collector_.snapshot(now);

        arcmsgs::msg::ExplorationMetrics message;
        message.header.stamp = node_->now();
        message.header.frame_id = "world";
        message.drone_id = droneId_;
        message.state = static_cast<uint8_t>(std::max(0, snapshot.state));
        message.state_time_s = snapshot.stateTime;
        message.state_transitions = snapshot.stateTransitions;

        message.plan_attempts = counter_(snapshot, DiagCounter::PLAN_ATTEMPTS);
        message.plan_successes = counter_(snapshot, DiagCounter::PLAN_SUCCESSES);
        message.plan_failures = counter_(snapshot, DiagCounter::PLAN_FAILURES);
        message.plan_no_grid = counter_(snapshot, DiagCounter::PLAN_NO_GRID);

        message.astar_failures = counter_(snapshot, DiagCounter::ASTAR_FAILURES);
        message.kino_search_failures = counter_(snapshot, DiagCounter::KINO_SEARCH_FAILURES);
        message.kino_optimize_failures = counter_(snapshot, DiagCounter::KINO_OPTIMIZE_FAILURES);
        message.yaw_failures = counter_(snapshot, DiagCounter::YAW_FAILURES);
        message.grid_tsp_failures = counter_(snapshot, DiagCounter::GRID_TSP_FAILURES);
        message.frontier_tsp_failures = counter_(snapshot, DiagCounter::FRONTIER_TSP_FAILURES);
        message.refine_failures = counter_(snapshot, DiagCounter::REFINE_FAILURES);

        message.collision_static = counter_(snapshot, DiagCounter::COLLISION_STATIC);
        message.collision_swarm = counter_(snapshot, DiagCounter::COLLISION_SWARM);
        message.last_collision_distance = gauge_(snapshot, DiagGauge::LAST_COLLISION_DISTANCE);
        message.last_collision_other_id = static_cast<int32_t>(gauge_(snapshot, DiagGauge::LAST_COLLISION_OTHER_ID));

        message.pair_opt_requests = counter_(snapshot, DiagCounter::PAIR_OPT_REQUESTS);
        message.pair_opt_accepted = counter_(snapshot, DiagCounter::PAIR_OPT_ACCEPTED);
        message.pair_opt_rejected = counter_(snapshot, DiagCounter::PAIR_OPT_REJECTED);
        message.pair_opt_timeouts = counter_(snapshot, DiagCounter::PAIR_OPT_TIMEOUTS);
        message.pair_opt_optimize_failures = counter_(snapshot, DiagCounter::PAIR_OPT_OPTIMIZE_FAILURES);
        message.pair_opt_no_candidate = counter_(snapshot, DiagCounter::PAIR_OPT_NO_CANDIDATE);
        message.pair_opt_no_improvement = counter_(snapshot, DiagCounter::PAIR_OPT_NO_IMPROVEMENT);

        message.trajectory_published = counter_(snapshot, DiagCounter::TRAJECTORY_PUBLISHED);
        message.trajectory_dropped_old = counter_(snapshot, DiagCounter::TRAJECTORY_DROPPED_OLD);
        message.trajectory_completed = counter_(snapshot, DiagCounter::TRAJECTORY_COMPLETED);

        for (size_t i = 0; i < static_cast<size_t>(DiagStage::COUNT); ++ i) {
            const DiagStage stage = static_cast<DiagStage>(i);
            const DiagStageStats& stats = snapshot.stages[i];
            message.stage_names.push_back(DiagNames::stage(stage));
            message.stage_counts.push_back(stats.count);
            message.stage_last_ms.push_back(stats.last * 1000.0);
            message.stage_avg_ms.push_back(stats.average() * 1000.0);
            message.stage_max_ms.push_back(stats.max * 1000.0);
        }
        for (int i = 0; i < 7; ++ i) {
            message.state_names.push_back(DiagNames::state(i));
            message.state_times_s.push_back(snapshot.stateTimes[static_cast<size_t>(i)]);
            message.state_entries.push_back(snapshot.stateEntries[static_cast<size_t>(i)]);
        }
        for (size_t i = 0; i < static_cast<size_t>(DiagGauge::COUNT); ++ i) {
            message.gauge_names.push_back(DiagNames::gauge(static_cast<DiagGauge>(i)));
            message.gauge_values.push_back(snapshot.gauges[i]);
        }
        metricsPub_->publish(message);
    }
    void setDroneId(int droneId) {
        droneId_ = droneId;
    }

private:
    static uint32_t counter_(const DiagSnapshot& snapshot, DiagCounter value) {
        return snapshot.counters[static_cast<size_t>(value)];
    }
    static double gauge_(const DiagSnapshot& snapshot, DiagGauge value) {
        return snapshot.gauges[static_cast<size_t>(value)];
    }

private:
    rclcpp::Node::SharedPtr node_;
    DiagnosticsCollector& collector_;
    rclcpp::Publisher<arcmsgs::msg::ExplorationEvent>::SharedPtr eventPub_;
    rclcpp::Publisher<arcmsgs::msg::ExplorationMetrics>::SharedPtr metricsPub_;
    double lastMetricsTime_{-1.0};
    int droneId_{0};
};

} // arc
