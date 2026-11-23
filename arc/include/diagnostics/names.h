#pragma once

#include <diagnostics/types.h>

namespace arc {

class DiagNames {
public:
    static const char* stage(DiagStage value) {
        static const char* names[] = {
            "plan_total",
            "frontier_refresh",
            "grid_update",
            "grid_tour",
            "frontier_tour",
            "viewpoint_select",
            "viewpoint_refine",
            "astar",
            "kino_search",
            "kino_optimize",
            "traj_optimize",
            "yaw_optimize",
            "tsp",
            "pair_opt",
            "traj_publish",
        };
        return names[static_cast<size_t>(value)];
    }
    static const char* gauge(DiagGauge value) {
        static const char* names[] = {
            "frontier_count",
            "dormant_frontier_count",
            "assigned_grid_count",
            "viewpoint_count",
            "last_collision_distance",
            "last_collision_other_id",
            "plan_success_rate",
            "last_plan_duration_ms",
            "tsp_last_duration_ms",
            "pair_opt_last_rtt_ms",
            "grid_tour_length",
            "frontier_tour_length",
            "dropped_events",
        };
        return names[static_cast<size_t>(value)];
    }
    static const char* state(int value) {
        static const char* names[] = {
            "INIT",
            "WAIT_TRIGGER",
            "PLAN_TRAJ",
            "PUB_TRAJ",
            "EXEC_TRAJ",
            "IDLE",
            "FINISH",
            "UNKNOWN",
        };
        if (value < 0 || value >= 7) {
            return names[7];
        }
        return names[value];
    }
};

} // arc
