#pragma once

#include <array>
#include <cstdint>
#include <string>

namespace arc {

enum class DiagStage : uint8_t {
    PLAN_TOTAL = 0,
    FRONTIER_REFRESH,
    GRID_UPDATE,
    GRID_TOUR,
    FRONTIER_TOUR,
    VIEWPOINT_SELECT,
    VIEWPOINT_REFINE,
    ASTAR,
    KINO_SEARCH,
    KINO_OPTIMIZE,
    TRAJ_OPTIMIZE,
    YAW_OPTIMIZE,
    TSP,
    PAIR_OPT,
    TRAJ_PUBLISH,
    COUNT
};

enum class DiagCounter : uint8_t {
    PLAN_ATTEMPTS = 0,
    PLAN_SUCCESSES,
    PLAN_FAILURES,
    PLAN_NO_GRID,
    ASTAR_FAILURES,
    KINO_SEARCH_FAILURES,
    KINO_OPTIMIZE_FAILURES,
    YAW_FAILURES,
    GRID_TSP_FAILURES,
    FRONTIER_TSP_FAILURES,
    REFINE_FAILURES,
    COLLISION_STATIC,
    COLLISION_SWARM,
    PAIR_OPT_REQUESTS,
    PAIR_OPT_ACCEPTED,
    PAIR_OPT_REJECTED,
    PAIR_OPT_TIMEOUTS,
    PAIR_OPT_OPTIMIZE_FAILURES,
    PAIR_OPT_NO_CANDIDATE,
    PAIR_OPT_NO_IMPROVEMENT,
    TRAJECTORY_PUBLISHED,
    TRAJECTORY_DROPPED_OLD,
    TRAJECTORY_COMPLETED,
    COUNT
};

enum class DiagGauge : uint8_t {
    FRONTIER_COUNT = 0,
    DORMANT_FRONTIER_COUNT,
    ASSIGNED_GRID_COUNT,
    VIEWPOINT_COUNT,
    LAST_COLLISION_DISTANCE,
    LAST_COLLISION_OTHER_ID,
    PLAN_SUCCESS_RATE,
    LAST_PLAN_DURATION_MS,
    TSP_LAST_DURATION_MS,
    PAIR_OPT_LAST_RTT_MS,
    GRID_TOUR_LENGTH,
    FRONTIER_TOUR_LENGTH,
    DROPPED_EVENTS,
    COUNT
};

struct DiagRecord {
    double stamp{0.0};
    std::string source{"exploration"};
    std::string eventType;
    std::string stage;
    std::string reason;
    std::string code;
    int droneId{0};
    int otherDroneId{-1};
    int fromState{-1};
    int toState{-1};
    double duration{0.0};
    double value{0.0};
    double distance{0.0};
    int gridCount{0};
    int frontierCount{0};
    int viewpointCount{0};
    std::string detail;
};

struct DiagStageStats {
    uint32_t count{0};
    double last{0.0};
    double total{0.0};
    double max{0.0};

    double average() const {
        return count > 0 ? total / static_cast<double>(count) : 0.0;
    }
};

struct DiagSnapshot {
    int state{-1};
    double stateTime{0.0};
    uint32_t stateTransitions{0};
    uint32_t droppedEvents{0};
    std::array<uint32_t, static_cast<size_t>(DiagCounter::COUNT)> counters{};
    std::array<DiagStageStats, static_cast<size_t>(DiagStage::COUNT)> stages{};
    std::array<double, 8> stateTimes{};
    std::array<uint32_t, 8> stateEntries{};
    std::array<double, static_cast<size_t>(DiagGauge::COUNT)> gauges{};
};

} // arc
