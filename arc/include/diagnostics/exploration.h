#pragma once

#include <base/iarc.h>
#include <diagnostics/codes.h>
#include <diagnostics/format.h>
#include <diagnostics/recorder.h>
#include <explore/tsp/tspdef.h>
#include <fastplanner/fastplanneroptions.h>
#include <search/astar.h>

namespace arc {

class DiagFailures {
public:
    static DiagFailure gridTsp(
        const TspSolveResult& result,
        int gridCount,
        int frontierCount,
        int viewpointCount,
        double seconds
    ) {
        DiagFailure failure;
        failure.stage = DiagStage::GRID_TOUR;
        failure.reason = result.success ? "grid_tsp_failed" : DiagCodes::tspFailure(result.failure);
        failure.code = "TSP";
        failure.detail = DiagDetail().add("grids", gridCount).str();
        failure.duration = seconds;
        failure.gridCount = gridCount;
        failure.frontierCount = frontierCount;
        failure.viewpointCount = viewpointCount;
        failure.counter = DiagCounter::GRID_TSP_FAILURES;
        return failure;
    }
    static DiagFailure frontierTsp(
        const TspSolveResult& result,
        int gridCount,
        int frontierCount,
        int viewpointCount,
        double seconds
    ) {
        DiagFailure failure;
        failure.stage = DiagStage::FRONTIER_TOUR;
        failure.reason = result.success ? "frontier_tsp_failed" : DiagCodes::tspFailure(result.failure);
        failure.code = "TSP";
        failure.detail = DiagDetail().add("frontiers", frontierCount).str();
        failure.duration = seconds;
        failure.gridCount = gridCount;
        failure.frontierCount = frontierCount;
        failure.viewpointCount = viewpointCount;
        failure.counter = DiagCounter::FRONTIER_TSP_FAILURES;
        return failure;
    }
    static DiagFailure astar(
        uint8_t status,
        const Eigen::Vector3d& start,
        const Eigen::Vector3d& goal,
        int viewpointCount,
        double seconds
    ) {
        DiagFailure failure;
        failure.stage = DiagStage::ASTAR;
        failure.reason = DiagCodes::astarReason(status);
        failure.code = DiagCodes::astarStatus(status);
        failure.detail = DiagDetail()
            .add("start", DiagFormat::vectorParen(start))
            .add("goal", DiagFormat::vectorParen(goal))
            .str();
        failure.duration = seconds;
        failure.viewpointCount = viewpointCount;
        failure.counter = DiagCounter::ASTAR_FAILURES;
        return failure;
    }
    static DiagFailure kino(
        const PlannerResult& result,
        double pathLength,
        int viewpointCount,
        double seconds
    ) {
        DiagFailure failure;
        failure.stage = result.code == PlannerResult::SEARCHFAILED
            ? DiagStage::KINO_SEARCH
            : DiagStage::KINO_OPTIMIZE;
        failure.reason = DiagCodes::plannerReason(result);
        failure.code = DiagCodes::planResult(result.code);
        failure.detail = DiagDetail()
            .add("pathLength", pathLength)
            .add("searchStatus", DiagCodes::kinoStatus(result.searchStatus))
            .add("optimizeStatus", DiagCodes::bsplineStatus(result.optimizeStatus))
            .str();
        failure.duration = seconds;
        failure.viewpointCount = viewpointCount;
        failure.counter = result.code == PlannerResult::SEARCHFAILED
            ? DiagCounter::KINO_SEARCH_FAILURES
            : DiagCounter::KINO_OPTIMIZE_FAILURES;
        return failure;
    }
    static DiagFailure trajOptimize(
        const PlannerResult& result,
        double pathLength,
        int viewpointCount,
        double seconds
    ) {
        DiagFailure failure;
        failure.stage = DiagStage::TRAJ_OPTIMIZE;
        failure.reason = DiagCodes::plannerReason(result);
        failure.code = DiagCodes::planResult(result.code);
        failure.detail = DiagDetail()
            .add("pathLength", pathLength)
            .add("optimizeStatus", DiagCodes::bsplineStatus(result.optimizeStatus))
            .str();
        failure.duration = seconds;
        failure.viewpointCount = viewpointCount;
        failure.counter = DiagCounter::KINO_OPTIMIZE_FAILURES;
        return failure;
    }
    static DiagFailure yaw(
        const PlannerResult& result,
        double targetYaw,
        int viewpointCount,
        double seconds
    ) {
        DiagFailure failure;
        failure.stage = DiagStage::YAW_OPTIMIZE;
        failure.reason = "yaw_failed";
        failure.code = DiagCodes::planResult(result.code);
        failure.detail = DiagDetail()
            .add("targetYaw", targetYaw)
            .add("optimizeStatus", DiagCodes::yawStatus(result.optimizeStatus))
            .str();
        failure.duration = seconds;
        failure.viewpointCount = viewpointCount;
        failure.counter = DiagCounter::YAW_FAILURES;
        return failure;
    }
    static DiagFailure refine(
        int groups,
        int viewpointCount,
        double seconds
    ) {
        DiagFailure failure;
        failure.stage = DiagStage::VIEWPOINT_REFINE;
        failure.reason = "refine_failed";
        failure.code = "GRAPH";
        failure.detail = DiagDetail().add("groups", groups).str();
        failure.duration = seconds;
        failure.viewpointCount = viewpointCount;
        failure.counter = DiagCounter::REFINE_FAILURES;
        return failure;
    }
    static DiagFailure noViewpoint(
        const std::string& detail,
        int viewpointCount
    ) {
        DiagFailure failure;
        failure.stage = DiagStage::VIEWPOINT_SELECT;
        failure.reason = "no_viewpoint";
        failure.code = "EMPTY";
        failure.detail = detail;
        failure.viewpointCount = viewpointCount;
        return failure;
    }
    static DiagFailure planSummary(
        int status,
        const std::string& detail,
        double seconds,
        int gridCount,
        int frontierCount,
        int viewpointCount
    ) {
        DiagFailure failure;
        failure.stage = DiagStage::PLAN_TOTAL;
        failure.reason = "plan_failed";
        failure.code = DiagCodes::explorationStatus(status);
        failure.detail = detail;
        failure.duration = seconds;
        failure.gridCount = gridCount;
        failure.frontierCount = frontierCount;
        failure.viewpointCount = viewpointCount;
        failure.counter = DiagCounter::PLAN_FAILURES;
        return failure;
    }
};

} // arc
