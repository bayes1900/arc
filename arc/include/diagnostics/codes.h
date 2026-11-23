#pragma once

#include <base/iarc.h>
#include <explore/exploration/explorationdef.h>
#include <explore/swarm/pairoptimizer.h>
#include <explore/tsp/tspdef.h>
#include <fastplanner/fastplanneroptions.h>
#include <search/astar.h>

namespace arc {

class DiagCodes {
public:
    static const char* explorationStatus(int value) {
        switch (value) {
            case 0: return "SUCCEED";
            case 1: return "FAIL";
            case 2: return "NO_GRID";
            default: return "UNKNOWN";
        }
    }
    static const char* astarStatus(uint8_t value) {
        switch (value) {
            case 0: return "DONE";
            case 1: return "NOPATH";
            case 2: return "TIMEOUT";
            case 3: return "OUTSIDE";
            case 4: return "NOMEM";
            default: return "UNKNOWN";
        }
    }
    static const char* kinoStatus(uint8_t value) {
        switch (value) {
            case 0: return "DONE";
            case 1: return "HORIZON";
            case 2: return "NEAR_END";
            case 3: return "NOPATH";
            case 4: return "TIMEOUT";
            case 5: return "OUTSIDE";
            case 6: return "NOMEM";
            default: return "UNKNOWN";
        }
    }
    static const char* planResult(uint8_t value) {
        switch (value) {
            case 0: return "SUCCESS";
            case 1: return "INPUTINVALID";
            case 2: return "SEARCHFAILED";
            case 3: return "SAMPLEFAILED";
            case 4: return "PARAMETERIZEFAILED";
            case 5: return "OPTIMIZEFAILED";
            case 6: return "TRAJECTORYINVALID";
            default: return "UNKNOWN";
        }
    }
    static const char* bsplineStatus(uint8_t value) {
        switch (value) {
            case 0: return "SUCCEEDED";
            case 1: return "INPUTINVALID";
            case 2: return "INFEASIBLE";
            case 3: return "SOLVERFAILED";
            case 4: return "NOSOLUTION";
            case 5: return "MAXEVALREACHED";
            case 6: return "MAXTIMEREACHED";
            default: return "UNKNOWN";
        }
    }
    static const char* yawStatus(uint8_t value) {
        switch (value) {
            case 0: return "SUCCEEDED";
            case 1: return "INPUTINVALID";
            case 2: return "SOLVERFAILED";
            case 3: return "NOSOLUTION";
            default: return "UNKNOWN";
        }
    }
    static const char* tspFailure(TspFailure failure) {
        return TspFailureName::get(failure);
    }
    static const char* pairOptFailure(PairOptFailure failure) {
        switch (failure) {
            case PairOptFailure::NONE: return "none";
            case PairOptFailure::INVALID_DRONE: return "invalid_drone";
            case PairOptFailure::SELF_COOLDOWN: return "self_cooldown";
            case PairOptFailure::NO_CANDIDATE: return "no_candidate";
            case PairOptFailure::NO_GRIDS: return "no_grids";
            case PairOptFailure::ALLOCATION_FAILED: return "allocation_failed";
            case PairOptFailure::INVALID_REALLOCATION: return "invalid_reallocation";
            case PairOptFailure::COST_INCREASED: return "cost_increased";
            default: return "unknown";
        }
    }
    static const char* astarReason(uint8_t status) {
        switch (status) {
            case AstarSearchStatus::NOPATH: return "astar_nopath";
            case AstarSearchStatus::TIMEOUT: return "astar_timeout";
            case AstarSearchStatus::OUTSIDE: return "astar_outside";
            case AstarSearchStatus::NOMEM: return "astar_nomem";
            default: return "astar_failed";
        }
    }
    static const char* plannerReason(const PlannerResult& result) {
        if (result.code == PlannerResult::SEARCHFAILED) {
            switch (result.searchStatus) {
                case KinoAstarSearchStatus::NOPATH: return "kino_nopath";
                case KinoAstarSearchStatus::TIMEOUT: return "kino_timeout";
                case KinoAstarSearchStatus::OUTSIDE: return "kino_outside";
                case KinoAstarSearchStatus::NOMEM: return "kino_nomem";
                default: return "kino_search_failed";
            }
        }
        switch (result.code) {
            case PlannerResult::INPUTINVALID: return "input_invalid";
            case PlannerResult::SAMPLEFAILED: return "sample_failed";
            case PlannerResult::PARAMETERIZEFAILED: return "parameterize_failed";
            case PlannerResult::OPTIMIZEFAILED: return "optimize_failed";
            case PlannerResult::TRAJECTORYINVALID: return "trajectory_invalid";
            default: return "plan_failed";
        }
    }
};

} // arc
