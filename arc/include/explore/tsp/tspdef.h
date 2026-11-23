#pragma once

#include <base/iarc.h>

namespace arc {

struct TspProblem {
    Eigen::MatrixXd cost;
    int depot{0};
    int vehicles{1};
    int capacity{0};
    std::vector<int> demands;
    bool acvrp{false};
    int minSize{-1};
    int maxSize{-1};
};

struct TspSolution {
    std::vector<int> tour;
    std::vector<std::vector<int>> tours;
    bool feasible{false};
};

enum class TspFailure : uint8_t {
    NONE = 0,
    INVALID_PROBLEM,
    WRITE_PROBLEM_FAILED,
    WRITE_PARAMS_FAILED,
    SERVICE_UNAVAILABLE,
    SERVICE_TIMEOUT,
    SERVICE_EXCEPTION,
    SOLVER_FAILED,
    READ_TOUR_FAILED,
    EMPTY_TOUR
};

struct TspSolveResult {
    bool success{false};
    TspFailure failure{TspFailure::NONE};
    double seconds{0.0};
    double writeSeconds{0.0};
    double serviceSeconds{0.0};
    double readSeconds{0.0};
};

class TspFailureName {
public:
    static const char* get(TspFailure failure) {
        switch (failure) {
            case TspFailure::NONE: return "none";
            case TspFailure::INVALID_PROBLEM: return "tsp_invalid_problem";
            case TspFailure::WRITE_PROBLEM_FAILED: return "tsp_write_problem_failed";
            case TspFailure::WRITE_PARAMS_FAILED: return "tsp_write_params_failed";
            case TspFailure::SERVICE_UNAVAILABLE: return "tsp_service_unavailable";
            case TspFailure::SERVICE_TIMEOUT: return "tsp_service_timeout";
            case TspFailure::SERVICE_EXCEPTION: return "tsp_service_exception";
            case TspFailure::SOLVER_FAILED: return "tsp_solver_failed";
            case TspFailure::READ_TOUR_FAILED: return "tsp_read_tour_failed";
            case TspFailure::EMPTY_TOUR: return "tsp_empty_tour";
            default: return "tsp_unknown_failure";
        }
    }
};

} // arc