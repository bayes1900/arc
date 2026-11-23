#pragma once

#include <base/iarc.h>
#include <explore/grid/hgrid.h>
#include <explore/tsp/tspsolver.h>

namespace arc {

enum class AllocationFailure : uint8_t {
    NONE = 0,
    TOO_FEW_DRONES,
    NO_CANDIDATES,
    TSP_FAILED,
    EMPTY_TOUR,
    NO_GRID_ASSIGNED
};

struct AllocationResult {
    bool success{false};
    AllocationFailure failure{AllocationFailure::NONE};
    TspFailure tspFailure{TspFailure::NONE};
    double seconds{0.0};
    double matrixSeconds{0.0};
    double tspSeconds{0.0};
    size_t selfCount{0};
    size_t otherCount{0};
};

class GridAllocator {
public:
    GridAllocator(HGrid& hgrid, TspSolver& solver)
        : hgrid_(hgrid)
        , solver_(solver)
    {}
    bool allocate(
        const std::vector<Eigen::Vector3d>& positions,
        const std::vector<Eigen::Vector3d>& velocities,
        const std::vector<std::vector<int>>& consistentFirstIds,
        const std::vector<std::vector<int>>& consistentSecondIds,
        const std::vector<int>& candidateGridIds,
        std::vector<int>& selfGridIds,
        std::vector<int>& otherGridIds
    ) {
        return allocateWithDiagnostics(
            positions,
            velocities,
            consistentFirstIds,
            consistentSecondIds,
            candidateGridIds,
            selfGridIds,
            otherGridIds
        ).success;
    }
    AllocationResult allocateWithDiagnostics(
        const std::vector<Eigen::Vector3d>& positions,
        const std::vector<Eigen::Vector3d>& velocities,
        const std::vector<std::vector<int>>& consistentFirstIds,
        const std::vector<std::vector<int>>& consistentSecondIds,
        const std::vector<int>& candidateGridIds,
        std::vector<int>& selfGridIds,
        std::vector<int>& otherGridIds
    ) {
        const auto start = std::chrono::steady_clock::now();
        double matrixSeconds = 0.0;
        double tspSeconds = 0.0;
        auto finish = [&](bool success, AllocationFailure failure) {
            AllocationResult result;
            result.success = success;
            result.failure = failure;
            result.seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
            result.matrixSeconds = matrixSeconds;
            result.tspSeconds = tspSeconds;
            result.selfCount = selfGridIds.size();
            result.otherCount = otherGridIds.size();
            return result;
        };

        selfGridIds.clear();
        otherGridIds.clear();
        if (positions.size() < 2) {
            return finish(false, AllocationFailure::TOO_FEW_DRONES);
        }
        if (candidateGridIds.empty()) {
            return finish(false, AllocationFailure::NO_CANDIDATES);
        }

        if (candidateGridIds.size() == 1) {
            const int gridId = candidateGridIds.front();
            const std::vector<int> emptyFirstRegion;
            const std::vector<int>& first = consistentFirstIds.empty() ? emptyFirstRegion : consistentFirstIds.front();
            const double d1 = hgrid_.costDroneToGrid(positions[0], gridId, first);
            const double d2 = hgrid_.costDroneToGrid(positions[1], gridId, emptyFirstRegion);
            if (d1 <= d2) {
                selfGridIds = candidateGridIds;
            } else {
                otherGridIds = candidateGridIds;
            }
            return finish(true, AllocationFailure::NONE);
        }

        const int droneCount = static_cast<int>(positions.size());
        Eigen::MatrixXd cost;
        const auto matrixStart = std::chrono::steady_clock::now();
        hgrid_.buildCostMatrix(positions, velocities, consistentFirstIds, consistentSecondIds, candidateGridIds, cost);
        matrixSeconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - matrixStart).count();

        int totalUnknown = 0;
        std::vector<int> unknownCounts;
        unknownCounts.reserve(candidateGridIds.size());
        for (int id : candidateGridIds) {
            const int unknown = hgrid_.unknownCellsOf(id);
            unknownCounts.push_back(unknown);
            totalUnknown += unknown;
        }
        const int capacity = std::max(1, static_cast<int>(totalUnknown * 0.75 * 0.1));

        TspProblem problem;
        problem.cost = cost;
        problem.vehicles = droneCount;
        problem.acvrp = true;
        problem.capacity = capacity;
        problem.depot = 0;
        problem.demands.assign(static_cast<size_t>(cost.rows()), 0);
        for (size_t i = 0; i < candidateGridIds.size(); ++ i) {
            const int node = 1 + droneCount + static_cast<int>(i);
            if (node < static_cast<int>(problem.demands.size())) {
                problem.demands[static_cast<size_t>(node)] = static_cast<int>(unknownCounts[i] * 0.1);
            }
        }

        TspSolution solution;
        const auto tspStart = std::chrono::steady_clock::now();
        const TspSolveResult tspResult = solver_.solveWithDiagnostics(problem, solution);
        tspSeconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - tspStart).count();
        if (!tspResult.success) {
            AllocationResult result = finish(false, AllocationFailure::TSP_FAILED);
            result.tspFailure = tspResult.failure;
            return result;
        }
        if (solution.tours.empty()) {
            return finish(false, AllocationFailure::EMPTY_TOUR);
        }
        std::vector<bool> assignedGrids(candidateGridIds.size(), false);
        for (size_t vehicle = 0; vehicle < solution.tours.size(); ++ vehicle) {
            std::vector<int>* target = nullptr;
            if (vehicle == 0) {
                target = &selfGridIds;
            } else if (vehicle == 1) {
                target = &otherGridIds;
            } else {
                continue;
            }
            for (int node : solution.tours[vehicle]) {
                const int index = node - (1 + droneCount);
                if (index < 0 || index >= static_cast<int>(candidateGridIds.size())) {
                    continue;
                }
                if (assignedGrids[static_cast<size_t>(index)]) {
                    continue;
                }
                assignedGrids[static_cast<size_t>(index)] = true;
                target->push_back(candidateGridIds[static_cast<size_t>(index)]);
            }
        }
        int covered = 0;
        for (bool value : assignedGrids) {
            if (value) {
                ++covered;
            }
        }
        if (covered == 0) {
            return finish(false, AllocationFailure::NO_GRID_ASSIGNED);
        }
        return finish(true, AllocationFailure::NONE);
    }

private:
    HGrid& hgrid_;
    TspSolver& solver_;
};

} // arc
