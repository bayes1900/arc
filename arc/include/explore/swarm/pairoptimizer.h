#pragma once

#include <base/iarc.h>
#include <explore/grid/hgrid.h>
#include <explore/swarm/gridallocator.h>
#include <explore/swarm/swarmdef.h>
#include <explore/swarm/swarmstate.h>

#include <algorithm>
#include <unordered_set>

namespace arc {

enum class PairOptFailure : uint8_t {
    NONE = 0,
    INVALID_DRONE,
    SELF_COOLDOWN,
    NO_CANDIDATE,
    NO_GRIDS,
    ALLOCATION_FAILED,
    INVALID_REALLOCATION,
    COST_INCREASED
};

struct PairOptResult {
    bool success{false};
    PairOptFailure failure{PairOptFailure::NONE};
    AllocationFailure allocationFailure{AllocationFailure::NONE};
    TspFailure tspFailure{TspFailure::NONE};
    double seconds{0.0};
    double matrixSeconds{0.0};
    double tspSeconds{0.0};
    double prevCost{0.0};
    double curCost{0.0};
    size_t selfCount{0};
    size_t otherCount{0};
};

class PairOptimizer {
public:
    PairOptimizer(HGrid& hgrid, GridAllocator& allocator, const SwarmParams& params)
        : hgrid_(hgrid)
        , allocator_(allocator)
        , params_(params)
    {}
    int selectNeighborWithDiagnostics(
        const SwarmState& state,
        int selfId,
        double now,
        PairOptFailure& failure
    ) const {
        failure = PairOptFailure::NO_CANDIDATE;
        if (selfId < 1 || selfId > state.droneNum()) {
            failure = PairOptFailure::INVALID_DRONE;
            return -1;
        }
        const DroneState& self = state.state(selfId);
        if (now - self.lastPairOptAttemptTime < params_.pairOptAttemptInterval) {
            failure = PairOptFailure::SELF_COOLDOWN;
            return -1;
        }
        int selectedDroneId = -1;
        double longestIdleInterval = -1.0;
        for (int id = 1; id <= state.droneNum(); ++id) {
            if (id <= selfId) {
                continue;
            }
            const DroneState& other = state.state(id);
            if (now - other.stamp > params_.droneStateTimeout) {
                continue;
            }
            if (now - other.lastPairOptAttemptTime < params_.pairOptAttemptInterval) {
                continue;
            }
            if (now - state.recentInteractTime(id) < params_.pairOptCooldownInterval) {
                continue;
            }
            if (self.assignedGridIds.size() + other.assignedGridIds.size() == 0) {
                continue;
            }
            const double interval = now - state.recentInteractTime(id);
            if (interval <= longestIdleInterval) {
                continue;
            }
            selectedDroneId = id;
            longestIdleInterval = interval;
        }
        if (selectedDroneId > 0) {
            failure = PairOptFailure::NONE;
        }
        return selectedDroneId;
    }
    bool optimizePair(
        const SwarmState& state,
        int selfId,
        int otherId,
        std::vector<int>& selfGridIds,
        std::vector<int>& otherGridIds
    ) const {
        return optimizePairWithDiagnostics(
            state, selfId, otherId, selfGridIds, otherGridIds).success;
    }
    PairOptResult optimizePairWithDiagnostics(
        const SwarmState& state,
        int selfId,
        int otherId,
        std::vector<int>& selfGridIds,
        std::vector<int>& otherGridIds
    ) const {
        const auto start = std::chrono::steady_clock::now();
        double matrixSeconds = 0.0;
        double tspSeconds = 0.0;
        double prevCost = 0.0;
        double curCost = 0.0;
        auto finish = [&](bool success, PairOptFailure failure) {
            PairOptResult result;
            result.success = success;
            result.failure = failure;
            result.seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
            result.matrixSeconds = matrixSeconds;
            result.tspSeconds = tspSeconds;
            result.prevCost = prevCost;
            result.curCost = curCost;
            result.selfCount = selfGridIds.size();
            result.otherCount = otherGridIds.size();
            return result;
        };

        if (selfId < 1 || otherId < 1 || selfId == otherId ||
            selfId > state.droneNum() || otherId > state.droneNum()) {
            return finish(false, PairOptFailure::INVALID_DRONE);
        }
        const DroneState& self = state.state(selfId);
        const DroneState& other = state.state(otherId);

        std::unordered_set<int> candidateGridIdSet;
        for (int id : self.assignedGridIds) {
            candidateGridIdSet.insert(id);
        }
        for (int id : other.assignedGridIds) {
            candidateGridIdSet.insert(id);
        }
        std::vector<int> activeGridIds;
        std::vector<int> unallocatedGridIds;
        hgrid_.activeGrids(activeGridIds);
        state.findUnallocated(activeGridIds, unallocatedGridIds);
        for (int id : unallocatedGridIds) {
            candidateGridIdSet.insert(id);
        }
        if (candidateGridIdSet.empty()) {
            return finish(false, PairOptFailure::NO_GRIDS);
        }
        std::vector<int> candidateGridIds(candidateGridIdSet.begin(), candidateGridIdSet.end());

        std::vector<int> selfConsistentFirst;
        std::vector<int> selfConsistentSecond;
        std::vector<int> otherConsistentFirst;
        std::vector<int> otherConsistentSecond;
        hgrid_.buildConsistentSets(self.assignedGridIds, self.assignedGridIds, selfConsistentFirst, selfConsistentSecond);
        hgrid_.buildConsistentSets(other.assignedGridIds, other.assignedGridIds, otherConsistentFirst, otherConsistentSecond);

        const AllocationResult allocation = allocator_.allocateWithDiagnostics(
            {self.pos, other.pos},
            {Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()},
            {selfConsistentFirst, otherConsistentFirst},
            {selfConsistentSecond, otherConsistentSecond},
            candidateGridIds,
            selfGridIds,
            otherGridIds
        );
        matrixSeconds = allocation.matrixSeconds;
        tspSeconds = allocation.tspSeconds;
        if (!allocation.success) {
            PairOptResult result = finish(false, PairOptFailure::ALLOCATION_FAILED);
            result.allocationFailure = allocation.failure;
            result.tspFailure = allocation.tspFailure;
            return result;
        }

        if (!self.assignedGridIds.empty() && !selfGridIds.empty() && !hgrid_.sameCoarseParent(self.assignedGridIds.front(), selfGridIds.front())) {
            return finish(false, PairOptFailure::INVALID_REALLOCATION);
        }
        if (!other.assignedGridIds.empty() && !otherGridIds.empty() && !hgrid_.sameCoarseParent(other.assignedGridIds.front(), otherGridIds.front())) {
            return finish(false, PairOptFailure::INVALID_REALLOCATION);
        }

        bool hasOverlap = false;
        for (int id : self.assignedGridIds) {
            if (std::find(other.assignedGridIds.begin(), other.assignedGridIds.end(), id) != other.assignedGridIds.end()) {
                hasOverlap = true;
                break;
            }
        }
        const bool needsInitialAllocation =
            self.assignedGridIds.empty() || other.assignedGridIds.empty();
        if (!hasOverlap && !needsInitialAllocation && !selfGridIds.empty() && !otherGridIds.empty()) {
            const std::vector<std::vector<int>> firsts = {selfConsistentFirst, otherConsistentFirst};
            const std::vector<std::vector<int>> seconds = {selfConsistentSecond, otherConsistentSecond};
            prevCost =
                computeGridPathCost_(self.pos, self.assignedGridIds, selfConsistentFirst, firsts, seconds) +
                computeGridPathCost_(other.pos, other.assignedGridIds, otherConsistentFirst, firsts, seconds);
            curCost =
                computeGridPathCost_(self.pos, selfGridIds, selfConsistentFirst, firsts, seconds) +
                computeGridPathCost_(other.pos, otherGridIds, otherConsistentFirst, firsts, seconds);
            if (curCost > prevCost + params_.pairOptCostMargin) {
                return finish(false, PairOptFailure::COST_INCREASED);
            }
        }
        return finish(true, PairOptFailure::NONE);
    }

private:
    double computeGridPathCost_(
        const Eigen::Vector3d& pos,
        const std::vector<int>& gridIds,
        const std::vector<int>& firstIds,
        const std::vector<std::vector<int>>& firsts,
        const std::vector<std::vector<int>>& seconds
    ) const {
        if (gridIds.empty()) {
            return 0.0;
        }
        double cost = hgrid_.costDroneToGrid(pos, gridIds.front(), firstIds);
        for (size_t i = 0; i + 1 < gridIds.size(); ++ i) {
            cost += hgrid_.costGridToGrid(
                gridIds[i], gridIds[i + 1], firsts, seconds,
                static_cast<int>(firsts.size()));
        }
        return cost;
    }
    HGrid& hgrid_;
    GridAllocator& allocator_;
    SwarmParams params_;
};

} // arc
