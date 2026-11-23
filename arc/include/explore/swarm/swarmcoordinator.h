#pragma once

#include <base/iarc.h>
#include <diagnostics/exploration.h>
#include <explore/exploration/explorationdata.h>
#include <explore/grid/hgrid.h>
#include <explore/swarm/gridallocator.h>
#include <explore/swarm/pairoptimizer.h>
#include <explore/swarm/swarmdef.h>
#include <explore/swarm/swarmstate.h>
#include <explore/swarm/swarmtransport.h>

namespace arc {

class SwarmCoordinator {
public:
    SwarmCoordinator(
        int droneId,
        int droneNum,
        HGrid& hgrid,
        GridAllocator& allocator,
        SwarmTransport& transport,
        ExplorationData& data,
        const SwarmParams& params = {},
        DiagnosticsRecorder* diagnostics = nullptr
    )
        : droneId_(droneId)
        , droneNum_(droneNum)
        , hgrid_(hgrid)
        , optimizer_(hgrid, allocator, params)
        , transport_(transport)
        , explorationData_(data)
        , params_(params)
        , swarmState_(droneId, droneNum)
        , lastPairOptRequestStamps_(static_cast<size_t>(droneNum), 0.0)
        , lastPairOptResponseStamps_(static_cast<size_t>(droneNum), 0.0)
        , noopDiagnostics_()
        , diagnostics_(diagnostics ? diagnostics : &noopDiagnostics_)
    {
        swarmState_.state(droneId_).assignedGridIds = explorationData_.gridTour.assignedGridIds.values();
    }
    void setEgoState(const Eigen::Vector3d& pos, const Eigen::Vector3d& vel, double yaw) {
        DroneState& ego = swarmState_.state(droneId_);
        ego.pos = pos;
        ego.vel = vel;
        ego.yaw = yaw;
    }
    void tick(double now, bool allowInitiate = true) {
        if (lastStatePublishTime_ < 0.0 || now - lastStatePublishTime_ >= params_.droneStateInterval) {
            lastStatePublishTime_ = now;
            DroneState state = swarmState_.state(droneId_);
            state.droneId = droneId_;
            state.assignedGridIds = explorationData_.gridTour.assignedGridIds.values();
            state.stamp = now;
            state.lastPairOptAttemptTime = lastPairOptAttemptTime_;
            swarmState_.updateState(state);
            transport_.sendDroneState(state);
        }

        if (waitingForResponse_ && pairOptRequestStamp_ >= 0.0 && now - pairOptRequestStamp_ > params_.pairOptResponseTimeout) {
            waitingForResponse_ = false;
            PairOptInfo info;
            info.reason = "pair_opt_timeout";
            info.otherDroneId = pairOptTargetId_;
            info.duration = now - pairOptRequestStamp_;
            info.counter = DiagCounter::PAIR_OPT_TIMEOUTS;
            info.detail = DiagDetail().add("timeout", params_.pairOptResponseTimeout).str();
            diagnostics_->pairOpt(now, info);
        }
        if (allowInitiate && !waitingForResponse_ && (lastPairOptSelectionTime_ < 0.0 || now - lastPairOptSelectionTime_ >= params_.pairOptSelectionInterval)) {
            lastPairOptSelectionTime_ = now;
            PairOptFailure selectionFailure = PairOptFailure::NONE;
            const int otherId = optimizer_.selectNeighborWithDiagnostics(
                swarmState_, droneId_, now, selectionFailure);
            if (otherId <= 0) {
                if (selectionFailure != PairOptFailure::SELF_COOLDOWN &&
                    (lastPairOptNoCandidateEventTime_ < 0.0 || now - lastPairOptNoCandidateEventTime_ > 1.0)) {
                    lastPairOptNoCandidateEventTime_ = now;
                    PairOptInfo info;
                    info.reason = "pair_opt_no_candidate";
                    info.duration = 0.0;
                    info.counter = DiagCounter::PAIR_OPT_NO_CANDIDATE;
                    info.detail = DiagDetail()
                        .add("failure", DiagCodes::pairOptFailure(selectionFailure))
                        .str();
                    diagnostics_->pairOpt(now, info);
                }
                return;
            }
            std::vector<int> selfGridIds;
            std::vector<int> otherGridIds;
            const PairOptResult optimization = optimizer_.optimizePairWithDiagnostics(
                swarmState_, droneId_, otherId, selfGridIds, otherGridIds);
            diagnostics_->recordStage(DiagStage::PAIR_OPT, optimization.seconds);
            if (!optimization.success) {
                PairOptInfo info;
                info.otherDroneId = otherId;
                info.duration = optimization.seconds;
                info.selfCount = static_cast<int>(selfGridIds.size());
                info.otherCount = static_cast<int>(otherGridIds.size());
                info.detail = DiagDetail()
                    .add("failure", DiagCodes::pairOptFailure(optimization.failure))
                    .add("tsp", DiagCodes::tspFailure(optimization.tspFailure))
                    .add("matrixMs", optimization.matrixSeconds * 1000.0)
                    .add("tspMs", optimization.tspSeconds * 1000.0)
                    .add("prevCost", optimization.prevCost)
                    .add("curCost", optimization.curCost)
                    .str();
                if (optimization.failure == PairOptFailure::INVALID_REALLOCATION ||
                    optimization.failure == PairOptFailure::COST_INCREASED) {
                    info.reason = "pair_opt_no_improvement";
                    info.counter = DiagCounter::PAIR_OPT_NO_IMPROVEMENT;
                } else if (optimization.failure == PairOptFailure::NO_GRIDS) {
                    info.reason = "pair_opt_no_candidate";
                    info.counter = DiagCounter::PAIR_OPT_NO_CANDIDATE;
                } else {
                    info.reason = "pair_opt_optimize_failed";
                    info.counter = DiagCounter::PAIR_OPT_OPTIMIZE_FAILURES;
                }
                diagnostics_->pairOpt(now, info);
                lastPairOptAttemptTime_ = now;
                swarmState_.state(droneId_).lastPairOptAttemptTime = now;
                swarmState_.setRecentInteractTime(droneId_, now);
                swarmState_.setRecentInteractTime(otherId, now);
                return;
            }
            PairOptRequest request;
            request.fromDroneId = droneId_;
            request.toDroneId = otherId;
            request.stamp = now;
            request.selfGridIds = selfGridIds;
            request.otherGridIds = otherGridIds;
            pendingSelfGridIds_ = selfGridIds;
            pendingOtherGridIds_ = otherGridIds;
            pairOptRequestStamp_ = now;
            pairOptTargetId_ = otherId;
            waitingForResponse_ = true;
            lastPairOptAttemptTime_ = now;
            swarmState_.state(droneId_).lastPairOptAttemptTime = now;
            PairOptInfo info;
            info.reason = "pair_opt_request";
            info.otherDroneId = otherId;
            info.duration = optimization.seconds;
            info.selfCount = static_cast<int>(selfGridIds.size());
            info.otherCount = static_cast<int>(otherGridIds.size());
            info.counter = DiagCounter::PAIR_OPT_REQUESTS;
            info.detail = DiagDetail()
                .add("selfGrids", selfGridIds.size())
                .add("otherGrids", otherGridIds.size())
                .add("matrixMs", optimization.matrixSeconds * 1000.0)
                .add("tspMs", optimization.tspSeconds * 1000.0)
                .str();
            diagnostics_->pairOpt(now, info);
            for (int i = 0; i < params_.repeatSendCount; ++i) {
                transport_.sendPairOpt(request);
            }
        }
    }
    void onDroneState(const DroneState& state) {
        if (state.droneId == droneId_) {
            return;
        }
        swarmState_.updateState(state);
    }
    void onPairOptRequest(const PairOptRequest& request) {
        if (request.fromDroneId == droneId_ || request.toDroneId != droneId_) {
            return;
        }
        if (request.fromDroneId < 1 || request.fromDroneId > droneNum_) {
            return;
        }
        double& lastStamp = lastPairOptRequestStamps_[static_cast<size_t>(request.fromDroneId - 1)];
        if (request.stamp <= lastStamp + 1e-6) {
            return;
        }
        lastStamp = request.stamp;
        if (request.stamp - lastPairOptAttemptTime_ < params_.pairOptAttemptInterval) {
            PairOptInfo info;
            info.reason = "pair_opt_rejected";
            info.otherDroneId = request.fromDroneId;
            info.selfCount = static_cast<int>(request.otherGridIds.size());
            info.otherCount = static_cast<int>(request.selfGridIds.size());
            info.counter = DiagCounter::PAIR_OPT_REJECTED;
            info.detail = "reason=attempt_interval";
            diagnostics_->pairOpt(request.stamp, info);
            sendResponse_(request.fromDroneId, request.stamp, PairOptStatus::REJECTED);
            return;
        }

        swarmState_.state(droneId_).assignedGridIds = request.otherGridIds;
        swarmState_.state(request.fromDroneId).assignedGridIds = request.selfGridIds;
        swarmState_.state(droneId_).lastPairOptAttemptTime = request.stamp;
        lastPairOptAttemptTime_ = request.stamp;
        swarmState_.setRecentInteractTime(droneId_, request.stamp);
        swarmState_.setRecentInteractTime(request.fromDroneId, request.stamp);
        explorationData_.gridTour.assignedGridIds.assign(request.otherGridIds);
        reallocationEvent_ = true;
        PairOptInfo info;
        info.reason = "pair_opt_accepted";
        info.otherDroneId = request.fromDroneId;
        info.selfCount = static_cast<int>(request.otherGridIds.size());
        info.otherCount = static_cast<int>(request.selfGridIds.size());
        info.counter = DiagCounter::PAIR_OPT_ACCEPTED;
        info.detail = "role=receiver";
        diagnostics_->pairOpt(request.stamp, info);
        sendResponse_(request.fromDroneId, request.stamp, PairOptStatus::ACCEPTED);
    }
    void onPairOptResponse(const PairOptResponse& response) {
        if (response.fromDroneId == droneId_ || response.toDroneId != droneId_) {
            return;
        }
        if (response.fromDroneId < 1 || response.fromDroneId > droneNum_) {
            return;
        }
        double& lastStamp = lastPairOptResponseStamps_[static_cast<size_t>(response.fromDroneId - 1)];
        if (response.stamp <= lastStamp + 1e-6) {
            return;
        }
        lastStamp = response.stamp;
        if (!waitingForResponse_ || std::fabs(pairOptRequestStamp_ - response.stamp) > 1e-5) {
            return;
        }
        waitingForResponse_ = false;
        if (response.status != PairOptStatus::ACCEPTED) {
            PairOptInfo info;
            info.reason = "pair_opt_rejected";
            info.otherDroneId = response.fromDroneId;
            info.duration = response.stamp - pairOptRequestStamp_;
            info.selfCount = static_cast<int>(pendingSelfGridIds_.size());
            info.otherCount = static_cast<int>(pendingOtherGridIds_.size());
            info.counter = DiagCounter::PAIR_OPT_REJECTED;
            info.detail = "reason=response_rejected";
            diagnostics_->pairOpt(response.stamp, info);
            return;
        }
        swarmState_.state(droneId_).assignedGridIds = pendingSelfGridIds_;
        swarmState_.state(response.fromDroneId).assignedGridIds = pendingOtherGridIds_;
        swarmState_.setRecentInteractTime(droneId_, response.stamp);
        swarmState_.setRecentInteractTime(response.fromDroneId, response.stamp);
        explorationData_.gridTour.assignedGridIds.assign(pendingSelfGridIds_);
        reallocationEvent_ = true;
        const double rtt = std::max(0.0, response.stamp - pairOptRequestStamp_);
        diagnostics_->setGauge(DiagGauge::PAIR_OPT_LAST_RTT_MS, rtt * 1000.0);
        PairOptInfo info;
        info.reason = "pair_opt_accepted";
        info.otherDroneId = response.fromDroneId;
        info.duration = rtt;
        info.selfCount = static_cast<int>(pendingSelfGridIds_.size());
        info.otherCount = static_cast<int>(pendingOtherGridIds_.size());
        info.counter = DiagCounter::PAIR_OPT_ACCEPTED;
        info.detail = "role=requester";
        diagnostics_->pairOpt(response.stamp, info);
    }
    const SwarmState& swarmState() const {
        return swarmState_;
    }
    SwarmState& swarmState() {
        return swarmState_;
    }
    const std::vector<int>& selfGridIds() const {
        return explorationData_.gridTour.assignedGridIds.values();
    }
    bool waitingForResponse() const {
        return waitingForResponse_;
    }
    bool consumeReallocation() {
        const bool reallocated = reallocationEvent_;
        reallocationEvent_ = false;
        return reallocated;
    }

private:
    void sendResponse_(int toDroneId, double stamp, int status) {
        PairOptResponse response;
        response.fromDroneId = droneId_;
        response.toDroneId = toDroneId;
        response.stamp = stamp;
        response.status = status;
        for (int i = 0; i < params_.repeatSendCount; ++i) {
            transport_.sendPairOptResponse(response);
        }
    }

private:
    int droneId_;
    int droneNum_;
    HGrid& hgrid_;
    PairOptimizer optimizer_;
    SwarmTransport& transport_;
    ExplorationData& explorationData_;
    SwarmParams params_;
    SwarmState swarmState_;
    double lastStatePublishTime_{-1.0};
    double lastPairOptSelectionTime_{-1.0};
    double lastPairOptAttemptTime_{-1.0};
    double pairOptRequestStamp_{-1.0};
    std::vector<double> lastPairOptRequestStamps_;
    std::vector<double> lastPairOptResponseStamps_;
    bool waitingForResponse_{false};
    bool reallocationEvent_{false};
    std::vector<int> pendingSelfGridIds_;
    std::vector<int> pendingOtherGridIds_;
    DiagnosticsRecorder noopDiagnostics_;
    DiagnosticsRecorder* diagnostics_{nullptr};
    int pairOptTargetId_{-1};
    double lastPairOptNoCandidateEventTime_{-1.0};
};

} // arc