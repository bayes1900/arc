#pragma once

#include <base/iarc.h>
#include <diagnostics/recorder.h>
#include <explore/exploration/explorationdata.h>
#include <explore/exploration/explorationdef.h>
#include <explore/exploration/explorationmanager.h>

namespace arc {

enum class ExplorationState : uint8_t {
    INIT         = 0,
    WAIT_TRIGGER = 1,
    PLAN_TRAJ    = 2,
    PUB_TRAJ     = 3,
    EXEC_TRAJ    = 4,
    IDLE         = 5,
    FINISH       = 6,
};

struct ExplorationFsmParams {
    double initWaitDuration{2.0};
    double replanThresh1{0.2};
    double replanThresh2{0.2};
    double replanThresh3{1.5};
    double idleDuration{100.0};
};

class ExplorationFsm {
public:
    ExplorationFsm(
        ExplorationManager& manager,
        ExplorationFsmParams params,
        DiagnosticsRecorder* diagnostics = nullptr
    )
        : manager_(manager)
        , params_(params)
        , noopDiagnostics_()
        , diagnostics_(diagnostics ? diagnostics : &noopDiagnostics_)
    {}
    void tick(
        double now,
        const OdomState& odom,
        const OdomState* planningStart = nullptr,
        bool updateFrontier = true
    ) {
        const ExplorationState previousState = state_;
        switch (state_) {
            case ExplorationState::INIT: {
                if (initTime_ < 0.0) {
                    initTime_ = now;
                }
                if (now - initTime_ >= params_.initWaitDuration) {
                    transitionReason_ = "init_done";
                    state_ = ExplorationState::WAIT_TRIGGER;
                }
                break;
            }
            case ExplorationState::WAIT_TRIGGER: {
                if (trigger_) {
                    trigger_ = false;
                    transitionReason_ = triggerReason_;
                    state_ = ExplorationState::PLAN_TRAJ;
                }
                break;
            }
            case ExplorationState::PLAN_TRAJ: {
                const bool replanOnly = avoidCollision_ && collisionReplanCount_ <= 2;
                avoidCollision_ = false;
                if (!replanOnly && updateFrontier && manager_.refreshFrontiers(odom.pos) == 0) {
                    enterIdle_(now, "no_frontier");
                    break;
                }
                const OdomState& planningOdom = planningStart ? *planningStart : odom;
                const int status = replanOnly
                    ? manager_.replanToCurrentViewpoint(planningOdom)
                    : manager_.planExploreMotion(planningOdom);
                if (status == ExplorationStatus::SUCCEED) {
                    if (!replanOnly) {
                        collisionReplanCount_ = 0;
                    }
                    transitionReason_ = "plan_success";
                    trajectoryPublished_ = false;
                    state_ = ExplorationState::PUB_TRAJ;
                } else if (status == ExplorationStatus::NO_GRID) {
                    enterIdle_(now, "no_grid");
                }
                break;
            }
            case ExplorationState::PUB_TRAJ: {
                if (trajectoryPublished_) {
                    transitionReason_ = "trajectory_published";
                    trajectoryPublished_ = false;
                    trajectoryFinished_ = false;
                    execStart_ = trajectoryStartTime_ >= 0.0 ? trajectoryStartTime_ : now;
                    trajectoryStartTime_ = -1.0;
                    state_ = ExplorationState::EXEC_TRAJ;
                }
                break;
            }
            case ExplorationState::EXEC_TRAJ: {
                const bool collisionReplan = collision_;
                const bool frontierReplan = frontierCovered_;
                const bool finishedReplan = trajectoryFinished_;
                const bool timeoutReplan = execStart_ >= 0.0 && now - execStart_ > params_.replanThresh3;
                if (collisionReplan || frontierReplan || finishedReplan || timeoutReplan) {
                    if (collisionReplan) {
                        transitionReason_ = "collision";
                    } else if (frontierReplan) {
                        transitionReason_ = "frontier_covered";
                    } else if (finishedReplan) {
                        transitionReason_ = "trajectory_finished";
                    } else {
                        transitionReason_ = "replan_timeout";
                    }
                    collision_ = false;
                    frontierCovered_ = false;
                    trajectoryFinished_ = false;
                    state_ = ExplorationState::PLAN_TRAJ;
                }
                break;
            }
            case ExplorationState::IDLE: {
                if (trigger_) {
                    trigger_ = false;
                    transitionReason_ = triggerReason_;
                    state_ = ExplorationState::PLAN_TRAJ;
                } else if (idleStart_ >= 0.0 && now - idleStart_ > params_.idleDuration) {
                    transitionReason_ = "idle_timeout";
                    state_ = ExplorationState::FINISH;
                }
                break;
            }
            case ExplorationState::FINISH: {
                break;
            }
        }
        if (state_ != previousState) {
            diagnostics_->fsmTransition(
                static_cast<int>(previousState),
                static_cast<int>(state_),
                transitionReason_.c_str(),
                now
            );
        }
    }
    void trigger() {
        trigger("trigger");
    }
    void trigger(const std::string& reason) {
        trigger_ = true;
        triggerReason_ = reason;
    }
    void onTrajectoryPublished(double startTime = -1.0) {
        trajectoryStartTime_ = startTime;
        trajectoryPublished_ = true;
    }
    void onTrajectoryFinished() {
        trajectoryFinished_ = true;
    }
    void onCollisionDetected() {
        collision_ = true;
        avoidCollision_ = true;
        collisionReplanCount_ += 1;
    }
    void onFrontierCovered() {
        frontierCovered_ = true;
    }
    void reset() {
        state_ = ExplorationState::INIT;
        initTime_ = -1.0;
        execStart_ = -1.0;
        idleStart_ = -1.0;
        trigger_ = false;
        triggerReason_ = "trigger";
        transitionReason_ = "none";
        trajectoryPublished_ = false;
        trajectoryStartTime_ = -1.0;
        trajectoryFinished_ = false;
        collision_ = false;
        avoidCollision_ = false;
        collisionReplanCount_ = 0;
        frontierCovered_ = false;
    }
    ExplorationState state() const {
        return state_;
    }
    const ExplorationFsmParams& params() const {
        return params_;
    }

private:
    void enterIdle_(double now, const std::string& reason) {
        idleStart_ = now;
        transitionReason_ = reason;
        state_ = ExplorationState::IDLE;
    }

private:
    ExplorationManager& manager_;
    ExplorationFsmParams params_;
    DiagnosticsRecorder noopDiagnostics_;
    DiagnosticsRecorder* diagnostics_{nullptr};
    ExplorationState state_{ExplorationState::INIT};
    double initTime_{-1.0};
    double execStart_{-1.0};
    double idleStart_{-1.0};
    double trajectoryStartTime_{-1.0};
    bool trigger_{false};
    bool trajectoryPublished_{false};
    bool trajectoryFinished_{false};
    bool collision_{false};
    bool avoidCollision_{false};
    int collisionReplanCount_{0};
    bool frontierCovered_{false};
    std::string triggerReason_{"trigger"};
    std::string transitionReason_{"none"};
};

} // arc
