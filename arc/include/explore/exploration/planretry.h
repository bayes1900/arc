#pragma once

#include <algorithm>

namespace arc {

class PlanRetryPolicy {
public:
    void reset() {
        lastAttemptTime_ = -1.0;
        delay_ = 0.0;
        consecutiveFailures_ = 0;
    }
    bool shouldAttempt(double now) const {
        if (lastAttemptTime_ < 0.0) {
            return true;
        }
        return now - lastAttemptTime_ >= delay_ - 1e-9;
    }
    void onAttempt(double now) {
        lastAttemptTime_ = now;
    }
    void onSuccess() {
        delay_ = 0.0;
        consecutiveFailures_ = 0;
    }
    void onFailure(double now, double baseDelay, double maxDelay, double factor) {
        lastAttemptTime_ = now;
        consecutiveFailures_ += 1;
        if (consecutiveFailures_ == 1) {
            delay_ = baseDelay;
            return;
        }
        delay_ = std::min(maxDelay, delay_ * factor);
    }
    void onNoGrid() {
        delay_ = 0.0;
        consecutiveFailures_ = 0;
    }
    int consecutiveFailures() const {
        return consecutiveFailures_;
    }
    double currentDelay() const {
        return delay_;
    }

private:
    double lastAttemptTime_{-1.0};
    double delay_{0.0};
    int consecutiveFailures_{0};
};

} // arc
