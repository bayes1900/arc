#pragma once

#include <diagnostics/types.h>

#include <algorithm>
#include <deque>
#include <mutex>
#include <vector>

namespace arc {

class DiagnosticsCollector {
public:
    DiagnosticsCollector() = default;

    void setEnabled(bool enabled) {
        std::lock_guard<std::mutex> lock(mutex_);
        enabled_ = enabled;
    }
    bool enabled() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return enabled_;
    }
    void setEventQueueSize(size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        eventQueueSize_ = std::max<size_t>(1, size);
    }
    void initializeState(int state, double now) {
        std::lock_guard<std::mutex> lock(mutex_);
        state_ = state;
        lastStateTime_ = now;
        if (state >= 0 && state < static_cast<int>(stateEntries_.size())) {
            stateEntries_[static_cast<size_t>(state)] = 1;
        }
    }
    void recordStateTransition(int from, int to, double now, const std::string& reason) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!enabled_) {
            return;
        }
        const int previous = from >= 0 ? from : state_;
        const double elapsed = lastStateTime_ >= 0.0 ? std::max(0.0, now - lastStateTime_) : 0.0;
        if (lastStateTime_ >= 0.0 && previous >= 0 &&
            previous < static_cast<int>(stateTimes_.size())) {
            stateTimes_[static_cast<size_t>(previous)] += elapsed;
        }
        state_ = to;
        lastStateTime_ = now;
        stateTransitions_ += 1;
        if (to >= 0 && to < static_cast<int>(stateEntries_.size())) {
            stateEntries_[static_cast<size_t>(to)] += 1;
        }
        DiagRecord record;
        record.stamp = now;
        record.eventType = "fsm_transition";
        record.stage = "fsm";
        record.reason = reason;
        record.fromState = previous;
        record.toState = to;
        record.value = elapsed;
        pushEvent_(std::move(record));
    }
    void recordStage(DiagStage stage, double seconds) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!enabled_) {
            return;
        }
        const size_t index = static_cast<size_t>(stage);
        if (index >= stages_.size()) {
            return;
        }
        DiagStageStats& stats = stages_[index];
        stats.count += 1;
        stats.last = seconds;
        stats.total += seconds;
        stats.max = std::max(stats.max, seconds);
    }
    void recordEvent(DiagRecord record) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!enabled_) {
            return;
        }
        pushEvent_(std::move(record));
    }
    void increment(DiagCounter counter, uint32_t value = 1) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!enabled_) {
            return;
        }
        const size_t index = static_cast<size_t>(counter);
        if (index < counters_.size()) {
            counters_[index] += value;
        }
    }
    void setGauge(DiagGauge gauge, double value) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!enabled_) {
            return;
        }
        const size_t index = static_cast<size_t>(gauge);
        if (index < gauges_.size()) {
            gauges_[index] = value;
        }
    }
    std::vector<DiagRecord> takeEvents() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<DiagRecord> events(events_.begin(), events_.end());
        events_.clear();
        return events;
    }
    DiagSnapshot snapshot(double now) const {
        std::lock_guard<std::mutex> lock(mutex_);
        DiagSnapshot result;
        result.state = state_;
        result.stateTransitions = stateTransitions_;
        result.droppedEvents = droppedEvents_;
        result.counters = counters_;
        result.stages = stages_;
        result.stateTimes = stateTimes_;
        result.stateEntries = stateEntries_;
        result.gauges = gauges_;
        if (state_ >= 0 && state_ < static_cast<int>(result.stateTimes.size())) {
            const double elapsed = std::max(0.0, now - lastStateTime_);
            result.stateTime = elapsed;
            result.stateTimes[static_cast<size_t>(state_)] += elapsed;
        }
        return result;
    }

private:
    void pushEvent_(DiagRecord record) {
        if (events_.size() >= eventQueueSize_) {
            events_.pop_front();
            droppedEvents_ += 1;
        }
        events_.push_back(std::move(record));
    }

private:
    mutable std::mutex mutex_;
    bool enabled_{true};
    size_t eventQueueSize_{1000};
    int state_{-1};
    double lastStateTime_{-1.0};
    uint32_t stateTransitions_{0};
    uint32_t droppedEvents_{0};
    std::array<uint32_t, static_cast<size_t>(DiagCounter::COUNT)> counters_{};
    std::array<DiagStageStats, static_cast<size_t>(DiagStage::COUNT)> stages_{};
    std::array<double, 8> stateTimes_{};
    std::array<uint32_t, 8> stateEntries_{};
    std::array<double, static_cast<size_t>(DiagGauge::COUNT)> gauges_{};
    std::deque<DiagRecord> events_;
};

} // arc
