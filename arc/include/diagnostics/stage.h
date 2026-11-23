#pragma once

#include <diagnostics/collector.h>

#include <chrono>

namespace arc {

class StageTimer {
public:
    StageTimer(DiagnosticsCollector* collector, DiagStage stage)
        : collector_(collector)
        , stage_(stage)
        , start_(std::chrono::steady_clock::now())
    {}
    ~StageTimer() {
        if (collector_) {
            collector_->recordStage(stage_, elapsed());
        }
    }
    StageTimer(const StageTimer&) = delete;
    StageTimer& operator=(const StageTimer&) = delete;
    StageTimer(StageTimer&& other) noexcept
        : collector_(other.collector_)
        , stage_(other.stage_)
        , start_(other.start_)
    {
        other.collector_ = nullptr;
    }
    StageTimer& operator=(StageTimer&&) = delete;

    double elapsed() const {
        return std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start_
        ).count();
    }

private:
    DiagnosticsCollector* collector_{nullptr};
    DiagStage stage_;
    std::chrono::steady_clock::time_point start_;
};


} // arc
