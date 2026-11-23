#pragma once

#include <base/ros.h>

#include <base/iarc.h>

namespace arc {

class TimeStats {
public:
    TimeStats() = default;

    TimeStats(std::string name, bool enabled = false)
        : name_(std::move(name))
        , enabled_(enabled)
    {}
    void setLogger(rclcpp::Logger logger) {
        logger_ = std::move(logger);
        hasLogger_ = true;
    }
    void setLogInterval(double seconds) {
        logInterval_ = std::max(0.0, seconds);
    }
    void setEnabled(bool enabled) { enabled_ = enabled; }
    bool enabled() const { return enabled_; }
    double last() const { return last_; }
    double average() const { return count_ > 0 ? total_ / count_ : 0.0; }
    double max() const { return max_; }
    int count() const { return count_; }

    void record(double seconds) {
        if (!enabled_) {
            return;
        }
        last_ = seconds;
        total_ += seconds;
        max_ = std::max(max_, seconds);
        ++ count_;
        if (!hasLogger_) {
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(now - lastLogTime_).count();
        if (!hasLogged_ || logInterval_ <= 0.0 || elapsed >= logInterval_) {
            RCLCPP_INFO(
                logger_,
                "%s t: cur: %.6f, avg: %.6f, max: %.6f",
                name_.c_str(), last_, average(), max_
            );
            lastLogTime_ = now;
            hasLogged_ = true;
        }
    }

private:
    std::string name_;
    bool enabled_{false};
    rclcpp::Logger logger_{rclcpp::get_logger("timestats")};
    bool hasLogger_{false};
    double logInterval_{3.0};
    bool hasLogged_{false};
    std::chrono::steady_clock::time_point lastLogTime_;
    double last_{0.0};
    double total_{0.0};
    double max_{0.0};
    int count_{0};
};

} // arc
