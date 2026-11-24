#pragma once

#include <base/base.h>
#include <lio/lioconfig.h>
#include <lio/liotypes.h>

namespace arc {

class PointPreprocessor {
public:
    explicit PointPreprocessor(const LioConfig& config) : config_(config) {
        config_.blindSq = config_.blind * config_.blind;
        ensure(config_.lidarType != LidarType::Avia, "Livox CustomMsg is handled by the ROS node adapter, not the core preprocessor");
    }
    void process(const LidarFrame& frame, LioCloud& output) {
        output.clear();
        if (frame.points.empty()) {
            return;
        }
        switch (config_.lidarType) {
            case LidarType::Sim: {
                processSim_(frame, output);
                break;
            }
            case LidarType::Generic:
            case LidarType::Ouster: {
                if (frame.hasTime) {
                    processTimed_(frame, output);
                } else {
                    processUntimed_(frame, output);
                }
                break;
            }
            default: {
                throw std::runtime_error("Invalid lidarType");
                break;
            }
        }
    }

private:
    void processTimed_(const LidarFrame& frame, LioCloud& output) {
        output.reserve(frame.points.size());
        for (size_t i = 0; i < frame.points.size(); ++ i) {
            if (config_.pointFilterNum > 1 && static_cast<int>(i) % config_.pointFilterNum != 0) {
                continue;
            }
            const LidarPoint& point = frame.points[i];
            if (!validPoint_(point)) {
                continue;
            }
            LioPoint added;
            added.x = point.x;
            added.y = point.y;
            added.z = point.z;
            added.intensity = point.intensity;
            added.time = point.time;
            output.push_back(added);
        }
    }
    void processUntimed_(const LidarFrame& frame, LioCloud& output) {
        const int scanLine = std::max(config_.scanLine, 1);
        std::vector<bool> isFirst(scanLine, true);
        std::vector<double> yawFirst(scanLine, 0.0);
        std::vector<double> timeLast(scanLine, 0.0);
        const double omega = std::max(config_.scanRate, 1) * 360.0;

        output.reserve(frame.points.size());
        for (size_t i = 0; i < frame.points.size(); ++ i) {
            const LidarPoint& point = frame.points[i];
            if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
                continue;
            }
            if (point.ring >= scanLine) {
                continue;
            }
            const int layer = point.ring;
            const double yaw = std::atan2(point.y, point.x) * 57.2957;

            float pointTime = 0.0f;
            if (isFirst[static_cast<size_t>(layer)]) {
                yawFirst[static_cast<size_t>(layer)] = yaw;
                timeLast[static_cast<size_t>(layer)] = 0.0;
                isFirst[static_cast<size_t>(layer)] = false;
                continue;
            }

            if (yaw <= yawFirst[static_cast<size_t>(layer)]) {
                pointTime = static_cast<float>((yawFirst[static_cast<size_t>(layer)] - yaw) / omega);
            } else {
                pointTime = static_cast<float>((yawFirst[static_cast<size_t>(layer)] - yaw + 360.0) / omega);
            }
            if (pointTime < timeLast[static_cast<size_t>(layer)]) {
                pointTime += static_cast<float>(360.0 / omega);
            }
            timeLast[static_cast<size_t>(layer)] = pointTime;

            if (config_.pointFilterNum > 1 && static_cast<int>(i) % config_.pointFilterNum != 0) {
                continue;
            }
            if (!validRange_(point)) {
                continue;
            }

            LioPoint added;
            added.x = point.x;
            added.y = point.y;
            added.z = point.z;
            added.intensity = point.intensity;
            added.time = pointTime;
            output.push_back(added);
        }
    }
    void processSim_(const LidarFrame& frame, LioCloud& output) {
        output.reserve(frame.points.size());
        for (const LidarPoint& point : frame.points) {
            if (!validPoint_(point)) {
                continue;
            }
            LioPoint added;
            added.x = point.x;
            added.y = point.y;
            added.z = point.z;
            added.intensity = point.intensity;
            added.time = 0.0f;
            output.push_back(added);
        }
    }
    bool validPoint_(const LidarPoint& point) const {
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
            return false;
        }
        return validRange_(point);
    }
    bool validRange_(const LidarPoint& point) const {
        const double range2 = static_cast<double>(point.x) * point.x
            + static_cast<double>(point.y) * point.y
            + static_cast<double>(point.z) * point.z;
        return range2 > config_.blindSq;
    }

private:
    LioConfig config_;
};

} // arc
