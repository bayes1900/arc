#pragma once

#include <base/base.h>
#include <lio/estimator.h>
#include <lio/lioconfig.h>
#include <lio/liotypes.h>
#include <lio/preprocessor.h>
#include <condition_variable>
#include <thread>

namespace arc {

class LioPipeline {
public:
    explicit LioPipeline(const LioConfig& config, bool publishCloud = true)
        : estimator_(config)
        , preprocessor_(config)
        , publishCloud_(publishCloud)
    {}
    ~LioPipeline() {
        stop();
    }
    LioPipeline(const LioPipeline&) = delete;
    LioPipeline& operator=(const LioPipeline&) = delete;

    void start() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (running_) {
            return;
        }
        running_ = true;
        worker_ = std::thread([this]() { run_(); });
    }
    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            running_ = false;
        }
        condition_.notify_all();
        if (worker_.joinable()) {
            worker_.join();
        }
    }
    void addLidar(LidarFrame frame) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!lidarBuffer_.empty() && frame.stamp < lastLidarStamp_) {
                std::cerr << "[lio] lidar loop back, clear buffer" << std::endl;
                lidarBuffer_.clear();
            }
            const double stamp = frame.stamp;
            lidarBuffer_.push_back(std::move(frame));
            lastLidarStamp_ = stamp;
            while (lidarBuffer_.size() > MAX_LIDAR_BUFFER) {
                lidarBuffer_.pop_front();
            }
        }
        condition_.notify_all();
    }
    void addImu(const ImuSample& imu) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!imuBuffer_.empty() && imu.time < lastImuStamp_) {
                std::cerr << "[lio] imu loop back, clear buffer" << std::endl;
                imuBuffer_.clear();
            }
            imuBuffer_.push_back(imu);
            lastImuStamp_ = imu.time;
            while (imuBuffer_.size() > MAX_IMU_BUFFER) {
                imuBuffer_.pop_front();
            }
        }
        condition_.notify_all();
    }
    bool getOutput(LioOutput& output) {
        std::lock_guard<std::mutex> lock(outputMutex_);
        if (!hasOutput_) {
            return false;
        }
        output = output_;
        return true;
    }

private:
    void run_() {
        while (true) {
            MeasurementGroup measurement;
            LidarFrame frame;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                condition_.wait(lock, [this]() {
                    return !running_ || (!lidarBuffer_.empty() && !imuBuffer_.empty());
                });
                if (!running_) {
                    return;
                }

                frame = std::move(lidarBuffer_.front());
                condition_.wait(lock, [this, endTime = frame.endTime]() {
                    return !running_ || (!imuBuffer_.empty() && imuBuffer_.back().time >= endTime);
                });
                if (!running_) {
                    return;
                }

                measurement.lidarBeginTime = frame.stamp;
                measurement.lidarEndTime = frame.endTime;
                measurement.lidar = std::make_shared<LioCloud>();

                while (!imuBuffer_.empty() && imuBuffer_.front().time <= frame.endTime) {
                    measurement.imu.push_back(imuBuffer_.front());
                    imuBuffer_.pop_front();
                }
                lidarBuffer_.pop_front();
            }
            try {
                preprocessor_.process(frame, *measurement.lidar);
            } catch (const std::exception& error) {
                std::cerr << "[lio] preprocess failed: " << error.what() << std::endl;
                continue;
            }

            bool processed = false;
            try {
                processed = estimator_.process(measurement);
            } catch (const std::exception& error) {
                std::cerr << "[lio] estimator failed: " << error.what() << std::endl;
                continue;
            }
            if (!processed) {
                continue;
            }
            LioOutput output;
            output.valid = true;
            output.stamp = measurement.lidarEndTime;
            output.position = estimator_.position();
            output.rotation = estimator_.rotation();
            output.velocity = estimator_.velocity();
            if (publishCloud_) {
                output.cloud = estimator_.registeredCloud();
            }
            {
                std::lock_guard<std::mutex> lock(outputMutex_);
                output_ = std::move(output);
                hasOutput_ = true;
            }
        }
    }

private:
    static constexpr size_t MAX_LIDAR_BUFFER = 20;
    static constexpr size_t MAX_IMU_BUFFER = 5000;

    LioEstimator estimator_;
    PointPreprocessor preprocessor_;
    bool publishCloud_{true};

    std::mutex mutex_;
    std::condition_variable condition_;
    std::deque<LidarFrame> lidarBuffer_;
    std::deque<ImuSample> imuBuffer_;
    double lastLidarStamp_{-1.0};
    double lastImuStamp_{-1.0};
    bool running_{false};
    std::thread worker_;

    std::mutex outputMutex_;
    LioOutput output_;
    bool hasOutput_{false};
};

} // arc
