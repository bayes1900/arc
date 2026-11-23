#pragma once

#include <base/base.h>
#include <vio/vio.h>

namespace arc {

class VioPipeline {
public:
    VioPipeline(
        const VioConfig& config,
        const Calibration<double>& calibration,
        const Eigen::Vector3d& gravity=Eigen::Vector3d(0.0, 0.0, -9.81)
    )
        : config_(config)
        , calibration_(calibration)
        , gravity_(gravity)
    {
        if (calibration_.intrinsics.empty()) {
            throw std::invalid_argument("VIO calibration has no cameras");
        }
        if (calibration_.intrinsics.size() != calibration_.resolution.size() ||
            calibration_.intrinsics.size() != calibration_.tIC.size()) {
            throw std::invalid_argument("VIO calibration sizes do not match");
        }
        if (!gravity_.allFinite() || gravity_.norm() < 1e-6) {
            throw std::invalid_argument("invalid gravity vector");
        }
        opticalFlow_ = OpticalFlowFactory::getOpticalFlow(config_, calibration_);
        estimator_ = VioEstimatorFactory::getVioEstimator(config_, calibration_, gravity_);
        opticalFlow_->outputQueue = &estimator_->visionDataQueue;
        estimator_->outStateQueue = &stateQueue_;
        stateQueue_.set_capacity(100);
    }
    ~VioPipeline() { stop(); }
    VioPipeline(const VioPipeline&) = delete;
    VioPipeline& operator=(const VioPipeline&) = delete;

public:
    void start(
        int64_t tNs,
        const Sophus::SE3d& tWI,
        const Eigen::Vector3d& velWI,
        const Eigen::Vector3d& bg,
        const Eigen::Vector3d& ba
    ) {
        if (started_) {
            return;
        }
        if (tNs <= 0) {
            throw std::invalid_argument("invalid VIO start timestamp");
        }
        clearImu_();
        minImuTNs_ = tNs;
        estimator_->initialize(tNs, tWI, velWI, bg, ba);
        started_ = true;
    }
    void startAutoAt(int64_t tNs, const Eigen::Vector3d& bg, const Eigen::Vector3d& ba) {
        if (started_) {
            return;
        }
        clearImu_();
        minImuTNs_ = tNs;
        estimator_->initialize(bg, ba);
        started_ = true;
    }
    void startAuto(const Eigen::Vector3d& bg, const Eigen::Vector3d& ba) {
        if (started_) {
            return;
        }
        minImuTNs_ = 0;
        estimator_->initialize(bg, ba);
        started_ = true;
    }
    void addImage(const OpticalFlowInput::Ptr& input) {
        if (!opticalFlow_ || !input) {
            throw std::invalid_argument("null optical flow input");
        }
        if (input->imgData.size() != calibration_.intrinsics.size()) {
            throw std::invalid_argument("image count does not match calibration");
        }
        if (input->tNs <= 0) {
            throw std::invalid_argument("invalid image timestamp");
        }
        if (opticalFlow_->inputQueue.size() >= opticalFlow_->inputQueue.capacity()) {
            OpticalFlowInput::Ptr dropped;
            opticalFlow_->inputQueue.try_pop(dropped);
            ++ droppedImages_;
        }
        opticalFlow_->inputQueue.try_push(input);
    }
    void addImu(const ImuData::Ptr& data) {
        if (!estimator_ || !data) {
            throw std::invalid_argument("null IMU data");
        }
        if (!data->accel.allFinite() || !data->gyro.allFinite()) {
            throw std::invalid_argument("non-finite IMU data");
        }
        if (data->tNs < minImuTNs_) {
            return;
        }
        if (estimator_->imuDataQueue.size() >= estimator_->imuDataQueue.capacity()) {
            ImuData::Ptr dropped;
            estimator_->imuDataQueue.try_pop(dropped);
            ++ droppedImu_;
        }
        estimator_->imuDataQueue.try_push(data);
    }
    size_t droppedImages() const { return droppedImages_; }
    size_t droppedImu() const { return droppedImu_; }
    bool tryPopState(PoseVelBiasState::Ptr& state) { return stateQueue_.try_pop(state); }
    void stop() {
        if (opticalFlow_) {
            opticalFlow_->inputQueue.push(nullptr);
            opticalFlow_.reset();
        }
        if (estimator_) {
            if (started_) {
                auto dummy = std::make_shared<ImuData>();
                dummy->tNs = std::numeric_limits<int64_t>::max() / 2;
                estimator_->imuDataQueue.push(dummy);
                estimator_->visionDataQueue.push(nullptr);
            }
            estimator_.reset();
        }
        started_ = false;
    }
    bool started() const { return started_; }
    void clearImu_() {
        if (!estimator_) {
            return;
        }
        ImuData::Ptr data;
        while (estimator_->imuDataQueue.try_pop(data)) {
        }
    }
    const VioConfig& config() const { return config_; }
    const Calibration<double>& calibration() const { return calibration_; }

private:
    VioConfig config_;
    Calibration<double> calibration_;
    Eigen::Vector3d gravity_;
    OpticalFlowBase::Ptr opticalFlow_;
    VioEstimatorBase::Ptr estimator_;
    tbb::concurrent_bounded_queue<PoseVelBiasState::Ptr> stateQueue_;
    bool started_{false};
    int64_t minImuTNs_{0};
    size_t droppedImages_{0};
    size_t droppedImu_{0};
};

} // arc
