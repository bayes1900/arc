#pragma once

#include <base/base.h>
#include <vio/opticalflow/base.h>
#include <vio/estimator/state.h>

namespace arc {

struct VioVisualizationData {
    using Ptr = std::shared_ptr<VioVisualizationData>;
    int64_t tNs;
    std::vector<Sophus::SE3d> states;
    std::vector<Sophus::SE3d> frames;
    std::vector<Eigen::Vector3d> points;
    std::vector<int> pointIds;
    OpticalFlowResult::Ptr optFlowRes;
    std::vector<std::vector<Eigen::Vector4d>> projections;
};

class VioEstimatorBase {
public:
    using Ptr = std::shared_ptr<VioEstimatorBase>;
    virtual ~VioEstimatorBase() = default;

    VioEstimatorBase()
        : outStateQueue(nullptr)
        , outMargQueue(nullptr)
        , outVisQueue(nullptr)
    {
        visionDataQueue.set_capacity(10);
        imuDataQueue.set_capacity(300);
        lastProcessedTNs = 0;
        finished = false;
    }

public:
    std::atomic<int64_t> lastProcessedTNs;
    std::atomic<bool> finished;
    tbb::concurrent_bounded_queue<OpticalFlowResult::Ptr> visionDataQueue;
    tbb::concurrent_bounded_queue<ImuData::Ptr> imuDataQueue;
    tbb::concurrent_bounded_queue<PoseVelBiasState::Ptr>* outStateQueue = nullptr;
    tbb::concurrent_bounded_queue<MargData::Ptr>* outMargQueue = nullptr;
    tbb::concurrent_bounded_queue<VioVisualizationData::Ptr>* outVisQueue = nullptr;
    virtual void initialize(
        int64_t tNs,
        const Sophus::SE3d& tWI,
        const Eigen::Vector3d& velWI,
        const Eigen::Vector3d& bg,
        const Eigen::Vector3d& ba
    ) = 0;
    virtual void initialize(const Eigen::Vector3d& bg, const Eigen::Vector3d& ba) = 0;
    virtual void maybeJoin() = 0;
    virtual void drainInputQueues() {
        while (!imuDataQueue.empty()) {
            ImuData::Ptr d;
            imuDataQueue.pop(d);
        }
        while (!visionDataQueue.empty()) {
            OpticalFlowResult::Ptr d;
            visionDataQueue.pop(d);
        }
    }
    virtual Sophus::SE3d getTWIInit() = 0;
    virtual void setMaxStates(size_t val) = 0;
    virtual void setMaxKfs(size_t val) = 0;
    virtual void addIMUToQueue(const ImuData::Ptr& data) = 0;
    virtual void addVisionToQueue(const OpticalFlowResult::Ptr& data) = 0;
};


} // arc
