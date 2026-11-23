#pragma once

#include <base/base.h>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <vio/estimator/accumulator.h>
#include <vio/estimator/stats.h>
#include <vio/estimator/marghelper.h>
#include <vio/estimator/scba.h>
#include <vio/estimator/sqrtba.h>
#include <vio/estimator/base.h>
#include <vio/imu/preintegration.h>
#include <vio/linearization/factory.h>

namespace arc {

class SqrtKeypointVioEstimator : public VioEstimatorBase, public SqrtBundleAdjustmentBase {
public:
    using Ptr = std::shared_ptr<SqrtKeypointVioEstimator>;
    using SqrtBundleAdjustmentBase::RelLinData;
    using SqrtBundleAdjustmentBase::AbsLinData;
    using BundleAdjustmentBase::computeError;
    using BundleAdjustmentBase::getCurrentPoints;
    using BundleAdjustmentBase::computeDelta;
    using BundleAdjustmentBase::computeProjections;
    using BundleAdjustmentBase::triangulate;
    using BundleAdjustmentBase::backup;
    using BundleAdjustmentBase::restore;
    using BundleAdjustmentBase::getPoseStateWithLin;
    using BundleAdjustmentBase::computeModelCostChange;
    using SqrtBundleAdjustmentBase::linearizeHelper;
    using SqrtBundleAdjustmentBase::linearizeAbsHelper;
    using SqrtBundleAdjustmentBase::linearizeRel;
    using SqrtBundleAdjustmentBase::linearizeAbs;
    using SqrtBundleAdjustmentBase::updatePoints;
    using SqrtBundleAdjustmentBase::updatePointsAbs;
    using SqrtBundleAdjustmentBase::linearizeMargPrior;
    using SqrtBundleAdjustmentBase::computeMargPriorError;
    using SqrtBundleAdjustmentBase::computeMargPriorModelCostChange;
    using SqrtBundleAdjustmentBase::checkNullspace;
    using SqrtBundleAdjustmentBase::checkEigenvalues;
    SqrtKeypointVioEstimator(const Eigen::Vector3d& g, const arc::Calibration<double>& calib, const VioConfig& config);
    void initialize(
        int64_t tNs,
        const Sophus::SE3d& tWI,
        const Eigen::Vector3d& velWI,
        const Eigen::Vector3d& bg,
        const Eigen::Vector3d& ba
   ) override;
    void initialize(const Eigen::Vector3d& bg, const Eigen::Vector3d& ba) override;
    virtual ~SqrtKeypointVioEstimator() { maybeJoin(); }
    void maybeJoin() override {
        if (processingThread_) {
            processingThread_->join();
            processingThread_.reset();
        }
    }
    void addIMUToQueue(const ImuData::Ptr& data) override;
    void addVisionToQueue(const OpticalFlowResult::Ptr& data) override;
    typename ImuData::Ptr popFromImuDataQueue();
    bool measure(
        const OpticalFlowResult::Ptr& optFlowMeas, const typename IntegratedImuMeasurement::Ptr& meas);
    void optimizeAndMarg(
        const std::map<int64_t, int>& numPointsConnected, const std::unordered_set<KeypointId>& lostLandmarks);
    void marginalize(
        const std::map<int64_t, int>& numPointsConnected, const std::unordered_set<KeypointId>& lostLandmarks);
    void optimize();
    void debugFinalize() override;
    void logMargNullspace();
    Eigen::VectorXd checkMargNullspace() const;
    Eigen::VectorXd checkMargEigenvalues() const;
    int64_t getTNs() const { return frameStates.at(lastStateTNs_).getState().tNs; }
    const Sophus::SE3d& getTWI() const { return frameStates.at(lastStateTNs_).getState().tWI; }
    const Vec3d& getVelWI() const { return frameStates.at(lastStateTNs_).getState().velWI; }
    const PoseVelBiasState& getState() const { return frameStates.at(lastStateTNs_).getState(); }
    PoseVelBiasState getState(int64_t tNs) const {
        PoseVelBiasState state;
        auto it = frameStates.find(tNs);
        if (it != frameStates.end()) {
            return it->second.getState();
        }
        auto it2 = framePoses.find(tNs);
        if (it2 != framePoses.end()) {
            state.tWI = it2->second.getPose();
        }
        return state;
    }
    void setMaxStates(size_t val) override { maxStates_ = val; }
    void setMaxKfs(size_t val) override { maxKfs_ = val; }
    std::vector<Sophus::SE3d> getFrameStates() const {
        std::vector<Sophus::SE3d> res;
        for (const auto& kv : frameStates) {
            res.push_back(kv.second.getState().tWI);
        }
        return res;
    }
    std::vector<Sophus::SE3d> getFramePoses() const {
        std::vector<Sophus::SE3d> res;
        for (const auto& kv : framePoses) {
            res.push_back(kv.second.getPose());
        }
        return res;
    }
    std::map<int64_t, Sophus::SE3d> getAllPosesMap() const {
        std::map<int64_t, Sophus::SE3d> res;
        for (const auto& kv : framePoses) {
            res[kv.first] = kv.second.getPose();
        }
        for (const auto& kv : frameStates) {
            res[kv.first] = kv.second.getState().tWI;
        }
        return res;
    }
    Sophus::SE3d getTWIInit() override { return tWIInit_; }

private:
    using BundleAdjustmentBase::framePoses;
    using BundleAdjustmentBase::frameStates;
    using BundleAdjustmentBase::lmdb;
    using BundleAdjustmentBase::obsStdDev;
    using BundleAdjustmentBase::huberThresh;
    using BundleAdjustmentBase::calib;

private:
    bool takeKf_;
    int framesAfterKf_;
    std::set<int64_t> kfIds_;
    int64_t lastStateTNs_;
    std::map<int64_t, IntegratedImuMeasurement> imuMeas_;
    const Vec3d g_;
    std::map<int64_t, OpticalFlowResult::Ptr> prevOptFlowRes_;
    std::map<int64_t, int> numPointsKf_;
    MargLinData margData_;
    MargLinData nullspaceMargData_;
    Vec3d gyroBiasSqrtWeight_, accelBiasSqrtWeight_;
    size_t maxStates_;
    size_t maxKfs_;
    Sophus::SE3d tWIInit_;
    bool initialized_;
    bool optStarted_;
    VioConfig config_;

    constexpr static double VEE_FACTOR = double(2.0);
    constexpr static double INITIAL_VEE = double(2.0);
    double lambda_, minLambda_, maxLambda_, lambdaVee_;
    std::shared_ptr<std::thread> processingThread_;
    ExecutionStats statsAll_;
    ExecutionStats statsSums_;
};

inline SqrtKeypointVioEstimator::SqrtKeypointVioEstimator(
    const Eigen::Vector3d& g_, const arc::Calibration<double>& calib_, const VioConfig& config_)
    : takeKf_(true)
, framesAfterKf_(0)
, g_(g_.cast<double>())
, initialized_(false)
, config_(config_)
, lambda_(config_.vioLmLambdaInitial)
, minLambda_(config_.vioLmLambdaMin)
, maxLambda_(config_.vioLmLambdaMax)
, lambdaVee_(2)
{
    obsStdDev = double(config_.vioObsStdDev);
    huberThresh = double(config_.vioObsHuberThresh);
    calib = calib_.cast<double>();
    margData_.isSqrt = config_.vioSqrtMarg;
    margData_.h.setZero(15, 15);
    margData_.b.setZero(15);
    nullspaceMargData_.isSqrt = margData_.isSqrt;
    nullspaceMargData_.h.setZero(15, 15);
    nullspaceMargData_.b.setZero(15);
    if (margData_.isSqrt) {
        margData_.h.diagonal().head<3>().setConstant(std::sqrt(double(config_.vioInitPoseWeight)));
        margData_.h(5, 5) = std::sqrt(double(config_.vioInitPoseWeight));
        margData_.h.diagonal().segment<3>(9).array() = std::sqrt(double(config_.vioInitBaWeight));
        margData_.h.diagonal().segment<3>(12).array() = std::sqrt(double(config_.vioInitBgWeight));
    } else {
        margData_.h.diagonal().head<3>().setConstant(double(config_.vioInitPoseWeight));
        margData_.h(5, 5) = double(config_.vioInitPoseWeight);
        margData_.h.diagonal().segment<3>(9).array() = double(config_.vioInitBaWeight);
        margData_.h.diagonal().segment<3>(12).array() = double(config_.vioInitBgWeight);
    }
    gyroBiasSqrtWeight_ = calib.gyroBiasStd.array().inverse();
    accelBiasSqrtWeight_ = calib.accelBiasStd.array().inverse();
    maxStates_ = config_.vioMaxStates;
    maxKfs_ = config_.vioMaxKfs;
    optStarted_ = false;
    visionDataQueue.set_capacity(10);
    imuDataQueue.set_capacity(300);
}
inline void SqrtKeypointVioEstimator::initialize(
    int64_t tNs,
    const Sophus::SE3d& tWI,
    const Eigen::Vector3d& velWI,
    const Eigen::Vector3d& bg,
    const Eigen::Vector3d& ba
) {
    initialized_ = true;
    tWIInit_ = tWI.cast<double>();
    lastStateTNs_ = tNs;
    imuMeas_[tNs] = IntegratedImuMeasurement(tNs, bg.cast<double>(), ba.cast<double>());
    frameStates[tNs] = PoseVelBiasStateWithLin(
        tNs, tWIInit_, velWI.cast<double>(), bg.cast<double>(), ba.cast<double>(), true);
    margData_.order.absOrderMap[tNs] = std::make_pair(0, 15);
    margData_.order.totalSize = 15;
    margData_.order.items = 1;
    nullspaceMargData_.order = margData_.order;
    initialize(bg, ba);
}
inline void SqrtKeypointVioEstimator::initialize(const Eigen::Vector3d& bg_, const Eigen::Vector3d& ba_) {
    Vec3d bgInit = bg_.cast<double>();
    Vec3d baInit = ba_.cast<double>();
    auto procFunc = [&, bg = bgInit, ba = baInit] {
        try {
        OpticalFlowResult::Ptr prevFrame, currFrame;
        IntegratedImuMeasurement::Ptr meas;
        const Vec3d accelCov = calib.dicreteTimeAccelNoiseStd().array().square();
        const Vec3d gyroCov = calib.dicreteTimeGyroNoiseStd().array().square();
        ImuData::Ptr data = popFromImuDataQueue();
        ensure(data != nullptr, "first IMU measurement is nullptr");
        data->accel = calib.calibAccelBias.getCalibrated(data->accel);
        data->gyro = calib.calibGyroBias.getCalibrated(data->gyro);
        while (true) {
            visionDataQueue.pop(currFrame);
            if (config_.vioEnforceRealtime) {
                while (!visionDataQueue.empty())
                    visionDataQueue.pop(currFrame);
            }
            if (!currFrame.get()) {
                break;
            }
            if (!initialized_) {
                while (data->tNs < currFrame->tNs) {
                    data = popFromImuDataQueue();
                    if (!data)
                        break;
                    data->accel = calib.calibAccelBias.getCalibrated(data->accel);
                    data->gyro = calib.calibGyroBias.getCalibrated(data->gyro);
                }
                Vec3d velWIInit;
                velWIInit.setZero();
                tWIInit_.setQuaternion(Eigen::Quaternion<double>::FromTwoVectors(data->accel, Vec3d::UnitZ()));
                lastStateTNs_ = currFrame->tNs;
                imuMeas_[lastStateTNs_] = IntegratedImuMeasurement(lastStateTNs_, bg, ba);
                frameStates[lastStateTNs_] = PoseVelBiasStateWithLin(lastStateTNs_, tWIInit_, velWIInit, bg, ba, true);
                margData_.order.absOrderMap[lastStateTNs_] = std::make_pair(0, 15);
                margData_.order.totalSize = 15;
                margData_.order.items = 1;
                if (config_.vioDebug) {
                    std::cout << "Setting up filter: tNs " << lastStateTNs_ << std::endl;
                    std::cout << "tWI\n" << tWIInit_.matrix() << std::endl;
                    std::cout << "velWI " << velWIInit.transpose() << std::endl;
                }
                if (config_.vioDebug || config_.vioExtendedLogging) {
                    logMargNullspace();
                }
                initialized_ = true;
            }
            if (prevFrame) {
                auto lastState = frameStates.at(lastStateTNs_);
                meas.reset(new IntegratedImuMeasurement(
                    prevFrame->tNs, lastState.getState().biasGyro, lastState.getState().biasAccel));
                ensure(prevFrame->tNs < currFrame->tNs, "duplicate frame timestamps");
                while (data->tNs <= prevFrame->tNs) {
                    data = popFromImuDataQueue();
                    if (!data)
                        break;
                    data->accel = calib.calibAccelBias.getCalibrated(data->accel);
                    data->gyro = calib.calibGyroBias.getCalibrated(data->gyro);
                }
                while (data->tNs <= currFrame->tNs) {
                    meas->integrate(*data, accelCov, gyroCov);
                    data = popFromImuDataQueue();
                    if (!data)
                        break;
                    data->accel = calib.calibAccelBias.getCalibrated(data->accel);
                    data->gyro = calib.calibGyroBias.getCalibrated(data->gyro);
                }
                if (meas->getStartTNs() + meas->getDtNs() < currFrame->tNs) {
                    if (!data.get())
                        break;
                    int64_t tmp = data->tNs;
                    data->tNs = currFrame->tNs;
                    meas->integrate(*data, accelCov, gyroCov);
                    data->tNs = tmp;
                }
            }
            measure(currFrame, meas);
            prevFrame = currFrame;
        }
        if (outVisQueue)
            outVisQueue->push(nullptr);
        if (outMargQueue)
            outMargQueue->push(nullptr);
        if (outStateQueue)
            outStateQueue->push(nullptr);
        finished = true;
        if (config_.vioDebug) {
            std::cout << "Finished VIOFilter " << std::endl;
        }
        } catch (const std::exception& error) {
            std::cerr << "VIO processing failed: " << error.what() << std::endl;
            finished = true;
            if (outVisQueue)
                outVisQueue->push(nullptr);
            if (outMargQueue)
                outMargQueue->push(nullptr);
            if (outStateQueue)
                outStateQueue->push(nullptr);
        }
    };
    processingThread_.reset(new std::thread(procFunc));
}
inline void SqrtKeypointVioEstimator::addIMUToQueue(const ImuData::Ptr& data) {
    if (!data) {
        throw std::invalid_argument("null IMU data");
    }
    imuDataQueue.emplace(data);
}
inline void SqrtKeypointVioEstimator::addVisionToQueue(const OpticalFlowResult::Ptr& data) {
    if (!data) {
        throw std::invalid_argument("null optical flow result");
    }
    visionDataQueue.push(data);
}
inline ImuData::Ptr SqrtKeypointVioEstimator::popFromImuDataQueue() {
    ImuData::Ptr data;
    imuDataQueue.pop(data);
    return data;
}
inline bool SqrtKeypointVioEstimator::measure(
    const OpticalFlowResult::Ptr& optFlowMeas, const IntegratedImuMeasurement::Ptr& meas) {
    if (!optFlowMeas) {
        throw std::invalid_argument("null optical flow result");
    }
    statsSums_.add("frameId", optFlowMeas->tNs).setFormat("none");
    Timer tTotal;
    if (meas.get()) {
        ensure(frameStates[lastStateTNs_].getState().tNs == meas->getStartTNs(), "VIO check failed");
        ensure(optFlowMeas->tNs == meas->getDtNs() + meas->getStartTNs(), "VIO check failed");
        ensure(meas->getDtNs() > 0, "VIO check failed");
        PoseVelBiasState nextState = frameStates.at(lastStateTNs_).getState();
        meas->predictState(frameStates.at(lastStateTNs_).getState(), g_, nextState);
        lastStateTNs_ = optFlowMeas->tNs;
        nextState.tNs = optFlowMeas->tNs;
        frameStates[lastStateTNs_] = PoseVelBiasStateWithLin(nextState);
        imuMeas_[meas->getStartTNs()] = *meas;
    }
    prevOptFlowRes_[optFlowMeas->tNs] = optFlowMeas;
    int connected0 = 0;
    std::map<int64_t, int> numPointsConnected;
    std::unordered_set<int> unconnectedObs0;
    for (size_t i = 0; i < optFlowMeas->observations.size(); ++ i) {
        TimeCamId tcidTarget(optFlowMeas->tNs, i);
        for (const auto& kvObs : optFlowMeas->observations[i]) {
            int kptId = kvObs.first;
            if (lmdb.landmarkExists(kptId)) {
                const TimeCamId& tcidHost = lmdb.getLandmark(kptId).hostKfId;
                KeypointObservation kobs;
                kobs.kptId = kptId;
                kobs.pos = kvObs.second.translation().cast<double>();
                lmdb.addObservation(tcidTarget, kobs);
                if (numPointsConnected.count(tcidHost.frameId) == 0) {
                    numPointsConnected[tcidHost.frameId] = 0;
                }
                numPointsConnected[tcidHost.frameId]++;
                if (i == 0)
                    ++ connected0;
            } else {
                if (i == 0) {
                    unconnectedObs0.emplace(kptId);
                }
            }
        }
    }
    if (double(connected0) / (connected0 + unconnectedObs0.size()) < double(config_.vioNewKfKeypointsThresh) &&
        framesAfterKf_ > config_.vioMinFramesAfterKf)
        takeKf_ = true;
    if (config_.vioDebug) {
        std::cout << "connected0 " << connected0 << " unconnected0 " << unconnectedObs0.size() << std::endl;
    }
    if (takeKf_) {
        takeKf_ = false;
        framesAfterKf_ = 0;
        kfIds_.emplace(lastStateTNs_);
        TimeCamId tcidl(optFlowMeas->tNs, 0);
        int numPointsAdded = 0;
        for (int lmId : unconnectedObs0) {
            std::map<TimeCamId, KeypointObservation> kpObs;
            for (const auto& kv : prevOptFlowRes_) {
                for (size_t k = 0; k < kv.second->observations.size(); ++ k) {
                    auto it = kv.second->observations[k].find(lmId);
                    if (it != kv.second->observations[k].end()) {
                        TimeCamId tcido(kv.first, k);
                        KeypointObservation kobs;
                        kobs.kptId = lmId;
                        kobs.pos = it->second.translation().cast<double>();
                        kpObs[tcido] = kobs;
                    }
                }
            }
            bool validKp = false;
            const double minTriangDistance2 = double(config_.vioMinTriangulationDist * config_.vioMinTriangulationDist);
            for (const auto& kvObs : kpObs) {
                if (validKp)
                    break;
                TimeCamId tcido = kvObs.first;
                const Vec2d p0 = optFlowMeas->observations.at(0).at(lmId).translation().cast<double>();
                const Vec2d p1 = prevOptFlowRes_[tcido.frameId]
                                    ->observations[tcido.camId]
                                    .at(lmId)
                                    .translation()
                                    .cast<double>();
                Vec4d p0Point, p1Point;
                bool valid1 = calib.intrinsics[0].unproject(p0, p0Point);
                bool valid2 = calib.intrinsics[tcido.camId].unproject(p1, p1Point);
                if (!valid1 || !valid2)
                    continue;
                Sophus::SE3d T_i0_i1 = getPoseStateWithLin(tcidl.frameId).getPose().inverse() *
                    getPoseStateWithLin(tcido.frameId).getPose();
                Sophus::SE3d T_0_1 = calib.tIC[0].inverse() * T_i0_i1 * calib.tIC[tcido.camId];
                if (T_0_1.translation().squaredNorm() < minTriangDistance2)
                    continue;
                Vec4d p0Triangulated = triangulate(p0Point.head<3>(), p1Point.head<3>(), T_0_1);
                if (p0Triangulated.array().isFinite().all() && p0Triangulated[3] > 0 && p0Triangulated[3] < 3.0) {
                    Keypoint kptPos;
                    kptPos.hostKfId = tcidl;
                    kptPos.direction = StereographicParam::project(p0Triangulated);
                    kptPos.invDist = p0Triangulated[3];
                    lmdb.addLandmark(lmId, kptPos);
                    ++ numPointsAdded;
                    validKp = true;
                }
            }
            if (validKp) {
                for (const auto& kvObs : kpObs) {
                    lmdb.addObservation(kvObs.first, kvObs.second);
                }
            }
        }
        numPointsKf_[optFlowMeas->tNs] = numPointsAdded;
    } else {
        ++ framesAfterKf_;
    }
    std::unordered_set<KeypointId> lostLandmarks;
    if (config_.vioMargLostLandmarks) {
        for (const auto& kv : lmdb.getLandmarks()) {
            bool connected = false;
            for (size_t i = 0; i < optFlowMeas->observations.size(); ++ i) {
                if (optFlowMeas->observations[i].count(kv.first) > 0)
                    connected = true;
            }
            if (!connected) {
                lostLandmarks.emplace(kv.first);
            }
        }
    }
    optimizeAndMarg(numPointsConnected, lostLandmarks);
    if (outStateQueue) {
        PoseVelBiasStateWithLin p = frameStates.at(lastStateTNs_);
        PoseVelBiasState::Ptr data(new PoseVelBiasState(p.getState()));
        outStateQueue->push(data);
    }
    if (outVisQueue) {
        VioVisualizationData::Ptr data(new VioVisualizationData);
        data->tNs = lastStateTNs_;
        for (const auto& kv : frameStates) {
            data->states.emplace_back(kv.second.getState().tWI.cast<double>());
        }
        for (const auto& kv : framePoses) {
            data->frames.emplace_back(kv.second.getPose().cast<double>());
        }
        getCurrentPoints(data->points, data->pointIds);
        data->projections.resize(optFlowMeas->observations.size());
        computeProjections(data->projections, lastStateTNs_);
        data->optFlowRes = prevOptFlowRes_[lastStateTNs_];
        outVisQueue->push(data);
    }
    lastProcessedTNs = lastStateTNs_;
    statsSums_.add("measure", tTotal.elapsed()).setFormat("ms");
    return true;
}
inline void SqrtKeypointVioEstimator::logMargNullspace() {
    nullspaceMargData_.order = margData_.order;
    if (config_.vioDebug) {
        std::cout << "======== Marg nullspace ==========" << std::endl;
        statsSums_.add("marg_ns", checkMargNullspace());
        std::cout << "=================================" << std::endl;
    } else {
        statsSums_.add("marg_ns", checkMargNullspace());
    }
    statsSums_.add("marg_ev", checkMargEigenvalues());
}
inline Eigen::VectorXd SqrtKeypointVioEstimator::checkMargNullspace() const {
    return checkNullspace(nullspaceMargData_, frameStates, framePoses, config_.vioDebug);
}
inline Eigen::VectorXd SqrtKeypointVioEstimator::checkMargEigenvalues() const {
    return checkEigenvalues(nullspaceMargData_, false);
}
inline void SqrtKeypointVioEstimator::marginalize(
    const std::map<int64_t, int>& numPointsConnected, const std::unordered_set<KeypointId>& lostLandmarks) {
    if (!optStarted_)
        return;
    Timer tTotal;
    if (framePoses.size() > maxKfs_ || frameStates.size() >= maxStates_) {
        const int statesToRemove = frameStates.size() - maxStates_ + 1;
        auto it = frameStates.cbegin();
        for (int i = 0; i < statesToRemove; ++ i)
            ++ it;
        int64_t lastStateToMarg = it->first;
        AbsOrderMap aom;
        std::set<int64_t> posesToMarg;
        for (const auto& kv : framePoses) {
            aom.absOrderMap[kv.first] = std::make_pair(aom.totalSize, 6);
            if (kfIds_.count(kv.first) == 0)
                posesToMarg.emplace(kv.first);
            ensure(margData_.order.absOrderMap.at(kv.first) == aom.absOrderMap.at(kv.first), "VIO check failed");
            aom.totalSize += 6;
            ++ aom.items;
        }
        std::set<int64_t> statesToMargVelBias;
        std::set<int64_t> statesToMargAll;
        for (const auto& kv : frameStates) {
            if (kv.first > lastStateToMarg)
                break;
            if (kv.first != lastStateToMarg) {
                if (kfIds_.count(kv.first) > 0) {
                    statesToMargVelBias.emplace(kv.first);
                } else {
                    statesToMargAll.emplace(kv.first);
                }
            }
            aom.absOrderMap[kv.first] = std::make_pair(aom.totalSize, 15);
            if (aom.items < margData_.order.absOrderMap.size())
                ensure(margData_.order.absOrderMap.at(kv.first) == aom.absOrderMap.at(kv.first), "VIO check failed");
            aom.totalSize += 15;
            ++ aom.items;
        }
        auto kfIdsAll = kfIds_;
        std::set<int64_t> kfsToMarg;
        while (kfIds_.size() > maxKfs_ && !statesToMargVelBias.empty()) {
            int64_t idToMarg = -1;
            if (kfIds_.size() > 2) {
                auto endMinus2 = std::prev(kfIds_.end(), 2);
                for (auto it = kfIds_.begin(); it != endMinus2; ++ it) {
                    if (numPointsConnected.count(*it) == 0 ||
                        (numPointsConnected.at(*it) / static_cast<float>(numPointsKf_.at(*it)) <
                            config_.vioKfMargFeatureRatio)) {
                        idToMarg = *it;
                        break;
                    }
                }
            }
            if (kfIds_.size() > 2 && idToMarg < 0) {
                auto endMinus2 = std::prev(kfIds_.end(), 2);
                int64_t lastKf = *kfIds_.crbegin();
                double minScore = std::numeric_limits<double>::max();
                int64_t minScoreId = -1;
                for (auto it1 = kfIds_.begin(); it1 != endMinus2; ++ it1) {
                    double denom = 0;
                    for (auto it2 = kfIds_.begin(); it2 != endMinus2; ++ it2) {
                        denom += 1 /
                            ((framePoses.at(*it1).getPose().translation() - framePoses.at(*it2).getPose().translation())
                                    .norm() +
                                double(1e-5));
                    }
                    double score = std::sqrt((framePoses.at(*it1).getPose().translation() -
                                       frameStates.at(lastKf).getState().tWI.translation())
                                                 .norm()) *
                        denom;
                    if (score < minScore) {
                        minScoreId = *it1;
                        minScore = score;
                    }
                }
                idToMarg = minScoreId;
            }
            ensure(idToMarg >= 0, "VIO check failed");
            kfsToMarg.emplace(idToMarg);
            posesToMarg.emplace(idToMarg);
            kfIds_.erase(idToMarg);
        }
        if (config_.vioDebug) {
            std::cout << "states_to_remove " << statesToRemove << std::endl;
            std::cout << "poses_to_marg.size() " << posesToMarg.size() << std::endl;
            std::cout << "states_to_marg.size() " << statesToMargAll.size() << std::endl;
            std::cout << "state_to_marg_vel_bias.size() " << statesToMargVelBias.size() << std::endl;
            std::cout << "kfs_to_marg.size() " << kfsToMarg.size() << std::endl;
        }
        Timer tActualMarg;
        size_t asize = aom.totalSize;
        bool isLinSqrt = isLinearizationSqrt(config_.vioLinearizationType);
        MatXd Q2Jp_or_H;
        VecXd Q2r_or_b;
        {
            Timer tLinearize;
            LinearizationBase::Options lqrOptions;
            lqrOptions.lbOptions.huberParameter = huberThresh;
            lqrOptions.lbOptions.obsStdDev = obsStdDev;
            lqrOptions.linearizationType = config_.vioLinearizationType;
            ImuLinData ild = {g_, gyroBiasSqrtWeight_, accelBiasSqrtWeight_, {}};
            for (const auto& kv : imuMeas_) {
                int64_t startT = kv.second.getStartTNs();
                int64_t endT = kv.second.getStartTNs() + kv.second.getDtNs();
                if (aom.absOrderMap.count(startT) == 0 || aom.absOrderMap.count(endT) == 0)
                    continue;
                ild.imuMeas[kv.first] = &kv.second;
            }
            auto lqr = LinearizationBase::create(
                this, aom, lqrOptions, &margData_, &ild, &kfsToMarg, &lostLandmarks, lastStateToMarg);
            lqr->linearizeProblem();
            lqr->performQR();
            if (isLinSqrt && margData_.isSqrt) {
                lqr->getDenseQ2JpQ2r(Q2Jp_or_H, Q2r_or_b);
            } else {
                lqr->getDenseHB(Q2Jp_or_H, Q2r_or_b);
            }
            statsSums_.add("marg_linearize", tLinearize.elapsed()).setFormat("ms");
        }
        if (outMargQueue && !kfsToMarg.empty()) {
            {
                MargData::Ptr m(new MargData);
                m->aom = aom;
                if (isLinSqrt && margData_.isSqrt) {
                    m->absH = (Q2Jp_or_H.transpose() * Q2Jp_or_H).cast<double>();
                    m->absB = (Q2Jp_or_H.transpose() * Q2r_or_b).cast<double>();
                } else {
                    m->absH = Q2Jp_or_H.cast<double>();
                    m->absB = Q2r_or_b.cast<double>();
                }
                m->framePoses = framePoses;
                m->frameStates = frameStates;
                m->kfsAll = kfIdsAll;
                m->kfsToMarg = kfsToMarg;
                m->useImu = true;
                for (int64_t t : m->kfsAll) {
                    m->optFlowRes.emplace_back(prevOptFlowRes_.at(t));
                }
                outMargQueue->push(m);
            }
        }
        std::set<int> idxToKeep, idxToMarg;
        for (const auto& kv : aom.absOrderMap) {
            if (kv.second.second == 6) {
                int startIdx = kv.second.first;
                if (posesToMarg.count(kv.first) == 0) {
                    for (size_t i = 0; i < 6; ++ i)
                        idxToKeep.emplace(startIdx + i);
                } else {
                    for (size_t i = 0; i < 6; ++ i)
                        idxToMarg.emplace(startIdx + i);
                }
            } else {
                ensure(kv.second.second == 15, "VIO check failed");
                int startIdx = kv.second.first;
                if (statesToMargAll.count(kv.first) > 0) {
                    for (size_t i = 0; i < 15; ++ i)
                        idxToMarg.emplace(startIdx + i);
                } else if (statesToMargVelBias.count(kv.first) > 0) {
                    for (size_t i = 0; i < 6; ++ i)
                        idxToKeep.emplace(startIdx + i);
                    for (size_t i = 6; i < 15; ++ i)
                        idxToMarg.emplace(startIdx + i);
                } else {
                    ensure(kv.first == lastStateToMarg, "VIO check failed");
                    for (size_t i = 0; i < 15; ++ i)
                        idxToKeep.emplace(startIdx + i);
                }
            }
        }
        if (config_.vioDebug) {
            std::cout << "keeping " << idxToKeep.size() << " marg " << idxToMarg.size() << " total " << asize
                      << std::endl;
            std::cout << "last_state_to_marg " << lastStateToMarg << " framePoses " << framePoses.size()
                      << " frameStates " << frameStates.size() << std::endl;
        }
        if (config_.vioDebug || config_.vioExtendedLogging) {
            MatXd Q2Jp_or_H_nullspace;
            VecXd Q2r_or_b_nullspace;
            LinearizationBase::Options lqrOptions;
            lqrOptions.lbOptions.huberParameter = huberThresh;
            lqrOptions.lbOptions.obsStdDev = obsStdDev;
            lqrOptions.linearizationType = config_.vioLinearizationType;
            nullspaceMargData_.order = margData_.order;
            ImuLinData ild = {g_, gyroBiasSqrtWeight_, accelBiasSqrtWeight_, {}};
            for (const auto& kv : imuMeas_) {
                int64_t startT = kv.second.getStartTNs();
                int64_t endT = kv.second.getStartTNs() + kv.second.getDtNs();
                if (aom.absOrderMap.count(startT) == 0 || aom.absOrderMap.count(endT) == 0)
                    continue;
                ild.imuMeas[kv.first] = &kv.second;
            }
            auto lqr = LinearizationBase::create(
                this, aom, lqrOptions, &nullspaceMargData_, &ild, &kfsToMarg, &lostLandmarks, lastStateToMarg);
            lqr->linearizeProblem();
            lqr->performQR();
            if (isLinSqrt && margData_.isSqrt) {
                lqr->getDenseQ2JpQ2r(Q2Jp_or_H_nullspace, Q2r_or_b_nullspace);
            } else {
                lqr->getDenseHB(Q2Jp_or_H_nullspace, Q2r_or_b_nullspace);
            }
            MatXd nullspace_sqrt_H_new;
            VecXd nullspaceSqrtBNew;
            if (isLinSqrt && margData_.isSqrt) {
                MargHelper::marginalizeHelperSqrtToSqrt(Q2Jp_or_H_nullspace,
                    Q2r_or_b_nullspace,
                    idxToKeep,
                    idxToMarg,
                    nullspace_sqrt_H_new,
                    nullspaceSqrtBNew);
            } else if (margData_.isSqrt) {
                MargHelper::marginalizeHelperSqToSqrt(Q2Jp_or_H_nullspace,
                    Q2r_or_b_nullspace,
                    idxToKeep,
                    idxToMarg,
                    nullspace_sqrt_H_new,
                    nullspaceSqrtBNew);
            } else {
                MargHelper::marginalizeHelperSqToSq(Q2Jp_or_H_nullspace,
                    Q2r_or_b_nullspace,
                    idxToKeep,
                    idxToMarg,
                    nullspace_sqrt_H_new,
                    nullspaceSqrtBNew);
            }
            nullspaceMargData_.h = nullspace_sqrt_H_new;
            nullspaceMargData_.b = nullspaceSqrtBNew;
        }
        MatXd marg_H_new;
        VecXd margBNew;
        {
            Timer t;
            if (isLinSqrt && margData_.isSqrt) {
                MargHelper::marginalizeHelperSqrtToSqrt(
                    Q2Jp_or_H, Q2r_or_b, idxToKeep, idxToMarg, marg_H_new, margBNew);
            } else if (margData_.isSqrt) {
                MargHelper::marginalizeHelperSqToSqrt(
                    Q2Jp_or_H, Q2r_or_b, idxToKeep, idxToMarg, marg_H_new, margBNew);
            } else {
                MargHelper::marginalizeHelperSqToSq(
                    Q2Jp_or_H, Q2r_or_b, idxToKeep, idxToMarg, marg_H_new, margBNew);
            }
            statsSums_.add("marg_helper", t.elapsed()).setFormat("ms");
        }
        {
            ensure(frameStates.at(lastStateToMarg).isLinearized() == false, "VIO check failed");
            frameStates.at(lastStateToMarg).setLinTrue();
        }
        for (const int64_t id : statesToMargAll) {
            frameStates.erase(id);
            imuMeas_.erase(id);
            prevOptFlowRes_.erase(id);
        }
        for (const int64_t id : statesToMargVelBias) {
            const PoseVelBiasStateWithLin& state = frameStates.at(id);
            PoseStateWithLin pose(state);
            framePoses[id] = pose;
            frameStates.erase(id);
            imuMeas_.erase(id);
        }
        for (const int64_t id : posesToMarg) {
            framePoses.erase(id);
            prevOptFlowRes_.erase(id);
        }
        lmdb.removeKeyframes(kfsToMarg, posesToMarg, statesToMargAll);
        if (config_.vioMargLostLandmarks) {
            for (const auto& lmId : lostLandmarks)
                lmdb.removeLandmark(lmId);
        }
        AbsOrderMap margOrderNew;
        for (const auto& kv : framePoses) {
            margOrderNew.absOrderMap[kv.first] = std::make_pair(margOrderNew.totalSize, 6);
            margOrderNew.totalSize += 6;
            ++ margOrderNew.items;
        }
        {
            margOrderNew.absOrderMap[lastStateToMarg] = std::make_pair(margOrderNew.totalSize, 15);
            margOrderNew.totalSize += 15;
            ++ margOrderNew.items;
        }
        margData_.h = marg_H_new;
        margData_.b = margBNew;
        margData_.order = margOrderNew;
        ensure(size_t(margData_.h.cols()) == margData_.order.totalSize, "VIO check failed");
        VecXd delta;
        computeDelta(margData_.order, delta);
        margData_.b -= margData_.h * delta;
        if (config_.vioDebug || config_.vioExtendedLogging) {
            VecXd delta;
            computeDelta(margData_.order, delta);
            nullspaceMargData_.b -= nullspaceMargData_.h * delta;
        }
        statsSums_.add("marg", tActualMarg.elapsed()).setFormat("ms");
        if (config_.vioDebug) {
            std::cout << "marginalizaon done!!" << std::endl;
        }
        if (config_.vioDebug || config_.vioExtendedLogging) {
            Timer t;
            logMargNullspace();
            statsSums_.add("marg_log", t.elapsed()).setFormat("ms");
        }
    }
    statsSums_.add("marginalize", tTotal.elapsed()).setFormat("ms");
}
inline void SqrtKeypointVioEstimator::optimize() {
    if (config_.vioDebug) {
        std::cout << "=================================" << std::endl;
    }
    if (optStarted_ || frameStates.size() > 4) {
        optStarted_ = true;
        ExecutionStats stats;
        Timer timerTotal;
        Timer timerIteration;
        AbsOrderMap aom;
        for (const auto& kv : framePoses) {
            aom.absOrderMap[kv.first] = std::make_pair(aom.totalSize, 6);
            ensure(margData_.order.absOrderMap.at(kv.first) == aom.absOrderMap.at(kv.first), "VIO check failed");
            aom.totalSize += 6;
            ++ aom.items;
        }
        for (const auto& kv : frameStates) {
            aom.absOrderMap[kv.first] = std::make_pair(aom.totalSize, 15);
            if (aom.items < margData_.order.absOrderMap.size())
                ensure(margData_.order.absOrderMap.at(kv.first) == aom.absOrderMap.at(kv.first), "VIO check failed");
            aom.totalSize += 15;
            ++ aom.items;
        }
        lambda_ = double(config_.vioLmLambdaInitial);
        stats.add("num_cams", this->framePoses.size()).setFormat("count");
        stats.add("num_lms", this->lmdb.numLandmarks()).setFormat("count");
        stats.add("num_obs", this->lmdb.numObservations()).setFormat("count");
        LinearizationBase::Options lqrOptions;
        lqrOptions.lbOptions.huberParameter = huberThresh;
        lqrOptions.lbOptions.obsStdDev = obsStdDev;
        lqrOptions.linearizationType = config_.vioLinearizationType;
        std::unique_ptr<LinearizationBase> lqr;
        ImuLinData ild = {g_, gyroBiasSqrtWeight_, accelBiasSqrtWeight_, {}};
        for (const auto& kv : imuMeas_) {
            ild.imuMeas[kv.first] = &kv.second;
        }
        {
            Timer t;
            lqr = LinearizationBase::create(this, aom, lqrOptions, &margData_, &ild);
            stats.add("allocateLMB", t.reset()).setFormat("ms");
            lqr->logProblemStats(stats);
        }
        bool terminated = false;
        bool converged = false;
        std::string message;
        int it = 0;
        int itRejected = 0;
        for (; it <= config_.vioMaxIterations && !terminated;) {
            if (it > 0) {
                timerIteration.reset();
            }
            double errorTotal = 0;
            VecXd Jp_column_norm2;
            {
                Timer t;
                bool numericallyValid;
                errorTotal = lqr->linearizeProblem(&numericallyValid);
                ensure(numericallyValid, "VIO check failed");
                stats.add("linearizeProblem", t.reset()).setFormat("ms");
                lqr->performQR();
                stats.add("performQR", t.reset()).setFormat("ms");
            }
            if (config_.vioDebug) {
                std::cout << "[LINEARIZE] Error: " << errorTotal << " num points " << std::endl;
                std::cout << "Iteration " << it << " " << errorTotal << std::endl;
            }
            for (int j = 0; it <= config_.vioMaxIterations && !terminated; ++ j) {
                if (j > 0) {
                    timerIteration.reset();
                    if (config_.vioDebug) {
                        std::cout << "Iteration " << it << ", backtracking" << std::endl;
                    }
                }
                {}
                VecXd inc;
                {
                    Timer t;
                    MatXd H;
                    VecXd b;
                    lqr->getDenseHB(H, b);
                    stats.add("getDenseHB", t.reset()).setFormat("ms");
                    int iter = 0;
                    bool incValid = false;
                    constexpr int MAX_NUM_ITER = 3;
                    while (iter < MAX_NUM_ITER && !incValid) {
                        VecXd Hdiag_lambda = (H.diagonal() * lambda_).cwiseMax(minLambda_);
                        MatXd H_copy = H;
                        H_copy.diagonal() += Hdiag_lambda;
                        Eigen::LDLT<Eigen::Ref<MatXd>> ldlt(H_copy);
                        inc = ldlt.solve(b);
                        stats.add("solve", t.reset()).setFormat("ms");
                        if (!inc.array().isFinite().all()) {
                            lambda_ = lambdaVee_ * lambda_;
                            lambdaVee_ *= VEE_FACTOR;
                        } else {
                            incValid = true;
                        }
                        ++ iter;
                    }
                    if (!incValid) {
                        std::cerr << "Still invalid inc after " << MAX_NUM_ITER << " iterations." << std::endl;
                    }
                }
                backup();
                double lDiff = 0;
                {
                    inc = -inc;
                    Timer t;
                    lDiff = lqr->backSubstitute(inc);
                    stats.add("backSubstitute", t.reset()).setFormat("ms");
                }
                for (auto& [frameId, state] : framePoses) {
                    int idx = aom.absOrderMap.at(frameId).first;
                    state.applyInc(inc.segment<6>(idx));
                }
                for (auto& [frameId, state] : frameStates) {
                    int idx = aom.absOrderMap.at(frameId).first;
                    state.applyInc(inc.segment<15>(idx));
                }
                double stepNorminf = inc.array().abs().maxCoeff();
                double afterUpdateMargPriorError = 0;
                double afterUpdateVisionAndInertialError = 0;
                {
                    Timer t;
                    computeError(afterUpdateVisionAndInertialError);
                    computeMargPriorError(margData_, afterUpdateMargPriorError);
                    double afterUpdateImuError = 0, afterBgError = 0, afterBaError = 0;
                    ScBundleAdjustmentBase::computeImuError(aom,
                        afterUpdateImuError,
                        afterBgError,
                        afterBaError,
                        frameStates,
                        imuMeas_,
                        gyroBiasSqrtWeight_.array().square(),
                        accelBiasSqrtWeight_.array().square(),
                        g_);
                    afterUpdateVisionAndInertialError += afterUpdateImuError + afterBgError + afterBaError;
                    stats.add("computerError2", t.reset()).setFormat("ms");
                }
                double afterErrorTotal = afterUpdateVisionAndInertialError + afterUpdateMargPriorError;
                double fDiff;
                bool stepIsValid = false;
                bool stepIsSuccessful = false;
                double relativeDecrease = 0;
                {
                    fDiff = errorTotal - afterErrorTotal;
                    relativeDecrease = fDiff / lDiff;
                    if (config_.vioDebug) {
                        std::cout << "\t[EVAL] error: " << afterErrorTotal << ", f_diff " << fDiff << " l_diff "
                                  << lDiff << " step_quality " << relativeDecrease << " step_size " << stepNorminf
                                  << "\n";
                    }
                    stepIsValid = lDiff > 0;
                    stepIsSuccessful = stepIsValid && relativeDecrease > 0;
                }
                double iterationTime = timerIteration.elapsed();
                double cumulativeTime = timerTotal.elapsed();
                stats.add("iteration", iterationTime).setFormat("ms");
                if (stepIsSuccessful) {
                    ensure(stepIsValid, "VIO check failed");
                    if (config_.vioDebug) {
                        std::cout << "\t[ACCEPTED] error: " << afterErrorTotal << ", lambda: " << lambda_
                                  << ", it_time: " << iterationTime << "s, total_time: " << cumulativeTime << "s\n";
                    }
                    lambda_ *= std::max<double>(double(1.0) / 3, 1 - std::pow<double>(2 * relativeDecrease - 1, 3));
                    lambda_ = std::max(minLambda_, lambda_);
                    lambdaVee_ = INITIAL_VEE;
                    ++ it;
                    if ((fDiff > 0 && fDiff < double(1e-6)) || stepNorminf < double(1e-4)) {
                        converged = true;
                        terminated = true;
                    }
                    break;
                } else {
                    std::string reason = stepIsValid ? "REJECTED" : "INVALID";
                    if (config_.vioDebug) {
                        std::cout << "\t[" << reason << "] error: " << afterErrorTotal << ", lambda: " << lambda_
                                  << ", it_time: " << iterationTime << "s, total_time: " << cumulativeTime << "s\n";
                    }
                    lambda_ = lambdaVee_ * lambda_;
                    lambdaVee_ *= VEE_FACTOR;
                    restore();
                    ++ it;
                    ++ itRejected;
                    if (lambda_ > maxLambda_) {
                        terminated = true;
                        message = "Solver did not converge and reached maximum damping lambda";
                    }
                }
            }
        }
        stats.add("optimize", timerTotal.elapsed()).setFormat("ms");
        stats.add("num_it", it).setFormat("count");
        stats.add("num_it_rejected", itRejected).setFormat("count");
        statsAll_.mergeAll(stats);
        statsSums_.mergeSums(stats);
        if (config_.vioDebug) {
            if (!converged) {
                if (terminated) {
                    std::cout << "Solver terminated early after " << it << " iterations: " << message << "\n";
                } else {
                    std::cout << "Solver did not converge after maximum number of " << it << " iterations\n";
                }
            }
            stats.print();
            std::cout << "=================================" << std::endl;
        }
    }
}
inline void SqrtKeypointVioEstimator::optimizeAndMarg(
    const std::map<int64_t, int>& numPointsConnected, const std::unordered_set<KeypointId>& lostLandmarks) {
    optimize();
    marginalize(numPointsConnected, lostLandmarks);
}
inline void SqrtKeypointVioEstimator::debugFinalize() {
    std::cout << "=== stats all ===\n";
    statsAll_.print();
    std::cout << "=== stats sums ===\n";
    statsSums_.print();
}

} // arc
