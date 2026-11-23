#pragma once

#include <base/base.h>
#include <sophus/se2.hpp>
#include <tbb/blocked_range.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/parallel_for.h>
#include <vio/opticalflow/base.h>
#include <vio/opticalflow/patch.h>
#include <vio/image/pyramid.h>
#include <vio/opticalflow/detector.h>

namespace arc {

template <typename Pattern>
class FrameToFrameOpticalFlow : public OpticalFlowBase {
public:
    using PatchT = OpticalFlowPatch<Pattern>;

    FrameToFrameOpticalFlow(const VioConfig& config, const arc::Calibration<double>& calib)
        : tNs_(-1)
        , frameCounter_(0)
        , lastKeypointId_(0)
        , config_(config)
    {
        inputQueue.set_capacity(10);
        this->calib_ = calib.cast<Scalar>();
        patchCoord = PatchT::PATTERN2.template cast<float>();
        if (calib.intrinsics.size() > 1) {
            Eigen::Matrix4d Ed;
            Sophus::SE3d T_i_j = calib.tIC[0].inverse() * calib.tIC[1];
            computeEssential(T_i_j, Ed);
            e_ = Ed.cast<Scalar>();
        }
        processingThread_.reset(new std::thread(&FrameToFrameOpticalFlow::processingLoop, this));
    }
    ~FrameToFrameOpticalFlow() {
        processingThread_->join();
    }
    void processingLoop() {
        OpticalFlowInput::Ptr inputPtr;
        while (true) {
            inputQueue.pop(inputPtr);
            if (!inputPtr.get()) {
                if (outputQueue)
                    outputQueue->push(nullptr);
                break;
            }
            processFrame(inputPtr->tNs, inputPtr);
        }
    }
    void processFrame(int64_t currTNs, OpticalFlowInput::Ptr& newImgVec) {
        for (const auto& v : newImgVec->imgData) {
            if (!v.img.get())
                return;
        }
        if (tNs_ < 0) {
            tNs_ = currTNs;
            transforms_.reset(new OpticalFlowResult);
            transforms_->observations.resize(calib_.intrinsics.size());
            transforms_->tNs = tNs_;
            pyramid_.reset(new std::vector<arc::ManagedImagePyr<uint16_t>>);
            pyramid_->resize(calib_.intrinsics.size());
            tbb::parallel_for(
                tbb::blocked_range<size_t>(0, calib_.intrinsics.size()), [&](const tbb::blocked_range<size_t>& r) {
                    for (size_t i = r.begin(); i != r.end(); ++ i) {
                        pyramid_->at(i).setFromImage(*newImgVec->imgData[i].img, config_.opticalFlowLevels);
                    }
                });
            transforms_->inputImages = newImgVec;
            addPoints();
            filterPoints();
        } else {
            tNs_ = currTNs;
            oldPyramid_ = pyramid_;
            pyramid_.reset(new std::vector<arc::ManagedImagePyr<uint16_t>>);
            pyramid_->resize(calib_.intrinsics.size());
            tbb::parallel_for(
                tbb::blocked_range<size_t>(0, calib_.intrinsics.size()), [&](const tbb::blocked_range<size_t>& r) {
                    for (size_t i = r.begin(); i != r.end(); ++ i) {
                        pyramid_->at(i).setFromImage(*newImgVec->imgData[i].img, config_.opticalFlowLevels);
                    }
                });
            OpticalFlowResult::Ptr newTransforms;
            newTransforms.reset(new OpticalFlowResult);
            newTransforms->observations.resize(calib_.intrinsics.size());
            newTransforms->tNs = tNs_;
            for (size_t i = 0; i < calib_.intrinsics.size(); ++ i) {
                trackPoints(
                    oldPyramid_->at(i), pyramid_->at(i), transforms_->observations[i], newTransforms->observations[i]);
            }
            transforms_ = newTransforms;
            transforms_->inputImages = newImgVec;
            addPoints();
            filterPoints();
        }
        if (outputQueue && frameCounter_ % config_.opticalFlowSkipFrames == 0) {
            outputQueue->push(transforms_);
        }
        ++ frameCounter_;
    }
    void trackPoints(
        const arc::ManagedImagePyr<uint16_t>& pyr1,
        const arc::ManagedImagePyr<uint16_t>& pyr2,
        const std::map<KeypointId, Eigen::AffineCompact2f>& transformMap1,
        std::map<KeypointId, Eigen::AffineCompact2f>& transformMap2
    ) const {
        size_t numPoints = transformMap1.size();
        std::vector<KeypointId> ids;
        std::vector<Eigen::AffineCompact2f> initVec;
        ids.reserve(numPoints);
        initVec.reserve(numPoints);
        for (const auto& kv : transformMap1) {
            ids.push_back(kv.first);
            initVec.push_back(kv.second);
        }
        tbb::concurrent_unordered_map<KeypointId,
            Eigen::AffineCompact2f,
            std::hash<KeypointId>,
            std::equal_to<KeypointId>>
            result;
        auto computeFunc = [&](const tbb::blocked_range<size_t>& range) {
            for (size_t r = range.begin(); r != range.end(); ++ r) {
                const KeypointId id = ids[r];
                const Eigen::AffineCompact2f& transform1 = initVec[r];
                Eigen::AffineCompact2f transform2 = transform1;
                bool valid = trackPoint(pyr1, pyr2, transform1, transform2);
                if (valid) {
                    Eigen::AffineCompact2f transform1Recovered = transform2;
                    valid = trackPoint(pyr2, pyr1, transform2, transform1Recovered);
                    if (valid) {
                        Scalar dist2 = (transform1.translation() - transform1Recovered.translation()).squaredNorm();
                        if (dist2 < config_.opticalFlowMaxRecoveredDist2) {
                            result[id] = transform2;
                        }
                    }
                }
            }
        };
        tbb::blocked_range<size_t> range(0, numPoints);
        tbb::parallel_for(range, computeFunc);
        transformMap2.clear();
        transformMap2.insert(result.begin(), result.end());
    }
    bool trackPoint(
        const arc::ManagedImagePyr<uint16_t>& oldPyr,
        const arc::ManagedImagePyr<uint16_t>& pyr,
        const Eigen::AffineCompact2f& oldTransform,
        Eigen::AffineCompact2f& transform
    ) const {
        bool patchValid = true;
        transform.linear().setIdentity();
        for (int level = config_.opticalFlowLevels; level >= 0 && patchValid; level--) {
            const Scalar scale = 1 << level;
            transform.translation() /= scale;
            PatchT p(oldPyr.lvl(level), oldTransform.translation() / scale);
            patchValid &= p.valid;
            if (patchValid) {
                patchValid &= trackPointAtLevel(pyr.lvl(level), p, transform);
            }
            transform.translation() *= scale;
        }
        transform.linear() = oldTransform.linear() * transform.linear();
        return patchValid;
    }
    bool trackPointAtLevel(const Image<const uint16_t>& img2, const PatchT& dp, Eigen::AffineCompact2f& transform) const {
        bool patchValid = true;
        for (int iteration = 0; patchValid && iteration < config_.opticalFlowMaxIterations; ++ iteration) {
            typename PatchT::VectorP res;
            typename PatchT::Matrix2P transformedPat = transform.linear().matrix() * PatchT::PATTERN2;
            transformedPat.colwise() += transform.translation();
            patchValid &= dp.residual(img2, transformedPat, res);
            if (patchValid) {
                const Vector3 inc = -dp.hSe2InvJSe2T * res;
                patchValid &= inc.array().isFinite().all();
                patchValid &= inc.lpNorm<Eigen::Infinity>() < 1e6;
                if (patchValid) {
                    transform *= SE2::exp(inc).matrix();
                    const int filterMargin = 2;
                    patchValid &= img2.inBounds(transform.translation(), filterMargin);
                }
            }
        }
        return patchValid;
    }
    void addPoints() {
        std::vector<Eigen::Vector2d> pts0;
        for (const auto& kv : transforms_->observations.at(0)) {
            pts0.emplace_back(kv.second.translation().cast<double>());
        }
        KeypointsData kd;
        detectKeypoints(pyramid_->at(0).lvl(0), kd, config_.opticalFlowDetectionGridSize, 1, pts0);
        std::map<KeypointId, Eigen::AffineCompact2f> newPoses0, newPoses1;
        for (size_t i = 0; i < kd.corners.size(); ++ i) {
            Eigen::AffineCompact2f transform;
            transform.setIdentity();
            transform.translation() = kd.corners[i].cast<Scalar>();
            transforms_->observations.at(0)[lastKeypointId_] = transform;
            newPoses0[lastKeypointId_] = transform;
            ++ lastKeypointId_;
        }
        if (calib_.intrinsics.size() > 1) {
            trackPoints(pyramid_->at(0), pyramid_->at(1), newPoses0, newPoses1);
            for (const auto& kv : newPoses1) {
                transforms_->observations.at(1).emplace(kv);
            }
        }
    }
    void filterPoints() {
        if (calib_.intrinsics.size() < 2) {
            return;
        }
        std::set<KeypointId> lmToRemove;
        std::vector<KeypointId> kpid;
        std::vector<Eigen::Vector2f> proj0, proj1;
        for (const auto& kv : transforms_->observations.at(1)) {
            auto it = transforms_->observations.at(0).find(kv.first);
            if (it != transforms_->observations.at(0).end()) {
                proj0.emplace_back(it->second.translation());
                proj1.emplace_back(kv.second.translation());
                kpid.emplace_back(kv.first);
            }
        }
        std::vector<Eigen::Vector4f> p3d0, p3d1;
        std::vector<bool> p3d0Success, p3d1Success;
        calib_.intrinsics[0].unproject(proj0, p3d0, p3d0Success);
        calib_.intrinsics[1].unproject(proj1, p3d1, p3d1Success);
        for (size_t i = 0; i < p3d0Success.size(); ++ i) {
            if (p3d0Success[i] && p3d1Success[i]) {
                const double epipolarError = std::abs(p3d0[i].transpose() * e_ * p3d1[i]);
                if (epipolarError > config_.opticalFlowEpipolarError) {
                    lmToRemove.emplace(kpid[i]);
                }
            } else {
                lmToRemove.emplace(kpid[i]);
            }
        }
        for (int id : lmToRemove) {
            transforms_->observations.at(1).erase(id);
        }
    }

private:
    int64_t tNs_;
    size_t frameCounter_;
    KeypointId lastKeypointId_;
    VioConfig config_;
    arc::Calibration<Scalar> calib_;
    OpticalFlowResult::Ptr transforms_;
    std::shared_ptr<std::vector<arc::ManagedImagePyr<uint16_t>>> oldPyramid_, pyramid_;
    Matrix4 e_;
    std::shared_ptr<std::thread> processingThread_;
};

} // arc
