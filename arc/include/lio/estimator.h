#pragma once

#include <base/base.h>
#include <lio/imuprocessor.h>
#include <lio/lioconfig.h>
#include <lio/liotypes.h>
#include <lio/mapmanager.h>
#include <base/manifold/math.h>
#include <lio/geometry.h>
#include <lio/ekf.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/impl/filter.hpp>
#include <pcl/filters/impl/voxel_grid.hpp>
#include <pcl/impl/pcl_base.hpp>

#include <omp.h>

namespace arc {

class LioEstimator {
public:
    explicit LioEstimator(const LioConfig& config)
        : config_(config)
        , imuProcessor_(config)
        , map_(config)
        , downsampleSurf_()
    {
        downsampleSurf_.setLeafSize(
            static_cast<float>(config_.filterSizeSurf),
            static_cast<float>(config_.filterSizeSurf),
            static_cast<float>(config_.filterSizeSurf)
        );
        initEkf_();
        state_ = kf_.state();
    }
    bool process(const MeasurementGroup& meas) {
        if (flgFirstScan_) {
            firstLidarTime_ = meas.lidarBeginTime;
            flgFirstScan_ = false;
            return false;
        }
        imuProcessor_.process(meas, kf_, *featsUndistort_);
        state_ = kf_.state();
        posLid_ = state_.pos + state_.rot * state_.offsetTLI;

        if (featsUndistort_->empty()) {
            std::cerr << "[lio] No point, skip this scan" << std::endl;
            return false;
        }
        flgEKFInited_ = (meas.lidarBeginTime - firstLidarTime_) >= INIT_TIME;
        map_.segmentFov(posLid_);

        downsampleSurf_.setInputCloud(featsUndistort_);
        downsampleSurf_.filter(*featsDownBody_);
        featsDownSize_ = static_cast<int>(featsDownBody_->size());

        if (map_.empty()) {
            if (featsDownSize_ > 5) {
                map_.setDownsampleParam(config_.filterSizeMap);
                featsDownWorld_->resize(static_cast<size_t>(featsDownSize_));
                for (int i = 0; i < featsDownSize_; ++ i) {
                    featsDownWorld_->points[static_cast<size_t>(i)] =
                        bodyToWorld_(featsDownBody_->points[static_cast<size_t>(i)]);
                }
                map_.build(featsDownWorld_->points);
            }
            return false;
        }
        if (featsDownSize_ < 5) {
            std::cerr << "[lio] No point, skip this scan" << std::endl;
            return false;
        }
        normvec_->resize(static_cast<size_t>(featsDownSize_));
        featsDownWorld_->resize(static_cast<size_t>(featsDownSize_));
        nearestPoints_.resize(static_cast<size_t>(featsDownSize_));
        pointSelectedSurf_.resize(static_cast<size_t>(featsDownSize_), 0);
        resLast_.resize(static_cast<size_t>(featsDownSize_), 0.0f);

        kf_.update(LASER_POINT_COV);
        state_ = kf_.state();
        posLid_ = state_.pos + state_.rot * state_.offsetTLI;

        mapIncremental_();
        return true;
    }
    const LioState& state() const {
        return state_;
    }
    MatXd covariance() const {
        return kf_.covariance();
    }
    Vec3d position() const {
        return state_.pos;
    }
    Mat3d rotation() const {
        return state_.rot.toRotationMatrix();
    }
    Vec3d velocity() const {
        return state_.vel;
    }
    LioCloud::Ptr registeredCloud() const {
        auto cloud = std::make_shared<LioCloud>();
        cloud->resize(featsUndistort_->size());
        const Mat3d rotWL = state_.rot.toRotationMatrix() * state_.offsetRLI.toRotationMatrix();
        const Vec3d transWL = state_.rot * state_.offsetTLI + state_.pos;
        for (size_t i = 0; i < featsUndistort_->size(); ++ i) {
            const LioPoint& point = featsUndistort_->points[i];
            const Vec3d world = rotWL * Vec3d(point.x, point.y, point.z) + transWL;
            cloud->points[i].x = static_cast<float>(world.x());
            cloud->points[i].y = static_cast<float>(world.y());
            cloud->points[i].z = static_cast<float>(world.z());
            cloud->points[i].intensity = point.intensity;
        }
        return cloud;
    }

private:
    void initEkf_() {
        kf_.initProcessModel(lioGetF, lioDfDx, lioDfDw);
        kf_.initMeasurementModel(
            [this](LioState& state, LioEkf::MeasurementData& data) {
                measurementModel_(state, data);
            },
            config_.maxIterations);
    }

    void measurementModel_(LioState& state, LioEkf::MeasurementData& ekfomData) {
        laserCloudOri_->clear();
        corrNormvec_->clear();

#ifdef _OPENMP
#pragma omp parallel
#endif
        {
            std::vector<float> pointSearchSqDis(kNumMatchPoints);
#ifdef _OPENMP
#pragma omp for
#endif
            for (int i = 0; i < featsDownSize_; ++ i) {
                LioPoint& pointBody = featsDownBody_->points[static_cast<size_t>(i)];
                LioPoint& pointWorld = featsDownWorld_->points[static_cast<size_t>(i)];

                const Vec3d pBody(pointBody.x, pointBody.y, pointBody.z);
                const Vec3d pGlobal = state.rot * (state.offsetRLI * pBody + state.offsetTLI) + state.pos;
                pointWorld.x = static_cast<float>(pGlobal.x());
                pointWorld.y = static_cast<float>(pGlobal.y());
                pointWorld.z = static_cast<float>(pGlobal.z());
                pointWorld.intensity = pointBody.intensity;

                LioPointVector& pointsNear = nearestPoints_[static_cast<size_t>(i)];
                pointsNear.clear();

                if (ekfomData.converge) {
                    map_.nearestSearch(pointWorld, kNumMatchPoints, pointsNear, pointSearchSqDis);
                    pointSelectedSurf_[static_cast<size_t>(i)] = pointsNear.size() < kNumMatchPoints
                        ? false
                        : pointSearchSqDis[kNumMatchPoints - 1] > 5.0f ? false : true;
                }

                if (!pointSelectedSurf_[static_cast<size_t>(i)]) {
                    continue;
                }

                Eigen::Matrix<float, 4, 1> plane;
                pointSelectedSurf_[static_cast<size_t>(i)] = false;
                if (estiPlane(plane, pointsNear, 0.1f)) {
                    const float pd2 = plane(0) * pointWorld.x
                        + plane(1) * pointWorld.y
                        + plane(2) * pointWorld.z
                        + plane(3);
                    const float score = 1.0f - 0.9f * std::fabs(pd2) / std::sqrt(pBody.norm());
                    if (score > 0.9f) {
                        pointSelectedSurf_[static_cast<size_t>(i)] = true;
                        LioPoint& normal = normvec_->points[static_cast<size_t>(i)];
                        normal.x = plane(0);
                        normal.y = plane(1);
                        normal.z = plane(2);
                        normal.intensity = pd2;
                        resLast_[static_cast<size_t>(i)] = std::fabs(pd2);
                    }
                }
            }
        }

        effctFeatNum_ = 0;
        laserCloudOri_->resize(static_cast<size_t>(featsDownSize_));
        corrNormvec_->resize(static_cast<size_t>(featsDownSize_));
        for (int i = 0; i < featsDownSize_; ++ i) {
            if (!pointSelectedSurf_[static_cast<size_t>(i)]) {
                continue;
            }
            laserCloudOri_->points[static_cast<size_t>(effctFeatNum_)] =
                featsDownBody_->points[static_cast<size_t>(i)];
            corrNormvec_->points[static_cast<size_t>(effctFeatNum_)] =
                normvec_->points[static_cast<size_t>(i)];
            ++ effctFeatNum_;
        }

        if (effctFeatNum_ < 1) {
            ekfomData.valid = false;
            std::cerr << "[lio] No effective points" << std::endl;
            return;
        }

        ekfomData.hX = Eigen::MatrixXd::Zero(effctFeatNum_, 12);
        ekfomData.h.resize(effctFeatNum_);

        for (int i = 0; i < effctFeatNum_; ++ i) {
            const LioPoint& laserPoint = laserCloudOri_->points[static_cast<size_t>(i)];
            const Vec3d pointThisBe(laserPoint.x, laserPoint.y, laserPoint.z);
            const Mat3d pointBeCross = manifold::hat(pointThisBe);
            const Vec3d pointThis = state.offsetRLI * pointThisBe + state.offsetTLI;
            const Mat3d pointCross = manifold::hat(pointThis);

            const LioPoint& normPoint = corrNormvec_->points[static_cast<size_t>(i)];
            const Vec3d normVec(normPoint.x, normPoint.y, normPoint.z);
            const Vec3d c = state.rot.conjugate() * normVec;
            const Vec3d a = pointCross * c;

            if (config_.extrinsicEstimate) {
                const Vec3d b = pointBeCross * state.offsetRLI.conjugate().toRotationMatrix() * c;
                ekfomData.hX.block<1, 12>(i, 0) << normPoint.x, normPoint.y, normPoint.z,
                    a(0), a(1), a(2),
                    b(0), b(1), b(2),
                    c(0), c(1), c(2);
            } else {
                ekfomData.hX.block<1, 12>(i, 0) << normPoint.x, normPoint.y, normPoint.z,
                    a(0), a(1), a(2),
                    0.0, 0.0, 0.0,
                    c(0), c(1), c(2);
            }
            ekfomData.h(i) = -normPoint.intensity;
        }
    }

    void mapIncremental_() {
        LioPointVector pointToAdd;
        LioPointVector pointNoNeedDownsample;
        pointToAdd.reserve(static_cast<size_t>(featsDownSize_));
        pointNoNeedDownsample.reserve(static_cast<size_t>(featsDownSize_));

        for (int i = 0; i < featsDownSize_; ++ i) {
            const LioPoint& world = featsDownWorld_->points[static_cast<size_t>(i)];
            const LioPointVector& pointsNear = nearestPoints_[static_cast<size_t>(i)];

            if (!pointsNear.empty() && flgEKFInited_) {
                bool needAdd = true;
                LioPoint midPoint;
                midPoint.x = std::floor(world.x / config_.filterSizeMap) * config_.filterSizeMap
                    + 0.5f * config_.filterSizeMap;
                midPoint.y = std::floor(world.y / config_.filterSizeMap) * config_.filterSizeMap
                    + 0.5f * config_.filterSizeMap;
                midPoint.z = std::floor(world.z / config_.filterSizeMap) * config_.filterSizeMap
                    + 0.5f * config_.filterSizeMap;
                const float dist = calcDist(world, midPoint);

                if (std::fabs(pointsNear[0].x - midPoint.x) > 0.5f * config_.filterSizeMap
                    && std::fabs(pointsNear[0].y - midPoint.y) > 0.5f * config_.filterSizeMap
                    && std::fabs(pointsNear[0].z - midPoint.z) > 0.5f * config_.filterSizeMap) {
                    pointNoNeedDownsample.push_back(world);
                    continue;
                }

                for (int readd = 0; readd < kNumMatchPoints; ++ readd) {
                    if (pointsNear.size() < static_cast<size_t>(kNumMatchPoints)) {
                        break;
                    }
                    if (calcDist(pointsNear[static_cast<size_t>(readd)], midPoint) < dist) {
                        needAdd = false;
                        break;
                    }
                }
                if (needAdd) {
                    pointToAdd.push_back(world);
                }
            } else {
                pointToAdd.push_back(world);
            }
        }

        map_.addPoints(pointToAdd, true);
        map_.addPoints(pointNoNeedDownsample, false);
    }

    Vec3d bodyToWorld_(const Vec3d& point) const {
        return state_.rot * (state_.offsetRLI * point + state_.offsetTLI) + state_.pos;
    }

    LioPoint bodyToWorld_(const LioPoint& point) const {
        const Vec3d world = bodyToWorld_(Vec3d(point.x, point.y, point.z));
        LioPoint output;
        output.x = static_cast<float>(world.x());
        output.y = static_cast<float>(world.y());
        output.z = static_cast<float>(world.z());
        output.intensity = point.intensity;
        return output;
    }

private:
    static constexpr double INIT_TIME = 0.1;
    static constexpr double LASER_POINT_COV = 0.001;

    LioConfig config_;
    ImuProcessor imuProcessor_;
    IncrementalMap map_;
    pcl::VoxelGrid<LioPoint> downsampleSurf_;
    LioEkf kf_;
    LioState state_;

    LioCloud::Ptr featsUndistort_{std::make_shared<LioCloud>()};
    LioCloud::Ptr featsDownBody_{std::make_shared<LioCloud>()};
    LioCloud::Ptr featsDownWorld_{std::make_shared<LioCloud>()};
    LioCloud::Ptr normvec_{std::make_shared<LioCloud>()};
    LioCloud::Ptr laserCloudOri_{std::make_shared<LioCloud>()};
    LioCloud::Ptr corrNormvec_{std::make_shared<LioCloud>()};

    std::vector<LioPointVector> nearestPoints_;
    std::vector<uint8_t> pointSelectedSurf_;
    std::vector<float> resLast_;

    int featsDownSize_{0};
    int effctFeatNum_{0};
    double firstLidarTime_{0.0};
    Vec3d posLid_{Vec3d::Zero()};
    bool flgFirstScan_{true};
    bool flgEKFInited_{false};
};

} // arc
