#pragma once

#include <base/base.h>
#include <lio/lioconfig.h>
#include <lio/liotypes.h>
#include <base/manifold/math.h>
#include <lio/geometry.h>
#include <lio/filter.h>

namespace arc {

class ImuProcessor {
public:
    explicit ImuProcessor(const LioConfig& config)
        : lidarType_(config.lidarType)
        , lidarRwrtImu_(config.extrinsicR)
        , lidarTwrtImu_(config.extrinsicT)
        , covAccScale_(Vec3d::Constant(config.accCov))
        , covGyrScale_(Vec3d::Constant(config.gyrCov))
        , covBiasGyr_(Vec3d::Constant(config.biasGyrCov))
        , covBiasAcc_(Vec3d::Constant(config.biasAccCov))
    {
        Q_ = processNoiseCov();
        covAcc_ = Vec3d(0.1, 0.1, 0.1);
        covGyr_ = Vec3d(0.1, 0.1, 0.1);
        meanAcc_ = Vec3d(0.0, 0.0, -1.0);
        meanGyr_ = Vec3d::Zero();
        angvelLast_ = Vec3d::Zero();
        accSLast_ = Vec3d::Zero();
    }
    void reset() {
        meanAcc_ = Vec3d(0.0, 0.0, -1.0);
        meanGyr_ = Vec3d::Zero();
        angvelLast_ = Vec3d::Zero();
        accSLast_ = Vec3d::Zero();
        imuNeedInit_ = true;
        initIterNum_ = 1;
        lastImu_.reset();
        imuPoses_.clear();
        lastLidarEndTime_ = 0.0;
    }
    void process(const MeasurementGroup& meas, LioEkf& kf, LioCloud& undistorted) {
        if (meas.imu.empty()) {
            return;
        }
        ensure(meas.lidar != nullptr, "LIO measurement cloud is null");
        if (imuNeedInit_) {
            initialize_(meas, kf, initIterNum_);
            if (initIterNum_ > MAX_INI_COUNT) {
                imuNeedInit_ = false;
                covAcc_ = covAccScale_;
                covGyr_ = covGyrScale_;
                std::clog << "[lio] IMU initialization done" << std::endl;
            }
            return;
        }
        undistort_(meas, kf, undistorted);
    }

private:
    void initialize_(const MeasurementGroup& meas, LioEkf& kf, int& count) {
        Vec3d curAcc = Vec3d::Zero();
        Vec3d curGyr = Vec3d::Zero();
        if (bFirstFrame_) {
            reset();
            count = 1;
            bFirstFrame_ = false;
            const ImuSample& first = meas.imu.front();
            meanAcc_ = first.accel;
            meanGyr_ = first.gyro;
        }
        for (const ImuSample& imu : meas.imu) {
            curAcc = imu.accel;
            curGyr = imu.gyro;
            meanAcc_ += (curAcc - meanAcc_) / static_cast<double>(count);
            meanGyr_ += (curGyr - meanGyr_) / static_cast<double>(count);
            covAcc_ = covAcc_ * (count - 1.0) / count + (curAcc - meanAcc_).cwiseProduct(curAcc - meanAcc_) * (count - 1.0) / (static_cast<double>(count) * count);
            covGyr_ = covGyr_ * (count - 1.0) / count + (curGyr - meanGyr_).cwiseProduct(curGyr - meanGyr_) * (count - 1.0) / (static_cast<double>(count) * count);
            ++ count;
        }
        LioState initState = kf.state();
        initState.grav = LioS2(LioVect3(-meanAcc_ / meanAcc_.norm() * kGravity));
        initState.bg = LioVect3(meanGyr_);
        initState.offsetTLI = LioVect3(lidarTwrtImu_);
        initState.offsetRLI = LioSo3(lidarRwrtImu_);
        kf.setState(initState);

        LioEkf::Covariance initP = kf.covariance();
        initP.setIdentity();
        initP(6, 6) = initP(7, 7) = initP(8, 8) = 0.00001;
        initP(9, 9) = initP(10, 10) = initP(11, 11) = 0.00001;
        initP(15, 15) = initP(16, 16) = initP(17, 17) = 0.0001;
        initP(18, 18) = initP(19, 19) = initP(20, 20) = 0.001;
        initP(21, 21) = initP(22, 22) = 0.00001;
        kf.setCovariance(initP);

        lastImu_ = meas.imu.back();
    }
    void undistort_(const MeasurementGroup& meas, LioEkf& kf, LioCloud& undistorted) {
        std::deque<ImuSample> imu = meas.imu;
        if (lastImu_) {
            imu.push_front(*lastImu_);
        }
        if (imu.size() < 2) {
            return;
        }
        const double imuEndTime = imu.back().time;
        double pclBeginTime = meas.lidarBeginTime;
        double pclEndTime = meas.lidarEndTime;

        if (lidarType_ == LidarType::Sim) {
            pclBeginTime = lastLidarEndTime_;
            pclEndTime = meas.lidarBeginTime;
        }
        undistorted = *meas.lidar;
        std::sort(
            undistorted.points.begin(),
            undistorted.points.end(),
            [](const LioPoint& lhs, const LioPoint& rhs) {
                return lhs.time < rhs.time;
            }
        );
        LioState imuState = kf.state();
        imuPoses_.clear();
        imuPoses_.push_back(setPose6D(
            0.0,
            accSLast_,
            angvelLast_,
            imuState.vel,
            imuState.pos,
            imuState.rot.toRotationMatrix())
        );
        Vec3d angvelAvg = Vec3d::Zero();
        Vec3d accAvg = Vec3d::Zero();
        Vec3d accImu = Vec3d::Zero();
        Vec3d velImu = Vec3d::Zero();
        Vec3d posImu = Vec3d::Zero();
        Mat3d rotImu = Mat3d::Identity();
        double dt = 0.0;

        LioInput input;
        for (auto it = imu.begin(); it < imu.end() - 1; ++ it) {
            const ImuSample& head = *it;
            const ImuSample& tail = *(it + 1);

            if (tail.time < lastLidarEndTime_) {
                continue;
            }
            angvelAvg = 0.5 * (head.gyro + tail.gyro);
            accAvg = 0.5 * (head.accel + tail.accel);
            accAvg = accAvg * kGravity / meanAcc_.norm();

            if (head.time < lastLidarEndTime_) {
                dt = tail.time - lastLidarEndTime_;
            } else {
                dt = tail.time - head.time;
            }
            input.acc = LioVect3(accAvg);
            input.gyro = LioVect3(angvelAvg);
            Q_.block<3, 3>(0, 0).diagonal() = covGyr_;
            Q_.block<3, 3>(3, 3).diagonal() = covAcc_;
            Q_.block<3, 3>(6, 6).diagonal() = covBiasGyr_;
            Q_.block<3, 3>(9, 9).diagonal() = covBiasAcc_;
            kf.predict(dt, Q_, input);

            imuState = kf.state();
            angvelLast_ = angvelAvg - imuState.bg;
            accSLast_ = imuState.rot * (accAvg - imuState.ba);
            for (int i = 0; i < 3; ++ i) {
                accSLast_[i] += imuState.grav[i];
            }
            const double offsetTime = tail.time - pclBeginTime;
            imuPoses_.push_back(setPose6D(
                offsetTime,
                accSLast_,
                angvelLast_,
                imuState.vel,
                imuState.pos,
                imuState.rot.toRotationMatrix()
            ));
        }
        const double note = pclEndTime > imuEndTime ? 1.0 : -1.0;
        dt = note * (pclEndTime - imuEndTime);
        kf.predict(dt, Q_, input);

        imuState = kf.state();
        lastImu_ = meas.imu.back();
        lastLidarEndTime_ = pclEndTime;

        if (undistorted.points.empty() || lidarType_ == LidarType::Sim) {
            return;
        }
        auto pointIt = undistorted.points.end() - 1;
        for (auto poseIt = imuPoses_.end() - 1; poseIt != imuPoses_.begin(); -- poseIt) {
            const ImuPose& head = *(poseIt - 1);
            const ImuPose& tail = *poseIt;
            rotImu = head.rot;
            velImu = head.vel;
            posImu = head.pos;
            accImu = tail.acc;
            angvelAvg = tail.gyr;

            while (pointIt->time > head.offsetTime) {
                dt = pointIt->time - head.offsetTime;
                const Mat3d rotI = rotImu * manifold::so3Exp(angvelAvg, dt);
                const Vec3d pointI(pointIt->x, pointIt->y, pointIt->z);
                const Vec3d transEI = posImu + velImu * dt + 0.5 * accImu * dt * dt - imuState.pos;
                const Vec3d compensated = imuState.offsetRLI.conjugate() * (imuState.rot.conjugate() * (rotI * (imuState.offsetRLI * pointI + imuState.offsetTLI) + transEI) - imuState.offsetTLI);
                pointIt->x = compensated.x();
                pointIt->y = compensated.y();
                pointIt->z = compensated.z();
                if (pointIt == undistorted.points.begin()) {
                    break;
                }
                -- pointIt;
            }
        }
    }

private:
    static constexpr int MAX_INI_COUNT = 10;
    LidarType lidarType_{LidarType::Generic};
    Mat3d lidarRwrtImu_{Mat3d::Identity()};
    Vec3d lidarTwrtImu_{Vec3d::Zero()};

    LioEkf::ProcessNoiseCovariance Q_;
    Vec3d covAcc_{Vec3d::Constant(0.1)};
    Vec3d covGyr_{Vec3d::Constant(0.1)};
    Vec3d covAccScale_{Vec3d::Constant(0.1)};
    Vec3d covGyrScale_{Vec3d::Constant(0.1)};
    Vec3d covBiasGyr_{Vec3d::Constant(0.0001)};
    Vec3d covBiasAcc_{Vec3d::Constant(0.0001)};

    double lastLidarEndTime_{0.0};
    int initIterNum_{1};
    bool bFirstFrame_{true};
    bool imuNeedInit_{true};

    Vec3d meanAcc_{Vec3d(0.0, 0.0, -1.0)};
    Vec3d meanGyr_{Vec3d::Zero()};
    Vec3d angvelLast_{Vec3d::Zero()};
    Vec3d accSLast_{Vec3d::Zero()};
    std::optional<ImuSample> lastImu_;
    std::vector<ImuPose> imuPoses_;
};

} // arc
