#pragma once

#include <pcl/point_types.h>

#include <pcl/point_cloud.h>

#include <base/ros.h>

#include <base/iarc.h>
#include <env/mapdef.h>
#include <env/esdf.h>
#include <env/raycaster.h>

namespace arc {

class SdfMap {
public:
    SdfMap(rclcpp::Node::SharedPtr node)
        : mp_(node)
        , md_(mp_)
        , rc_(mp_)
        , esdf_(mp_, md_)
    {}

public:
    MapParams& params() { return mp_; }
    const MapParams& params() const { return mp_; }
    MapData& data() { return md_; }
    const MapData& data() const { return md_; }
    void requestUpdatedBoxReset() {
        md_.resetUpdatedBox = true;
    }

public:
    bool needsESDFUpdate() const {
        return esdfNeedUpdate_;
    }
    void updateESDF() {
        if (!esdfNeedUpdate_) {
            return;
        }
        esdf_.update();
        esdfNeedUpdate_ = false;
    }
    double getDistWithGrad(const Eigen::Vector3d& pos, Eigen::Vector3d& grad) const {
        if (!mp_.isInMap(pos)) {
            grad.setZero();
            return 0.0;
        }
        const Eigen::Vector3d posM = pos - 0.5 * mp_.resolution * Eigen::Vector3d::Ones();
        Eigen::Vector3i idx = mp_.p2i(posM);
        const Eigen::Vector3i maxIdx = mp_.voxelShape - Eigen::Vector3i::Constant(2);
        idx = idx.cwiseMax(0).cwiseMin(maxIdx.cwiseMax(0));
        const Eigen::Vector3d idxPos = mp_.i2p(idx);
        const Eigen::Vector3d diff = (pos - idxPos) * mp_.resolutionInv;

        double values[2][2][2];
        for (int x = 0; x < 2; ++ x) {
            for (int y = 0; y < 2; ++ y) {
                for (int z = 0; z < 2; ++ z) {
                    const Eigen::Vector3i sampleIdx = idx + Eigen::Vector3i(x, y, z);
                    values[x][y][z] = md_.getDistance(sampleIdx);
                }
            }
        }
        const double v00 = (1.0 - diff(0)) * values[0][0][0] + diff(0) * values[1][0][0];
        const double v01 = (1.0 - diff(0)) * values[0][0][1] + diff(0) * values[1][0][1];
        const double v10 = (1.0 - diff(0)) * values[0][1][0] + diff(0) * values[1][1][0];
        const double v11 = (1.0 - diff(0)) * values[0][1][1] + diff(0) * values[1][1][1];
        const double v0 = (1.0 - diff(1)) * v00 + diff(1) * v10;
        const double v1 = (1.0 - diff(1)) * v01 + diff(1) * v11;
        const double dist = (1.0 - diff(2)) * v0 + diff(2) * v1;

        grad(2) = (v1 - v0) * mp_.resolutionInv;
        grad(1) = ((1.0 - diff(2)) * (v10 - v00) + diff(2) * (v11 - v01)) * mp_.resolutionInv;
        grad(0) = (1.0 - diff(2)) * (1.0 - diff(1)) * (values[1][0][0] - values[0][0][0]);
        grad(0) += (1.0 - diff(2)) * diff(1) * (values[1][1][0] - values[0][1][0]);
        grad(0) += diff(2) * (1.0 - diff(1)) * (values[1][0][1] - values[0][0][1]);
        grad(0) += diff(2) * diff(1) * (values[1][1][1] - values[0][1][1]);
        grad(0) *= mp_.resolutionInv;
        return dist;
    }
    bool isLineFree(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2) const {
        if (!mp_.isInMap(p1) || !mp_.isInMap(p2)) {
            return false;
        }
        RayCaster caster(mp_);
        if (!caster.input(p1, p2)) {
            return md_.getInflateOccupancy(p2) == VoxelInflateOccupancyCode::FREE;
        }
        Eigen::Vector3i idx;
        while (caster.nextId(idx)) {
            if (!mp_.isInMap(idx) ||
                md_.getInflateOccupancy(idx) != VoxelInflateOccupancyCode::FREE) {
                return false;
            }
        }
        return md_.getInflateOccupancy(p2) == VoxelInflateOccupancyCode::FREE;
    }
    Eigen::Vector3d findClosestPoint(const Eigen::Vector3d& pos, const Eigen::Vector3d& cameraPos) const {
        // this function assumes cameraPos is inside map
        Eigen::Vector3d ray2pos = pos - cameraPos;
        Eigen::Vector3d ray2ori = mp_.origin - cameraPos;
        Eigen::Vector3d ray2end = mp_.endpoint - cameraPos;
        double t = 1.0;
        for (int i = 0; i < 3; ++ i) {
            if (std::fabs(ray2pos[i]) > 1e-8) {
                double tEnd = ray2end[i] / ray2pos[i];
                if (tEnd > 0 && tEnd < t) {
                    t = tEnd;
                }
                double tOri = ray2ori[i] / ray2pos[i];
                if (tOri > 0 && tOri < t) {
                    t = tOri;
                }
            }
        }
        if (t < 1e-3) {
            return cameraPos;
        }
        return cameraPos + (t - 1e-3) * ray2pos;
    }
    void updateFramePoints(const pcl::PointCloud<pcl::PointXYZ>& points, size_t numPoints, const Eigen::Vector3d& cameraPos) {
        numPoints = std::min(numPoints, points.points.size());
        if (numPoints == 0) {
            return;
        }
        if (!mp_.isInMap(cameraPos)) {
            return;
        }
        raycastIndex_ += 1;
        if (raycastIndex_ == 0) {
            std::fill(md_.raycastIndexes.begin(), md_.raycastIndexes.end(), 0);
            raycastIndex_ = 1;
        }
        Eigen::Vector3i idx;
        if (md_.resetUpdatedBox) {
            md_.updateMin = cameraPos;
            md_.updateMax = cameraPos;
            md_.resetUpdatedBox = false;
        }
        for (size_t i = 0; i < numPoints; ++ i) {
            const auto& pt = points.points[i];
            Eigen::Vector3d pos(pt.x, pt.y, pt.z);
            uint8_t occ = VoxelOccupancyCode::OCCUPIED;
            if (!mp_.isInMap(pos)) {
                pos = findClosestPoint(pos, cameraPos);
                occ = VoxelOccupancyCode::FREE;
            }
            double length = (pos - cameraPos).norm();
            if (length > mp_.maxRayLength) {
                pos = (pos - cameraPos) / length * mp_.maxRayLength + cameraPos;
                occ = VoxelOccupancyCode::FREE;
            }
            Eigen::Vector3i posIdx = mp_.p2i(pos);
            size_t voxelIdx = mp_.getVoxelIdx(posIdx);
            md_.updateMin = md_.updateMin.cwiseMin(pos);
            md_.updateMax = md_.updateMax.cwiseMax(pos);
            md_.setFrameOccupancy(voxelIdx, occ, raycastIndex_);
            rc_.input(pos, cameraPos);
            rc_.nextId(idx);
            while (rc_.nextId(idx)) {
                size_t rayVoxelIdx = mp_.getVoxelIdx(idx);
                md_.setFrameOccupancy(rayVoxelIdx, VoxelOccupancyCode::FREE, raycastIndex_);
            }
        }
        if (initialized_) {
            md_.allMin = md_.allMin.cwiseMin(md_.updateMin);
            md_.allMax = md_.allMax.cwiseMax(md_.updateMax);
        } else {
            md_.allMin = md_.updateMin;
            md_.allMax = md_.updateMax;
            initialized_ = true;
        }

        const int boundStep = mp_.inflateStep() + 1;
        Eigen::Vector3i localMin = mp_.p2i(md_.updateMin) - Eigen::Vector3i::Constant(boundStep);
        Eigen::Vector3i localMax = mp_.p2i(md_.updateMax) + Eigen::Vector3i::Constant(boundStep);
        mp_.boundIndex(localMin);
        mp_.boundIndex(localMax);
        md_.localBoundMin = localMin;
        md_.localBoundMax = localMax;

        md_.mergeFrameOccupancy(raycastIndex_);
        md_.updateInflateOccupancy();
        esdfNeedUpdate_ = true;
    }
    std::vector<size_t> takeNewVoxelIndexes() {
        std::vector<size_t> indexes = std::move(md_.newVoxelIndexes);
        md_.newVoxelIndexes.clear();
        return indexes;
    }
    void applyExternalVoxels(const std::vector<size_t>& voxelIdxs, const std::vector<uint8_t>& occupancy) {
        const size_t count = std::min(voxelIdxs.size(), occupancy.size());
        if (count == 0) {
            return;
        }
        Eigen::Vector3d chunkMin = Eigen::Vector3d::Constant(std::numeric_limits<double>::max());
        Eigen::Vector3d chunkMax = Eigen::Vector3d::Constant(-std::numeric_limits<double>::max());
        Eigen::Vector3i indexMin = Eigen::Vector3i::Constant(std::numeric_limits<int>::max());
        Eigen::Vector3i indexMax = Eigen::Vector3i::Constant(std::numeric_limits<int>::min());
        bool updated = false;
        for (size_t i = 0; i < count; ++ i) {
            const size_t voxelIdx = voxelIdxs[i];
            if (voxelIdx >= md_.numVoxels) {
                continue;
            }
            md_.occupancyLogits[voxelIdx] = occupancy[i] ? mp_.lmax : mp_.lmin;
            md_.dirtyIndexes.push_back(voxelIdx);
            const Eigen::Vector3i index = mp_.voxelIdxToIndex(voxelIdx);
            const Eigen::Vector3d pos = mp_.i2p(index);
            indexMin = indexMin.cwiseMin(index);
            indexMax = indexMax.cwiseMax(index);
            chunkMin = chunkMin.cwiseMin(pos);
            chunkMax = chunkMax.cwiseMax(pos);
            updated = true;
        }
        if (!updated) {
            return;
        }
        md_.allMin = md_.allMin.cwiseMin(chunkMin);
        md_.allMax = md_.allMax.cwiseMax(chunkMax);
        const int boundStep = mp_.inflateStep() + 1;
        Eigen::Vector3i localMin = indexMin - Eigen::Vector3i::Constant(boundStep);
        Eigen::Vector3i localMax = indexMax + Eigen::Vector3i::Constant(boundStep);
        mp_.boundIndex(localMin);
        mp_.boundIndex(localMax);
        md_.localBoundMin = localMin;
        md_.localBoundMax = localMax;
        md_.updateInflateOccupancy();
        ++ md_.updateRevision;
    }
    void resetBuffer() {
        md_.reset();
        raycastIndex_ = 0;
        initialized_ = false;
        esdfNeedUpdate_ = false;
    }

private:
    MapParams mp_;
    MapData md_;
    RayCaster rc_;
    Esdf esdf_;

private:
    bool initialized_{false};
    bool esdfNeedUpdate_{false};
    uint8_t raycastIndex_{0};
};

} // arc
