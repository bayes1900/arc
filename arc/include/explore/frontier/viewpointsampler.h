#pragma once

#include <base/iarc.h>
#include <env/raycaster.h>
#include <env/sdfmap.h>
#include <explore/frontier/frontierdef.h>
#include <explore/frontier/fovmodel.h>

namespace arc {

class ViewpointSampler {
public:
    ViewpointSampler(SdfMap& map, const FrontierParams& params)
        : mp_(map.params())
        , md_(map.data())
        , params_(params)
        , raycaster_(map.params())
        , fov_()
    {}
    void sampleViewpoints(Frontier& frontier) {
        frontier.viewpoints.clear();
        if (frontier.downsampledCells.empty()) {
            return;
        }
        cellRayStarts_.resize(frontier.downsampledCells.size());
        for (size_t i = 0; i < frontier.downsampledCells.size(); ++ i) {
            cellRayStarts_[i] = (frontier.downsampledCells[i] - mp_.origin) * mp_.resolutionInv;
        }
        const int radiusNum = std::max(1, params_.candidateRnum);
        const double radiusStep = (params_.candidateRmax - params_.candidateRmin) / static_cast<double>(radiusNum);
        for (double radius = params_.candidateRmin; radius <= params_.candidateRmax + 1e-3; radius += radiusStep) {
            for (double phi = -__pi__; phi < __pi__; phi += params_.candidateDphi) {
                const Eigen::Vector3d sample = frontier.center + radius * Eigen::Vector3d(std::cos(phi), std::sin(phi), 0.0);
                if (!mp_.isInMap(sample) || !mp_.isInBox(sample) ||
                    md_.getInflateOccupancy(sample) != VoxelInflateOccupancyCode::FREE ||
                    isNearUnknown_(sample)) {
                    continue;
                }
                const double yaw = meanVisibleYaw_(frontier.downsampledCells, sample);
                const int visibleCellCount = countVisibleCells_(sample, yaw, frontier.downsampledCells);
                if (visibleCellCount > params_.minVisibleNum) {
                    Viewpoint viewpoint;
                    viewpoint.pos = sample;
                    viewpoint.yaw = yaw;
                    viewpoint.visibleCellCount = visibleCellCount;
                    frontier.viewpoints.push_back(viewpoint);
                }
            }
        }
    }
    int viewGain(const Eigen::Vector3d& pos, double yaw) {
        fov_.setPose(pos, yaw);
        Eigen::Vector3d bmin;
        Eigen::Vector3d bmax;
        fov_.getFOVBoundingBox(bmin, bmax);

        int gain = 0;
        for (double x = bmin(0); x <= bmax(0); x += 0.8) {
            for (double y = bmin(1); y <= bmax(1); y += 0.8) {
                for (double z = bmin(2); z <= bmax(2); z += 0.8) {
                    const Eigen::Vector3d sample(x, y, z);
                    if (fov_.insideFOV(sample) && md_.getOccupancy(sample) == VoxelOccupancyCode::UNKNOWN) {
                        ++ gain;
                    }
                }
            }
        }
        return gain;
    }

private:
    double meanVisibleYaw_(const std::vector<Eigen::Vector3d>& cells, const Eigen::Vector3d& pos) const {
        const Eigen::Vector3d reference = (cells.front() - pos).normalized();
        double average = 0.0;
        for (size_t i = 1; i < cells.size(); ++ i) {
            const Eigen::Vector3d direction = (cells[i] - pos).normalized();
            double yaw = std::acos(std::clamp(direction.dot(reference), -1.0, 1.0));
            if (reference.cross(direction)(2) < 0.0) {
                yaw = -yaw;
            }
            average += yaw;
        }
        average = average / static_cast<double>(cells.size()) + std::atan2(reference(1), reference(0));
        return misc::normalizeAngle(average);
    }
    int countVisibleCells_(const Eigen::Vector3d& pos, double yaw, const std::vector<Eigen::Vector3d>& cells) {
        fov_.setPose(pos, yaw);
        int count = 0;
        Eigen::Vector3i index;
        for (size_t i = 0; i < cells.size(); ++ i) {
            const Eigen::Vector3d& cell = cells[i];
            if (!fov_.insideFOV(cell)) {
                continue;
            }
            raycaster_.inputNormalized(cellRayStarts_[i], pos);
            bool visible = true;
            while (raycaster_.nextId(index)) {
                if (md_.isRayBlocked(index)) {
                    visible = false;
                    break;
                }
            }
            if (visible) {
                ++ count;
            }
        }
        return count;
    }
    bool isNearUnknown_(const Eigen::Vector3d& pos) const {
        const int voxelNum = static_cast<int>(std::floor(params_.minCandidateClearance / mp_.resolution));
        const Eigen::Vector3i center = mp_.p2i(pos);
        for (int x = -voxelNum; x <= voxelNum; ++ x) {
            for (int y = -voxelNum; y <= voxelNum; ++ y) {
                for (int z = -1; z <= 1; ++ z) {
                    const Eigen::Vector3i index = center + Eigen::Vector3i(x, y, z);
                    if (md_.getOccupancy(index) == VoxelOccupancyCode::UNKNOWN) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

private:
    const MapParams& mp_;
    const MapData& md_;
    FrontierParams params_;
    RayCaster raycaster_;
    FovModel fov_;
    std::vector<Eigen::Vector3d> cellRayStarts_;
};

} // arc
