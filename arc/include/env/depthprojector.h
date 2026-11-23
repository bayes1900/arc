#pragma once

#include <pcl/point_types.h>

#include <pcl/point_cloud.h>

#include <opencv2/core.hpp>

#include <base/iarc.h>

namespace arc {

struct DepthProjectorParams {
    double fx{0.0};
    double fy{0.0};
    double cx{0.0};
    double cy{0.0};
    double minDepth{0.0};
    double maxDepth{0.0};
    int margin{0};
    double scalingFactor{1000.0};
    int skipPixel{1};
};

class DepthProjector {
public:
    DepthProjector() = default;

    void init(const DepthProjectorParams& params) {
        params_ = params;
        count_ = 0;
        cloud_.points.clear();
    }

public:
    const pcl::PointCloud<pcl::PointXYZ>& cloud() const {
        return cloud_;
    }
    size_t size() const {
        return count_;
    }
    double scalingFactor() const {
        return params_.scalingFactor;
    }
    void project(const cv::Mat& depthImage, const Eigen::Vector3d& cameraPos, const Eigen::Matrix3d& cameraR) {
        const int rows = depthImage.rows;
        const int cols = depthImage.cols;
        const double invFactor = 1.0 / params_.scalingFactor;
        const size_t maxPoints = static_cast<size_t>(rows / params_.skipPixel + 1) * static_cast<size_t>(cols / params_.skipPixel + 1);
        count_ = 0;
        if (cloud_.points.size() < maxPoints) {
            cloud_.points.resize(maxPoints);
        }
        for (int v = params_.margin; v < rows - params_.margin; v += params_.skipPixel) {
            const uint16_t* row = depthImage.ptr<uint16_t>(v);
            for (int u = params_.margin; u < cols - params_.margin; u += params_.skipPixel) {
                const uint16_t rawDepth = row[u];
                double depth = rawDepth * invFactor;
                if (rawDepth == 0 || depth > params_.maxDepth) {
                    depth = params_.maxDepth;
                } else if (depth < params_.minDepth) {
                    continue;
                }
                const Eigen::Vector3d ptCur(
                    (u - params_.cx) * depth / params_.fx,
                    (v - params_.cy) * depth / params_.fy,
                    depth
                );
                const Eigen::Vector3d ptWorld = cameraR * ptCur + cameraPos;
                auto& pt = cloud_.points[count_ ++];
                pt.x = ptWorld.x();
                pt.y = ptWorld.y();
                pt.z = ptWorld.z();
            }
        }
        cloud_.points.resize(count_);
        cloud_.width = count_;
        cloud_.height = 1;
        cloud_.is_dense = true;
    }

private:
    DepthProjectorParams params_;
    pcl::PointCloud<pcl::PointXYZ> cloud_;
    size_t count_{0};
};

} // arc
