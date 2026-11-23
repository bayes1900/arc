#pragma once

#include <algorithm>

#include <Eigen/Dense>
#include <sophus/se3.hpp>

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <base/base.h>
#include <vio/image/image.h>
#include <vio/viodef.h>

namespace arc {

inline void computeEssential(const Sophus::SE3d& T_0_1, Eigen::Matrix4d& E) {
    E.setZero();
    const Eigen::Vector3d t01 = T_0_1.translation();
    const Eigen::Matrix3d R_0_1 = T_0_1.rotationMatrix();
    E.topLeftCorner<3, 3>() = Sophus::SO3d::hat(t01.normalized()) * R_0_1;
}

namespace {
const int EDGE_THRESHOLD = 19;
}
inline void detectKeypoints(
    const Image<const uint16_t>& imgRaw,
    KeypointsData& kd,
    int PATCH_SIZE = 32,
    int numPointsCell = 1,
    const std::vector<Eigen::Vector2d>& currentPoints = std::vector<Eigen::Vector2d>()
) {
    kd.corners.clear();
    kd.cornerAngles.clear();
    kd.cornerDescriptors.clear();
    const size_t xStart = (imgRaw.w % PATCH_SIZE) / 2;
    const size_t xStop = xStart + PATCH_SIZE * (imgRaw.w / PATCH_SIZE - 1);
    const size_t yStart = (imgRaw.h % PATCH_SIZE) / 2;
    const size_t yStop = yStart + PATCH_SIZE * (imgRaw.h / PATCH_SIZE - 1);
    Eigen::MatrixXi cells;
    cells.setZero(imgRaw.h / PATCH_SIZE + 1, imgRaw.w / PATCH_SIZE + 1);
    for (const Eigen::Vector2d& p : currentPoints) {
        if (p[0] >= xStart && p[1] >= yStart && p[0] < xStop + PATCH_SIZE && p[1] < yStop + PATCH_SIZE) {
            int x = (p[0] - xStart) / PATCH_SIZE;
            int y = (p[1] - yStart) / PATCH_SIZE;
            cells(y, x) += 1;
        }
    }
    for (size_t x = xStart; x <= xStop; x += PATCH_SIZE) {
        for (size_t y = yStart; y <= yStop; y += PATCH_SIZE) {
            if (cells((y - yStart) / PATCH_SIZE, (x - xStart) / PATCH_SIZE) > 0)
                continue;
            const Image<const uint16_t> subImgRaw = imgRaw.subImage(x, y, PATCH_SIZE, PATCH_SIZE);
            cv::Mat subImg(PATCH_SIZE, PATCH_SIZE, CV_8U);
            for (int y = 0; y < PATCH_SIZE; ++ y) {
                uchar* subPtr = subImg.ptr(y);
                for (int x = 0; x < PATCH_SIZE; ++ x) {
                    subPtr[x] = (subImgRaw(x, y) >> 8);
                }
            }
            int pointsAdded = 0;
            int threshold = 40;
            while (pointsAdded < numPointsCell && threshold >= 5) {
                std::vector<cv::KeyPoint> points;
                cv::FAST(subImg, points, threshold);
                std::sort(points.begin(), points.end(), [](const cv::KeyPoint& a, const cv::KeyPoint& b) -> bool {
                    return a.response > b.response;
                });
                for (size_t i = 0; i < points.size() && pointsAdded < numPointsCell; ++ i)
                    if (imgRaw.inBounds(x + points[i].pt.x, y + points[i].pt.y, EDGE_THRESHOLD)) {
                        kd.corners.emplace_back(x + points[i].pt.x, y + points[i].pt.y);
                        ++ pointsAdded;
                    }
                threshold /= 2;
            }
        }
    }
}

} // arc
