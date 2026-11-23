#pragma once

#include <base/iarc.h>

namespace arc {

struct GridParams {
    double gridSize{5.0};
    int minUnknown{4000};
    int minFree{3000};
    double consistentCost{-5.0};
    double consistentCost2{8.0};
    std::optional<Eigen::Vector3d> boxMin;
    std::optional<Eigen::Vector3d> boxMax;
};

struct GridInfo {
    int unknownCount{0};
    Eigen::Vector3d center{Eigen::Vector3d::Zero()};
    std::unordered_set<int> frontierIndices;
    bool updatedThisPass{false};
    bool shouldPartition{false};
    bool isActive{false};
    bool wasRelevant{false};
    bool isRelevantNow{false};
    Eigen::Vector3d boxMin{Eigen::Vector3d::Zero()};
    Eigen::Vector3d boxMax{Eigen::Vector3d::Zero()};
    std::vector<Eigen::Vector3d> vertices;
    std::vector<Eigen::Vector3d> normals;
};

} // arc
