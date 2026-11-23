#pragma once

#include <base/iarc.h>
#include <explore/grid/grididlist.h>

namespace arc {

struct ViewpointData {
    std::vector<Eigen::Vector3d> positions;
    std::vector<double> yaws;
    std::vector<Eigen::Vector3d> frontierCenters;
};

struct GridTourData {
    GridIdList assignedGridIds;
    std::vector<int> previousGridIds;
    std::vector<int> consistentFirstIds;
    std::vector<int> consistentSecondIds;
    std::vector<Eigen::Vector3d> centers;
};

struct FrontierTourData {
    std::vector<int> indices;
    std::vector<Eigen::Vector3d> path;
    std::vector<int> refinedIndices;
    std::vector<Eigen::Vector3d> refinedViewpoints;
    std::vector<double> refinedViewpointYaws;
};

struct ExplorationPlanningData {
    Eigen::Vector3d nextViewpoint{Eigen::Vector3d::Zero()};
    double nextViewpointYaw{0.0};
    std::vector<Eigen::Vector3d> pathToNextViewpoint;
    int planCount{0};
};

struct ExplorationData {
    ViewpointData viewpoints;
    GridTourData gridTour;
    FrontierTourData frontierTour;
    ExplorationPlanningData planning;
};

} // arc