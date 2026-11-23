#pragma once

#include <base/iarc.h>
#include <base/ros.h>
#include <planning/bspline.h>

namespace arc {

struct BSplineOptCost {
    union {
        struct {
            uint32_t SMOOTHNESS  : 1;
            uint32_t DISTANCE    : 1;
            uint32_t FEASIBILITY : 1;
            uint32_t START       : 1;
            uint32_t END         : 1;
            uint32_t GUIDE       : 1;
            uint32_t WAYPOINTS   : 1;
            uint32_t VIEWCONS    : 1;
            uint32_t MINTIME     : 1;
            uint32_t SWARM       : 1;
        };
        uint32_t costFlags{0};
    };
};

struct BSplineOptParams {
    double smoothWeight{5.0};
    double distanceWeight{10.0};
    double feasibilityWeight{2.0};
    double startWeight{100.0};
    double endWeight{0.5};
    double guideWeight{1.5};
    double waypointWeight{7.0};
    double viewWeight{0.0};
    double viewParallelWeight{1.0};
    double timeWeight{1.5};
    double swarmWeight{5.0};

    double distanceThreshold{0.7};
    double maxVel{1.5};
    double maxAcc{1.0};
    double swarmSafeDistance{1.0};
    double startPositionWeight{10.0};
    double timeLowerBoundWeight{10.0};

    int maxEval{2000};
    double maxTime{0.05};
    double xtolRel{1e-5};
    double minInterval{1e-3};
    double maxInterval{5.0};

    BSplineOptParams() = default;

    BSplineOptParams(rclcpp::Node::SharedPtr node) {
        smoothWeight = misc::getParams<double>(node, "bsplineopt.smoothWeight", smoothWeight);
        distanceWeight = misc::getParams<double>(node, "bsplineopt.distanceWeight", distanceWeight);
        feasibilityWeight = misc::getParams<double>(node, "bsplineopt.feasibilityWeight", feasibilityWeight);
        startWeight = misc::getParams<double>(node, "bsplineopt.startWeight", startWeight);
        endWeight = misc::getParams<double>(node, "bsplineopt.endWeight", endWeight);
        guideWeight = misc::getParams<double>(node, "bsplineopt.guideWeight", guideWeight);
        waypointWeight = misc::getParams<double>(node, "bsplineopt.waypointWeight", waypointWeight);
        viewWeight = misc::getParams<double>(node, "bsplineopt.viewWeight", viewWeight);
        viewParallelWeight = misc::getParams<double>(node, "bsplineopt.viewParallelWeight", viewParallelWeight);
        timeWeight = misc::getParams<double>(node, "bsplineopt.timeWeight", timeWeight);
        swarmWeight = misc::getParams<double>(node, "bsplineopt.swarmWeight", swarmWeight);

        distanceThreshold = misc::getParams<double>(node, "bsplineopt.distanceThreshold", distanceThreshold);
        maxVel = misc::getParams<double>(node, "bsplineopt.maxVel", maxVel);
        maxAcc = misc::getParams<double>(node, "bsplineopt.maxAcc", maxAcc);
        swarmSafeDistance = misc::getParams<double>(node, "bsplineopt.swarmSafeDistance", swarmSafeDistance);
        startPositionWeight = misc::getParams<double>(node, "bsplineopt.startPositionWeight", startPositionWeight);
        timeLowerBoundWeight = misc::getParams<double>(node, "bsplineopt.timeLowerBoundWeight", timeLowerBoundWeight);

        maxEval = misc::getParams<int>(node, "bsplineopt.maxEval", maxEval);
        maxTime = misc::getParams<double>(node, "bsplineopt.maxTime", maxTime);
        xtolRel = misc::getParams<double>(node, "bsplineopt.xtolRel", xtolRel);
        minInterval = misc::getParams<double>(node, "bsplineopt.minInterval", minInterval);
        maxInterval = misc::getParams<double>(node, "bsplineopt.maxInterval", maxInterval);
    }
};

struct BSplineOptView {
    Eigen::Vector3d point{Eigen::Vector3d::Zero()};
    Eigen::Vector3d direction{Eigen::Vector3d::UnitX()};
    int index{-1};
};

struct BSplineOptSwarm {
    BSpline trajectory;
    double startTime{0.0};
};

struct BSplineOptRequest {
    Eigen::MatrixXd points;
    double interval{0.1};
    BSplineOptCost cost;

    std::vector<Eigen::Vector3d> startState;
    std::vector<Eigen::Vector3d> endState;
    std::vector<Eigen::Vector3d> guidePoints;
    std::vector<Eigen::Vector3d> waypoints;
    std::vector<int> waypointIndices;

    std::optional<BSplineOptView> view;
    std::vector<BSplineOptSwarm> swarm;

    double planStartTime{0.0};
    double timeLowerBound{0.0};
};

struct BSplineOptResult {
    static constexpr uint8_t SUCCEEDED       = 0;
    static constexpr uint8_t INPUTINVALID    = 1;
    static constexpr uint8_t INFEASIBLE      = 2;
    static constexpr uint8_t SOLVERFAILED    = 3;
    static constexpr uint8_t NOSOLUTION      = 4;
    static constexpr uint8_t MAXEVALREACHED  = 5;
    static constexpr uint8_t MAXTIMEREACHED  = 6;
    static constexpr uint8_t UNKNOWN         = 255;

    uint8_t code{255};
    int iterations{0};
    double cost{0.0};
    Eigen::MatrixXd points;
    double interval{0.0};
};

} // arc
