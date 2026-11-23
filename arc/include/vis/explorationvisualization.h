#pragma once

#include <base/ros.h>

#include <base/iarc.h>
#include <explore/exploration/explorationdata.h>
#include <explore/frontier/frontiermanager.h>
#include <explore/grid/hgrid.h>
#include <planning/trajectory.h>
#include <vis/planningvisualization.h>

namespace arc {

class ExplorationVisualization {
public:
    explicit ExplorationVisualization(rclcpp::Node::SharedPtr node)
        : vis_(std::move(node))
    {}
    void drawFrontiers(const std::vector<Frontier>& frontiers) {
        const int count = static_cast<int>(frontiers.size());
        for (int i = 0; i < count; ++ i) {
            const double hue = count > 0 ? static_cast<double>(i) / static_cast<double>(count) : 0.0;
            vis_.drawCubes(
                frontiers[i].cells, 0.1, PlanningVisualization::getColor(hue, 0.4),
                "frontier", i, PlanningVisualization::FRONTIER
            );
        }
        for (int i = count; i < lastFrontierCount_; ++ i) {
            vis_.drawCubes(
                {}, 0.1, Eigen::Vector4d(0.0, 0.0, 0.0, 1.0),
                "frontier", i, PlanningVisualization::FRONTIER
            );
        }
        lastFrontierCount_ = count;
    }
    void drawGrid(HGrid& grid, int droneId) {
        if (droneId != 1) {
            return;
        }
        std::vector<Eigen::Vector3d> pts1;
        std::vector<Eigen::Vector3d> pts2;
        grid.buildWireframe(pts1, pts2);
        vis_.drawLines(
            pts1, pts2, 0.05, Eigen::Vector4d(1.0, 0.0, 1.0, 0.5),
            "partition", 1, PlanningVisualization::VIEWPOINT
        );

        std::vector<Eigen::Vector3d> points;
        std::vector<std::string> texts;
        grid.buildLabels(points, texts);
        for (int i = 0; i < static_cast<int>(points.size()); ++ i) {
            vis_.drawText(
                points[i], texts[i], 1.0, Eigen::Vector4d(0.0, 0.0, 0.0, 1.0),
                "text", i, PlanningVisualization::VIEWPOINT
            );
        }
        for (int i = static_cast<int>(points.size()); i < lastGridTextCount_; ++ i) {
            vis_.drawText(
                Eigen::Vector3d::Zero(), std::string(), 1.0,
                Eigen::Vector4d(0.0, 0.0, 0.0, 1.0),
                "text", i, PlanningVisualization::VIEWPOINT
            );
        }
        lastGridTextCount_ = static_cast<int>(points.size());
    }
    void drawGridTour(const ExplorationData& data, int droneId, int droneNum) {
        const double hue = static_cast<double>(std::max(droneId - 1, 0)) /
            static_cast<double>(std::max(droneNum, 1));
        vis_.drawLines(
            data.gridTour.centers, 0.05, PlanningVisualization::getColor(hue),
            "grid_tour", 0, PlanningVisualization::VIEWPOINT
        );
    }
    void drawTrajectory(const LocalTrajectory& trajectory, int droneId, int droneNum) {
        const double hue = static_cast<double>(std::max(droneId - 1, 0)) /
            static_cast<double>(std::max(droneNum, 1));
        std::vector<Eigen::Vector3d> points;
        if (trajectory.valid()) {
            for (double t = 0.0; t <= trajectory.duration + 1e-6; t += 0.01) {
                points.push_back(Eigen::Vector3d(trajectory.position.evalt(t).head<3>()));
            }
        }
        vis_.drawSpheres(
            points, 0.1, PlanningVisualization::getColor(hue),
            "B-Spline", 0, PlanningVisualization::TRAJECTORY
        );
    }

private:
    PlanningVisualization vis_;
    int lastFrontierCount_{0};
    int lastGridTextCount_{0};
};

} // arc
