#pragma once

#include <base/ros.h>

#include <base/iarc.h>
#include <diagnostics/exploration.h>
#include <env/raycaster.h>
#include <env/sdfmap.h>
#include <explore/exploration/explorationdata.h>
#include <explore/exploration/explorationdef.h>
#include <explore/frontier/frontiermanager.h>
#include <explore/frontier/viewnode.h>
#include <explore/grid/hgrid.h>
#include <explore/tsp/tspsolver.h>
#include <fastplanner/fastplanner.h>
#include <search/astar.h>
#include <search/graphsearch.h>

namespace arc {

class ExplorationManager {
public:
    ExplorationManager(
        rclcpp::Node::SharedPtr node,
        SdfMap& map,
        FrontierManager& frontierManager,
        HGrid& hgrid,
        ViewNodeContext& viewNode,
        TspSolver& tspSolver,
        FastPlanner& planner,
        const ExplorationParams& params,
        const AstarOptions& astarOptions = {},
        DiagnosticsRecorder* diagnostics = nullptr
    )
        : node_(std::move(node))
        , map_(map)
        , frontierManager_(frontierManager)
        , hgrid_(hgrid)
        , viewNode_(viewNode)
        , tspSolver_(tspSolver)
        , planner_(planner)
        , params_(params)
        , astar_(map, astarOptions)
        , raycaster_(map.params())
        , noopDiagnostics_()
        , diagnostics_(diagnostics ? diagnostics : &noopDiagnostics_)
    {}
    size_t refreshFrontiers(const Eigen::Vector3d& pos) {
        auto timer = diagnostics_->stage(DiagStage::FRONTIER_REFRESH);
        frontierManager_.detectFrontiers();
        frontierManager_.sampleNewFrontierViewpoints();
        frontierManager_.selectTopViewpoints(pos, data_.viewpoints.positions, data_.viewpoints.yaws, data_.viewpoints.frontierCenters);
        frontierManager_.refreshCostMatrix();
        return frontierManager_.activeFrontierCount();
    }
    bool updateGridTour(const OdomState& odom, std::vector<int>& gridOrder, bool init=false) {
        gridOrder.clear();
        {
            auto updateTimer = diagnostics_->stage(DiagStage::GRID_UPDATE);
            hgrid_.inputFrontiers(data_.viewpoints.frontierCenters);
            hgrid_.updateGridData(
                params_.droneId,
                data_.gridTour.assignedGridIds,
                data_.gridTour.previousGridIds,
                data_.gridTour.consistentFirstIds,
                data_.gridTour.consistentSecondIds
            );
        }
        if (data_.gridTour.assignedGridIds.empty()) {
            data_.gridTour.centers.clear();
            return false;
        }
        auto tourTimer = diagnostics_->stage(DiagStage::GRID_TOUR);
        Eigen::MatrixXd cost;
        if (init) {
            hgrid_.buildCostMatrix({odom.pos}, {odom.vel}, {{}}, {{}}, data_.gridTour.assignedGridIds.values(), cost);
        } else {
            hgrid_.buildCostMatrix({odom.pos}, {odom.vel}, {data_.gridTour.consistentFirstIds}, {data_.gridTour.consistentSecondIds}, data_.gridTour.assignedGridIds.values(), cost);
        }
        const TspSolveResult tspResult = solveSingleVehicleTourWithDiagnostics_(cost, data_.gridTour.assignedGridIds.values(), gridOrder);
        if (!tspResult.success || gridOrder.empty()) {
            data_.gridTour.centers.clear();
            diagnostics_->planFailure(
                node_->now().seconds(),
                DiagFailures::gridTsp(
                    tspResult,
                    static_cast<int>(data_.gridTour.assignedGridIds.size()),
                    0,
                    static_cast<int>(data_.viewpoints.positions.size()),
                    tourTimer.elapsed()
                )
            );
            return false;
        }
        data_.gridTour.assignedGridIds.replaceOrder(gridOrder);
        hgrid_.buildGridCenters(data_.gridTour.assignedGridIds.values(), odom.pos, data_.gridTour.centers);
        data_.gridTour.previousGridIds = data_.gridTour.assignedGridIds.values();
        return true;
    }
    int planExploreMotion(const OdomState& odom) {
        auto totalTimer = diagnostics_->stage(DiagStage::PLAN_TOTAL);
        diagnostics_->planAttempt();
        if (isExplorationExhausted_) {
            PlanNoGridInfo info;
            info.duration = totalTimer.elapsed();
            info.reason = "exhausted";
            info.detail = DiagDetail().add("pos", DiagFormat::vectorParen(odom.pos)).str();
            diagnostics_->planNoGrid(node_->now().seconds(), info);
            return ExplorationStatus::NO_GRID;
        }
        data_.frontierTour.indices.clear();
        std::vector<int> assignedGridIds;
        updateGridAndFrontierTour(odom, assignedGridIds);
        const std::vector<int>& frontierIndices = data_.frontierTour.indices;

        int status = ExplorationStatus::FAIL;
        if (assignedGridIds.empty()) {
            status = ExplorationStatus::NO_GRID;
        } else if (frontierIndices.empty()) {
            status = planToNearestFrontier_(odom);
        } else if (frontierIndices.size() == 1) {
            status = planToSingleFrontier_(odom, assignedGridIds, frontierIndices);
        } else {
            status = planToMultipleFrontiers_(odom, frontierIndices);
        }
        if (params_.finishUnreachableFrontier &&
            status == ExplorationStatus::SUCCEED &&
            frontierIndices.size() == 1 &&
            data_.viewpoints.positions.size() <= 2 &&
            (data_.planning.nextViewpoint - odom.pos).norm() < 0.8
        ) {
            RCLCPP_WARN(node_->get_logger(), "last frontier reached without coverage, entering IDLE");
            isExplorationExhausted_ = true;
            status = ExplorationStatus::NO_GRID;
        }
        const double totalDuration = totalTimer.elapsed();
        const double distance = (data_.planning.nextViewpoint - odom.pos).norm();
        const std::string detail = DiagDetail()
            .add("pos", DiagFormat::vectorParen(odom.pos))
            .add("next", DiagFormat::vectorParen(data_.planning.nextViewpoint))
            .add("dist", distance)
            .add("yaw", odom.yaw[0])
            .add("nextYaw", data_.planning.nextViewpointYaw)
            .add("traj", planner_.localTrajectory().duration)
            .str();
        const double eventTime = node_->now().seconds();
        if (status == ExplorationStatus::SUCCEED) {
            PlanSuccessInfo info;
            info.duration = totalDuration;
            info.distance = distance;
            info.gridCount = static_cast<int>(assignedGridIds.size());
            info.frontierCount = static_cast<int>(frontierIndices.size());
            info.viewpointCount = static_cast<int>(data_.viewpoints.positions.size());
            info.detail = detail;
            diagnostics_->planSuccess(eventTime, info);
        } else if (status == ExplorationStatus::FAIL) {
            diagnostics_->planFailure(
                eventTime,
                DiagFailures::planSummary(
                    status,
                    detail,
                    totalDuration,
                    static_cast<int>(assignedGridIds.size()),
                    static_cast<int>(frontierIndices.size()),
                    static_cast<int>(data_.viewpoints.positions.size())
                )
            );
        } else {
            PlanNoGridInfo info;
            info.duration = totalDuration;
            info.reason = "no_grid";
            info.detail = detail;
            diagnostics_->planNoGrid(eventTime, info);
        }
        RCLCPP_DEBUG(
            node_->get_logger(),
            "planExploreMotion assignedGrids=%zu frontiers=%zu viewpoints=%zu pos=(%.2f,%.2f,%.2f) "
            "next=(%.2f,%.2f,%.2f) dist=%.2f traj=%.2f status=%s",
            assignedGridIds.size(), frontierIndices.size(), data_.viewpoints.positions.size(),
            odom.pos(0), odom.pos(1), odom.pos(2),
            data_.planning.nextViewpoint(0), data_.planning.nextViewpoint(1), data_.planning.nextViewpoint(2),
            distance,
            planner_.localTrajectory().duration,
            DiagCodes::explorationStatus(status)
        );
        return status;
    }
    const ExplorationData& data() const {
        return data_;
    }
    ExplorationData& data() {
        return data_;
    }
    const ExplorationParams& params() const {
        return params_;
    }
    bool isFrontierCovered() const {
        return frontierManager_.isAnyFrontierCovered();
    }
    int replanToCurrentViewpoint(const OdomState& odom) {
        if (!hasPlanningGoal_) {
            return planExploreMotion(odom);
        }
        return planToViewpoint(odom, data_.planning.nextViewpoint, data_.planning.nextViewpointYaw);
    }

private:
    void updateGridAndFrontierTour(const OdomState& odom, std::vector<int>& assignedGridIds) {
        if (!updateGridTour(odom, assignedGridIds)) {
            assignedGridIds.clear();
            data_.frontierTour.indices.clear();
            data_.frontierTour.path.clear();
            return;
        }
        data_.frontierTour.indices.clear();
        data_.frontierTour.path.clear();
        hgrid_.frontiersInGrid(assignedGridIds, data_.frontierTour.indices);
        if (data_.frontierTour.indices.empty()) {
            return;
        }
        Eigen::Vector3d nextGridPosition;
        std::vector<Eigen::Vector3d> nextGridPositions;
        if (hgrid_.nextGrid(assignedGridIds, nextGridPosition)) {
            nextGridPositions = {nextGridPosition};
        }
        const auto frontierOrder = buildFrontierTour(odom, data_.frontierTour.indices, nextGridPositions);
        if (!frontierOrder) {
            data_.frontierTour.indices.clear();
            return;
        }
        data_.frontierTour.indices = *frontierOrder;
    }
    std::optional<std::vector<int>> buildFrontierTour(
        const OdomState& odom,
        const std::vector<int>& frontierIndices,
        const std::vector<Eigen::Vector3d>& nextGridPositions
    ) {
        data_.frontierTour.path.clear();
        if (frontierIndices.empty()) {
            return std::nullopt;
        }
        auto tourTimer = diagnostics_->stage(DiagStage::FRONTIER_TOUR);
        Eigen::MatrixXd cost;
        frontierManager_.buildSwarmCostMatrix({odom.pos}, {odom.vel}, {odom.yaw[0]}, frontierIndices, nextGridPositions, cost);

        std::vector<int> frontierOrder;
        const TspSolveResult tspResult = solveSingleVehicleTourWithDiagnostics_(cost, frontierIndices, frontierOrder);
        if (!tspResult.success || frontierOrder.empty()) {
            diagnostics_->planFailure(
                node_->now().seconds(),
                DiagFailures::frontierTsp(
                    tspResult,
                    0,
                    static_cast<int>(frontierIndices.size()),
                    static_cast<int>(data_.viewpoints.positions.size()),
                    tourTimer.elapsed()
                )
            );
            return std::nullopt;
        }
        frontierManager_.buildPathForTour(odom.pos, frontierOrder, data_.frontierTour.path);
        if (!nextGridPositions.empty()) {
            data_.frontierTour.path.push_back(nextGridPositions.front());
        }
        return frontierOrder;
    }
    void refineFrontierTour(
        const OdomState& odom,
        const std::vector<std::vector<Eigen::Vector3d>>& viewpointCandidates,
        const std::vector<std::vector<double>>& yawCandidates,
        std::vector<Eigen::Vector3d>& refinedViewpoints,
        std::vector<double>& refinedViewpointYaws
    ) {
        auto selectTimer = diagnostics_->stage(DiagStage::VIEWPOINT_REFINE);
        refinedViewpoints.clear();
        refinedViewpointYaws.clear();

        GraphSearch<ViewNode> graph;
        std::vector<std::unique_ptr<ViewNode>> ownedNodes;
        std::vector<ViewNode*> viewNodePath;
        if (!buildViewGraph_(odom, viewpointCandidates, yawCandidates, graph, ownedNodes, viewNodePath)) {
            diagnostics_->planFailure(
                node_->now().seconds(),
                DiagFailures::refine(
                    static_cast<int>(viewpointCandidates.size()),
                    static_cast<int>(data_.viewpoints.positions.size()),
                    selectTimer.elapsed()
                )
            );
            return;
        }
        for (size_t i = 1; i < viewNodePath.size(); ++i) {
            refinedViewpoints.push_back(viewNodePath[i]->pos);
            refinedViewpointYaws.push_back(viewNodePath[i]->yaw);
        }
    }
    int planToViewpoint(const OdomState& odom, const Eigen::Vector3d& nextViewpoint, double nextViewpointYaw) {
        if (!params_.planTrajectory) {
            return ExplorationStatus::SUCCEED;
        }
        map_.updateESDF();
        const double yawDifference = std::fabs(nextViewpointYaw - odom.yaw[0]);
        const double timeLowerBound = std::min(yawDifference, 2.0 * __pi__ - yawDifference) / std::max(viewNode_.params().yd, 1e-6);
        const bool useInitialPlan = data_.planning.planCount < params_.initPlanNum;
        double pathLength = 0.0;
        if (!findPathToViewpoint_(odom, nextViewpoint, useInitialPlan, pathLength)) {
            return ExplorationStatus::FAIL;
        }
        if (!planPathToViewpoint_(odom, nextViewpoint, pathLength, timeLowerBound, useInitialPlan)) {
            return ExplorationStatus::FAIL;
        }
        if (!planYawToViewpoint_(odom, nextViewpointYaw)) {
            return ExplorationStatus::FAIL;
        }
        return ExplorationStatus::SUCCEED;
    }
    void shortenPath(std::vector<Eigen::Vector3d>& path) {
        if (path.empty()) {
            return;
        }
        std::vector<Eigen::Vector3d> shortenedPath = {path.front()};
        for (size_t i = 1; i + 1 < path.size(); ++i) {
            if ((path[i] - shortenedPath.back()).norm() > params_.shortenDist) {
                shortenedPath.push_back(path[i]);
                continue;
            }
            raycaster_.input(shortenedPath.back(), path[i + 1]);
            Eigen::Vector3i index;
            while (raycaster_.nextId(index)) {
                if (map_.data().getInflateOccupancy(index) != VoxelInflateOccupancyCode::FREE ||
                    map_.data().getOccupancy(index) == VoxelOccupancyCode::UNKNOWN) {
                    shortenedPath.push_back(path[i]);
                    break;
                }
            }
        }
        if ((path.back() - shortenedPath.back()).norm() > 1e-3) {
            shortenedPath.push_back(path.back());
        }
        if (shortenedPath.size() == 2) {
            shortenedPath.insert(shortenedPath.begin() + 1, 0.5 * (shortenedPath.front() + shortenedPath.back()));
        }
        path = shortenedPath;
    }
    bool findPathToViewpoint_(
        const OdomState& odom,
        const Eigen::Vector3d& nextViewpoint,
        bool useInitialPlan,
        double& pathLength
    ) {
        auto searchTimer = diagnostics_->stage(DiagStage::ASTAR);
        astar_.reset();
        const uint8_t status = astar_.search(odom.pos, nextViewpoint, useInitialPlan);
        if (status != AstarSearchStatus::DONE) {
            diagnostics_->planFailure(
                node_->now().seconds(),
                DiagFailures::astar(
                    status,
                    odom.pos,
                    nextViewpoint,
                    static_cast<int>(data_.viewpoints.positions.size()),
                    searchTimer.elapsed()
                )
            );
            RCLCPP_WARN(
                node_->get_logger(),
                "astar failed status=%s start=(%.2f,%.2f,%.2f) goal=(%.2f,%.2f,%.2f)",
                DiagCodes::astarStatus(status),
                odom.pos(0), odom.pos(1), odom.pos(2), nextViewpoint(0), nextViewpoint(1), nextViewpoint(2)
            );
            return false;
        }
        data_.planning.pathToNextViewpoint = astar_.getPath();
        shortenPath(data_.planning.pathToNextViewpoint);
        if (data_.planning.pathToNextViewpoint.size() < 2) {
            data_.planning.pathToNextViewpoint = {odom.pos, nextViewpoint};
        }
        pathLength = misc::pathLength(data_.planning.pathToNextViewpoint);
        return true;
    }
    std::vector<Eigen::Vector3d> farGoalPath_() const {
        std::vector<Eigen::Vector3d> shortenedPath = {data_.planning.pathToNextViewpoint.front()};
        double shortenedLength = 0.0;
        for (size_t i = 1; i < data_.planning.pathToNextViewpoint.size() && shortenedLength < params_.radiusFar; ++i) {
            const Eigen::Vector3d currentPoint = data_.planning.pathToNextViewpoint[i];
            shortenedLength += (currentPoint - shortenedPath.back()).norm();
            shortenedPath.push_back(currentPoint);
        }
        return shortenedPath;
    }
    bool planPathToViewpoint_(
        const OdomState& odom,
        const Eigen::Vector3d& nextViewpoint,
        double pathLength,
        double timeLowerBound,
        bool useInitialPlan
    ) {
        if (pathLength < params_.radiusClose || useInitialPlan) {
            auto optimizeTimer = diagnostics_->stage(DiagStage::TRAJ_OPTIMIZE);
            const PlannerResult result = planner_.planExploreTrajectory(data_.planning.pathToNextViewpoint, odom.vel, odom.acc, timeLowerBound);
            if (!result.succeeded()) {
                diagnostics_->planFailure(
                    node_->now().seconds(),
                    DiagFailures::trajOptimize(
                        result,
                        pathLength,
                        static_cast<int>(data_.viewpoints.positions.size()),
                        optimizeTimer.elapsed()
                    )
                );
                RCLCPP_WARN(
                    node_->get_logger(),
                    "planExploreTrajectory failed code=%s optimizeStatus=%u pathLength=%.2f",
                    DiagCodes::planResult(result.code),
                    static_cast<unsigned>(result.optimizeStatus),
                    pathLength
                );
                return false;
            }
            if (data_.planning.planCount < params_.initPlanNum) {
                data_.planning.planCount += 1;
            }
            return true;
        }
        if (pathLength > params_.radiusFar) {
            auto optimizeTimer = diagnostics_->stage(DiagStage::TRAJ_OPTIMIZE);
            const std::vector<Eigen::Vector3d> shortenedPath = farGoalPath_();
            const PlannerResult result = planner_.planExploreTrajectory(shortenedPath, odom.vel, odom.acc, timeLowerBound);
            if (!result.succeeded()) {
                diagnostics_->planFailure(
                    node_->now().seconds(),
                    DiagFailures::trajOptimize(
                        result,
                        pathLength,
                        static_cast<int>(data_.viewpoints.positions.size()),
                        optimizeTimer.elapsed()
                    )
                );
                RCLCPP_WARN(
                    node_->get_logger(),
                    "planExploreTrajectory(far) failed code=%s optimizeStatus=%u pathLength=%.2f",
                    DiagCodes::planResult(result.code),
                    static_cast<unsigned>(result.optimizeStatus),
                    pathLength
                );
                return false;
            }
            return true;
        }
        const PlannerResult result = planner_.kinodynamicReplan(
            odom.pos, odom.vel, odom.acc, nextViewpoint, Eigen::Vector3d::Zero(), timeLowerBound);
        if (result.searchTime > 0.0) {
            diagnostics_->recordStage(DiagStage::KINO_SEARCH, result.searchTime);
        }
        if (result.optimizeTime > 0.0) {
            diagnostics_->recordStage(DiagStage::KINO_OPTIMIZE, result.optimizeTime);
        }
        if (!result.succeeded()) {
            diagnostics_->planFailure(
                node_->now().seconds(),
                DiagFailures::kino(
                    result,
                    pathLength,
                    static_cast<int>(data_.viewpoints.positions.size()),
                    result.searchTime + result.optimizeTime
                )
            );
            RCLCPP_WARN(
                node_->get_logger(),
                "kinodynamicReplan failed code=%s searchStatus=%s optimizeStatus=%u pathLength=%.2f",
                DiagCodes::planResult(result.code),
                DiagCodes::kinoStatus(result.searchStatus),
                static_cast<unsigned>(result.optimizeStatus),
                pathLength
            );
            return false;
        }
        return true;
    }
    bool planYawToViewpoint_(const OdomState& odom, double nextViewpointYaw) {
        auto yawTimer = diagnostics_->stage(DiagStage::YAW_OPTIMIZE);
        const PlannerResult result = planner_.planYawExplore(
            odom.yaw, nextViewpointYaw, true);
        if (!result.succeeded()) {
            diagnostics_->planFailure(
                node_->now().seconds(),
                DiagFailures::yaw(
                    result,
                    nextViewpointYaw,
                    static_cast<int>(data_.viewpoints.positions.size()),
                    yawTimer.elapsed()
                )
            );
            RCLCPP_WARN(
                node_->get_logger(),
                "planYawExplore failed code=%s",
                DiagCodes::planResult(result.code)
            );
            return false;
        }
        return true;
    }
    void setNextViewpoint_(const Eigen::Vector3d& viewpoint, double yaw) {
        data_.planning.nextViewpoint = viewpoint;
        data_.planning.nextViewpointYaw = yaw;
        hasPlanningGoal_ = true;
    }
    void useRefinedViewpoint_() {
        setNextViewpoint_(data_.frontierTour.refinedViewpoints.front(), data_.frontierTour.refinedViewpointYaws.front());
    }
    bool buildViewGraph_(
        const OdomState& odom,
        const std::vector<std::vector<Eigen::Vector3d>>& viewpointCandidates,
        const std::vector<std::vector<double>>& yawCandidates,
        GraphSearch<ViewNode>& graph,
        std::vector<std::unique_ptr<ViewNode>>& ownedNodes,
        std::vector<ViewNode*>& viewNodePath
    ) {
        std::vector<int> previousLayer;
        std::vector<int> currentLayer;
        auto addNode = [&](const Eigen::Vector3d& position, double yaw) {
            ownedNodes.push_back(std::make_unique<ViewNode>(viewNode_, position, yaw));
            const int index = static_cast<int>(ownedNodes.size()) - 1;
            graph.addNode(ownedNodes.back().get());
            return index;
        };

        const int first = addNode(odom.pos, odom.yaw[0]);
        ownedNodes[first]->vel = odom.vel;
        previousLayer.push_back(first);
        int lastViewpointNode = -1;
        for (size_t i = 0; i < viewpointCandidates.size(); ++i) {
            if (viewpointCandidates[i].empty()) {
                continue;
            }
            for (size_t j = 0; j < viewpointCandidates[i].size(); ++j) {
                const int node = addNode(viewpointCandidates[i][j], yawCandidates[i][j]);
                for (int previous : previousLayer) {
                    graph.addEdge(previous, node);
                }
                currentLayer.push_back(node);
                if (i == viewpointCandidates.size() - 1) {
                    lastViewpointNode = node;
                    break;
                }
            }
            previousLayer = currentLayer;
            currentLayer.clear();
        }
        if (lastViewpointNode < 0) {
            return false;
        }
        return graph.dijkstraSearch(first, lastViewpointNode, viewNodePath);
    }
    int planToNearestFrontier_(const OdomState& odom) {
        const Eigen::Vector3d gridCenter = data_.gridTour.centers.size() > 1 ? data_.gridTour.centers[1] : odom.pos;
        double bestCost = std::numeric_limits<double>::max();
        int bestViewpointIndex = -1;
        std::vector<Eigen::Vector3d> segment;
        {
            auto selectTimer = diagnostics_->stage(DiagStage::VIEWPOINT_SELECT);
            for (size_t i = 0; i < data_.viewpoints.positions.size(); ++ i) {
                const double cost = viewNode_.costBetween(gridCenter, data_.viewpoints.frontierCenters[i], 0.0, 0.0, Eigen::Vector3d::Zero(), 0.0, segment);
                if (cost < bestCost) {
                    bestCost = cost;
                    bestViewpointIndex = static_cast<int>(i);
                }
            }
        }
        if (bestViewpointIndex < 0) {
            diagnostics_->planFailure(
                node_->now().seconds(),
                DiagFailures::noViewpoint(
                    DiagDetail().add("frontiers", data_.viewpoints.positions.size()).str(),
                    static_cast<int>(data_.viewpoints.positions.size())
                )
            );
            return ExplorationStatus::FAIL;
        }
        setNextViewpoint_(data_.viewpoints.positions[bestViewpointIndex], data_.viewpoints.yaws[bestViewpointIndex]);
        return planToViewpoint(odom, data_.planning.nextViewpoint, data_.planning.nextViewpointYaw);
    }
    int planToSingleFrontier_(
        const OdomState& odom,
        const std::vector<int>& assignedGridIds,
        const std::vector<int>& frontierIndices
    ) {
        data_.frontierTour.refinedIndices = {frontierIndices.front()};
        std::vector<std::vector<Eigen::Vector3d>> viewpointCandidates;
        std::vector<std::vector<double>> yawCandidates;
        {
            auto selectTimer = diagnostics_->stage(DiagStage::VIEWPOINT_SELECT);
            frontierManager_.selectViewpoints(odom.pos, {frontierIndices.front()}, params_.topViewNum, params_.maxDecay, viewpointCandidates, yawCandidates);
        }
        if (viewpointCandidates.empty() || viewpointCandidates.front().empty()) {
            diagnostics_->planFailure(
                node_->now().seconds(),
                DiagFailures::noViewpoint(
                    DiagDetail().add("frontiers", 1).str(),
                    static_cast<int>(data_.viewpoints.positions.size())
                )
            );
            return ExplorationStatus::FAIL;
        }
        if (assignedGridIds.size() <= 1) {
            double bestCost = std::numeric_limits<double>::max();
            int bestViewpointIndex = -1;
            std::vector<Eigen::Vector3d> segment;
            for (size_t i = 0; i < viewpointCandidates.front().size(); ++i) {
                const double cost = viewNode_.costBetween(
                    odom.pos, viewpointCandidates.front()[i], odom.yaw[0], yawCandidates.front()[i], odom.vel, 0.0, segment);
                if (cost < bestCost) {
                    bestCost = cost;
                    bestViewpointIndex = static_cast<int>(i);
                }
            }
            if (bestViewpointIndex < 0) {
                diagnostics_->planFailure(
                    node_->now().seconds(),
                    DiagFailures::noViewpoint(
                        DiagDetail().add("candidates", viewpointCandidates.front().size()).str(),
                        static_cast<int>(data_.viewpoints.positions.size())
                    )
                );
                return ExplorationStatus::FAIL;
            }
            setNextViewpoint_(
                viewpointCandidates.front()[bestViewpointIndex],
                yawCandidates.front()[bestViewpointIndex]);
        } else {
            Eigen::Vector3d nextGridPosition;
            double nextGridYaw = 0.0;
            if (hgrid_.nextGrid(assignedGridIds, nextGridPosition, nextGridYaw)) {
                viewpointCandidates.push_back({nextGridPosition});
                yawCandidates.push_back({nextGridYaw});
            }
            refineFrontierTour(odom, viewpointCandidates, yawCandidates, data_.frontierTour.refinedViewpoints, data_.frontierTour.refinedViewpointYaws);
            if (data_.frontierTour.refinedViewpoints.empty()) {
                return ExplorationStatus::FAIL;
            }
            useRefinedViewpoint_();
        }
        data_.frontierTour.refinedViewpoints = {data_.planning.nextViewpoint};
        data_.frontierTour.refinedViewpointYaws = {data_.planning.nextViewpointYaw};
        return planToViewpoint(odom, data_.planning.nextViewpoint, data_.planning.nextViewpointYaw);
    }
    int planToMultipleFrontiers_(
        const OdomState& odom,
        const std::vector<int>& frontierIndices
    ) {
        data_.frontierTour.refinedIndices.clear();
        const int refinedFrontierCount = std::min(static_cast<int>(frontierIndices.size()), params_.refinedNum);
        for (int i = 0; i < refinedFrontierCount; ++i) {
            const Eigen::Vector3d point = data_.viewpoints.positions[frontierIndices[i]];
            data_.frontierTour.refinedIndices.push_back(frontierIndices[i]);
            if ((point - odom.pos).norm() > params_.refinedRadius && data_.frontierTour.refinedIndices.size() >= 2) {
                break;
            }
        }
        std::vector<std::vector<Eigen::Vector3d>> viewpointCandidates;
        std::vector<std::vector<double>> yawCandidates;
        frontierManager_.selectViewpoints(
            odom.pos, data_.frontierTour.refinedIndices, params_.topViewNum, params_.maxDecay, viewpointCandidates, yawCandidates);
        refineFrontierTour(odom, viewpointCandidates, yawCandidates, data_.frontierTour.refinedViewpoints, data_.frontierTour.refinedViewpointYaws);
        if (data_.frontierTour.refinedViewpoints.empty()) {
            return ExplorationStatus::FAIL;
        }
        useRefinedViewpoint_();
        return planToViewpoint(odom, data_.planning.nextViewpoint, data_.planning.nextViewpointYaw);
    }

    void decodeGridTour_(
        const std::vector<int>& tour,
        const std::vector<int>& candidateIds,
        std::vector<int>& orderedIds
    ) const {
        constexpr int droneCount = 1;
        orderedIds.clear();
        for (int id : tour) {
            if (id <= droneCount) {
                continue;
            }
            const int index = id - (1 + droneCount);
            if (index >= 0 && index < static_cast<int>(candidateIds.size())) {
                orderedIds.push_back(candidateIds[index]);
            }
        }
    }
    TspSolveResult solveSingleVehicleTourWithDiagnostics_(
        const Eigen::MatrixXd& cost,
        const std::vector<int>& candidateIds,
        std::vector<int>& orderedIds
    ) {
        TspProblem problem;
        problem.cost = cost;
        problem.vehicles = 1;
        problem.acvrp = false;
        TspSolution solution;
        TspSolveResult result = tspSolver_.solveWithDiagnostics(problem, solution);
        diagnostics_->recordStage(DiagStage::TSP, result.seconds);
        if (result.seconds > 0.3) {
            RCLCPP_WARN(
                node_->get_logger(),
                "tsp slow total=%.3fs write=%.3fs service=%.3fs read=%.3fs dim=%d",
                result.seconds,
                result.writeSeconds,
                result.serviceSeconds,
                result.readSeconds,
                static_cast<int>(cost.rows())
            );
        }
        if (!result.success) {
            return result;
        }
        decodeGridTour_(solution.tour, candidateIds, orderedIds);
        if (orderedIds.empty()) {
            result.success = false;
            result.failure = TspFailure::EMPTY_TOUR;
        }
        return result;
    }
    rclcpp::Node::SharedPtr node_;
    SdfMap& map_;
    FrontierManager& frontierManager_;
    HGrid& hgrid_;
    ViewNodeContext& viewNode_;
    TspSolver& tspSolver_;
    FastPlanner& planner_;
    ExplorationParams params_;
    ExplorationData data_;
    Astar astar_;
    RayCaster raycaster_;
    DiagnosticsRecorder noopDiagnostics_;
    DiagnosticsRecorder* diagnostics_{nullptr};
    bool isExplorationExhausted_{false};
    bool hasPlanningGoal_{false};
};

} // arc
