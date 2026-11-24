#pragma once

#include <pcl/point_types.h>

#include <pcl/point_cloud.h>

#include <base/ros.h>

#include <algorithm>
#include <cmath>
#include <string>

#include <base/iarc.h>

#include <env/depthprojector.h>
#include <env/multimapmanager.h>
#include <env/sdfmap.h>
#include <diagnostics/collector.h>
#include <diagnostics/explorationdiagnostics.h>
#include <diagnostics/recorder.h>
#include <explore/exploration/explorationfsm.h>
#include <explore/exploration/explorationmanager.h>
#include <explore/exploration/planretry.h>
#include <explore/frontier/frontiermanager.h>
#include <explore/frontier/viewnode.h>
#include <explore/grid/hgrid.h>
#include <explore/swarm/gridallocator.h>
#include <explore/swarm/swarmcoordinator.h>
#include <explore/tsp/tspsolver.h>
#include <fastplanner/fastplanner.h>
#include <swarmrosbridge.h>
#include <vis/explorationvisualization.h>

#include <arcmsgs/msg/drone_state.hpp>
#include <arcmsgs/msg/pair_opt.hpp>
#include <arcmsgs/msg/pair_opt_response.hpp>
#include <arcmsgs/msg/swarm_trajectory.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/string.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <pcl_conversions/pcl_conversions.h>

#if __has_include(<cv_bridge/cv_bridge.hpp>)
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif

namespace arc {

class ExplorationNode {
public:
    explicit ExplorationNode(rclcpp::Node::SharedPtr node)
        : node_(std::move(node))
        , map_(node_)
    {
        loadParams_();
        diagnosticsRecorder_.attach(&diagnostics_, explorationParams_.droneId);
        projector_.init(depthParams_);

        viewNode_ = std::make_unique<ViewNodeContext>(map_, viewNodeParams_, viewAstarOptions_);
        frontierManager_ = std::make_unique<FrontierManager>(map_, *viewNode_, frontierParams_);
        hgrid_ = std::make_unique<HGrid>(map_, gridParams_, viewAstarOptions_);
        multiMap_ = std::make_unique<MultiMapManager>(map_, multiMapParams_);
        hgrid_->setMultiMapManager(multiMap_.get());
        tspClient_ = std::make_unique<SwarmTspClient>(node_, explorationParams_.droneId);
        tspSolver_ = std::make_unique<TspSolver>(
            tspWorkDir_,
            "exploration",
            [this](const std::string& parameterFile, bool acvrp) {
                return tspClient_->solve(parameterFile, acvrp);
            }
        );
        planner_ = std::make_unique<FastPlanner>(node_, map_, plannerOptions_);
        planner_->initSwarm(explorationParams_.droneId, explorationParams_.droneNum);
        manager_ = std::make_unique<ExplorationManager>(
            node_,
            map_,
            *frontierManager_,
            *hgrid_,
            *viewNode_,
            *tspSolver_,
            *planner_,
            explorationParams_,
            astarOptions_,
            &diagnosticsRecorder_
        );
        fsm_ = std::make_unique<ExplorationFsm>(*manager_, fsmParams_, &diagnosticsRecorder_);
        diagnosticsRecorder_.initializeState(static_cast<int>(fsm_->state()), node_->now().seconds());
        visualization_ = std::make_unique<ExplorationVisualization>(node_);

        allocator_ = std::make_unique<GridAllocator>(*hgrid_, *tspSolver_);
        swarmTransport_ = std::make_unique<RosSwarmTransport>(node_);
        coordinator_ = std::make_unique<SwarmCoordinator>(
            explorationParams_.droneId,
            explorationParams_.droneNum,
            *hgrid_,
            *allocator_,
            *swarmTransport_,
            manager_->data(),
            swarmParams_,
            &diagnosticsRecorder_
        );
        mapTransport_ = std::make_unique<RosMapTransport>(node_, *multiMap_);
        mapTransport_->setStampInterval(multiMapParams_.minStampInterval);
        trajectoryTransport_ = std::make_unique<RosTrajectoryTransport>(node_);
        trajectoryTransport_->setHandler(
            [this](int droneId, const LocalTrajectory& trajectory) {
                if (droneId == explorationParams_.droneId) {
                    return;
                }
                planner_->updateSwarmTrajectory(droneId, trajectory);
                if (fsm_->state() != ExplorationState::EXEC_TRAJ) {
                    return;
                }
                double distance = 0.0;
                if (planner_->checkSwarmCollision(droneId, distance)) {
                    return;
                }
                if (diagnosticsAdapter_) {
                    CollisionInfo info;
                    info.swarm = true;
                    info.distance = distance;
                    info.otherDroneId = droneId;
                    info.position = odom_.pos;
                    diagnosticsAdapter_->collision(node_->now().seconds(), info);
                }
                fsm_->onCollisionDetected();
            }
        );

        initRos_();
        diagnosticsAdapter_ = std::make_unique<ExplorationDiagnostics>(
            node_, diagnostics_, diagnosticsRecorder_, explorationParams_.droneId);
    }

private:
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::Image,
        geometry_msgs::msg::PoseStamped
    >;

    void loadParams_() {
        depthParams_.fx = node_->declare_parameter<double>("sensor.fx", 387.229);
        depthParams_.fy = node_->declare_parameter<double>("sensor.fy", 387.229);
        depthParams_.cx = node_->declare_parameter<double>("sensor.cx", 321.046);
        depthParams_.cy = node_->declare_parameter<double>("sensor.cy", 243.449);
        depthParams_.minDepth = node_->declare_parameter<double>("sensor.minDepth", 0.2);
        depthParams_.maxDepth = node_->declare_parameter<double>("sensor.maxDepth", 4.6);
        depthParams_.margin = node_->declare_parameter<int>("sensor.margin", 2);
        depthParams_.scalingFactor = node_->declare_parameter<double>("sensor.scalingFactor", 1000.0);
        depthParams_.skipPixel = node_->declare_parameter<int>("sensor.skipPixel", 4);

        frontierParams_.clusterMin = node_->declare_parameter<int>("frontier.clusterMin", frontierParams_.clusterMin);
        frontierParams_.clusterSizeXY = node_->declare_parameter<double>("frontier.clusterSizeXY", frontierParams_.clusterSizeXY);
        frontierParams_.minCandidateDist = node_->declare_parameter<double>("frontier.minCandidateDist", frontierParams_.minCandidateDist);
        frontierParams_.minCandidateClearance = node_->declare_parameter<double>("frontier.minCandidateClearance", frontierParams_.minCandidateClearance);
        frontierParams_.candidateDphi = node_->declare_parameter<double>("frontier.candidateDphi", frontierParams_.candidateDphi);
        frontierParams_.candidateRnum = node_->declare_parameter<int>("frontier.candidateRnum", frontierParams_.candidateRnum);
        frontierParams_.candidateRmin = node_->declare_parameter<double>("frontier.candidateRmin", frontierParams_.candidateRmin);
        frontierParams_.candidateRmax = node_->declare_parameter<double>("frontier.candidateRmax", frontierParams_.candidateRmax);
        frontierParams_.downSample = node_->declare_parameter<int>("frontier.downSample", frontierParams_.downSample);
        frontierParams_.minVisibleNum = node_->declare_parameter<int>("frontier.minVisibleNum", frontierParams_.minVisibleNum);
        frontierParams_.minViewFinishFraction = node_->declare_parameter<double>("frontier.minViewFinishFraction", frontierParams_.minViewFinishFraction);
        frontierParams_.minFrontierZ = node_->declare_parameter<double>("frontier.minFrontierZ", frontierParams_.minFrontierZ);

        gridParams_.gridSize = node_->declare_parameter<double>("grid.gridSize", gridParams_.gridSize);
        gridParams_.minUnknown = node_->declare_parameter<int>("grid.minUnknown", gridParams_.minUnknown);
        gridParams_.minFree = node_->declare_parameter<int>("grid.minFree", gridParams_.minFree);
        gridParams_.consistentCost = node_->declare_parameter<double>("grid.consistentCost", gridParams_.consistentCost);
        gridParams_.consistentCost2 = node_->declare_parameter<double>("grid.consistentCost2", gridParams_.consistentCost2);

        explorationParams_.refinedNum = node_->declare_parameter<int>("exploration.refinedNum", explorationParams_.refinedNum);
        explorationParams_.refinedRadius = node_->declare_parameter<double>("exploration.refinedRadius", explorationParams_.refinedRadius);
        explorationParams_.topViewNum = node_->declare_parameter<int>("exploration.topViewNum", explorationParams_.topViewNum);
        explorationParams_.maxDecay = node_->declare_parameter<double>("exploration.maxDecay", explorationParams_.maxDecay);
        explorationParams_.droneId = node_->declare_parameter<int>("exploration.droneId", explorationParams_.droneId);
        explorationParams_.droneNum = node_->declare_parameter<int>("exploration.droneNum", explorationParams_.droneNum);
        explorationParams_.initPlanNum = node_->declare_parameter<int>("exploration.initPlanNum", explorationParams_.initPlanNum);
        explorationParams_.replanTime = node_->declare_parameter<double>("exploration.replanTime", explorationParams_.replanTime);
        explorationParams_.radiusFar = node_->declare_parameter<double>("exploration.radiusFar", explorationParams_.radiusFar);
        explorationParams_.radiusClose = node_->declare_parameter<double>("exploration.radiusClose", explorationParams_.radiusClose);
        explorationParams_.shortenDist = node_->declare_parameter<double>("exploration.shortenDist", explorationParams_.shortenDist);
        explorationParams_.planRetryBaseDelay = node_->declare_parameter<double>(
            "exploration.planRetryBaseDelay", explorationParams_.planRetryBaseDelay);
        explorationParams_.planRetryMaxDelay = node_->declare_parameter<double>(
            "exploration.planRetryMaxDelay", explorationParams_.planRetryMaxDelay);
        explorationParams_.planRetryBackoffFactor = node_->declare_parameter<double>(
            "exploration.planRetryBackoffFactor", explorationParams_.planRetryBackoffFactor);
        explorationParams_.planRefreshAfterFailures = node_->declare_parameter<int>(
            "exploration.planRefreshAfterFailures", explorationParams_.planRefreshAfterFailures);
        explorationParams_.planTrajectory = node_->declare_parameter<bool>("exploration.planTrajectory", explorationParams_.planTrajectory);
        explorationParams_.finishUnreachableFrontier = node_->declare_parameter<bool>(
            "exploration.finishUnreachableFrontier",
            explorationParams_.finishUnreachableFrontier
        );
        autoTrigger_ = node_->declare_parameter<bool>("exploration.autoTrigger", autoTrigger_);

        multiMapParams_.droneId = explorationParams_.droneId;
        multiMapParams_.droneNum = explorationParams_.droneNum;
        const int chunkSize = static_cast<int>(
            node_->declare_parameter<int>("mapchunk.chunkSize", static_cast<int>(multiMapParams_.chunkSize))
        );
        multiMapParams_.chunkSize = static_cast<size_t>(std::max(1, chunkSize));
        multiMapParams_.minStampInterval = node_->declare_parameter<double>(
            "mapchunk.minStampInterval", multiMapParams_.minStampInterval
        );

        swarmParams_.droneStateInterval = node_->declare_parameter<double>("swarm.droneStateInterval", swarmParams_.droneStateInterval);
        swarmParams_.pairOptSelectionInterval = node_->declare_parameter<double>("swarm.optInterval", swarmParams_.pairOptSelectionInterval);
        swarmParams_.pairOptAttemptInterval = node_->declare_parameter<double>("swarm.attemptInterval", swarmParams_.pairOptAttemptInterval);
        swarmParams_.pairOptCooldownInterval = node_->declare_parameter<double>("swarm.pairOptInterval", swarmParams_.pairOptCooldownInterval);
        swarmParams_.pairOptCostMargin = node_->declare_parameter<double>("swarm.costMargin", swarmParams_.pairOptCostMargin);
        swarmParams_.droneStateTimeout = node_->declare_parameter<double>("swarm.stateTimeout", swarmParams_.droneStateTimeout);
        swarmParams_.pairOptResponseTimeout = node_->declare_parameter<double>("swarm.responseTimeout", swarmParams_.pairOptResponseTimeout);
        swarmParams_.repeatSendCount = node_->declare_parameter<int>("swarm.repeatSendNum", swarmParams_.repeatSendCount);

        fsmParams_.initWaitDuration = node_->declare_parameter<double>("fsm.initWaitDuration", fsmParams_.initWaitDuration);
        fsmParams_.replanThresh1 = node_->declare_parameter<double>("fsm.replanThresh1", fsmParams_.replanThresh1);
        fsmParams_.replanThresh2 = node_->declare_parameter<double>("fsm.replanThresh2", fsmParams_.replanThresh2);
        fsmParams_.replanThresh3 = node_->declare_parameter<double>("fsm.replanThresh3", fsmParams_.replanThresh3);
        fsmParams_.idleDuration = node_->declare_parameter<double>("fsm.idleDuration", fsmParams_.idleDuration);

        astarOptions_.resolution = node_->declare_parameter<double>("astar.resolution", astarOptions_.resolution);
        astarOptions_.lambdaHeu = node_->declare_parameter<double>("astar.lambdaHeu", astarOptions_.lambdaHeu);
        astarOptions_.maxSearchTime = node_->declare_parameter<double>("astar.maxSearchTime", astarOptions_.maxSearchTime);
        astarOptions_.collisionCheckStep = node_->declare_parameter<double>("astar.collisionCheckStep", astarOptions_.collisionCheckStep);
        astarOptions_.maxNodes = static_cast<size_t>(node_->declare_parameter<int>("astar.maxNodes", static_cast<int>(astarOptions_.maxNodes)));

        viewAstarOptions_.resolution = node_->declare_parameter<double>("astarView.resolution", viewAstarOptions_.resolution);
        viewAstarOptions_.lambdaHeu = node_->declare_parameter<double>("astarView.lambdaHeu", viewAstarOptions_.lambdaHeu);
        viewAstarOptions_.maxSearchTime = node_->declare_parameter<double>("astarView.maxSearchTime", viewAstarOptions_.maxSearchTime);
        viewAstarOptions_.collisionCheckStep = node_->declare_parameter<double>("astarView.collisionCheckStep", viewAstarOptions_.collisionCheckStep);
        viewAstarOptions_.maxNodes = static_cast<size_t>(node_->declare_parameter<int>("astarView.maxNodes", static_cast<int>(viewAstarOptions_.maxNodes)));

        viewNodeParams_.vm = node_->declare_parameter<double>("viewpoint.vm", viewNodeParams_.vm);
        viewNodeParams_.yd = node_->declare_parameter<double>("viewpoint.yd", viewNodeParams_.yd);
        viewNodeParams_.wDir = node_->declare_parameter<double>("viewpoint.wDir", viewNodeParams_.wDir);
        viewNodeParams_.noPathPenaltyFactor = node_->declare_parameter<double>("viewpoint.noPathPenaltyFactor", viewNodeParams_.noPathPenaltyFactor);
        viewNodeParams_.minNoPathPenaltyLength = node_->declare_parameter<double>("viewpoint.minNoPathPenaltyLength", viewNodeParams_.minNoPathPenaltyLength);
        viewNodeParams_.noPathPenaltyLength = node_->declare_parameter<double>("viewpoint.noPathPenaltyLength", viewNodeParams_.noPathPenaltyLength);

        plannerOptions_.controlPointDistance = node_->declare_parameter<double>("planner.controlPointDistance", plannerOptions_.controlPointDistance);
        plannerOptions_.maxYawRate = node_->declare_parameter<double>("planner.maxYawRate", plannerOptions_.maxYawRate);
        plannerOptions_.relaxTime2 = node_->declare_parameter<double>("planner.relaxTime2", plannerOptions_.relaxTime2);
        plannerOptions_.minTime = node_->declare_parameter<bool>("planner.minTime", plannerOptions_.minTime);
        plannerOptions_.optimistic = node_->declare_parameter<bool>("planner.optimistic", plannerOptions_.optimistic);
        plannerOptions_.collisionCheckRadius = node_->declare_parameter<double>("planner.collisionCheckRadius", plannerOptions_.collisionCheckRadius);
        plannerOptions_.collisionCheckStep = node_->declare_parameter<double>("planner.collisionCheckStep", plannerOptions_.collisionCheckStep);
        plannerOptions_.swarmSafeDistance = node_->declare_parameter<double>("planner.swarmSafeDistance", plannerOptions_.swarmSafeDistance);

        KinoAstarOptions& kino = plannerOptions_.kino;
        kino.resolution = node_->declare_parameter<double>("kino.resolution", kino.resolution);
        kino.maxTau = node_->declare_parameter<double>("kino.maxTau", kino.maxTau);
        kino.initMaxTau = node_->declare_parameter<double>("kino.initMaxTau", kino.initMaxTau);
        kino.maxVel = node_->declare_parameter<double>("kino.maxVel", kino.maxVel);
        kino.velMargin = node_->declare_parameter<double>("kino.velMargin", kino.velMargin);
        kino.maxAcc = node_->declare_parameter<double>("kino.maxAcc", kino.maxAcc);
        kino.timeWeight = node_->declare_parameter<double>("kino.timeWeight", kino.timeWeight);
        kino.horizon = node_->declare_parameter<double>("kino.horizon", kino.horizon);
        kino.lambdaHeu = node_->declare_parameter<double>("kino.lambdaHeu", kino.lambdaHeu);
        kino.collisionCheckStep = node_->declare_parameter<double>("kino.collisionCheckStep", kino.collisionCheckStep);
        kino.maxSearchTime = node_->declare_parameter<double>("kino.maxSearchTime", kino.maxSearchTime);
        kino.pathSampleInterval = node_->declare_parameter<double>("kino.pathSampleInterval", kino.pathSampleInterval);
        kino.maxNodes = static_cast<size_t>(node_->declare_parameter<int>("kino.maxNodes", static_cast<int>(kino.maxNodes)));

        YawOptParams& yaw = plannerOptions_.yaw;
        yaw.smoothWeight = node_->declare_parameter<double>("yaw.smoothWeight", yaw.smoothWeight);
        yaw.startWeight = node_->declare_parameter<double>("yaw.startWeight", yaw.startWeight);
        yaw.endWeight = node_->declare_parameter<double>("yaw.endWeight", yaw.endWeight);
        yaw.waypointWeight = node_->declare_parameter<double>("yaw.waypointWeight", yaw.waypointWeight);
        yaw.startPositionWeight = node_->declare_parameter<double>("yaw.startPositionWeight", yaw.startPositionWeight);
        yaw.maxEval = node_->declare_parameter<int>("yaw.maxEval", yaw.maxEval);
        yaw.maxTime = node_->declare_parameter<double>("yaw.maxTime", yaw.maxTime);
        yaw.xtolRel = node_->declare_parameter<double>("yaw.xtolRel", yaw.xtolRel);

        tspWorkDir_ = node_->declare_parameter<std::string>("tsp.workDir", "/tmp/arc_exploration_1");

        diagnostics_.setEnabled(node_->declare_parameter<bool>("diagnostics.enabled", true));
        diagPublishInterval_ = node_->declare_parameter<double>("diagnostics.publishInterval", 1.0);
        const int eventQueueSize = node_->declare_parameter<int>("diagnostics.eventQueueSize", 1000);
        diagnostics_.setEventQueueSize(static_cast<size_t>(std::max(1, eventQueueSize)));
    }

    void initRos_() {
        depthSub_ = std::make_unique<message_filters::Subscriber<sensor_msgs::msg::Image>>(
            node_.get(), "depth", rclcpp::QoS(50).get_rmw_qos_profile()
        );
        poseSub_ = std::make_unique<message_filters::Subscriber<geometry_msgs::msg::PoseStamped>>(
            node_.get(), "pose", rclcpp::QoS(50).get_rmw_qos_profile()
        );
        sync_ = std::make_unique<message_filters::Synchronizer<SyncPolicy>>(
            SyncPolicy(100), *depthSub_, *poseSub_
        );
        sync_->registerCallback(
            [this](const sensor_msgs::msg::Image::ConstSharedPtr& image,
                   const geometry_msgs::msg::PoseStamped::ConstSharedPtr& pose,
                   const auto&...) {
                depthPoseCallback_(image, pose);
            }
        );

        odomSub_ = node_->create_subscription<nav_msgs::msg::Odometry>(
            "odometry",
            50,
            [this](const nav_msgs::msg::Odometry::SharedPtr msg) {
                odomCallback_(msg);
            }
        );
        localTrajectoryPub_ = node_->create_publisher<arcmsgs::msg::SwarmTrajectory>(
            "/planning/bspline", rclcpp::QoS(100)
        );
        statePub_ = node_->create_publisher<std_msgs::msg::String>("/exploration/state", 10);
        localOccupancyPub_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>("/exploration/occupancy_local", 10);
        allOccupancyPub_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>("/exploration/occupancy_all", 10);
        inflatePub_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>("/exploration/occupancy_inflate", 10);
        depthCloudPub_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>("/exploration/depth_cloud", 10);
        droneStateSub_ = node_->create_subscription<arcmsgs::msg::DroneState>(
            "/swarm_expl/drone_state",
            rclcpp::QoS(10),
            [this](const arcmsgs::msg::DroneState::SharedPtr msg) {
                coordinator_->onDroneState(SwarmRosConverter::fromMessage(*msg));
            }
        );
        pairOptSub_ = node_->create_subscription<arcmsgs::msg::PairOpt>(
            "/swarm_expl/pair_opt",
            rclcpp::QoS(100),
            [this](const arcmsgs::msg::PairOpt::SharedPtr msg) {
                coordinator_->onPairOptRequest(SwarmRosConverter::fromMessage(*msg));
            }
        );
        pairOptResponseSub_ = node_->create_subscription<arcmsgs::msg::PairOptResponse>(
            "/swarm_expl/pair_opt_res",
            rclcpp::QoS(100),
            [this](const arcmsgs::msg::PairOptResponse::SharedPtr msg) {
                coordinator_->onPairOptResponse(SwarmRosConverter::fromMessage(*msg));
            }
        );
        goalSub_ = node_->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/goal_pose",
            1,
            [this](const geometry_msgs::msg::PoseStamped::SharedPtr) {
                triggerRequested_ = true;
            }
        );

        execTimer_ = node_->create_wall_timer(
            std::chrono::milliseconds(10),
            [this]() { execTimerCallback_(); }
        );
        safetyTimer_ = node_->create_wall_timer(
            std::chrono::milliseconds(50),
            [this]() { safetyTimerCallback_(); }
        );
        frontierTimer_ = node_->create_wall_timer(
            std::chrono::milliseconds(500),
            [this]() { frontierTimerCallback_(); }
        );
        visualizationTimer_ = node_->create_wall_timer(
            std::chrono::milliseconds(200),
            [this]() { visualizationTimerCallback_(); }
        );
        swarmTimer_ = node_->create_wall_timer(
            std::chrono::milliseconds(40),
            [this]() { swarmTimerCallback_(); }
        );
        mapChunkTimer_ = node_->create_wall_timer(
            std::chrono::milliseconds(100),
            [this]() { mapChunkTimerCallback_(); }
        );
        diagnosticsTimer_ = node_->create_wall_timer(
            std::chrono::milliseconds(50),
            [this]() { diagnosticsTimerCallback_(); }
        );
    }

    void odomCallback_(const nav_msgs::msg::Odometry::SharedPtr msg) {
        odom_.pos = Eigen::Vector3d(
            msg->pose.pose.position.x,
            msg->pose.pose.position.y,
            msg->pose.pose.position.z
        );
        odom_.vel = Eigen::Vector3d(
            msg->twist.twist.linear.x,
            msg->twist.twist.linear.y,
            msg->twist.twist.linear.z
        );
        odom_.acc.setZero();
        const Eigen::Quaterniond orientation(
            msg->pose.pose.orientation.w,
            msg->pose.pose.orientation.x,
            msg->pose.pose.orientation.y,
            msg->pose.pose.orientation.z
        );
        const Eigen::Matrix3d rotation = orientation.toRotationMatrix();
        const double yaw = std::atan2(rotation(1, 0), rotation(0, 0));
        odom_.yaw = Eigen::Vector3d(yaw, 0.0, 0.0);
        odomReady_ = true;
    }

    void depthPoseCallback_(
        const sensor_msgs::msg::Image::ConstSharedPtr& image,
        const geometry_msgs::msg::PoseStamped::ConstSharedPtr& pose
    ) {
        const Eigen::Vector3d cameraPos(
            pose->pose.position.x,
            pose->pose.position.y,
            pose->pose.position.z
        );
        if (!map_.params().isInMap(cameraPos)) {
            return;
        }
        cv_bridge::CvImageConstPtr imagePtr;
        try {
            imagePtr = cv_bridge::toCvShare(image, image->encoding);
        } catch (const cv_bridge::Exception& error) {
            RCLCPP_ERROR(node_->get_logger(), "cv_bridge: %s", error.what());
            return;
        }
        cv::Mat converted;
        const cv::Mat* depthImage = &imagePtr->image;
        if (image->encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
            imagePtr->image.convertTo(converted, CV_16UC1, projector_.scalingFactor());
            depthImage = &converted;
        }
        if (depthImage->type() != CV_16UC1) {
            RCLCPP_ERROR(node_->get_logger(), "unsupported depth image type: %d", depthImage->type());
            return;
        }
        const Eigen::Quaterniond cameraQ(
            pose->pose.orientation.w,
            pose->pose.orientation.x,
            pose->pose.orientation.y,
            pose->pose.orientation.z
        );
        const Eigen::Matrix3d cameraR = cameraQ.toRotationMatrix();
        projector_.project(*depthImage, cameraPos, cameraR);
        map_.updateFramePoints(projector_.cloud(), projector_.size(), cameraPos);
        multiMap_->updateMapChunk(map_.takeNewVoxelIndexes());
        sensorReady_ = true;
    }

    void execTimerCallback_() {
        if (!odomReady_ || !sensorReady_) {
            return;
        }
        const double now = node_->now().seconds();
        if (fsm_->state() == ExplorationState::WAIT_TRIGGER && (autoTrigger_ || triggerRequested_)) {
            triggerRequested_ = false;
            fsm_->trigger();
        }
        if (fsm_->state() == ExplorationState::EXEC_TRAJ) {
            const auto& trajectory = planner_->localTrajectory();
            if (trajectory.valid() && trajectory.startTime > 0.0) {
                const double tc = now - trajectory.startTime;
                if (tc > fsmParams_.replanThresh2 && manager_->isFrontierCovered()) {
                    fsm_->onFrontierCovered();
                }
            }
            if (!trajectoryFinishedSent_ && trajectoryNearEnd_(now)) {
                trajectoryFinishedSent_ = true;
                fsm_->onTrajectoryFinished();
            }
        }
        if (fsm_->state() == ExplorationState::PUB_TRAJ) {
            publishPendingTrajectory_(now);
        }
        if (fsm_->state() == ExplorationState::PLAN_TRAJ) {
            tryPlan_(now);
        } else {
            const ExplorationState previousState = fsm_->state();
            fsm_->tick(now, odom_);
            if (fsm_->state() == ExplorationState::PLAN_TRAJ &&
                previousState != ExplorationState::PLAN_TRAJ) {
                firstPlanAttempt_ = true;
            }
        }
    }

    void safetyTimerCallback_() {
        if (fsm_->state() != ExplorationState::EXEC_TRAJ) {
            collisionActive_ = false;
            return;
        }
        double staticDistance = 0.0;
        if (!planner_->checkTrajectoryCollision(staticDistance)) {
            if (!collisionActive_) {
                CollisionInfo info;
                info.swarm = false;
                info.distance = staticDistance;
                info.position = odom_.pos;
                diagnosticsAdapter_->collision(node_->now().seconds(), info);
                collisionActive_ = true;
                fsm_->onCollisionDetected();
            }
            return;
        }
        collisionActive_ = false;
    }

    void frontierTimerCallback_() {
        if (!odomReady_ || !sensorReady_) {
            return;
        }
        if (fsm_->state() == ExplorationState::INIT ||
            fsm_->state() == ExplorationState::WAIT_TRIGGER) {
            manager_->refreshFrontiers(odom_.pos);
            std::vector<int> gridIds;
            manager_->updateGridTour(odom_, gridIds, true);
        }
    }

    void swarmTimerCallback_() {
        if (!odomReady_ ||
            fsm_->state() == ExplorationState::INIT ||
            fsm_->state() == ExplorationState::FINISH) {
            return;
        }
        const double now = node_->now().seconds();
        coordinator_->setEgoState(odom_.pos, odom_.vel, odom_.yaw[0]);
        const ExplorationState state = fsm_->state();
        const bool allowInitiate =
            state == ExplorationState::IDLE ||
            state == ExplorationState::PLAN_TRAJ ||
            state == ExplorationState::WAIT_TRIGGER;
        coordinator_->tick(now, allowInitiate);
        if (coordinator_->consumeReallocation() &&
            fsm_->state() == ExplorationState::IDLE &&
            !manager_->data().gridTour.assignedGridIds.empty()) {
            fsm_->trigger("reallocation");
        }
        if (fsm_->state() == ExplorationState::EXEC_TRAJ) {
            const auto& trajectory = planner_->localTrajectory();
            if (trajectory.valid() &&
                (lastSwarmTrajectoryPublishTime_ < 0.0 ||
                 now - lastSwarmTrajectoryPublishTime_ > 0.5)) {
                lastSwarmTrajectoryPublishTime_ = now;
                trajectoryTransport_->publish(explorationParams_.droneId, trajectory);
            }
        }
    }

    void mapChunkTimerCallback_() {
        if (!odomReady_) {
            return;
        }
        mapTransport_->tick(node_->now().seconds());
    }

    void diagnosticsTimerCallback_() {
        if (!diagnosticsAdapter_ || !diagnostics_.enabled()) {
            return;
        }
        const double now = node_->now().seconds();
        logFrontierStats_(now);
        diagnosticsAdapter_->tick(
            now,
            diagPublishInterval_,
            manager_->data(),
            frontierManager_->activeFrontierCount(),
            frontierManager_->dormantFrontierCount()
        );
    }

    void logFrontierStats_(double now) {
        if (lastFrontierStatsTime_ >= 0.0 && now - lastFrontierStatsTime_ < 10.0) {
            return;
        }
        lastFrontierStatsTime_ = now;
        const auto& frontiers = frontierManager_->frontiers();
        if (frontiers.empty()) {
            return;
        }
        std::vector<int> counts;
        counts.reserve(frontiers.size());
        for (const auto& frontier : frontiers) {
            counts.push_back(static_cast<int>(frontier.cells.size()));
        }
        std::sort(counts.begin(), counts.end());
        const int count = static_cast<int>(counts.size());
        RCLCPP_INFO(
            node_->get_logger(),
            "frontier stats n=%d min=%d p50=%d p90=%d max=%d",
            count,
            counts.front(),
            counts[count / 2],
            counts[count * 9 / 10],
            counts.back()
        );
    }

    void visualizationTimerCallback_() {
        if (!sensorReady_) {
            return;
        }
        visualization_->drawFrontiers(frontierManager_->frontiers());
        visualization_->drawGrid(*hgrid_, explorationParams_.droneId);
        const int colorNum = std::max(explorationParams_.droneNum, 5);
        visualization_->drawGridTour(
            manager_->data(), explorationParams_.droneId, colorNum);
        visualization_->drawTrajectory(
            planner_->localTrajectory(), explorationParams_.droneId, colorNum);

        std_msgs::msg::String state;
        state.data = stateName_();
        statePub_->publish(state);

        publishDepthCloud_();
        publishMapLayers_();
    }

    OdomState makePlanningStart_(double planStartTime) const {
        OdomState planningStart = odom_;
        const auto& previous = planner_->localTrajectory();
        if (!previous.valid() || previous.startTime <= 0.0) {
            return planningStart;
        }
        const double elapsed = planStartTime - previous.startTime;
        const double sampleTime = std::max(0.0, std::min(elapsed, previous.duration));
        if (previous.position.valid()) {
            planningStart.pos = Eigen::Vector3d(previous.position.evalt(sampleTime).head<3>());
        }
        if (previous.velocity.valid()) {
            planningStart.vel = Eigen::Vector3d(previous.velocity.evalt(sampleTime).head<3>());
        }
        if (previous.acceleration.valid()) {
            planningStart.acc = Eigen::Vector3d(previous.acceleration.evalt(sampleTime).head<3>());
        }
        if (previous.yaw.valid()) {
            planningStart.yaw(0) = previous.yaw.evalt(sampleTime)(0);
        }
        if (previous.yawRate.valid()) {
            planningStart.yaw(1) = previous.yawRate.evalt(sampleTime)(0);
        }
        if (previous.yawAcceleration.valid()) {
            planningStart.yaw(2) = previous.yawAcceleration.evalt(sampleTime)(0);
        }
        return planningStart;
    }

    void tryPlan_(double now) {
        if (fsm_->state() != ExplorationState::PLAN_TRAJ) {
            return;
        }
        if (!planRetry_.shouldAttempt(now)) {
            return;
        }
        planRetry_.onAttempt(now);
        const double planStartTime = now + explorationParams_.replanTime;
        OdomState planningStart = makePlanningStart_(planStartTime);
        if (!map_.params().isInBox(planningStart.pos)) {
            if (map_.params().isInBox(odom_.pos)) {
                planningStart = odom_;
            } else {
                return;
            }
        }
        const bool updateFrontier = firstPlanAttempt_;
        fsm_->tick(now, odom_, &planningStart, updateFrontier);
        firstPlanAttempt_ = false;
        if (fsm_->state() == ExplorationState::PUB_TRAJ) {
            planRetry_.onSuccess();
            plannedStartTime_ = std::max(planStartTime, node_->now().seconds());
            planner_->setLocalTrajectoryStartTime(plannedStartTime_);
            return;
        }
        if (fsm_->state() == ExplorationState::PLAN_TRAJ) {
            planRetry_.onFailure(
                now,
                explorationParams_.planRetryBaseDelay,
                explorationParams_.planRetryMaxDelay,
                explorationParams_.planRetryBackoffFactor);
            if (planRetry_.consecutiveFailures() >= explorationParams_.planRefreshAfterFailures) {
                firstPlanAttempt_ = true;
            }
            return;
        }
        planRetry_.onNoGrid();
    }

    void publishPendingTrajectory_(double now) {
        if (fsm_->state() != ExplorationState::PUB_TRAJ) {
            return;
        }
        const auto& trajectory = planner_->localTrajectory();
        if (!trajectory.valid() || trajectory.id == publishedTrajectoryId_) {
            return;
        }
        if (trajectory.startTime > now) {
            return;
        }
        auto publishTimer = diagnosticsRecorder_.stage(DiagStage::TRAJ_PUBLISH);
        publishedTrajectoryId_ = trajectory.id;
        trajectoryFinishedSent_ = false;
        publishLocalTrajectory_(trajectory);
        trajectoryTransport_->publish(explorationParams_.droneId, trajectory);
        lastSwarmTrajectoryPublishTime_ = now;
        fsm_->onTrajectoryPublished(trajectory.startTime);
        diagnosticsAdapter_->trajectoryPublished(now, trajectory.id, trajectory.duration);
    }

    bool trajectoryNearEnd_(double now) const {
        const auto& trajectory = planner_->localTrajectory();
        if (!trajectory.valid() || trajectory.startTime <= 0.0) {
            return false;
        }
        const double tc = now - trajectory.startTime;
        return tc >= 0.0 && trajectory.duration - tc < fsmParams_.replanThresh1;
    }

    void publishLocalTrajectory_(const LocalTrajectory& trajectory) {
        if (!trajectory.valid()) {
            return;
        }
        arcmsgs::msg::SwarmTrajectory message;
        SwarmRosConverter::toMessage(explorationParams_.droneId, trajectory, message);
        localTrajectoryPub_->publish(message);
    }

    void publishCloud_(
        const pcl::PointCloud<pcl::PointXYZ>& cloud,
        const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr& publisher
    ) {
        sensor_msgs::msg::PointCloud2 message;
        pcl::toROSMsg(cloud, message);
        message.header.stamp = node_->now();
        message.header.frame_id = "world";
        publisher->publish(message);
    }

    void publishDepthCloud_() {
        if (projector_.size() == 0) {
            return;
        }
        publishCloud_(projector_.cloud(), depthCloudPub_);
    }

    void publishMapLayers_() {
        const MapParams& params = map_.params();
        const MapData& data = map_.data();
        if (data.updateRevision == publishedMapRevision_) {
            return;
        }
        publishedMapRevision_ = data.updateRevision;
        if ((data.allMax.array() < data.allMin.array()).any()) {
            publishCloud_(pcl::PointCloud<pcl::PointXYZ>(), localOccupancyPub_);
            publishCloud_(pcl::PointCloud<pcl::PointXYZ>(), allOccupancyPub_);
            publishCloud_(pcl::PointCloud<pcl::PointXYZ>(), inflatePub_);
            return;
        }

        Eigen::Vector3i localMin = data.localBoundMin;
        Eigen::Vector3i localMax = data.localBoundMax;
        params.boundIndex(localMin);
        params.boundIndex(localMax);

        pcl::PointCloud<pcl::PointXYZ>& localCloud = localCloud_;
        pcl::PointCloud<pcl::PointXYZ>& allCloud = allCloud_;
        pcl::PointCloud<pcl::PointXYZ>& inflateCloud = inflateCloud_;
        localCloud.clear();
        allCloud.clear();
        inflateCloud.clear();

        const double resolution = params.resolution;
        const double originX = params.origin(0);
        const double originY = params.origin(1);
        const double originZ = params.origin(2);
        for (int x = localMin(0); x <= localMax(0); ++ x) {
            const float px = static_cast<float>((x + 0.5) * resolution + originX);
            for (int y = localMin(1); y <= localMax(1); ++ y) {
                const float py = static_cast<float>((y + 0.5) * resolution + originY);
                size_t base = params.getVoxelIdx(x, y, localMin(2));
                for (int z = localMin(2); z <= localMax(2); ++ z, ++ base) {
                    const pcl::PointXYZ point(
                        px,
                        py,
                        static_cast<float>((z + 0.5) * resolution + originZ)
                    );
                    if (data.occupied[base]) {
                        inflateCloud.push_back(point);
                    }
                    if (data.occupancyLogits[base] > params.locc) {
                        localCloud.push_back(point);
                    }
                }
            }
        }

        Eigen::Vector3i allMin = params.p2i(data.allMin);
        Eigen::Vector3i allMax = params.p2i(data.allMax);
        params.boundIndex(allMin);
        params.boundIndex(allMax);
        for (int x = allMin(0); x <= allMax(0); ++ x) {
            const float px = static_cast<float>((x + 0.5) * resolution + originX);
            for (int y = allMin(1); y <= allMax(1); ++ y) {
                const float py = static_cast<float>((y + 0.5) * resolution + originY);
                size_t base = params.getVoxelIdx(x, y, allMin(2));
                for (int z = allMin(2); z <= allMax(2); ++ z, ++ base) {
                    if (data.occupancyLogits[base] > params.locc) {
                        allCloud.push_back(pcl::PointXYZ(
                            px,
                            py,
                            static_cast<float>((z + 0.5) * resolution + originZ)
                        ));
                    }
                }
            }
        }

        publishCloud_(localCloud, localOccupancyPub_);
        publishCloud_(allCloud, allOccupancyPub_);
        publishCloud_(inflateCloud, inflatePub_);
        logMapStats_(allCloud, localCloud, inflateCloud, data, node_->now().seconds());
    }

    void logMapStats_(
        const pcl::PointCloud<pcl::PointXYZ>& allCloud,
        const pcl::PointCloud<pcl::PointXYZ>& localCloud,
        const pcl::PointCloud<pcl::PointXYZ>& inflateCloud,
        const MapData& data,
        double now
    ) {
        if (lastMapStatsTime_ >= 0.0 && now - lastMapStatsTime_ < 10.0) {
            return;
        }
        lastMapStatsTime_ = now;
        if (allCloud.empty()) {
            RCLCPP_INFO(
                node_->get_logger(),
                "map stats all=0 local=%zu inflate=%zu",
                localCloud.size(),
                inflateCloud.size()
            );
            return;
        }
        float minX = allCloud.front().x;
        float maxX = allCloud.front().x;
        float minY = allCloud.front().y;
        float maxY = allCloud.front().y;
        float minZ = allCloud.front().z;
        float maxZ = allCloud.front().z;
        std::vector<int> zHist;
        zHist.assign(8, 0);
        for (const auto& point : allCloud.points) {
            minX = std::min(minX, point.x);
            maxX = std::max(maxX, point.x);
            minY = std::min(minY, point.y);
            maxY = std::max(maxY, point.y);
            minZ = std::min(minZ, point.z);
            maxZ = std::max(maxZ, point.z);
            const int bucket = static_cast<int>(std::floor(point.z * 2.0));
            if (bucket >= 0 && bucket < static_cast<int>(zHist.size())) {
                zHist[static_cast<size_t>(bucket)] += 1;
            }
        }
        std::string hist;
        for (size_t i = 0; i < zHist.size(); ++ i) {
            if (!hist.empty()) {
                hist += ",";
            }
            hist += std::to_string(zHist[i]);
        }
        RCLCPP_INFO(
            node_->get_logger(),
            "map stats all=%zu local=%zu inflate=%zu range=(%.1f..%.1f,%.1f..%.1f,%.1f..%.1f) "
            "allMin=(%.1f,%.1f,%.1f) allMax=(%.1f,%.1f,%.1f) zHist=%s",
            allCloud.size(),
            localCloud.size(),
            inflateCloud.size(),
            minX, maxX, minY, maxY, minZ, maxZ,
            data.allMin(0), data.allMin(1), data.allMin(2),
            data.allMax(0), data.allMax(1), data.allMax(2),
            hist.c_str()
        );
    }

    std::string stateName_() const {
        switch (fsm_->state()) {
            case ExplorationState::INIT: return "INIT";
            case ExplorationState::WAIT_TRIGGER: return "WAIT_TRIGGER";
            case ExplorationState::PLAN_TRAJ: return "PLAN_TRAJ";
            case ExplorationState::PUB_TRAJ: return "PUB_TRAJ";
            case ExplorationState::EXEC_TRAJ: return "EXEC_TRAJ";
            case ExplorationState::IDLE: return "IDLE";
            case ExplorationState::FINISH: return "FINISH";
            default: return "UNKNOWN";
        }
    }

private:
    rclcpp::Node::SharedPtr node_;
    SdfMap map_;
    DepthProjector projector_;
    DepthProjectorParams depthParams_;
    FrontierParams frontierParams_;
    GridParams gridParams_;
    ExplorationParams explorationParams_;
    ExplorationFsmParams fsmParams_;
    AstarOptions astarOptions_;
    AstarOptions viewAstarOptions_{0.3, 10000.0, 0.001, 0.1, 1000000};
    ViewNodeParams viewNodeParams_;
    FastPlannerOptions plannerOptions_;
    MultiMapManagerParams multiMapParams_;
    SwarmParams swarmParams_;
    std::string tspWorkDir_;
    bool autoTrigger_{true};
    DiagnosticsCollector diagnostics_;
    DiagnosticsRecorder diagnosticsRecorder_;
    std::unique_ptr<ExplorationDiagnostics> diagnosticsAdapter_;
    double diagPublishInterval_{1.0};
    double lastFrontierStatsTime_{-1.0};
    double lastMapStatsTime_{-1.0};

    std::unique_ptr<ViewNodeContext> viewNode_;
    std::unique_ptr<FrontierManager> frontierManager_;
    std::unique_ptr<HGrid> hgrid_;
    std::unique_ptr<MultiMapManager> multiMap_;
    std::unique_ptr<SwarmTspClient> tspClient_;
    std::unique_ptr<TspSolver> tspSolver_;
    std::unique_ptr<FastPlanner> planner_;
    std::unique_ptr<ExplorationManager> manager_;
    std::unique_ptr<ExplorationFsm> fsm_;
    std::unique_ptr<ExplorationVisualization> visualization_;
    std::unique_ptr<GridAllocator> allocator_;
    std::unique_ptr<RosSwarmTransport> swarmTransport_;
    std::unique_ptr<SwarmCoordinator> coordinator_;
    std::unique_ptr<RosMapTransport> mapTransport_;
    std::unique_ptr<RosTrajectoryTransport> trajectoryTransport_;

    OdomState odom_;
    bool odomReady_{false};
    bool sensorReady_{false};
    bool triggerRequested_{false};
    int publishedTrajectoryId_{0};
    bool trajectoryFinishedSent_{false};
    bool firstPlanAttempt_{false};
    PlanRetryPolicy planRetry_;
    bool collisionActive_{false};
    double plannedStartTime_{-1.0};
    double lastSwarmTrajectoryPublishTime_{-1.0};
    size_t publishedMapRevision_{0};

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odomSub_;
    std::unique_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> depthSub_;
    std::unique_ptr<message_filters::Subscriber<geometry_msgs::msg::PoseStamped>> poseSub_;
    std::unique_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
    rclcpp::Publisher<arcmsgs::msg::SwarmTrajectory>::SharedPtr localTrajectoryPub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr statePub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr localOccupancyPub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr allOccupancyPub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr inflatePub_;
    pcl::PointCloud<pcl::PointXYZ> localCloud_;
    pcl::PointCloud<pcl::PointXYZ> allCloud_;
    pcl::PointCloud<pcl::PointXYZ> inflateCloud_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr depthCloudPub_;
    rclcpp::Subscription<arcmsgs::msg::DroneState>::SharedPtr droneStateSub_;
    rclcpp::Subscription<arcmsgs::msg::PairOpt>::SharedPtr pairOptSub_;
    rclcpp::Subscription<arcmsgs::msg::PairOptResponse>::SharedPtr pairOptResponseSub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goalSub_;
    rclcpp::TimerBase::SharedPtr execTimer_;
    rclcpp::TimerBase::SharedPtr safetyTimer_;
    rclcpp::TimerBase::SharedPtr frontierTimer_;
    rclcpp::TimerBase::SharedPtr visualizationTimer_;
    rclcpp::TimerBase::SharedPtr swarmTimer_;
    rclcpp::TimerBase::SharedPtr mapChunkTimer_;
    rclcpp::TimerBase::SharedPtr diagnosticsTimer_;
};

} // arc
