#pragma once

#include <base/iarc.h>
#include <env/multimapmanager.h>
#include <explore/swarm/swarmdef.h>
#include <explore/swarm/swarmtransport.h>
#include <explore/tsp/tspdef.h>
#include <planning/trajectory.h>
#include <arcmsgs/msg/chunk_data.hpp>
#include <arcmsgs/msg/chunk_stamps.hpp>
#include <arcmsgs/msg/drone_state.hpp>
#include <arcmsgs/msg/idx_list.hpp>
#include <arcmsgs/msg/pair_opt.hpp>
#include <arcmsgs/msg/pair_opt_response.hpp>
#include <arcmsgs/msg/swarm_trajectory.hpp>
#include <arcmsgs/srv/solve_tsp.hpp>
#include <rclcpp/executors/single_threaded_executor.hpp>
#include <rclcpp/rclcpp.hpp>

namespace arc {

class SwarmRosConverter {
public:
    static void toMessage(const DroneState& state, arcmsgs::msg::DroneState& msg) {
        msg.drone_id = state.droneId;
        msg.grid_ids.assign(state.assignedGridIds.begin(), state.assignedGridIds.end());
        msg.recent_attempt_time = state.lastPairOptAttemptTime;
        msg.stamp = state.stamp;
        msg.pos.x = state.pos(0);
        msg.pos.y = state.pos(1);
        msg.pos.z = state.pos(2);
        msg.vel.x = state.vel(0);
        msg.vel.y = state.vel(1);
        msg.vel.z = state.vel(2);
        msg.yaw = state.yaw;
    }
    static DroneState fromMessage(const arcmsgs::msg::DroneState& msg) {
        DroneState state;
        state.droneId = msg.drone_id;
        state.assignedGridIds.assign(msg.grid_ids.begin(), msg.grid_ids.end());
        state.lastPairOptAttemptTime = msg.recent_attempt_time;
        state.stamp = msg.stamp;
        state.pos << msg.pos.x, msg.pos.y, msg.pos.z;
        state.vel << msg.vel.x, msg.vel.y, msg.vel.z;
        state.yaw = msg.yaw;
        return state;
    }
    static void toMessage(const PairOptRequest& request, arcmsgs::msg::PairOpt& msg) {
        msg.from_drone_id = request.fromDroneId;
        msg.to_drone_id = request.toDroneId;
        msg.stamp = request.stamp;
        msg.ego_ids.assign(request.selfGridIds.begin(), request.selfGridIds.end());
        msg.other_ids.assign(request.otherGridIds.begin(), request.otherGridIds.end());
    }
    static PairOptRequest fromMessage(const arcmsgs::msg::PairOpt& msg) {
        PairOptRequest request;
        request.fromDroneId = msg.from_drone_id;
        request.toDroneId = msg.to_drone_id;
        request.stamp = msg.stamp;
        request.selfGridIds.assign(msg.ego_ids.begin(), msg.ego_ids.end());
        request.otherGridIds.assign(msg.other_ids.begin(), msg.other_ids.end());
        return request;
    }
    static void toMessage(const PairOptResponse& response, arcmsgs::msg::PairOptResponse& msg) {
        msg.from_drone_id = response.fromDroneId;
        msg.to_drone_id = response.toDroneId;
        msg.status = response.status;
        msg.stamp = response.stamp;
    }
    static PairOptResponse fromMessage(const arcmsgs::msg::PairOptResponse& msg) {
        PairOptResponse response;
        response.fromDroneId = msg.from_drone_id;
        response.toDroneId = msg.to_drone_id;
        response.status = msg.status;
        response.stamp = msg.stamp;
        return response;
    }
    static void toMessage(const ChunkStamps& stamps, arcmsgs::msg::ChunkStamps& msg) {
        msg.from_drone_id = stamps.fromDroneId;
        msg.time = stamps.time;
        msg.idx_lists.resize(stamps.idxLists.size());
        for (size_t i = 0; i < stamps.idxLists.size(); ++ i) {
            msg.idx_lists[i].ids.assign(stamps.idxLists[i].begin(), stamps.idxLists[i].end());
        }
    }
    static ChunkStamps fromMessage(const arcmsgs::msg::ChunkStamps& msg) {
        ChunkStamps stamps;
        stamps.fromDroneId = msg.from_drone_id;
        stamps.time = msg.time;
        stamps.idxLists.resize(msg.idx_lists.size());
        for (size_t i = 0; i < msg.idx_lists.size(); ++ i) {
            stamps.idxLists[i].assign(msg.idx_lists[i].ids.begin(), msg.idx_lists[i].ids.end());
        }
        return stamps;
    }
    static void toMessage(const ChunkData& chunk, arcmsgs::msg::ChunkData& msg) {
        msg.from_drone_id = chunk.fromDroneId;
        msg.to_drone_id = chunk.toDroneId;
        msg.chunk_drone_id = chunk.chunkDroneId;
        msg.idx = chunk.idx;
        msg.voxel_adrs.assign(chunk.voxelAdrs.begin(), chunk.voxelAdrs.end());
        msg.voxel_occ.assign(chunk.voxelOcc.begin(), chunk.voxelOcc.end());
    }
    static ChunkData fromMessage(const arcmsgs::msg::ChunkData& msg) {
        ChunkData chunk;
        chunk.fromDroneId = msg.from_drone_id;
        chunk.toDroneId = msg.to_drone_id;
        chunk.chunkDroneId = msg.chunk_drone_id;
        chunk.idx = msg.idx;
        chunk.voxelAdrs.assign(msg.voxel_adrs.begin(), msg.voxel_adrs.end());
        chunk.voxelOcc.assign(msg.voxel_occ.begin(), msg.voxel_occ.end());
        return chunk;
    }
    static void toMessage(int droneId, const LocalTrajectory& trajectory, arcmsgs::msg::SwarmTrajectory& msg) {
        msg.drone_id = droneId;
        msg.traj_id = trajectory.id;
        msg.start_time = trajectory.startTime;
        msg.order = trajectory.position.order();
        msg.interval = trajectory.position.interval();
        const Eigen::MatrixXd& points = trajectory.position.controlPoints();
        msg.control_points.resize(static_cast<size_t>(points.rows()));
        for (Eigen::Index i = 0; i < points.rows(); ++ i) {
            msg.control_points[static_cast<size_t>(i)].x = points(i, 0);
            msg.control_points[static_cast<size_t>(i)].y = points(i, 1);
            msg.control_points[static_cast<size_t>(i)].z = points(i, 2);
        }
        msg.yaw_control_points.clear();
        msg.yaw_order = 0;
        msg.yaw_interval = 0.0;
        if (trajectory.yaw.valid()) {
            const Eigen::MatrixXd& yawPoints = trajectory.yaw.controlPoints();
            msg.yaw_order = trajectory.yaw.order();
            msg.yaw_interval = trajectory.yaw.interval();
            msg.yaw_control_points.resize(static_cast<size_t>(yawPoints.rows()));
            for (Eigen::Index i = 0; i < yawPoints.rows(); ++ i) {
                msg.yaw_control_points[static_cast<size_t>(i)] = yawPoints(i, 0);
            }
        }
    }
    static bool fromMessage(const arcmsgs::msg::SwarmTrajectory& msg, LocalTrajectory& trajectory) {
        if (msg.order < 1 || msg.interval <= 1e-6 ||
            static_cast<int>(msg.control_points.size()) < msg.order + 1) {
            return false;
        }
        Eigen::MatrixXd points(static_cast<Eigen::Index>(msg.control_points.size()), 3);
        for (size_t i = 0; i < msg.control_points.size(); ++ i) {
            points(static_cast<Eigen::Index>(i), 0) = msg.control_points[i].x;
            points(static_cast<Eigen::Index>(i), 1) = msg.control_points[i].y;
            points(static_cast<Eigen::Index>(i), 2) = msg.control_points[i].z;
        }
        trajectory = LocalTrajectory();
        trajectory.position.setUniformBSpline(points, msg.order, msg.interval);
        trajectory.id = static_cast<int>(msg.traj_id);
        trajectory.startTime = msg.start_time;
        trajectory.duration = trajectory.position.duration();
        trajectory.velocity = trajectory.position.getDerivativeBSpline();
        trajectory.acceleration = trajectory.velocity.getDerivativeBSpline();
        if (msg.yaw_order >= 1 && msg.yaw_interval > 1e-6 &&
            static_cast<int>(msg.yaw_control_points.size()) >= msg.yaw_order + 1) {
            Eigen::MatrixXd yawPoints(
                static_cast<Eigen::Index>(msg.yaw_control_points.size()), 1
            );
            for (size_t i = 0; i < msg.yaw_control_points.size(); ++ i) {
                yawPoints(static_cast<Eigen::Index>(i), 0) = msg.yaw_control_points[i];
            }
            trajectory.yaw.setUniformBSpline(yawPoints, msg.yaw_order, msg.yaw_interval);
            trajectory.yawRate = trajectory.yaw.getDerivativeBSpline();
            trajectory.yawAcceleration = trajectory.yawRate.getDerivativeBSpline();
        }
        return trajectory.valid();
    }
};

class RosSwarmTransport : public SwarmTransport {
public:
    explicit RosSwarmTransport(rclcpp::Node::SharedPtr node)
    {
        statePub_ = node->create_publisher<arcmsgs::msg::DroneState>(
            "/swarm_expl/drone_state", rclcpp::QoS(10)
        );
        pairOptPub_ = node->create_publisher<arcmsgs::msg::PairOpt>(
            "/swarm_expl/pair_opt", rclcpp::QoS(100)
        );
        pairOptResponsePub_ = node->create_publisher<arcmsgs::msg::PairOptResponse>(
            "/swarm_expl/pair_opt_res", rclcpp::QoS(100)
        );
    }
    void sendDroneState(const DroneState& state) override {
        arcmsgs::msg::DroneState msg;
        SwarmRosConverter::toMessage(state, msg);
        statePub_->publish(msg);
    }
    void sendPairOpt(const PairOptRequest& request) override {
        arcmsgs::msg::PairOpt msg;
        SwarmRosConverter::toMessage(request, msg);
        pairOptPub_->publish(msg);
    }
    void sendPairOptResponse(const PairOptResponse& response) override {
        arcmsgs::msg::PairOptResponse msg;
        SwarmRosConverter::toMessage(response, msg);
        pairOptResponsePub_->publish(msg);
    }

private:
    rclcpp::Publisher<arcmsgs::msg::DroneState>::SharedPtr statePub_;
    rclcpp::Publisher<arcmsgs::msg::PairOpt>::SharedPtr pairOptPub_;
    rclcpp::Publisher<arcmsgs::msg::PairOptResponse>::SharedPtr pairOptResponsePub_;
};

class RosMapTransport {
public:
    RosMapTransport(rclcpp::Node::SharedPtr node, MultiMapManager& manager)
        : node_(std::move(node))
        , manager_(manager)
    {
        stampPub_ = node_->create_publisher<arcmsgs::msg::ChunkStamps>(
            "/multi_map_manager/chunk_stamps", rclcpp::QoS(10)
        );
        chunkPub_ = node_->create_publisher<arcmsgs::msg::ChunkData>(
            "/multi_map_manager/chunk_data", rclcpp::QoS(100)
        );
        stampSub_ = node_->create_subscription<arcmsgs::msg::ChunkStamps>(
            "/multi_map_manager/chunk_stamps",
            rclcpp::QoS(100),
            [this](const arcmsgs::msg::ChunkStamps::SharedPtr msg) {
                const ChunkStamps stamps = SwarmRosConverter::fromMessage(*msg);
                for (const ChunkData& chunk : manager_.handleStamp(stamps)) {
                    sendChunk(chunk);
                }
            }
        );
        chunkSub_ = node_->create_subscription<arcmsgs::msg::ChunkData>(
            "/multi_map_manager/chunk_data",
            rclcpp::QoS(100),
            [this](const arcmsgs::msg::ChunkData::SharedPtr msg) {
                manager_.receiveChunk(SwarmRosConverter::fromMessage(*msg));
            }
        );
    }
    void setStampInterval(double interval) {
        stampInterval_ = std::max(0.05, interval);
    }
    void tick(double now) {
        if (lastStampTime_ < 0.0 || now - lastStampTime_ >= stampInterval_) {
            lastStampTime_ = now;
            arcmsgs::msg::ChunkStamps msg;
            SwarmRosConverter::toMessage(manager_.buildStamp(now), msg);
            stampPub_->publish(msg);
        }
        manager_.processReceivedChunks();
    }

private:
    void sendChunk(const ChunkData& chunk) {
        arcmsgs::msg::ChunkData msg;
        SwarmRosConverter::toMessage(chunk, msg);
        chunkPub_->publish(msg);
    }

private:
    rclcpp::Node::SharedPtr node_;
    MultiMapManager& manager_;
    double stampInterval_{0.4};
    double lastStampTime_{-1.0};
    rclcpp::Publisher<arcmsgs::msg::ChunkStamps>::SharedPtr stampPub_;
    rclcpp::Publisher<arcmsgs::msg::ChunkData>::SharedPtr chunkPub_;
    rclcpp::Subscription<arcmsgs::msg::ChunkStamps>::SharedPtr stampSub_;
    rclcpp::Subscription<arcmsgs::msg::ChunkData>::SharedPtr chunkSub_;
};

class RosTrajectoryTransport {
public:
    using Handler = std::function<void(int, const LocalTrajectory&)>;
    explicit RosTrajectoryTransport(rclcpp::Node::SharedPtr node)
        : node_(std::move(node))
    {
        pub_ = node_->create_publisher<arcmsgs::msg::SwarmTrajectory>(
            "/planning/swarm_traj", rclcpp::QoS(100)
        );
        sub_ = node_->create_subscription<arcmsgs::msg::SwarmTrajectory>(
            "/planning/swarm_traj",
            rclcpp::QoS(100),
            [this](const arcmsgs::msg::SwarmTrajectory::SharedPtr msg) {
                if (!handler_) {
                    return;
                }
                LocalTrajectory trajectory;
                if (SwarmRosConverter::fromMessage(*msg, trajectory)) {
                    handler_(msg->drone_id, trajectory);
                }
            }
        );
    }
    void setHandler(Handler handler) {
        handler_ = std::move(handler);
    }
    void publish(int droneId, const LocalTrajectory& trajectory) {
        if (!trajectory.valid()) {
            return;
        }
        arcmsgs::msg::SwarmTrajectory msg;
        SwarmRosConverter::toMessage(droneId, trajectory, msg);
        pub_->publish(msg);
    }

private:
    rclcpp::Node::SharedPtr node_;
    Handler handler_;
    rclcpp::Publisher<arcmsgs::msg::SwarmTrajectory>::SharedPtr pub_;
    rclcpp::Subscription<arcmsgs::msg::SwarmTrajectory>::SharedPtr sub_;
};

class SwarmTspClient {
public:
    SwarmTspClient(rclcpp::Node::SharedPtr node, int droneId)
        : node_(std::move(node))
        , droneId_(droneId)
    {
        clientNode_ = std::make_shared<rclcpp::Node>(
            "tsp_client_" + std::to_string(droneId_)
        );
        tspClient_ = clientNode_->create_client<arcmsgs::srv::SolveTsp>(
            "/solve_tsp_" + std::to_string(droneId_)
        );
        acvrpClient_ = clientNode_->create_client<arcmsgs::srv::SolveTsp>(
            "/solve_acvrp_" + std::to_string(droneId_)
        );
        executor_ = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
        executor_->add_node(clientNode_);
        thread_ = std::thread([this]() { executor_->spin(); });
    }
    ~SwarmTspClient() {
        if (executor_) {
            executor_->cancel();
        }
        if (thread_.joinable()) {
            thread_.join();
        }
    }
    TspSolveResult solve(const std::string& parameterFile, bool acvrp) {
        const auto start = std::chrono::steady_clock::now();
        auto finish = [&start](bool success, TspFailure failure) {
            TspSolveResult result;
            result.success = success;
            result.failure = failure;
            result.seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
            return result;
        };
        auto client = acvrp ? acvrpClient_ : tspClient_;
        if (!client) {
            return finish(false, TspFailure::INVALID_PROBLEM);
        }
        if (!client->wait_for_service(std::chrono::seconds(2))) {
            RCLCPP_WARN(
                node_->get_logger(),
                "TSP service for drone %d is not available",
                droneId_
            );
            return finish(false, TspFailure::SERVICE_UNAVAILABLE);
        }
        auto request = std::make_shared<arcmsgs::srv::SolveTsp::Request>();
        request->prob = acvrp ? 3 : 2;
        request->parameter_file = parameterFile;
        auto future = client->async_send_request(request);
        if (future.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
            RCLCPP_ERROR(
                node_->get_logger(),
                "TSP service for drone %d timed out",
                droneId_
            );
            return finish(false, TspFailure::SERVICE_TIMEOUT);
        }
        try {
            auto response = future.get();
            if (!response || !response->success) {
                return finish(false, TspFailure::SOLVER_FAILED);
            }
            return finish(true, TspFailure::NONE);
        } catch (const std::exception& error) {
            RCLCPP_ERROR(
                node_->get_logger(),
                "TSP service for drone %d failed: %s",
                droneId_,
                error.what()
            );
            return finish(false, TspFailure::SERVICE_EXCEPTION);
        }
    }

private:
    rclcpp::Node::SharedPtr node_;
    rclcpp::Node::SharedPtr clientNode_;
    rclcpp::Client<arcmsgs::srv::SolveTsp>::SharedPtr tspClient_;
    rclcpp::Client<arcmsgs::srv::SolveTsp>::SharedPtr acvrpClient_;
    rclcpp::executors::SingleThreadedExecutor::SharedPtr executor_;
    std::thread thread_;
    int droneId_{1};
};

} // arc
