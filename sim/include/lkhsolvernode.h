#pragma once

#include <base/iarc.h>
#include <explore/tsp/lkhinterface.h>

#include <arcmsgs/srv/solve_tsp.hpp>
#include <rclcpp/rclcpp.hpp>

#include <memory>
#include <string>

namespace arc {

class LkhSolverNode {
public:
    explicit LkhSolverNode(rclcpp::Node::SharedPtr node)
        : node_(std::move(node))
    {
        droneId_ = static_cast<int>(node_->declare_parameter<int>("drone_id", 1));
        problemId_ = static_cast<int>(node_->declare_parameter<int>("problem_id", 1));
        workDir_ = node_->declare_parameter<std::string>(
            "work_dir", "/tmp/arc_exploration_" + std::to_string(droneId_)
        );

        const std::string serviceName = problemId_ == 1
            ? "/solve_tsp_" + std::to_string(droneId_)
            : "/solve_acvrp_" + std::to_string(droneId_);
        server_ = node_->create_service<arcmsgs::srv::SolveTsp>(
            serviceName,
            [this](
                const std::shared_ptr<arcmsgs::srv::SolveTsp::Request> request,
                std::shared_ptr<arcmsgs::srv::SolveTsp::Response> response
            ) {
                onSolve_(request, response);
            }
        );
    }

private:
    void onSolve_(
        const std::shared_ptr<arcmsgs::srv::SolveTsp::Request> request,
        std::shared_ptr<arcmsgs::srv::SolveTsp::Response> response
    ) {
        std::string parameterFile = request->parameter_file;
        if (parameterFile.empty()) {
            const std::string base = request->prob == 3
                ? "exploration_acvrp"
                : "exploration_tsp";
            parameterFile = workDir_ + "/" + base + ".par";
        }
        const int result = solveMTSPWithLKH3(parameterFile.c_str());
        response->success = result == 0;
    }

private:
    rclcpp::Node::SharedPtr node_;
    rclcpp::Service<arcmsgs::srv::SolveTsp>::SharedPtr server_;
    int droneId_{1};
    int problemId_{1};
    std::string workDir_;
};

} // arc
