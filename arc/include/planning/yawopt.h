#pragma once

#include <base/iarc.h>
#include <planning/bspline.h>
#include <nlopt.hpp>

namespace arc {

struct YawOptParams {
    double smoothWeight{5.0};
    double startWeight{100.0};
    double endWeight{0.5};
    double waypointWeight{7.0};
    double startPositionWeight{10.0};

    int maxEval{2000};
    double maxTime{0.005};
    double xtolRel{1e-5};
};

struct YawOptRequest {
    Eigen::MatrixXd points;
    double interval{0.1};

    std::vector<double> startState;
    std::vector<double> endState;
    std::vector<double> waypoints;
    std::vector<int> waypointIndices;
};

struct YawOptResult {
    static constexpr uint8_t SUCCEEDED = 0;
    static constexpr uint8_t INPUTINVALID = 1;
    static constexpr uint8_t SOLVERFAILED = 2;
    static constexpr uint8_t NOSOLUTION = 3;

    uint8_t code{255};
    Eigen::MatrixXd points;
};

class YawOptimizer {
public:
    explicit YawOptimizer(const YawOptParams& params=YawOptParams())
        : params_(params)
    {}

public:
    YawOptResult optimize(const YawOptRequest& request) const {
        YawOptResult result;
        if (request.points.rows() < 4 || request.points.cols() != 1 || request.interval <= 0.0 || request.startState.empty()) {
            result.code = YawOptResult::INPUTINVALID;
            return result;
        }
        const int numPoints = request.points.rows();
        double pointDistance = 0.0;
        for (int i = 0; i + 1 < numPoints; ++ i) {
            pointDistance += std::abs(request.points(i + 1, 0) - request.points(i, 0));
        }
        pointDistance = std::max(pointDistance / static_cast<double>(numPoints), 1e-6);

        std::vector<double> variables(numPoints);
        for (int i = 0; i < numPoints; ++ i) {
            variables[i] = request.points(i, 0);
        }

        EvalContext context;
        context.request = &request;
        context.params = &params_;
        context.pointDistance = pointDistance;
        context.bestCost = std::numeric_limits<double>::max();
        context.bestVariables = variables;

        nlopt::opt optimizer(nlopt::LD_LBFGS, numPoints);
        optimizer.set_min_objective(costFunction_, &context);
        if (params_.maxEval > 0) {
            optimizer.set_maxeval(params_.maxEval);
        }
        if (params_.maxTime > 0.0) {
            optimizer.set_maxtime(params_.maxTime);
        }
        optimizer.set_xtol_rel(params_.xtolRel);
        optimizer.set_lower_bounds(std::vector<double>(numPoints, -std::numeric_limits<double>::infinity()));
        optimizer.set_upper_bounds(std::vector<double>(numPoints, std::numeric_limits<double>::infinity()));

        uint8_t statusCode = YawOptResult::SUCCEEDED;
        try {
            double finalCost = std::numeric_limits<double>::max();
            const nlopt::result status = optimizer.optimize(variables, finalCost);
            switch (status) {
                case nlopt::SUCCESS:
                case nlopt::STOPVAL_REACHED:
                case nlopt::FTOL_REACHED:
                case nlopt::XTOL_REACHED:
                case nlopt::MAXEVAL_REACHED:
                case nlopt::MAXTIME_REACHED: {
                    statusCode = YawOptResult::SUCCEEDED;
                    break;
                }
                default: {
                    statusCode = YawOptResult::NOSOLUTION;
                    break;
                }
            }
        } catch (const std::exception&) {
            result.code = YawOptResult::SOLVERFAILED;
            return result;
        }
        if (context.bestVariables.size() != static_cast<size_t>(numPoints)) {
            result.code = YawOptResult::NOSOLUTION;
            return result;
        }

        result.code = statusCode;
        result.points.resize(numPoints, 1);
        for (int i = 0; i < numPoints; ++ i) {
            result.points(i, 0) = context.bestVariables[i];
        }
        return result;
    }

private:
    struct EvalContext {
        const YawOptRequest* request{nullptr};
        const YawOptParams* params{nullptr};
        double pointDistance{1.0};
        double bestCost{std::numeric_limits<double>::max()};
        std::vector<double> bestVariables;
    };
    static double costFunction_(const std::vector<double>& variables, std::vector<double>& gradient, void* data) {
        auto* context = static_cast<EvalContext*>(data);
        const YawOptRequest& request = *context->request;
        const YawOptParams& params = *context->params;
        const int numPoints = static_cast<int>(variables.size());
        const double interval = request.interval;
        std::vector<double> pointGradient(variables.size(), 0.0);
        double cost = 0.0;

        for (int i = 0; i + 3 < numPoints; ++ i) {
            const double diff = (variables[i + 3] - 3.0 * variables[i + 2] + 3.0 * variables[i + 1] - variables[i]) / context->pointDistance;
            cost += params.smoothWeight * diff * diff;
            const double value = 2.0 * params.smoothWeight * diff / context->pointDistance;
            pointGradient[i] -= value;
            pointGradient[i + 1] += 3.0 * value;
            pointGradient[i + 2] -= 3.0 * value;
            pointGradient[i + 3] += value;
        }
        if (request.startState.size() >= 1) {
            const double error = (variables[0] + 4.0 * variables[1] + variables[2]) / 6.0 - request.startState[0];
            const double weight = params.startWeight * params.startPositionWeight;
            cost += weight * error * error;
            pointGradient[0] += weight * 2.0 * error / 6.0;
            pointGradient[1] += weight * 8.0 * error / 6.0;
            pointGradient[2] += weight * 2.0 * error / 6.0;
        }
        if (request.startState.size() >= 2) {
            const double error = (variables[2] - variables[0]) / (2.0 * interval) - request.startState[1];
            cost += params.startWeight * error * error;
            pointGradient[0] -= params.startWeight * error / interval;
            pointGradient[2] += params.startWeight * error / interval;
        }
        if (request.startState.size() >= 3) {
            const double error = (variables[0] - 2.0 * variables[1] + variables[2]) / (interval * interval) - request.startState[2];
            cost += params.startWeight * error * error;
            pointGradient[0] += params.startWeight * 2.0 * error / (interval * interval);
            pointGradient[1] -= params.startWeight * 4.0 * error / (interval * interval);
            pointGradient[2] += params.startWeight * 2.0 * error / (interval * interval);
        }

        const int end = numPoints - 1;
        if (request.endState.size() >= 1) {
            const double error = (variables[end - 2] + 4.0 * variables[end - 1] + variables[end]) / 6.0 - request.endState[0];
            cost += params.endWeight * error * error;
            pointGradient[end - 2] += params.endWeight * 2.0 * error / 6.0;
            pointGradient[end - 1] += params.endWeight * 8.0 * error / 6.0;
            pointGradient[end] += params.endWeight * 2.0 * error / 6.0;
        }
        if (request.endState.size() >= 2) {
            const double error = (variables[end] - variables[end - 2]) / (2.0 * interval) - request.endState[1];
            cost += params.endWeight * error * error;
            pointGradient[end - 2] -= params.endWeight * error / interval;
            pointGradient[end] += params.endWeight * error / interval;
        }

        const int waypointCount = std::min(
            static_cast<int>(request.waypoints.size()),
            static_cast<int>(request.waypointIndices.size())
        );
        for (int i = 0; i < waypointCount; ++ i) {
            const int index = request.waypointIndices[i];
            if (index < 0 || index + 2 >= numPoints) {
                continue;
            }
            const double error = (variables[index] + 4.0 * variables[index + 1] + variables[index + 2]) / 6.0 - request.waypoints[i];
            cost += params.waypointWeight * error * error;
            pointGradient[index] += params.waypointWeight * 2.0 * error / 6.0;
            pointGradient[index + 1] += params.waypointWeight * 8.0 * error / 6.0;
            pointGradient[index + 2] += params.waypointWeight * 2.0 * error / 6.0;
        }
        if (!gradient.empty()) {
            gradient = pointGradient;
        }
        if (cost < context->bestCost) {
            context->bestCost = cost;
            context->bestVariables = variables;
        }
        return cost;
    }

private:
    YawOptParams params_;
};

} // arc
