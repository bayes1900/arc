#pragma once

#include <base/ros.h>

#include <env/sdfmap.h>
#include <planning/bsplinecost.h>
#include <planning/bsplineoptdef.h>
#include <nlopt.hpp>

namespace arc {

class BSplineOptimizer {
public:
    BSplineOptimizer(rclcpp::Node::SharedPtr node, const SdfMap& map)
        : params_(std::move(node))
        , map_(map)
        , cost_(params_, map_)
    {}

public:
    BSplineOptResult optimize(const BSplineOptRequest& request) const {
        BSplineOptResult result;
        const int numPoints = request.points.rows();
        if (numPoints < 4 || request.points.cols() != 3 || request.interval <= 0.0) {
            result.code = BSplineOptResult::INPUTINVALID;
            return result;
        }
        const MapParams& mapParams = map_.params();
        Eigen::MatrixXd points = request.points;
        for (int i = 0; i < numPoints; ++ i) {
            for (int j = 0; j < 3; ++ j) {
                const double lowerBound = mapParams.boxMin(j) + 1e-3;
                const double upperBound = mapParams.boxMax(j) - 1e-3;
                if (lowerBound < upperBound) {
                    points(i, j) = std::min(std::max(points(i, j), lowerBound), upperBound);
                }
            }
        }
        const bool optimizeInterval = request.cost.MINTIME;
        const double fixedInterval = request.interval;
        const Eigen::Index segmentCount = points.rows() - 1;
        double ptAvgDist = std::max((points.bottomRows(segmentCount) - points.topRows(segmentCount)).rowwise().norm().mean(), 1e-6);

        Eigen::MatrixXd gradient;
        double timeGradient = 0.0;
        const double initialCost = cost_.evaluate(request, points, fixedInterval, ptAvgDist, gradient, timeGradient);
        if (!std::isfinite(initialCost)) {
            result.code = BSplineOptResult::INFEASIBLE;
            return result;
        }
        const int variableCount = 3 * numPoints + (optimizeInterval ? 1 : 0);
        std::vector<double> variables(variableCount, 0.0);
        for (int i = 0; i < numPoints; ++i) {
            for (int j = 0; j < 3; ++j) {
                variables[3 * i + j] = points(i, j);
            }
        }
        if (optimizeInterval) {
            variables.back() = fixedInterval;
        }
        EvalContext context;
        context.cost = &cost_;
        context.request = &request;
        context.numPoints = numPoints;
        context.ptAvgDist = ptAvgDist;
        context.fixedInterval = fixedInterval;
        context.optimizeInterval = optimizeInterval;
        context.bestCost = initialCost;
        context.bestVariables = variables;
        context.points.resize(numPoints, 3);
        context.pointGradient.resize(numPoints, 3);

        nlopt::opt optimizer(nlopt::LD_MMA, variableCount);
        optimizer.set_min_objective(costFunction_, &context);
        if (params_.maxEval > 0) {
            optimizer.set_maxeval(params_.maxEval);
        }
        if (params_.maxTime > 0.0) {
            optimizer.set_maxtime(params_.maxTime);
        }
        optimizer.set_xtol_rel(params_.xtolRel);

        std::vector<double> lower(variableCount, -std::numeric_limits<double>::infinity());
        std::vector<double> upper(variableCount, std::numeric_limits<double>::infinity());

        for (int i = 0; i < numPoints; ++ i) {
            for (int j = 0; j < 3; ++ j) {
                lower[3 * i + j] = mapParams.boxMin(j) + 1e-3;
                upper[3 * i + j] = mapParams.boxMax(j) - 1e-3;
            }
        }
        if (optimizeInterval) {
            lower.back() = minimumInterval_(request, numPoints);
            upper.back() = params_.maxInterval;
        }
        optimizer.set_lower_bounds(lower);
        optimizer.set_upper_bounds(upper);

        uint8_t statusCode = BSplineOptResult::SUCCEEDED;
        try {
            double finalCost = std::numeric_limits<double>::max();
            const nlopt::result status = optimizer.optimize(variables, finalCost);
            switch (status) {
                case nlopt::SUCCESS:
                case nlopt::STOPVAL_REACHED:
                case nlopt::FTOL_REACHED:
                case nlopt::XTOL_REACHED:
                case nlopt::ROUNDOFF_LIMITED: {
                    statusCode = BSplineOptResult::SUCCEEDED;
                    break;
                }
                case nlopt::MAXEVAL_REACHED: {
                    statusCode = BSplineOptResult::MAXEVALREACHED;
                    break;
                }
                case nlopt::MAXTIME_REACHED: {
                    statusCode = BSplineOptResult::MAXTIMEREACHED;
                    break;
                }
                default: {
                    statusCode = BSplineOptResult::UNKNOWN;
                    break;
                }
            }
        } catch (const std::exception&) {
            result.code = BSplineOptResult::SOLVERFAILED;
            return result;
        }
        if (context.bestVariables.size() != static_cast<size_t>(variableCount)) {
            result.code = BSplineOptResult::NOSOLUTION;
            return result;
        }
        for (int i = 0; i < numPoints; ++ i) {
            for (int j = 0; j < 3; ++ j) {
                points(i, j) = context.bestVariables[3 * i + j];
            }
        }
        double interval = optimizeInterval ? context.bestVariables.back() : fixedInterval;
        if (!optimizeInterval && request.timeLowerBound > 0.0) {
            const double minInterval = minimumInterval_(request, numPoints);
            if (static_cast<double>(numPoints - 3) * interval < request.timeLowerBound) {
                interval = minInterval;
            }
        }
        result.code = statusCode;
        result.iterations = context.iterations;
        result.cost = context.bestCost;
        result.points = std::move(points);
        result.interval = interval;
        return result;
    }

private:
    struct EvalContext {
        const BSplineCost* cost{nullptr};
        const BSplineOptRequest* request{nullptr};
        int numPoints{0};
        double ptAvgDist{1.0};
        double fixedInterval{0.1};
        bool optimizeInterval{false};
        double bestCost{std::numeric_limits<double>::max()};
        std::vector<double> bestVariables;
        Eigen::MatrixXd points;
        Eigen::MatrixXd pointGradient;
        int iterations{0};
    };
    static double costFunction_(const std::vector<double>& variables, std::vector<double>& gradient, void* data) {
        auto* context = static_cast<EvalContext*>(data);
        Eigen::MatrixXd& points = context->points;
        for (int i = 0; i < context->numPoints; ++ i) {
            for (int j = 0; j < 3; ++j) {
                points(i, j) = variables[3 * i + j];
            }
        }
        const double interval = context->optimizeInterval ? variables.back() : context->fixedInterval;
        Eigen::MatrixXd& pointGradient = context->pointGradient;
        double timeGradient = 0.0;
        double cost = context->cost->evaluate(*context->request, points, interval, context->ptAvgDist, pointGradient, timeGradient);
        if (!std::isfinite(cost)) {
            cost = std::numeric_limits<double>::max();
        }
        if (!gradient.empty()) {
            for (int i = 0; i < context->numPoints; ++ i) {
                for (int j = 0; j < 3; ++j) {
                    gradient[3 * i + j] = pointGradient(i, j);
                }
            }
            if (context->optimizeInterval) {
                gradient.back() = timeGradient;
            }
        }
        context->iterations += 1;
        if (cost < context->bestCost) {
            context->bestCost = cost;
            context->bestVariables = variables;
        }
        return cost;
    }
    double minimumInterval_(const BSplineOptRequest& request, int numPoints) const {
        double minimum = params_.minInterval;
        if (request.timeLowerBound > 0.0) {
            minimum = std::max(minimum, request.timeLowerBound / static_cast<double>(numPoints - 3));
        }
        return std::min(minimum, params_.maxInterval);
    }

private:
    BSplineOptParams params_;
    const SdfMap& map_;
    BSplineCost cost_;
};

} // arc
