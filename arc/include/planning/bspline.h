#pragma once

#include <base/iarc.h>

namespace arc {

struct BSplineOptions {
    double maxVel;
    double maxAcc;
    double maxExpandRatio;

    BSplineOptions() { memset(this, 0, sizeof(*this)); }
};

class BSpline {
public:
    BSpline() = default;
    BSpline(const BSplineOptions& options)
        : options_(options)
    {}

public:
    void setUniformBSpline(const Eigen::MatrixXd& points, int order, double interval) {
        points_ = points;
        order_ = order;
        numPoints_ = points.rows();
        numKnots_ = numPoints_ + order + 1;

        u_ = Eigen::VectorXd::LinSpaced(
            numKnots_,
            -static_cast<double>(order_) * interval,
            static_cast<double>(points_.rows()) * interval
        );
        ustart_ = u_(order_);
        uend_ = u_(numPoints_);
        duration_ = uend_ - ustart_;
    }
    void setUniformBSpline(const Eigen::MatrixXd& points, int order, const Eigen::VectorXd& knots) {
        points_ = points;
        order_ = order;
        numPoints_ = points.rows();
        numKnots_ = numPoints_ + order + 1;
        setKnots(knots);
    }
    void setKnots(const Eigen::VectorXd& knots) {
        if (numPoints_ == 0 || knots.size() != static_cast<Eigen::Index>(numPoints_ + order_ + 1)) {
            throw std::runtime_error("Invalid size of knots: " + std::to_string(knots.size()));
        }
        numKnots_ = knots.size();
        u_ = knots;
        ustart_ = u_(order_);
        uend_ = u_(numPoints_);
        duration_ = uend_ - ustart_;
    }

public:
    bool valid() const {
        return order_ >= 1 && numPoints_ >= static_cast<size_t>(order_ + 1) && u_.size() == static_cast<Eigen::Index>(numKnots_) && duration_ > 1e-9;
    }
    double duration() const { return duration_; }
    const Eigen::MatrixXd& controlPoints() const { return points_; }
    int order() const { return order_; }
    double interval() const {
        if (order_ < 1 || u_.size() < static_cast<Eigen::Index>(order_ + 2)) {
            return 0.0;
        }
        return u_(order_ + 1) - u_(order_);
    }

public:
    Eigen::VectorXd evalu(double u) const {
        if (!valid()) {
            return Eigen::VectorXd();
        }
        u = std::min<double>(std::max<double>(u, ustart_), uend_);
        int k = order_;
        const int kMax = numKnots_ - order_ - 2;
        while (k < kMax && u_(k + 1) <= u) {
            k += 1;
        };
        const Eigen::Index dims = points_.cols();
        Eigen::VectorXd result(dims);
        if (order_ <= 3 && dims <= 3) {
            Eigen::Matrix<double, 4, 3> d;
            for (int i = 0; i <= order_; ++ i) {
                d.row(i).head(dims) = points_.row(k - order_ + i);
            }
            for (int r = 1; r <= order_; ++ r) {
                for (int i = order_; i >= r; -- i) {
                    const int knotIndex = i + k - order_;
                    const double denominator = u_(i + 1 + k - r) - u_(knotIndex);
                    double alpha = 0.0;
                    if (std::abs(denominator) > 1e-12) {
                        alpha = (u - u_(knotIndex)) / denominator;
                    }
                    d.row(i).head(dims) =
                        (1.0 - alpha) * d.row(i - 1).head(dims) + alpha * d.row(i).head(dims);
                }
            }
            result = d.row(order_).head(dims).transpose();
            return result;
        }
        Eigen::MatrixXd d(order_ + 1, dims);
        for (int i = 0; i <= order_; ++ i) {
            d.row(i) = points_.row(k - order_ + i);
        }
        for (int r = 1; r <= order_; ++ r) {
            for (int i = order_; i >= r; -- i) {
                const int knotIndex = i + k - order_;
                const double denominator = u_(i + 1 + k - r) - u_(knotIndex);
                double alpha = 0.0;
                if (std::abs(denominator) > 1e-12) {
                    alpha = (u - u_(knotIndex)) / denominator;
                }
                d.row(i) = (1.0 - alpha) * d.row(i - 1) + alpha * d.row(i);
            }
        }
        return d.row(order_).transpose();
    }
    Eigen::VectorXd evalt(double t) const {
        double u = std::min<double>(std::max<double>(t, 0.0), duration_) + ustart_;
        return evalu(u);
    }
    BSpline getDerivativeBSpline() const {
        if (!valid()) {
            return BSpline();
        }
        BSpline derivative(options_);
        derivative.setUniformBSpline(
            getDerivativeControlPoints_(),
            order_ - 1,
            u_.segment(1, u_.size() - 2)
        );
        return derivative;
    }
    std::vector<BSpline> getRecursiveDerivatives(int k) const {
        if (k < 1) {
            return {};
        }
        std::vector<BSpline> derivatives;
        derivatives.push_back(getDerivativeBSpline());
        for (int i = 0; i < k - 1; ++i) {
            derivatives.push_back(derivatives.back().getDerivativeBSpline());
        }
        return derivatives;
    }
    void getBoundaryStates(int k, std::vector<Eigen::Vector3d>& start, std::vector<Eigen::Vector3d>& end) {
        start.clear();
        end.clear();
        if (!valid() || points_.cols() != 3) {
            return;
        }
        std::vector<BSpline> derivatives = getRecursiveDerivatives(k);
        start.push_back(Eigen::Vector3d(evalt(0.0).head<3>()));
        for (int i = 0; i < static_cast<int>(derivatives.size()); ++ i) {
            start.push_back(Eigen::Vector3d(derivatives[i].evalt(0.0).head<3>()));
        }
        end.push_back(Eigen::Vector3d(evalt(duration_).head<3>()));
        for (int i = 0; i < static_cast<int>(derivatives.size()); ++ i) {
            end.push_back(Eigen::Vector3d(derivatives[i].evalt(derivatives[i].duration_).head<3>()));
        }
    }

private:
    Eigen::MatrixXd getDerivativeControlPoints_() const {
        Eigen::MatrixXd dcp = Eigen::MatrixXd::Zero(points_.rows() - 1, points_.cols());
        for (int i = 0; i + 1 < points_.rows(); ++ i) {
            const double denominator = u_(i + order_ + 1) - u_(i + 1);
            if (std::abs(denominator) < 1e-12) {
                continue;
            }
            dcp.row(i) = order_ * (points_.row(i + 1) - points_.row(i)) / denominator;
        }
        return dcp;
    }

private:
    BSplineOptions options_;

private:
    Eigen::MatrixXd points_;
    Eigen::VectorXd u_;
    size_t numPoints_{0};
    size_t numKnots_{0};
    int order_{0};
    double ustart_{0};
    double uend_{0};
    double duration_{0};
};

} // arc
