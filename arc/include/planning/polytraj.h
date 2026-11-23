#pragma once

#include <base/iarc.h>

namespace arc {

struct Polynomial {
    double time;
    Vec6d cx;
    Vec6d cy;
    Vec6d cz;

    Polynomial(double time, const Vec6d& cx, const Vec6d& cy, const Vec6d& cz)
        : time(time)
        , cx(cx)
        , cy(cy)
        , cz(cz)
    {}
    double getTBasis(double t, int n, int k) const {
        if (k > n) {
            return 0.0;
        }
        int coefficient = 1;
        for (int i = n; i > n - k; -- i) {
            coefficient *= i;
        }
        return static_cast<double>(coefficient) * std::pow(t, n - k);
    }
    Eigen::Vector3d evaluate(double t, int k) const {
        Vec6d basis = Vec6d::Zero();
        for (int i = k; i < 6; ++ i) {
            basis(i) = getTBasis(t, i, k);
        }
        return Eigen::Vector3d(basis.dot(cx), basis.dot(cy), basis.dot(cz));
    }
};

struct PolynomialTrajectory {
    std::vector<Polynomial> segments;

    void reset() {
        segments.clear();
        timeSum_ = -1.0;
    }
    void addSegment(const Polynomial& poly) {
        segments.push_back(poly);
        timeSum_ = -1.0;
    }
    Eigen::Vector3d evaluate(double t, int k) const {
        if (segments.empty()) {
            return Eigen::Vector3d::Zero();
        }
        if (t <= 0.0) {
            return segments.front().evaluate(0.0, k);
        }
        const double total = getTotalTime();
        if (t >= total) {
            return segments.back().evaluate(segments.back().time, k);
        }
        int index = 0;
        double localTime = t;
        while (index + 1 < static_cast<int>(segments.size()) && localTime >= segments[index].time) {
            localTime -= segments[index].time;
            index += 1;
        }
        localTime = std::min(std::max(localTime, 0.0), segments[index].time);
        return segments[index].evaluate(localTime, k);
    }
    double getTotalTime() const {
        if (timeSum_ < 0.0) {
            timeSum_ = 0.0;
            for (const auto& segment : segments) {
                timeSum_ += segment.time;
            }
        }
        return timeSum_;
    }
    double getLength() const {
        const double total = getTotalTime();
        if (total <= 0.0) {
            return 0.0;
        }
        constexpr double sampleInterval = 0.01;
        double length = 0.0;
        Eigen::Vector3d previous = evaluate(0.0, 0);
        for (double t = sampleInterval; t < total; t += sampleInterval) {
            const Eigen::Vector3d current = evaluate(t, 0);
            length += (current - previous).norm();
            previous = current;
        }
        length += (evaluate(total, 0) - previous).norm();
        return length;
    }
    static bool waypointsTraj(
        const Eigen::MatrixXd& positions,
        const Eigen::Vector3d& startVel,
        const Eigen::Vector3d& endVel,
        const Eigen::Vector3d& startAcc,
        const Eigen::Vector3d& endAcc,
        const Eigen::VectorXd& times,
        PolynomialTrajectory& trajectory
    ) {
        if (positions.cols() != 3 || positions.rows() < 2 || times.size() != positions.rows() - 1) {
            return false;
        }
        const int segmentCount = static_cast<int>(times.size());
        for (int i = 0; i < segmentCount; ++ i) {
            if (times(i) <= 0.0) {
                return false;
            }
        }
        const auto factorial = [](int x) {
            int value = 1;
            for (int i = x; i > 0; -- i) {
                value *= i;
            }
            return value;
        };
        Eigen::VectorXd dx = Eigen::VectorXd::Zero(segmentCount * 6);
        Eigen::VectorXd dy = Eigen::VectorXd::Zero(segmentCount * 6);
        Eigen::VectorXd dz = Eigen::VectorXd::Zero(segmentCount * 6);
        for (int k = 0; k < segmentCount; ++ k) {
            dx(k * 6) = positions(k, 0);
            dx(k * 6 + 1) = positions(k + 1, 0);
            dy(k * 6) = positions(k, 1);
            dy(k * 6 + 1) = positions(k + 1, 1);
            dz(k * 6) = positions(k, 2);
            dz(k * 6 + 1) = positions(k + 1, 2);
            if (k == 0) {
                dx(k * 6 + 2) = startVel(0);
                dy(k * 6 + 2) = startVel(1);
                dz(k * 6 + 2) = startVel(2);
                dx(k * 6 + 4) = startAcc(0);
                dy(k * 6 + 4) = startAcc(1);
                dz(k * 6 + 4) = startAcc(2);
            }
            if (k == segmentCount - 1) {
                dx(k * 6 + 3) = endVel(0);
                dy(k * 6 + 3) = endVel(1);
                dz(k * 6 + 3) = endVel(2);
                dx(k * 6 + 5) = endAcc(0);
                dy(k * 6 + 5) = endAcc(1);
                dz(k * 6 + 5) = endAcc(2);
            }
        }
        Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(segmentCount * 6, segmentCount * 6);
        for (int k = 0; k < segmentCount; ++ k) {
            Eigen::Matrix<double, 6, 6> block = Eigen::Matrix<double, 6, 6>::Zero();
            for (int i = 0; i < 3; ++ i) {
                block(2 * i, i) = static_cast<double>(factorial(i));
                for (int j = i; j < 6; ++ j) {
                    block(2 * i + 1, j) = static_cast<double>(factorial(j)) /
                        static_cast<double>(factorial(j - i)) * std::pow(times(k), j - i);
                }
            }
            matrix.block<6, 6>(k * 6, k * 6) = block;
        }
        const int freeCount = 2 * segmentCount + 4;
        const int fixedCount = 2 * segmentCount - 2;
        const int derivativeCount = 6 * segmentCount;
        const int variableCount = freeCount + fixedCount;

        Eigen::MatrixXd constraints = Eigen::MatrixXd::Zero(derivativeCount, variableCount);
        if (segmentCount == 1) {
            constraints(0, 0) = 1.0;
            constraints(1, 3) = 1.0;
            constraints(2, 1) = 1.0;
            constraints(3, 4) = 1.0;
            constraints(4, 2) = 1.0;
            constraints(5, 5) = 1.0;
        } else {
            constraints(0, 0) = 1.0;
            constraints(2, 1) = 1.0;
            constraints(4, 2) = 1.0;
            constraints(1, 3) = 1.0;
            constraints(3, 2 * segmentCount + 4) = 1.0;
            constraints(5, 2 * segmentCount + 5) = 1.0;

            constraints(6 * (segmentCount - 1) + 0, 2 * segmentCount + 0) = 1.0;
            constraints(6 * (segmentCount - 1) + 1, 2 * segmentCount + 1) = 1.0;
            constraints(6 * (segmentCount - 1) + 2, 4 * segmentCount + 0) = 1.0;
            constraints(6 * (segmentCount - 1) + 3, 2 * segmentCount + 2) = 1.0;
            constraints(6 * (segmentCount - 1) + 4, 4 * segmentCount + 1) = 1.0;
            constraints(6 * (segmentCount - 1) + 5, 2 * segmentCount + 3) = 1.0;

            for (int j = 2; j < segmentCount; ++ j) {
                constraints(6 * (j - 1) + 0, 2 + 2 * (j - 1) + 0) = 1.0;
                constraints(6 * (j - 1) + 1, 2 + 2 * (j - 1) + 1) = 1.0;
                constraints(6 * (j - 1) + 2, 2 * segmentCount + 4 + 2 * (j - 2) + 0) = 1.0;
                constraints(6 * (j - 1) + 3, 2 * segmentCount + 4 + 2 * (j - 1) + 0) = 1.0;
                constraints(6 * (j - 1) + 4, 2 * segmentCount + 4 + 2 * (j - 2) + 1) = 1.0;
                constraints(6 * (j - 1) + 5, 2 * segmentCount + 4 + 2 * (j - 1) + 1) = 1.0;
            }
        }

        Eigen::PartialPivLU<Eigen::MatrixXd> lu(matrix);
        const Eigen::MatrixXd basis = lu.solve(constraints);

        Eigen::MatrixXd cost = Eigen::MatrixXd::Zero(derivativeCount, derivativeCount);
        for (int k = 0; k < segmentCount; ++ k) {
            for (int i = 3; i < 6; ++ i) {
                for (int j = 3; j < 6; ++ j) {
                    const int order = i + j - 5;
                    cost(k * 6 + i, k * 6 + j) =
                        static_cast<double>(i * (i - 1) * (i - 2) * j * (j - 1) * (j - 2)) /
                        static_cast<double>(order) * std::pow(times(k), order);
                }
            }
        }
        const Eigen::MatrixXd reducedCost = basis.transpose() * cost * basis;
        const Eigen::VectorXd dxReduced = constraints.transpose() * dx;
        const Eigen::VectorXd dyReduced = constraints.transpose() * dy;
        const Eigen::VectorXd dzReduced = constraints.transpose() * dz;
        const Eigen::VectorXd dxFree = dxReduced.head(freeCount);
        const Eigen::VectorXd dyFree = dyReduced.head(freeCount);
        const Eigen::VectorXd dzFree = dzReduced.head(freeCount);

        Eigen::VectorXd dxFixed = Eigen::VectorXd::Zero(fixedCount);
        Eigen::VectorXd dyFixed = Eigen::VectorXd::Zero(fixedCount);
        Eigen::VectorXd dzFixed = Eigen::VectorXd::Zero(fixedCount);
        if (fixedCount > 0) {
            const Eigen::MatrixXd freeFixed = reducedCost.block(0, freeCount, freeCount, fixedCount);
            const Eigen::MatrixXd fixedFixed = reducedCost.block(freeCount, freeCount, fixedCount, fixedCount);
            const Eigen::MatrixXd solver = fixedFixed.ldlt().solve(freeFixed.transpose());
            dxFixed = -solver * dxFree;
            dyFixed = -solver * dyFree;
            dzFixed = -solver * dzFree;
        }
        Eigen::VectorXd fullX(variableCount);
        Eigen::VectorXd fullY(variableCount);
        Eigen::VectorXd fullZ(variableCount);
        fullX << dxFree, dxFixed;
        fullY << dyFree, dyFixed;
        fullZ << dzFree, dzFixed;

        const Eigen::VectorXd px = basis * fullX;
        const Eigen::VectorXd py = basis * fullY;
        const Eigen::VectorXd pz = basis * fullZ;

        trajectory.reset();
        for (int i = 0; i < segmentCount; ++ i) {
            trajectory.addSegment(Polynomial(
                times(i),
                px.segment<6>(i * 6),
                py.segment<6>(i * 6),
                pz.segment<6>(i * 6)
            ));
        }
        return true;
    }

private:
    mutable double timeSum_{-1.0};
};

} // arc
