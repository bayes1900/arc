#pragma once

#include <base/iarc.h>

namespace arc {

struct ControlPointFitter {
    inline static Eigen::MatrixXd fitControlPoininterval(
        const std::vector<Eigen::Vector3d>& samples,
        const std::vector<Eigen::Vector3d>& boundaryDerivatives,
        int order,
        double interval
    ) {
        if (order < 3 || order > 5) {
            return {};
        }
        const int K = static_cast<int>(samples.size());
        const int numControlPoints = K + order - 1;
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(K + 4, numControlPoints);
        Eigen::VectorXd bx(K + 4), by(K + 4), bz(K + 4);
        Eigen::VectorXd ptToPos(order), ptToVel(order), ptToAcc(order);
        if (order == 3) {
            ptToPos << 1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0;
            ptToVel << -1.0 / (2.0 * interval), 0.0, 1.0 / (2.0 * interval);
            ptToAcc << 1.0 / (interval * interval), -2.0 / (interval * interval), 1.0 / (interval * interval);
        } else if (order == 4) {
            ptToPos << 1.0 / 24.0, 11.0 / 24.0, 11.0 / 24.0, 1.0 / 24.0;
            ptToVel << -1.0 / (6.0 * interval), -3.0 / (6.0 * interval), 3.0 / (6.0 * interval), 1.0 / (6.0 * interval);
            ptToAcc << 1.0 / (2.0 * interval * interval), -1.0 / (2.0 * interval * interval), -1.0 / (2.0 * interval * interval), 1.0 / (2.0 * interval * interval);
        } else {
            ptToPos << 1.0 / 120.0, 26.0 / 120.0, 66.0 / 120.0, 26.0 / 120.0, 1.0 / 120.0;
            ptToVel << -1.0 / (24.0 * interval), -10.0 / (24.0 * interval), 0.0, 10.0 / (24.0 * interval), 1.0 / (24.0 * interval);
            ptToAcc << 1.0 / (6.0 * interval * interval), 2.0 / (6.0 * interval * interval), -6.0 / (6.0 * interval * interval), 2.0 / (6.0 * interval * interval), 1.0 / (6.0 * interval * interval);
        }
        for (int i = 0; i < K; ++ i) {
            A.block(i, i, 1, order) = ptToPos.transpose();
        }
        A.block(K, 0, 1, order) = ptToVel.transpose();
        A.block(K + 1, K - 1, 1, order) = ptToVel.transpose();
        A.block(K + 2, 0, 1, order) = ptToAcc.transpose();
        A.block(K + 3, K - 1, 1, order) = ptToAcc.transpose();
        for (int i = 0; i < K; ++ i) {
            bx(i) = samples[i](0);
            by(i) = samples[i](1);
            bz(i) = samples[i](2);
        }
        for (int i = 0; i < 4; ++ i) {
            bx(K + i) = boundaryDerivatives[i](0);
            by(K + i) = boundaryDerivatives[i](1);
            bz(K + i) = boundaryDerivatives[i](2);
        }
        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(A);
        Eigen::MatrixXd controlPoints(numControlPoints, 3);
        controlPoints.col(0) = qr.solve(bx);
        controlPoints.col(1) = qr.solve(by);
        controlPoints.col(2) = qr.solve(bz);
        return controlPoints;
    }
};

} // arc
