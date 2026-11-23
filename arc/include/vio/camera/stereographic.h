#pragma once

#include <Eigen/Dense>

#include <base/base.h>

namespace arc {

class StereographicParam {
public:
    static inline Vec2d project(const Vec4d& p3d, Mat24d* dRDP = nullptr) {
        const double sqrt = p3d.head<3>().norm();
        const double norm = p3d[2] + sqrt;
        const double normInv = double(1) / norm;
        Vec2d res(p3d[0] * normInv, p3d[1] * normInv);
        if (dRDP) {
            double normInv2 = normInv * normInv;
            double tmp = -normInv2 / sqrt;
            (*dRDP).setZero();
            (*dRDP)(0, 0) = normInv + p3d[0] * p3d[0] * tmp;
            (*dRDP)(1, 0) = p3d[0] * p3d[1] * tmp;
            (*dRDP)(1, 1) = normInv + p3d[1] * p3d[1] * tmp;
            (*dRDP)(0, 1) = p3d[0] * p3d[1] * tmp;
            (*dRDP)(0, 2) = p3d[0] * norm * tmp;
            (*dRDP)(1, 2) = p3d[1] * norm * tmp;
            (*dRDP)(0, 3) = double(0);
            (*dRDP)(1, 3) = double(0);
        }
        return res;
    }
    static inline Vec4d unproject(const Vec2d& proj, Mat42d* dRDP = nullptr) {
        const double x2 = proj[0] * proj[0];
        const double y2 = proj[1] * proj[1];
        const double r2 = x2 + y2;
        const double normInv = double(2) / (double(1) + r2);
        Vec4d res(proj[0] * normInv, proj[1] * normInv, normInv - double(1), double(0));
        if (dRDP) {
            const double normInv2 = normInv * normInv;
            const double xy = proj[0] * proj[1];
            (*dRDP)(0, 0) = (normInv - x2 * normInv2);
            (*dRDP)(0, 1) = -xy * normInv2;
            (*dRDP)(1, 0) = -xy * normInv2;
            (*dRDP)(1, 1) = (normInv - y2 * normInv2);
            (*dRDP)(2, 0) = -proj[0] * normInv2;
            (*dRDP)(2, 1) = -proj[1] * normInv2;
            (*dRDP)(3, 0) = double(0);
            (*dRDP)(3, 1) = double(0);
        }
        return res;
    }
};

} // arc
