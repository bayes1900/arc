#pragma once

#include <cmath>
#include <Eigen/Core>
#include <base/manifold/math.h>
#include <base/manifold/so3.h>
#include <base/manifold/vector.h>

namespace arc {
namespace manifold {

template <typename Scalar = double, int Den = 1, int Num = 1, int Type = 3>
class ManifoldS2 {
public:
    using ScalarType = Scalar;
    using Vector = ManifoldVector<3, Scalar>;

    static constexpr int DOF = 2;
    static constexpr int DIM = 3;
    static constexpr int TYP = 1;
    static constexpr Scalar LENGTH = Scalar(Den) / Scalar(Num);

    ManifoldS2() {
        if constexpr (Type == 3) {
            vec = LENGTH * Vector(Eigen::Matrix<Scalar, 3, 1>(Scalar(0), Scalar(0), Scalar(1)));
        } else if constexpr (Type == 2) {
            vec = LENGTH * Vector(Eigen::Matrix<Scalar, 3, 1>(Scalar(0), Scalar(1), Scalar(0)));
        } else {
            vec = LENGTH * Vector(Eigen::Matrix<Scalar, 3, 1>(Scalar(1), Scalar(0), Scalar(0)));
        }
    }
    ManifoldS2(Scalar x, Scalar y, Scalar z) : vec(Eigen::Matrix<Scalar, 3, 1>(x, y, z)) {
        vec.normalize();
        vec *= LENGTH;
    }
    explicit ManifoldS2(const Vector& value) : vec(value) {
        vec.normalize();
        vec *= LENGTH;
    }
    void boxPlus(const Eigen::Matrix<Scalar, 2, 1>& delta, Scalar scale = Scalar(1)) {
        Eigen::Matrix<Scalar, 3, 2> bx;
        s2Bx(bx);
        const Vector bu = bx * delta;
        const ManifoldSo3<Scalar> rotation = ManifoldSo3<Scalar>::exp(bu, scale);
        vec = rotation.toRotationMatrix() * vec;
    }
    void boxMinus(Eigen::Matrix<Scalar, 2, 1>& result, const ManifoldS2& other) const {
        const Scalar vSin = (hat<Scalar>(vec) * other.vec).norm();
        const Scalar vCos = vec.transpose() * other.vec;
        const Scalar theta = std::atan2(vSin, vCos);
        if (vSin < tolerance<Scalar>()) {
            if (std::fabs(theta) > tolerance<Scalar>()) {
                result(0) = Scalar(3.1415926);
                result(1) = Scalar(0);
            } else {
                result.setZero();
            }
            return;
        }
        ManifoldS2 otherCopy = other;
        Eigen::Matrix<Scalar, 3, 2> bx;
        otherCopy.s2Bx(bx);
        result = theta / vSin * bx.transpose() * hat<Scalar>(other.vec) * vec;
    }
    void oPlus(const Eigen::Matrix<Scalar, 3, 1>& delta, Scalar scale = Scalar(1)) {
        const ManifoldSo3<Scalar> rotation = ManifoldSo3<Scalar>::exp(delta, scale);
        vec = rotation.toRotationMatrix() * vec;
    }
    void s2Hat(Eigen::Matrix<Scalar, 3, 3>& result) const {
        result = hat<Scalar>(vec);
    }
    void s2Bx(Eigen::Matrix<Scalar, 3, 2>& result) const {
        if constexpr (Type == 3) {
            if (vec(2) + LENGTH > tolerance<Scalar>()) {
                result << LENGTH - vec(0) * vec(0) / (LENGTH + vec(2)),
                    -vec(0) * vec(1) / (LENGTH + vec(2)),
                    -vec(0) * vec(1) / (LENGTH + vec(2)),
                    LENGTH - vec(1) * vec(1) / (LENGTH + vec(2)),
                    -vec(0), -vec(1);
                result /= LENGTH;
            } else {
                result.setZero();
                result(1, 1) = Scalar(-1);
                result(2, 0) = Scalar(1);
            }
        } else if constexpr (Type == 2) {
            if (vec(1) + LENGTH > tolerance<Scalar>()) {
                result << LENGTH - vec(0) * vec(0) / (LENGTH + vec(1)),
                    -vec(0) * vec(2) / (LENGTH + vec(1)),
                    -vec(0), -vec(2),
                    -vec(0) * vec(2) / (LENGTH + vec(1)),
                    LENGTH - vec(2) * vec(2) / (LENGTH + vec(1));
                result /= LENGTH;
            } else {
                result.setZero();
                result(1, 1) = Scalar(-1);
                result(2, 0) = Scalar(1);
            }
        } else {
            if (vec(0) + LENGTH > tolerance<Scalar>()) {
                result << -vec(1), -vec(2),
                    LENGTH - vec(1) * vec(1) / (LENGTH + vec(0)),
                    -vec(2) * vec(1) / (LENGTH + vec(0)),
                    -vec(2) * vec(1) / (LENGTH + vec(0)),
                    LENGTH - vec(2) * vec(2) / (LENGTH + vec(0));
                result /= LENGTH;
            } else {
                result.setZero();
                result(1, 1) = Scalar(-1);
                result(2, 0) = Scalar(1);
            }
        }
    }
    void s2NxYy(Eigen::Matrix<Scalar, 2, 3>& result) const {
        Eigen::Matrix<Scalar, 3, 2> bx;
        s2Bx(bx);
        result = Scalar(1) / (LENGTH * LENGTH) * bx.transpose() * hat<Scalar>(vec);
    }
    void s2Mx(Eigen::Matrix<Scalar, 3, 2>& result, const Eigen::Matrix<Scalar, 2, 1>& delta) const {
        Eigen::Matrix<Scalar, 3, 2> bx;
        s2Bx(bx);
        if (delta.norm() < tolerance<Scalar>()) {
            result = -hat<Scalar>(vec) * bx;
            return;
        }
        const Vector bu = bx * delta;
        // FAST_LIO passes scalar(1/2), which evaluates to integer zero. Keep
        // the identity rotation for numerical parity with upstream.
        const ManifoldSo3<Scalar> rotation = ManifoldSo3<Scalar>::exp(bu, Scalar(0));
        result = -rotation.toRotationMatrix() * hat<Scalar>(vec) * aMatrix<Scalar>(bu).transpose() * bx;
    }
    operator const Vector&() const {
        return vec;
    }
    const Vector& vector() const {
        return vec;
    }
    Scalar operator[](int index) const {
        return vec[index];
    }

public:
    Vector vec;
};

} // manifold
} // arc
