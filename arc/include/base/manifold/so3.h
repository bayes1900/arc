#pragma once

#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/so3.hpp>
#include <base/manifold/math.h>

namespace arc {
namespace manifold {

template <typename Scalar>
class ManifoldSo3;

template <typename Scalar>
inline ManifoldSo3<Scalar> expSO3(const Eigen::Matrix<Scalar, 3, 1>& omega, Scalar scale = Scalar(1));

template <typename Scalar>
inline Eigen::Matrix<Scalar, 3, 1> logSO3(const ManifoldSo3<Scalar>& orientation);

template <typename Scalar = double>
class ManifoldSo3 : public Sophus::SO3<Scalar> {
public:
    using Base = Sophus::SO3<Scalar>;
    using ScalarType = Scalar;

    static constexpr int DOF = 3;
    static constexpr int DIM = 3;
    static constexpr int TYP = 2;

    ManifoldSo3() = default;
    ManifoldSo3(const Base& value)
        : Base(value)
    {}
    ManifoldSo3(const Eigen::Quaternion<Scalar>& value)
        : Base(value)
    {}
    ManifoldSo3(Scalar w, Scalar x, Scalar y, Scalar z)
        : Base(Eigen::Quaternion<Scalar>(w, x, y, z))
    {}
    template <typename Derived>
    ManifoldSo3(const Eigen::MatrixBase<Derived>& matrix)
        : Base(typename Base::Transformation(matrix))
    {}
    ManifoldSo3 conjugate() const {
        return ManifoldSo3(this->inverse());
    }
    Eigen::Matrix<Scalar, 3, 3> toRotationMatrix() const {
        return this->matrix();
    }
    static ManifoldSo3 exp(const Eigen::Matrix<Scalar, 3, 1>& omega, Scalar scale = Scalar(1)) {
        const Scalar halfScale = scale / Scalar(2);
        const auto cosSinc = cosSincSqrt(halfScale * halfScale * omega.squaredNorm());
        const Eigen::Matrix<Scalar, 3, 1> vector = cosSinc.second * halfScale * omega;
        ManifoldSo3 result;
        auto& coefficients = result.unit_quaternion_nonconst().coeffs();
        coefficients(0) = vector(0);
        coefficients(1) = vector(1);
        coefficients(2) = vector(2);
        coefficients(3) = cosSinc.first;
        return result;
    }
    static Eigen::Matrix<Scalar, 3, 1> log(const ManifoldSo3& orientation) {
        const auto& coefficients = orientation.unit_quaternion().coeffs();
        const Eigen::Matrix<Scalar, 3, 1> vector = coefficients.template head<3>();
        Scalar norm = vector.norm();
        if (norm < tolerance<Scalar>()) {
            norm = tolerance<Scalar>();
        }
        const Scalar scale = Scalar(2) / norm * std::atan(norm / coefficients(3));
        return scale * vector;
    }
    void boxPlus(const Eigen::Matrix<Scalar, 3, 1>& delta, Scalar scale = Scalar(1)) {
        const Base& self = *this;
        *this = ManifoldSo3(self * static_cast<const Base&>(exp(delta, scale)));
    }
    void boxMinus(Eigen::Matrix<Scalar, 3, 1>& result, const ManifoldSo3& other) const {
        const Base product = static_cast<const Base&>(other).inverse() * static_cast<const Base&>(*this);
        result = log(ManifoldSo3(product));
    }
    void oPlus(const Eigen::Matrix<Scalar, 3, 1>& delta, Scalar scale = Scalar(1)) {
        const Base& self = *this;
        *this = ManifoldSo3(self * static_cast<const Base&>(exp(delta, scale)));
    }
};

// Free-function aliases over ManifoldSo3::exp/log.
// The implementation stays in ManifoldSo3 to preserve the upstream
// Taylor/quaternion path and the filter-critical scale semantics.
template <typename Scalar>
inline ManifoldSo3<Scalar> expSO3(const Eigen::Matrix<Scalar, 3, 1>& omega, Scalar scale) {
    return ManifoldSo3<Scalar>::exp(omega, scale);
}

template <typename Scalar>
inline Eigen::Matrix<Scalar, 3, 1> logSO3(const ManifoldSo3<Scalar>& orientation) {
    return ManifoldSo3<Scalar>::log(orientation);
}

} // manifold
} // arc
