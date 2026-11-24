#pragma once

#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>
#include <Eigen/Core>
#include <sophus/so3.hpp>

namespace arc {
namespace manifold {

template <typename T>
inline constexpr T tolerance() {
    return std::is_same_v<T, float> ? T(1e-5) : T(1e-11);
}

template <typename T>
inline Eigen::Matrix<T, 3, 3> hat(const Eigen::Matrix<T, 3, 1>& vector) {
    return Sophus::SO3<T>::hat(vector);
}

template <typename T>
inline std::pair<T, T> cosSincSqrt(const T& squaredAngle) {
    const T taylor0Bound = std::numeric_limits<T>::epsilon();
    const T taylor2Bound = std::sqrt(taylor0Bound);
    const T taylorNBound = std::sqrt(taylor2Bound);

    if (squaredAngle >= taylorNBound) {
        const T angle = std::sqrt(squaredAngle);
        return {std::cos(angle), std::sin(angle) / angle};
    }
    const T inverse[] = {
        T(1) / T(3), T(1) / T(4), T(1) / T(5),
        T(1) / T(6), T(1) / T(7), T(1) / T(8), T(1) / T(9)
    };
    T cosine = T(1);
    T sinc = T(1);
    T term = T(-0.5) * squaredAngle;
    for (int i = 0; i < 3; ++ i) {
        cosine += term;
        term *= inverse[2 * i];
        sinc += term;
        term *= -inverse[2 * i + 1] * squaredAngle;
    }
    return {cosine, sinc};
}

// Left Jacobian of SO(3); IKFoM calls this A_matrix.
template <typename T>
inline Eigen::Matrix<T, 3, 3> aMatrix(const Eigen::Matrix<T, 3, 1>& vector) {
    const T squaredNorm = vector.squaredNorm();
    const T norm = std::sqrt(squaredNorm);
    if (norm < tolerance<T>()) {
        return Eigen::Matrix<T, 3, 3>::Identity();
    }
    const Eigen::Matrix<T, 3, 3> skew = hat(vector);
    return Eigen::Matrix<T, 3, 3>::Identity()
        + (T(1) - std::cos(norm)) / squaredNorm * skew
        + (T(1) - std::sin(norm) / norm) / squaredNorm * skew * skew;
}

template <typename T>
inline Eigen::Matrix<T, 3, 3> leftJacobianSO3(const Eigen::Matrix<T, 3, 1>& vector) {
    return aMatrix(vector);
}

// Rodrigues matrix convenience; expSO3 in so3.h keeps the filter-critical
// quaternion path and scale semantics.
template <typename T, typename Ts>
inline Eigen::Matrix<T, 3, 3> so3Exp(const Eigen::Matrix<T, 3, 1>& omega, const Ts& dt) {
    const Eigen::Matrix<T, 3, 1> scaled = omega * static_cast<T>(dt);
    const T angle = scaled.norm();
    const Eigen::Matrix<T, 3, 3> eye = Eigen::Matrix<T, 3, 3>::Identity();
    if (angle > T(1e-7)) {
        const Eigen::Matrix<T, 3, 1> axis = scaled / angle;
        const Eigen::Matrix<T, 3, 3> skew = hat(axis);
        return eye + std::sin(angle) * skew + (T(1) - std::cos(angle)) * skew * skew;
    }
    return eye;
}

template <typename T>
inline Eigen::Matrix<T, 3, 3> so3Exp(const Eigen::Matrix<T, 3, 1>& omega) {
    return so3Exp(omega, T(1));
}

} // manifold
} // arc
