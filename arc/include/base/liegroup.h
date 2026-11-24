#pragma once

// SO(3) Jacobians used by VIO IMU preintegration.
// LIO IKFoM-style manifold adapters live in base/manifold.

#include <sophus/se3.hpp>
#include <base/base.h>

namespace arc {

template <typename Derived1, typename Derived2>
inline void rightJacobianSO3(const Eigen::MatrixBase<Derived1>& phi, const Eigen::MatrixBase<Derived2>& J_phi) {
    static_assert(Derived1::SizeAtCompileTime != Eigen::Dynamic, "Derived1 must be fixed size");
    static_assert(Derived2::SizeAtCompileTime != Eigen::Dynamic, "Derived2 must be fixed size");
    static_assert(Derived1::IsVectorAtCompileTime && Derived1::SizeAtCompileTime == 3, "Derived1 must be a vector of size 3");
    static_assert(Derived2::RowsAtCompileTime == 3 && Derived2::ColsAtCompileTime == 3, "Derived2 must be a 3x3 matrix");
    using Scalar = typename Derived1::Scalar;
    Eigen::MatrixBase<Derived2>& J = const_cast<Eigen::MatrixBase<Derived2>&>(J_phi);
    Scalar phi_norm2 = phi.squaredNorm();
    Eigen::Matrix<Scalar, 3, 3> phi_hat = Sophus::SO3<Scalar>::hat(phi);
    Eigen::Matrix<Scalar, 3, 3> phi_hat2 = phi_hat * phi_hat;
    J.setIdentity();
    if (phi_norm2 > Sophus::Constants<Scalar>::epsilon()) {
        Scalar phi_norm = std::sqrt(phi_norm2);
        Scalar phi_norm3 = phi_norm2 * phi_norm;
        J -= phi_hat * (1 - std::cos(phi_norm)) / phi_norm2;
        J += phi_hat2 * (phi_norm - std::sin(phi_norm)) / phi_norm3;
    } else {
        J -= phi_hat / 2;
        J += phi_hat2 / 6;
    }
}

template <typename Derived1, typename Derived2>
inline void rightJacobianInvSO3(const Eigen::MatrixBase<Derived1>& phi, const Eigen::MatrixBase<Derived2>& J_phi) {
    static_assert(Derived1::SizeAtCompileTime != Eigen::Dynamic, "Derived1 must be fixed size");
    static_assert(Derived2::SizeAtCompileTime != Eigen::Dynamic, "Derived2 must be fixed size");
    static_assert(Derived1::IsVectorAtCompileTime && Derived1::SizeAtCompileTime == 3, "Derived1 must be a vector of size 3");
    static_assert(Derived2::RowsAtCompileTime == 3 && Derived2::ColsAtCompileTime == 3, "Derived2 must be a 3x3 matrix");
    using Scalar = typename Derived1::Scalar;
    Eigen::MatrixBase<Derived2>& J = const_cast<Eigen::MatrixBase<Derived2>&>(J_phi);
    Scalar phi_norm2 = phi.squaredNorm();
    Eigen::Matrix<Scalar, 3, 3> phi_hat = Sophus::SO3<Scalar>::hat(phi);
    Eigen::Matrix<Scalar, 3, 3> phi_hat2 = phi_hat * phi_hat;
    J.setIdentity();
    J += phi_hat / 2;
    if (phi_norm2 > Sophus::Constants<Scalar>::epsilon()) {
        Scalar phi_norm = std::sqrt(phi_norm2);
        arc::ensure(phi_norm <= M_PI + Sophus::Constants<Scalar>::epsilon(), "VIO check failed");
        if (phi_norm < M_PI - Sophus::Constants<Scalar>::epsilonSqrt()) {
            J += phi_hat2 * (1 / phi_norm2 - (1 + std::cos(phi_norm)) / (2 * phi_norm * std::sin(phi_norm)));
        } else {
            J += phi_hat2 / (M_PI * M_PI);
        }
    } else {
        J += phi_hat2 / 12;
    }
}

template <typename Derived1, typename Derived2>
inline void leftJacobianInvSO3(const Eigen::MatrixBase<Derived1>& phi, const Eigen::MatrixBase<Derived2>& J_phi) {
    static_assert(Derived1::SizeAtCompileTime != Eigen::Dynamic, "Derived1 must be fixed size");
    static_assert(Derived2::SizeAtCompileTime != Eigen::Dynamic, "Derived2 must be fixed size");
    static_assert(Derived1::IsVectorAtCompileTime && Derived1::SizeAtCompileTime == 3, "Derived1 must be a vector of size 3");
    static_assert(Derived2::RowsAtCompileTime == 3 && Derived2::ColsAtCompileTime == 3, "Derived2 must be a 3x3 matrix");
    using Scalar = typename Derived1::Scalar;
    Eigen::MatrixBase<Derived2>& J = const_cast<Eigen::MatrixBase<Derived2>&>(J_phi);
    Scalar phi_norm2 = phi.squaredNorm();
    Eigen::Matrix<Scalar, 3, 3> phi_hat = Sophus::SO3<Scalar>::hat(phi);
    Eigen::Matrix<Scalar, 3, 3> phi_hat2 = phi_hat * phi_hat;
    J.setIdentity();
    J -= phi_hat / 2;
    if (phi_norm2 > Sophus::Constants<Scalar>::epsilon()) {
        Scalar phi_norm = std::sqrt(phi_norm2);
        arc::ensure(phi_norm <= M_PI + Sophus::Constants<Scalar>::epsilon(), "VIO check failed");
        if (phi_norm < M_PI - Sophus::Constants<Scalar>::epsilonSqrt()) {
            J += phi_hat2 * (1 / phi_norm2 - (1 + std::cos(phi_norm)) / (2 * phi_norm * std::sin(phi_norm)));
        } else {
            J += phi_hat2 / (M_PI * M_PI);
        }
    } else {
        J += phi_hat2 / 12;
    }
}

} // arc
