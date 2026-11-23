#pragma once

#include <sophus/se3.hpp>
#include <sophus/sim3.hpp>
#include <base/base.h>

namespace arc {

template <typename Scalar>
inline typename Sophus::SE3<Scalar>::Tangent se3Logd(const Sophus::SE3<Scalar>& se3) {
    typename Sophus::SE3<Scalar>::Tangent upsilon_omega;
    upsilon_omega.template tail<3>() = se3.so3().log();
    upsilon_omega.template head<3>() = se3.translation();
    return upsilon_omega;
}

template <typename Derived>
inline Sophus::SE3<typename Derived::Scalar> se3Expd(const Eigen::MatrixBase<Derived>& upsilon_omega) {
    static_assert(Derived::SizeAtCompileTime != Eigen::Dynamic, "Derived must be fixed size");
    static_assert(Derived::IsVectorAtCompileTime && Derived::SizeAtCompileTime == 6, "Derived must be a vector of size 6");
    using Scalar = typename Derived::Scalar;
    return Sophus::SE3<Scalar>(Sophus::SO3<Scalar>::exp(upsilon_omega.template tail<3>()), upsilon_omega.template head<3>());
}

template <typename Scalar>
inline typename Sophus::Sim3<Scalar>::Tangent sim3Logd(const Sophus::Sim3<Scalar>& sim3) {
    typename Sophus::Sim3<Scalar>::Tangent upsilon_omega_sigma;
    upsilon_omega_sigma.template tail<4>() = sim3.rxso3().log();
    upsilon_omega_sigma.template head<3>() = sim3.translation();
    return upsilon_omega_sigma;
}

template <typename Derived>
inline Sophus::Sim3<typename Derived::Scalar> sim3Expd(const Eigen::MatrixBase<Derived>& upsilon_omega_sigma) {
    static_assert(Derived::SizeAtCompileTime != Eigen::Dynamic, "Derived must be fixed size");
    static_assert(Derived::IsVectorAtCompileTime && Derived::SizeAtCompileTime == 7, "Derived must be a vector of size 7");
    using Scalar = typename Derived::Scalar;
    return Sophus::Sim3<Scalar>(Sophus::RxSO3<Scalar>::exp(upsilon_omega_sigma.template tail<4>()), upsilon_omega_sigma.template head<3>());
}

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
inline void leftJacobianSO3(const Eigen::MatrixBase<Derived1>& phi, const Eigen::MatrixBase<Derived2>& J_phi) {
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
        J += phi_hat * (1 - std::cos(phi_norm)) / phi_norm2;
        J += phi_hat2 * (phi_norm - std::sin(phi_norm)) / phi_norm3;
    } else {
        J += phi_hat / 2;
        J += phi_hat2 / 6;
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

template <typename Derived1, typename Derived2>
inline void rightJacobianSE3Decoupled(
    const Eigen::MatrixBase<Derived1>& phi, const Eigen::MatrixBase<Derived2>& J_phi) {
    static_assert(Derived1::SizeAtCompileTime != Eigen::Dynamic, "Derived1 must be fixed size");
    static_assert(Derived2::SizeAtCompileTime != Eigen::Dynamic, "Derived2 must be fixed size");
    static_assert(Derived1::IsVectorAtCompileTime && Derived1::SizeAtCompileTime == 6, "Derived1 must be a vector of size 6");
    static_assert(Derived2::RowsAtCompileTime == 6 && Derived2::ColsAtCompileTime == 6, "Derived2 must be a 6x6 matrix");
    using Scalar = typename Derived1::Scalar;
    Eigen::MatrixBase<Derived2>& J = const_cast<Eigen::MatrixBase<Derived2>&>(J_phi);
    J.setZero();
    Eigen::Matrix<Scalar, 3, 1> omega = phi.template tail<3>();
    rightJacobianSO3(omega, J.template bottomRightCorner<3, 3>());
    J.template topLeftCorner<3, 3>() = Sophus::SO3<Scalar>::exp(omega).inverse().matrix();
}

template <typename Derived1, typename Derived2>
inline void rightJacobianInvSE3Decoupled(
    const Eigen::MatrixBase<Derived1>& phi, const Eigen::MatrixBase<Derived2>& J_phi) {
    static_assert(Derived1::SizeAtCompileTime != Eigen::Dynamic, "Derived1 must be fixed size");
    static_assert(Derived2::SizeAtCompileTime != Eigen::Dynamic, "Derived2 must be fixed size");
    static_assert(Derived1::IsVectorAtCompileTime && Derived1::SizeAtCompileTime == 6, "Derived1 must be a vector of size 6");
    static_assert(Derived2::RowsAtCompileTime == 6 && Derived2::ColsAtCompileTime == 6, "Derived2 must be a 6x6 matrix");
    using Scalar = typename Derived1::Scalar;
    Eigen::MatrixBase<Derived2>& J = const_cast<Eigen::MatrixBase<Derived2>&>(J_phi);
    J.setZero();
    Eigen::Matrix<Scalar, 3, 1> omega = phi.template tail<3>();
    rightJacobianInvSO3(omega, J.template bottomRightCorner<3, 3>());
    J.template topLeftCorner<3, 3>() = Sophus::SO3<Scalar>::exp(omega).matrix();
}

template <typename Derived1, typename Derived2>
inline void rightJacobianSim3Decoupled(
    const Eigen::MatrixBase<Derived1>& phi, const Eigen::MatrixBase<Derived2>& J_phi) {
    static_assert(Derived1::SizeAtCompileTime != Eigen::Dynamic, "Derived1 must be fixed size");
    static_assert(Derived2::SizeAtCompileTime != Eigen::Dynamic, "Derived2 must be fixed size");
    static_assert(Derived1::IsVectorAtCompileTime && Derived1::SizeAtCompileTime == 7, "Derived1 must be a vector of size 7");
    static_assert(Derived2::RowsAtCompileTime == 7 && Derived2::ColsAtCompileTime == 7, "Derived2 must be a 7x7 matrix");
    using Scalar = typename Derived1::Scalar;
    Eigen::MatrixBase<Derived2>& J = const_cast<Eigen::MatrixBase<Derived2>&>(J_phi);
    J.setZero();
    Eigen::Matrix<Scalar, 4, 1> omega = phi.template tail<4>();
    rightJacobianSO3(omega.template head<3>(), J.template block<3, 3>(3, 3));
    J.template topLeftCorner<3, 3>() = Sophus::RxSO3<Scalar>::exp(omega).inverse().matrix();
    J(6, 6) = 1;
}

template <typename Derived1, typename Derived2>
inline void rightJacobianInvSim3Decoupled(
    const Eigen::MatrixBase<Derived1>& phi, const Eigen::MatrixBase<Derived2>& J_phi) {
    static_assert(Derived1::SizeAtCompileTime != Eigen::Dynamic, "Derived1 must be fixed size");
    static_assert(Derived2::SizeAtCompileTime != Eigen::Dynamic, "Derived2 must be fixed size");
    static_assert(Derived1::IsVectorAtCompileTime && Derived1::SizeAtCompileTime == 7, "Derived1 must be a vector of size 7");
    static_assert(Derived2::RowsAtCompileTime == 7 && Derived2::ColsAtCompileTime == 7, "Derived2 must be a 7x7 matrix");
    using Scalar = typename Derived1::Scalar;
    Eigen::MatrixBase<Derived2>& J = const_cast<Eigen::MatrixBase<Derived2>&>(J_phi);
    J.setZero();
    Eigen::Matrix<Scalar, 4, 1> omega = phi.template tail<4>();
    rightJacobianInvSO3(omega.template head<3>(), J.template block<3, 3>(3, 3));
    J.template topLeftCorner<3, 3>() = Sophus::RxSO3<Scalar>::exp(omega).matrix();
    J(6, 6) = 1;
}

} // arc
