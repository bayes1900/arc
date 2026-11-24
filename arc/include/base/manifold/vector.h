#pragma once

#include <Eigen/Core>

namespace arc {
namespace manifold {

template <int D, typename Scalar = double>
class ManifoldVector : public Eigen::Matrix<Scalar, D, 1> {
public:
    using Base = Eigen::Matrix<Scalar, D, 1>;
    using ScalarType = Scalar;

    static constexpr int DOF = D;
    static constexpr int DIM = D;
    static constexpr int TYP = 0;

    using Base::operator=;

    ManifoldVector()
        : Base(Base::Zero())
    {}
    ManifoldVector(const Base& value)
        : Base(value)
    {}
    template <typename Derived>
    ManifoldVector(const Eigen::MatrixBase<Derived>& value)
        : Base(value)
    {}
    void boxPlus(const Eigen::Matrix<Scalar, D, 1>& delta, Scalar scale = Scalar(1)) {
        *this += scale * delta;
    }
    void boxMinus(Eigen::Matrix<Scalar, D, 1>& result, const ManifoldVector& other) const {
        result = *this - other;
    }
    void oPlus(const Eigen::Matrix<Scalar, D, 1>& delta, Scalar scale = Scalar(1)) {
        *this += scale * delta;
    }
};

} // manifold
} // arc
