#pragma once

#include <Eigen/Dense>

#include <type_traits>

namespace arc {

template <class Derived>
struct EvalOrReference {
    using Type = typename Eigen::internal::eval<Derived>::type;
};

template <typename Scalar,
    int Rows,
    int Cols,
    int Options,
    int MaxRows,
    int MaxCols,
    int MapOptions,
    typename StrideType>
struct EvalOrReference<
    Eigen::Map<Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>, MapOptions, StrideType>> {
    using Type = const Eigen::MatrixBase<
        Eigen::Map<Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>, MapOptions, StrideType>>&;
};

template <class DerivedPoint3D, class DerivedPoint2D, class DerivedJ3DPtr, class DerivedJparamPtr, int N>
constexpr inline void checkProjectionDerivedTypes() {
    static_assert(DerivedPoint3D::IsVectorAtCompileTime &&
            (DerivedPoint3D::SizeAtCompileTime == 3 || DerivedPoint3D::SizeAtCompileTime == 4),
        "point3D must be a vector of size 3 or 4");
    static_assert(DerivedPoint2D::IsVectorAtCompileTime && DerivedPoint2D::SizeAtCompileTime == 2,
        "point2D must be a vector of size 2");
    if constexpr (!std::is_same_v<DerivedJ3DPtr, std::nullptr_t>) {
        static_assert(std::is_pointer_v<DerivedJ3DPtr>, "J3D must be a pointer");
        using DerivedJ3D = std::remove_pointer_t<DerivedJ3DPtr>;
        static_assert(DerivedJ3D::RowsAtCompileTime == 2 &&
                int(DerivedJ3D::ColsAtCompileTime) == int(DerivedPoint3D::SizeAtCompileTime),
            "J3D must be 2 x point3D size");
    }
    if constexpr (!std::is_same_v<DerivedJparamPtr, std::nullptr_t>) {
        static_assert(std::is_pointer_v<DerivedJparamPtr>, "Jparam must be a pointer");
        using DerivedJparam = std::remove_pointer_t<DerivedJparamPtr>;
        static_assert(DerivedJparam::RowsAtCompileTime == 2 && DerivedJparam::ColsAtCompileTime == N,
            "Jparam must be 2 x N");
    }
}

template <class DerivedPoint2D, class DerivedPoint3D, class DerivedJ2DPtr, class DerivedJparamPtr, int N>
constexpr inline void checkUnprojectionDerivedTypes() {
    static_assert(DerivedPoint2D::IsVectorAtCompileTime && DerivedPoint2D::SizeAtCompileTime == 2,
        "point2D must be a vector of size 2");
    static_assert(DerivedPoint3D::IsVectorAtCompileTime &&
            (DerivedPoint3D::SizeAtCompileTime == 3 || DerivedPoint3D::SizeAtCompileTime == 4),
        "point3D must be a vector of size 3 or 4");
    if constexpr (!std::is_same_v<DerivedJ2DPtr, std::nullptr_t>) {
        static_assert(std::is_pointer_v<DerivedJ2DPtr>, "J2D must be a pointer");
        using DerivedJ2D = std::remove_pointer_t<DerivedJ2DPtr>;
        static_assert(DerivedJ2D::ColsAtCompileTime == 2 &&
                int(DerivedJ2D::RowsAtCompileTime) == int(DerivedPoint3D::SizeAtCompileTime),
            "J2D must be point3D size x 2");
    }
    if constexpr (!std::is_same_v<DerivedJparamPtr, std::nullptr_t>) {
        static_assert(std::is_pointer_v<DerivedJparamPtr>, "Jparam must be a pointer");
        using DerivedJparam = std::remove_pointer_t<DerivedJparamPtr>;
        static_assert(DerivedJparam::ColsAtCompileTime == N &&
                int(DerivedJparam::RowsAtCompileTime) == int(DerivedPoint3D::SizeAtCompileTime),
            "Jparam must be point3D size x N");
    }
}

} // arc
