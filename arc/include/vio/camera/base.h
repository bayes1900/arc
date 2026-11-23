#pragma once

#include <vio/camera/doublesphere.h>
#include <vio/camera/extended.h>
#include <vio/camera/kannalabrandt4.h>
#include <vio/camera/pinhole.h>
#include <vio/camera/pinholeradtan8.h>
#include <vio/camera/unified.h>

#include <variant>

namespace arc {

using std::sqrt;

template <typename Scalar>
class GenericCamera {
    using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
    using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
    using Vec4 = Eigen::Matrix<Scalar, 4, 1>;
    using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Mat24 = Eigen::Matrix<Scalar, 2, 4>;
    using Mat42 = Eigen::Matrix<Scalar, 4, 2>;
    using Mat4 = Eigen::Matrix<Scalar, 4, 4>;
    using VariantT = std::variant<ExtendedUnifiedCamera<Scalar>,
        DoubleSphereCamera<Scalar>,
        KannalaBrandtCamera4<Scalar>,
        UnifiedCamera<Scalar>,
        PinholeCamera<Scalar>,
        PinholeRadtan8Camera<Scalar>>;

public:
    template <typename Scalar2>
    GenericCamera<Scalar2> cast() const {
        GenericCamera<Scalar2> res;
        std::visit([&](const auto& v) { res.variant = v.template cast<Scalar2>(); }, variant);
        return res;
    }
    int getN() const {
        int res;
        std::visit([&](const auto& v) { res = v.N; }, variant);
        return res;
    }
    std::string getName() const {
        std::string res;
        std::visit([&](const auto& v) { res = v.getName(); }, variant);
        return res;
    }
    void setFromInit(const Vec4& init) {
        std::visit([&](auto& v) { return v.setFromInit(init); }, variant);
    }
    void applyInc(const VecX& inc) {
        std::visit([&](auto& v) { return v += inc; }, variant);
    }
    VecX getParam() const {
        VecX res;
        std::visit([&](const auto& cam) { res = cam.getParam(); }, variant);
        return res;
    }

    template <typename DerivedJ3DPtr = std::nullptr_t>
    bool project(const Vec4& p3d, Vec2& proj, DerivedJ3DPtr dProjDP3d = nullptr) const {
        if constexpr (!std::is_same_v<DerivedJ3DPtr, std::nullptr_t>) {
            static_assert(std::is_pointer_v<DerivedJ3DPtr>);
            using DerivedJ3D = typename std::remove_pointer<DerivedJ3DPtr>::type;
            static_assert(DerivedJ3D::RowsAtCompileTime == 2 && DerivedJ3D::ColsAtCompileTime == 4,
                "DerivedJ3D must be a 2x4 matrix");
        }
        bool res;
        std::visit([&](const auto& cam) { res = cam.project(p3d, proj, dProjDP3d); }, variant);
        return res;
    }

    template <typename DerivedJ2DPtr = std::nullptr_t>
    bool unproject(const Vec2& proj, Vec4& p3d, DerivedJ2DPtr dP3dDProj = nullptr) const {
        if constexpr (!std::is_same_v<DerivedJ2DPtr, std::nullptr_t>) {
            static_assert(std::is_pointer_v<DerivedJ2DPtr>);
            using DerivedJ2D = typename std::remove_pointer<DerivedJ2DPtr>::type;
            static_assert(DerivedJ2D::RowsAtCompileTime == 4 && DerivedJ2D::ColsAtCompileTime == 2,
                "DerivedJ2D must be a 4x2 matrix");
        }
        bool res;
        std::visit([&](const auto& cam) { res = cam.unproject(proj, p3d, dP3dDProj); }, variant);
        return res;
    }

    template <typename DerivedJ3DPtr = std::nullptr_t>
    bool project(const Vec3& p3d, Vec2& proj, DerivedJ3DPtr dProjDP3d = nullptr) const {
        if constexpr (!std::is_same_v<DerivedJ3DPtr, std::nullptr_t>) {
            static_assert(std::is_pointer_v<DerivedJ3DPtr>);
            using DerivedJ3D = typename std::remove_pointer<DerivedJ3DPtr>::type;
            static_assert(DerivedJ3D::RowsAtCompileTime == 2 && DerivedJ3D::ColsAtCompileTime == 3,
                "DerivedJ3D must be a 2x3 matrix");
        }
        bool res;
        std::visit([&](const auto& cam) { res = cam.project(p3d, proj, dProjDP3d); }, variant);
        return res;
    }

    template <typename DerivedJ2DPtr = std::nullptr_t>
    bool unproject(const Vec2& proj, Vec3& p3d, DerivedJ2DPtr dP3dDProj = nullptr) const {
        if constexpr (!std::is_same_v<DerivedJ2DPtr, std::nullptr_t>) {
            static_assert(std::is_pointer_v<DerivedJ2DPtr>);
            using DerivedJ2D = typename std::remove_pointer<DerivedJ2DPtr>::type;
            static_assert(DerivedJ2D::RowsAtCompileTime == 3 && DerivedJ2D::ColsAtCompileTime == 2,
                "DerivedJ2D must be a 3x2 matrix");
        }
        bool res;
        std::visit([&](const auto& cam) { res = cam.unproject(proj, p3d, dP3dDProj); }, variant);
        return res;
    }
    void project(
        const std::vector<Vec3>& p3d,
        const Mat4& T_c_w,
        std::vector<Vec2>& proj,
        std::vector<bool>& projSuccess
    ) const {
        std::visit(
            [&](const auto& cam) {
                proj.resize(p3d.size());
                projSuccess.resize(p3d.size());
                for (size_t i = 0; i < p3d.size(); ++ i) {
                    projSuccess[i] = cam.project(T_c_w * p3d[i].homogeneous(), proj[i]);
                }
            },
            variant);
    }
    void project(
        const std::vector<Vec4>& p3d,
        const Mat4& T_c_w,
        std::vector<Vec2>& proj,
        std::vector<bool>& projSuccess
    ) const {
        std::visit(
            [&](const auto& cam) {
                proj.resize(p3d.size());
                projSuccess.resize(p3d.size());
                for (size_t i = 0; i < p3d.size(); ++ i) {
                    projSuccess[i] = cam.project(T_c_w * p3d[i], proj[i]);
                }
            },
            variant);
    }
    void project(
        const std::vector<Vec4>& p3d,
        std::vector<Vec2>& proj,
        std::vector<bool>& projSuccess
    ) const {
        std::visit(
            [&](const auto& cam) {
                proj.resize(p3d.size());
                projSuccess.resize(p3d.size());
                for (size_t i = 0; i < p3d.size(); ++ i) {
                    projSuccess[i] = cam.project(p3d[i], proj[i]);
                }
            },
            variant);
    }
    void project(
        const std::vector<Vec4>& p3d,
        const Mat4& T_c_w,
        std::vector<Vec2>& proj,
        std::vector<bool>& projSuccess,
        std::vector<Vec2>& polarAzimuthalAngle
    ) const {
        std::visit(
            [&](const auto& cam) {
                proj.resize(p3d.size());
                projSuccess.resize(p3d.size());
                polarAzimuthalAngle.resize(p3d.size());
                for (size_t i = 0; i < p3d.size(); ++ i) {
                    Vec4 p3dt = T_c_w * p3d[i];
                    projSuccess[i] = cam.project(p3dt, proj[i]);
                    Scalar r2 = p3dt[0] * p3dt[0] + p3dt[1] * p3dt[1];
                    Scalar r = std::sqrt(r2);
                    polarAzimuthalAngle[i][0] = std::atan2(r, p3dt[2]);
                    polarAzimuthalAngle[i][1] = std::atan2(p3dt[0], p3dt[1]);
                }
            },
            variant);
    }
    void unproject(
        const std::vector<Vec2>& proj,
        std::vector<Vec4>& p3d,
        std::vector<bool>& unprojSuccess
    ) const {
        std::visit(
            [&](const auto& cam) {
                p3d.resize(proj.size());
                unprojSuccess.resize(proj.size());
                for (size_t i = 0; i < p3d.size(); ++ i) {
                    unprojSuccess[i] = cam.unproject(proj[i], p3d[i]);
                }
            },
            variant);
    }
    static GenericCamera<Scalar> fromString(const std::string& name);

    VariantT variant;

private:
    template <int I>
    static void visitAllTypes_(GenericCamera<Scalar>& res, const std::string& name) {
        if constexpr (I >= 0) {
            using CamT = typename std::variant_alternative<I, VariantT>::type;
            if (CamT::getName() == name) {
                CamT val;
                res.variant = val;
            }
            visitAllTypes_<I - 1>(res, name);
        }
    }
};
} // arc
