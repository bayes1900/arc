#pragma once

#include <base/base.h>

#include <vio/camera/traits.h>

#include <base/liegroup.h>

namespace arc {

using std::sqrt;

template <typename Scalar = double>
class PinholeCamera {
public:
    static constexpr int N = 4;
    using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
    using Vec4 = Eigen::Matrix<Scalar, 4, 1>;
    using VecN = Eigen::Matrix<Scalar, N, 1>;
    using Mat24 = Eigen::Matrix<Scalar, 2, 4>;
    using Mat2N = Eigen::Matrix<Scalar, 2, N>;
    using Mat42 = Eigen::Matrix<Scalar, 4, 2>;
    using Mat4N = Eigen::Matrix<Scalar, 4, N>;
    PinholeCamera() { param_.setZero(); }
    explicit PinholeCamera(const VecN& p) { param_ = p; }

    template <class Scalar2>
    PinholeCamera<Scalar2> cast() const {
        return PinholeCamera<Scalar2>(param_.template cast<Scalar2>());
    }
    static std::string getName() { return "pinhole"; }

    template <class DerivedPoint3D,
        class DerivedPoint2D,
        class DerivedJ3D = std::nullptr_t,
        class DerivedJparam = std::nullptr_t>
    bool project(
        const Eigen::MatrixBase<DerivedPoint3D>& p3d,
        Eigen::MatrixBase<DerivedPoint2D>& proj,
        DerivedJ3D dProjDP3d = nullptr,
        DerivedJparam dProjDParam = nullptr
    ) const {
        checkProjectionDerivedTypes<DerivedPoint3D, DerivedPoint2D, DerivedJ3D, DerivedJparam, N>();
        const typename EvalOrReference<DerivedPoint3D>::Type p3dEval(p3d);
        const Scalar& fx = param_[0];
        const Scalar& fy = param_[1];
        const Scalar& cx = param_[2];
        const Scalar& cy = param_[3];
        const Scalar& x = p3dEval[0];
        const Scalar& y = p3dEval[1];
        const Scalar& z = p3dEval[2];
        proj[0] = fx * x / z + cx;
        proj[1] = fy * y / z + cy;
        const bool isValid = z >= Sophus::Constants<Scalar>::epsilonSqrt();
        if constexpr (!std::is_same_v<DerivedJ3D, std::nullptr_t>) {
            ensure(dProjDP3d, "VIO check failed");
            dProjDP3d->setZero();
            const Scalar z2 = z * z;
            (*dProjDP3d)(0, 0) = fx / z;
            (*dProjDP3d)(0, 2) = -fx * x / z2;
            (*dProjDP3d)(1, 1) = fy / z;
            (*dProjDP3d)(1, 2) = -fy * y / z2;
        } else {
            (void)(dProjDP3d);
        }
        if constexpr (!std::is_same_v<DerivedJparam, std::nullptr_t>) {
            ensure(dProjDParam, "VIO check failed");
            dProjDParam->setZero();
            (*dProjDParam)(0, 0) = x / z;
            (*dProjDParam)(0, 2) = Scalar(1);
            (*dProjDParam)(1, 1) = y / z;
            (*dProjDParam)(1, 3) = Scalar(1);
        } else {
            (void)(dProjDParam);
        }
        return isValid;
    }

    template <class DerivedPoint2D,
        class DerivedPoint3D,
        class DerivedJ2D = std::nullptr_t,
        class DerivedJparam = std::nullptr_t>
    bool unproject(
        const Eigen::MatrixBase<DerivedPoint2D>& proj,
        Eigen::MatrixBase<DerivedPoint3D>& p3d,
        DerivedJ2D dP3dDProj = nullptr,
        DerivedJparam dP3dDParam = nullptr
    ) const {
        checkUnprojectionDerivedTypes<DerivedPoint2D, DerivedPoint3D, DerivedJ2D, DerivedJparam, N>();
        const typename EvalOrReference<DerivedPoint2D>::Type projEval(proj);
        const Scalar& fx = param_[0];
        const Scalar& fy = param_[1];
        const Scalar& cx = param_[2];
        const Scalar& cy = param_[3];
        const Scalar mx = (projEval[0] - cx) / fx;
        const Scalar my = (projEval[1] - cy) / fy;
        const Scalar r2 = mx * mx + my * my;
        const Scalar norm = sqrt(Scalar(1) + r2);
        const Scalar normInv = Scalar(1) / norm;
        p3d.setZero();
        p3d[0] = mx * normInv;
        p3d[1] = my * normInv;
        p3d[2] = normInv;
        if constexpr (!std::is_same_v<DerivedJ2D, std::nullptr_t> || !std::is_same_v<DerivedJparam, std::nullptr_t>) {
            const Scalar dNormInvDR2 = -Scalar(0.5) * normInv * normInv * normInv;
            constexpr int SIZE_3D = DerivedPoint3D::SizeAtCompileTime;
            Eigen::Matrix<Scalar, SIZE_3D, 1> c0, c1;
            c0.setZero();
            c0(0) = (normInv + 2 * mx * mx * dNormInvDR2) / fx;
            c0(1) = (2 * my * mx * dNormInvDR2) / fx;
            c0(2) = 2 * mx * dNormInvDR2 / fx;
            c1.setZero();
            c1(0) = (2 * my * mx * dNormInvDR2) / fy;
            c1(1) = (normInv + 2 * my * my * dNormInvDR2) / fy;
            c1(2) = 2 * my * dNormInvDR2 / fy;
            if constexpr (!std::is_same_v<DerivedJ2D, std::nullptr_t>) {
                ensure(dP3dDProj, "VIO check failed");
                dP3dDProj->col(0) = c0;
                dP3dDProj->col(1) = c1;
            } else {
                (void)(dP3dDProj);
            }
            if constexpr (!std::is_same_v<DerivedJparam, std::nullptr_t>) {
                ensure(dP3dDParam, "VIO check failed");
                dP3dDParam->col(2) = -c0;
                dP3dDParam->col(3) = -c1;
                dP3dDParam->col(0) = -c0 * mx;
                dP3dDParam->col(1) = -c1 * my;
            } else {
                (void)(dP3dDParam);
            }
        } else {
            (void)(dP3dDProj);
            (void)(dP3dDParam);
        }
        return true;
    }
    void setFromInit(const Vec4& init) {
        param_[0] = init[0];
        param_[1] = init[1];
        param_[2] = init[2];
        param_[3] = init[3];
    }

    void operator+=(const VecN& inc) { param_ += inc; }
    const VecN& getParam() const { return param_; }
    static std::vector<PinholeCamera> getTestProjections() {
        std::vector<PinholeCamera> res;
        VecN vec1;
        vec1 << 460.76484651566468, 459.4051018049483, 365.8937161309615, 249.33499869752445;
        res.emplace_back(vec1);
        vec1 << 191.14799816648748, 191.13150946585135, 254.95857715233118, 256.8815466235898;
        res.emplace_back(vec1);
        return res;
    }
    static std::vector<Eigen::Vector2i> getTestResolutions() {
        std::vector<Eigen::Vector2i> res;
        res.emplace_back(752, 480);
        res.emplace_back(512, 512);
        return res;
    }

private:
    VecN param_;
};

} // arc
