#pragma once

#include <base/base.h>

#include <vio/camera/traits.h>

#include <base/liegroup.h>

namespace arc {

using std::sqrt;

template <typename Scalar>
class ExtendedUnifiedCamera {
public:
    static constexpr int N = 6;
    using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
    using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
    using Vec4 = Eigen::Matrix<Scalar, 4, 1>;
    using VecN = Eigen::Matrix<Scalar, N, 1>;
    using Mat24 = Eigen::Matrix<Scalar, 2, 4>;
    using Mat2N = Eigen::Matrix<Scalar, 2, N>;
    using Mat42 = Eigen::Matrix<Scalar, 4, 2>;
    using Mat4N = Eigen::Matrix<Scalar, 4, N>;
    ExtendedUnifiedCamera() { param_.setZero(); }
    explicit ExtendedUnifiedCamera(const VecN& p) { param_ = p; }

    template <class Scalar2>
    ExtendedUnifiedCamera<Scalar2> cast() const {
        return ExtendedUnifiedCamera<Scalar2>(param_.template cast<Scalar2>());
    }
    static std::string getName() { return "eucm"; }

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
        const Scalar& alpha = param_[4];
        const Scalar& beta = param_[5];
        const Scalar& x = p3dEval[0];
        const Scalar& y = p3dEval[1];
        const Scalar& z = p3dEval[2];
        const Scalar r2 = x * x + y * y;
        const Scalar rho2 = beta * r2 + z * z;
        const Scalar rho = sqrt(rho2);
        const Scalar norm = alpha * rho + (Scalar(1) - alpha) * z;
        const Scalar mx = x / norm;
        const Scalar my = y / norm;
        proj = Vec2(fx * mx + cx, fy * my + cy);
        const Scalar w = alpha > Scalar(0.5) ? (Scalar(1) - alpha) / alpha : alpha / (Scalar(1) - alpha);
        const bool isValid = (z > -w * rho);
        if constexpr (!std::is_same_v<DerivedJ3D, std::nullptr_t>) {
            ensure(dProjDP3d, "VIO check failed");
            const Scalar denom = norm * norm * rho;
            const Scalar mid = -(alpha * beta * x * y);
            const Scalar add = norm * rho;
            const Scalar addz = (alpha * z + (Scalar(1) - alpha) * rho);
            (*dProjDP3d).setZero();
            (*dProjDP3d)(0, 0) = fx * (add - x * x * alpha * beta);
            (*dProjDP3d)(1, 0) = fy * mid;
            (*dProjDP3d)(0, 1) = fx * mid;
            (*dProjDP3d)(1, 1) = fy * (add - y * y * alpha * beta);
            (*dProjDP3d)(0, 2) = -fx * x * addz;
            (*dProjDP3d)(1, 2) = -fy * y * addz;
            (*dProjDP3d) /= denom;
        } else {
            (void)(dProjDP3d);
        }
        if constexpr (!std::is_same_v<DerivedJparam, std::nullptr_t>) {
            ensure(dProjDParam, "VIO check failed");
            const Scalar norm2 = norm * norm;
            (*dProjDParam).setZero();
            (*dProjDParam)(0, 0) = mx;
            (*dProjDParam)(0, 2) = Scalar(1);
            (*dProjDParam)(1, 1) = my;
            (*dProjDParam)(1, 3) = Scalar(1);
            const Scalar tmpX = -fx * x / norm2;
            const Scalar tmpY = -fy * y / norm2;
            const Scalar tmp4 = (rho - z);
            (*dProjDParam)(0, 4) = tmpX * tmp4;
            (*dProjDParam)(1, 4) = tmpY * tmp4;
            const Scalar tmp5 = Scalar(0.5) * alpha * r2 / rho;
            (*dProjDParam)(0, 5) = tmpX * tmp5;
            (*dProjDParam)(1, 5) = tmpY * tmp5;
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
        const Scalar& alpha = param_[4];
        const Scalar& beta = param_[5];
        const Scalar mx = (projEval[0] - cx) / fx;
        const Scalar my = (projEval[1] - cy) / fy;
        const Scalar r2 = mx * mx + my * my;
        const Scalar gamma = Scalar(1) - alpha;
        const bool isValid = !static_cast<bool>(alpha > Scalar(0.5) && (r2 >= Scalar(1) / ((alpha - gamma) * beta)));
        const Scalar tmp1 = (Scalar(1) - alpha * alpha * beta * r2);
        const Scalar tmpSqrt = sqrt(Scalar(1) - (alpha - gamma) * beta * r2);
        const Scalar tmp2 = (alpha * tmpSqrt + gamma);
        const Scalar k = tmp1 / tmp2;
        const Scalar norm = sqrt(r2 + k * k);
        p3d.setZero();
        p3d(0) = mx / norm;
        p3d(1) = my / norm;
        p3d(2) = k / norm;
        if constexpr (!std::is_same_v<DerivedJ2D, std::nullptr_t> || !std::is_same_v<DerivedJparam, std::nullptr_t>) {
            const Scalar norm2 = norm * norm;
            const Scalar tmp2Sq = tmp2 * tmp2;
            const Scalar dKDR2 = Scalar(0.5) * alpha * beta * (-Scalar(2) * alpha * tmp2 + tmp1 * (alpha - gamma) / tmpSqrt) / tmp2Sq;
            const Scalar dNormInvDR2 = -Scalar(0.5) * (Scalar(1) + Scalar(2) * k * dKDR2) / norm2;
            constexpr int SIZE_3D = DerivedPoint3D::SizeAtCompileTime;
            Eigen::Matrix<Scalar, SIZE_3D, 1> c0, c1;
            c0.setZero();
            c0[0] = (1 + 2 * mx * mx * dNormInvDR2);
            c0[1] = (2 * my * mx * dNormInvDR2);
            c0[2] = 2 * mx * (k * dNormInvDR2 + dKDR2);
            c0 /= fx * norm;
            c1.setZero();
            c1[0] = (2 * my * mx * dNormInvDR2);
            c1[1] = (1 + 2 * my * my * dNormInvDR2);
            c1[2] = 2 * my * (k * dNormInvDR2 + dKDR2);
            c1 /= fy * norm;
            if constexpr (!std::is_same_v<DerivedJ2D, std::nullptr_t>) {
                ensure(dP3dDProj, "VIO check failed");
                dP3dDProj->col(0) = c0;
                dP3dDProj->col(1) = c1;
            } else {
                (void)(dP3dDProj);
            }
            if constexpr (!std::is_same_v<DerivedJparam, std::nullptr_t>) {
                ensure(dP3dDParam, "VIO check failed");
                dP3dDParam->setZero();
                (*dP3dDParam).col(0) = -c0 * mx;
                (*dP3dDParam).col(1) = -c1 * my;
                (*dP3dDParam).col(2) = -c0;
                (*dP3dDParam).col(3) = -c1;
                const Scalar dKDAlpha = (-Scalar(2) * alpha * beta * r2 * tmp2 -
                                               (tmpSqrt - alpha * beta * r2 / tmpSqrt - Scalar(1)) * tmp1) /
                    tmp2Sq;
                const Scalar dKDBeta = alpha * r2 * (Scalar(0.5) * tmp1 * (alpha - gamma) / tmpSqrt - alpha * tmp2) / tmp2Sq;
                const Scalar dNormInvDK = -k / norm2;
                (*dP3dDParam)(0, 4) = mx * dNormInvDK * dKDAlpha;
                (*dP3dDParam)(1, 4) = my * dNormInvDK * dKDAlpha;
                (*dP3dDParam)(2, 4) = (k * dNormInvDK + 1) * dKDAlpha;
                dP3dDParam->col(4) /= norm;
                (*dP3dDParam)(0, 5) = mx * dNormInvDK * dKDBeta;
                (*dP3dDParam)(1, 5) = my * dNormInvDK * dKDBeta;
                (*dP3dDParam)(2, 5) = (k * dNormInvDK + 1) * dKDBeta;
                dP3dDParam->col(5) /= norm;
            } else {
                (void)(dP3dDParam);
            }
        } else {
            (void)(dP3dDProj);
            (void)(dP3dDParam);
        }
        return isValid;
    }
    void setFromInit(const Vec4& init) {
        param_[0] = init[0];
        param_[1] = init[1];
        param_[2] = init[2];
        param_[3] = init[3];
        param_[4] = 0.5;
        param_[5] = 1;
    }

    void operator+=(const VecN& inc) {
        param_ += inc;

        param_[4] = std::clamp(param_[4], Scalar(0), Scalar(1));
        if (param_[5] < Sophus::Constants<Scalar>::epsilonSqrt()) {
            param_[5] = Sophus::Constants<Scalar>::epsilonSqrt();
        }
    }
    const VecN& getParam() const { return param_; }
    static std::vector<ExtendedUnifiedCamera> getTestProjections() {
        std::vector<ExtendedUnifiedCamera> res;
        VecN vec1;
        vec1 << 460.76484651566468, 459.4051018049483, 365.8937161309615, 249.33499869752445, 0.5903365915227143,
            1.127468196965374;
        res.emplace_back(vec1);
        vec1 << 191.14799816648748, 191.13150946585135, 254.95857715233118, 256.8815466235898, 0.6291060871161842,
            1.0418067403139693;
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
