#pragma once

#include <base/base.h>

#include <vio/camera/traits.h>

#include <base/liegroup.h>

namespace arc {

using std::sqrt;

template <typename Scalar = double>
class UnifiedCamera {
public:
    static constexpr int N = 5;
    using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
    using Vec4 = Eigen::Matrix<Scalar, 4, 1>;
    using VecN = Eigen::Matrix<Scalar, N, 1>;
    using Mat24 = Eigen::Matrix<Scalar, 2, 4>;
    using Mat2N = Eigen::Matrix<Scalar, 2, N>;
    using Mat42 = Eigen::Matrix<Scalar, 4, 2>;
    using Mat4N = Eigen::Matrix<Scalar, 4, N>;
    UnifiedCamera() { param_.setZero(); }
    explicit UnifiedCamera(const VecN& p) { param_ = p; }

    template <class Scalar2>
    UnifiedCamera<Scalar2> cast() const {
        return UnifiedCamera<Scalar2>(param_.template cast<Scalar2>());
    }
    static std::string getName() { return "ucm"; }

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
        const typename EvalOrReference<DerivedPoint3D>::Type p3dEval(p3d);
        const Scalar& fx = param_[0];
        const Scalar& fy = param_[1];
        const Scalar& cx = param_[2];
        const Scalar& cy = param_[3];
        const Scalar& alpha = param_[4];
        const Scalar& x = p3dEval[0];
        const Scalar& y = p3dEval[1];
        const Scalar& z = p3dEval[2];
        const Scalar r2 = x * x + y * y;
        const Scalar rho2 = r2 + z * z;
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
            const Scalar mid = -(alpha * x * y);
            const Scalar add = norm * rho;
            const Scalar addz = (alpha * z + (Scalar(1) - alpha) * rho);
            dProjDP3d->setZero();
            (*dProjDP3d)(0, 0) = fx * (add - x * x * alpha);
            (*dProjDP3d)(1, 0) = fy * mid;
            (*dProjDP3d)(0, 1) = fx * mid;
            (*dProjDP3d)(1, 1) = fy * (add - y * y * alpha);
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
        const Scalar& u = projEval[0];
        const Scalar& v = projEval[1];
        const Scalar xi = alpha / (Scalar(1) - alpha);
        const Scalar mxx = (u - cx) / fx;
        const Scalar myy = (v - cy) / fy;
        const Scalar mx = (Scalar(1) - alpha) * mxx;
        const Scalar my = (Scalar(1) - alpha) * myy;
        const Scalar r2 = mx * mx + my * my;
        const bool isValid = !static_cast<bool>((alpha > Scalar(0.5)) && (r2 >= Scalar(1) / ((Scalar(2) * alpha - Scalar(1)))));
        const Scalar xi2 = xi * xi;
        const Scalar n = sqrt(Scalar(1) + (Scalar(1) - xi2) * (r2));
        const Scalar m = (Scalar(1) + r2);
        const Scalar k = (xi + n) / m;
        p3d.setZero();
        p3d[0] = k * mx;
        p3d[1] = k * my;
        p3d[2] = k - xi;
        if constexpr (!std::is_same_v<DerivedJ2D, std::nullptr_t> || !std::is_same_v<DerivedJparam, std::nullptr_t>) {
            const Scalar dkDmx = -Scalar(2) * mx * (n + xi) / (m * m) + mx * (Scalar(1) - xi2) / (n * m);
            const Scalar dkDmy = -Scalar(2) * my * (n + xi) / (m * m) + my * (Scalar(1) - xi2) / (n * m);
            constexpr int SIZE_3D = DerivedPoint3D::SizeAtCompileTime;
            Eigen::Matrix<Scalar, SIZE_3D, 1> c0, c1;
            c0.setZero();
            c0(0) = (dkDmx * mx + k) / fx;
            c0(1) = dkDmx * my / fx;
            c0(2) = dkDmx / fx;
            c1.setZero();
            c1(0) = dkDmy * mx / fy;
            c1(1) = (dkDmy * my + k) / fy;
            c1(2) = dkDmy / fy;
            c0 *= (1 - alpha);
            c1 *= (1 - alpha);
            if constexpr (!std::is_same_v<DerivedJ2D, std::nullptr_t>) {
                ensure(dP3dDProj, "VIO check failed");
                dP3dDProj->col(0) = c0;
                dP3dDProj->col(1) = c1;
            } else {
                (void)(dP3dDProj);
            }
            if constexpr (!std::is_same_v<DerivedJparam, std::nullptr_t>) {
                ensure(dP3dDParam, "VIO check failed");
                const Scalar dXiDAlpha = Scalar(1) / ((Scalar(1) - alpha) * (Scalar(1) - alpha));
                const Scalar dMDAlpha = -Scalar(2) * (Scalar(1) - alpha) * (mxx * mxx + myy * myy);
                const Scalar dNDAlpha = -(mxx * mxx + myy * myy) / n;
                const Scalar dkDAlpha = ((dXiDAlpha + dNDAlpha) * m - dMDAlpha * (xi + n)) / (m * m);
                dP3dDParam->setZero();
                dP3dDParam->col(0) = -mxx * c0;
                dP3dDParam->col(1) = -myy * c1;
                dP3dDParam->col(2) = -c0;
                dP3dDParam->col(3) = -c1;
                (*dP3dDParam)(0, 4) = dkDAlpha * mx - k * mxx;
                (*dP3dDParam)(1, 4) = dkDAlpha * my - k * myy;
                (*dP3dDParam)(2, 4) = dkDAlpha - dXiDAlpha;
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
    }

    void operator+=(const VecN& inc) {
        param_ += inc;

        param_[4] = std::clamp(param_[4], Scalar(0), Scalar(1));
    }
    const VecN& getParam() const { return param_; }
    static std::vector<UnifiedCamera> getTestProjections() {
        std::vector<UnifiedCamera> res;
        VecN vec1;
        vec1 << 460.76484651566468, 459.4051018049483, 365.8937161309615, 249.33499869752445, 0.5903365915227143;
        res.emplace_back(vec1);
        vec1 << 191.14799816648748, 191.13150946585135, 254.95857715233118, 256.8815466235898, 0.6291060871161842;
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
