#pragma once

#include <base/base.h>

#include <vio/camera/traits.h>

#include <base/liegroup.h>

namespace arc {

using std::sqrt;

template <typename Scalar = double>
class DoubleSphereCamera {
public:
    static constexpr int N = 6;
    using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
    using Vec4 = Eigen::Matrix<Scalar, 4, 1>;
    using VecN = Eigen::Matrix<Scalar, N, 1>;
    using Mat24 = Eigen::Matrix<Scalar, 2, 4>;
    using Mat2N = Eigen::Matrix<Scalar, 2, N>;
    using Mat42 = Eigen::Matrix<Scalar, 4, 2>;
    using Mat4N = Eigen::Matrix<Scalar, 4, N>;
    DoubleSphereCamera() { param_.setZero(); }
    explicit DoubleSphereCamera(const VecN& p) { param_ = p; }

    template <class Scalar2>
    DoubleSphereCamera<Scalar2> cast() const {
        return DoubleSphereCamera<Scalar2>(param_.template cast<Scalar2>());
    }
    static std::string getName() { return "ds"; }

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
        const Scalar& xi = param_[4];
        const Scalar& alpha = param_[5];
        const Scalar& x = p3dEval[0];
        const Scalar& y = p3dEval[1];
        const Scalar& z = p3dEval[2];
        const Scalar xx = x * x;
        const Scalar yy = y * y;
        const Scalar zz = z * z;
        const Scalar r2 = xx + yy;
        const Scalar d1Sq = r2 + zz;
        const Scalar d1 = sqrt(d1Sq);
        const Scalar w1 = alpha > Scalar(0.5) ? (Scalar(1) - alpha) / alpha : alpha / (Scalar(1) - alpha);
        const Scalar w2 = (w1 + xi) / sqrt(Scalar(2) * w1 * xi + xi * xi + Scalar(1));
        const bool isValid = (z > -w2 * d1);
        const Scalar k = xi * d1 + z;
        const Scalar kk = k * k;
        const Scalar d2Sq = r2 + kk;
        const Scalar d2 = sqrt(d2Sq);
        const Scalar norm = alpha * d2 + (Scalar(1) - alpha) * k;
        const Scalar mx = x / norm;
        const Scalar my = y / norm;
        proj[0] = fx * mx + cx;
        proj[1] = fy * my + cy;
        if constexpr (!std::is_same_v<DerivedJ3D, std::nullptr_t>) {
            ensure(dProjDP3d, "VIO check failed");
            const Scalar norm2 = norm * norm;
            const Scalar xy = x * y;
            const Scalar tt2 = xi * z / d1 + Scalar(1);
            const Scalar dNormDR2 = (xi * (Scalar(1) - alpha) / d1 + alpha * (xi * k / d1 + Scalar(1)) / d2) / norm2;
            const Scalar tmp2 = ((Scalar(1) - alpha) * tt2 + alpha * k * tt2 / d2) / norm2;
            dProjDP3d->setZero();
            (*dProjDP3d)(0, 0) = fx * (Scalar(1) / norm - xx * dNormDR2);
            (*dProjDP3d)(1, 0) = -fy * xy * dNormDR2;
            (*dProjDP3d)(0, 1) = -fx * xy * dNormDR2;
            (*dProjDP3d)(1, 1) = fy * (Scalar(1) / norm - yy * dNormDR2);
            (*dProjDP3d)(0, 2) = -fx * x * tmp2;
            (*dProjDP3d)(1, 2) = -fy * y * tmp2;
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
            const Scalar tmp4 = (alpha - Scalar(1) - alpha * k / d2) * d1 / norm2;
            const Scalar tmp5 = (k - d2) / norm2;
            (*dProjDParam)(0, 4) = fx * x * tmp4;
            (*dProjDParam)(1, 4) = fy * y * tmp4;
            (*dProjDParam)(0, 5) = fx * x * tmp5;
            (*dProjDParam)(1, 5) = fy * y * tmp5;
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
        const Scalar& xi = param_[4];
        const Scalar& alpha = param_[5];
        const Scalar mx = (projEval[0] - cx) / fx;
        const Scalar my = (projEval[1] - cy) / fy;
        const Scalar r2 = mx * mx + my * my;
        const bool isValid = !static_cast<bool>(alpha > Scalar(0.5) && (r2 >= Scalar(1) / (Scalar(2) * alpha - Scalar(1))));
        const Scalar xi2Sq = alpha * alpha;
        const Scalar xi1Sq = xi * xi;
        const Scalar sqrt2 = sqrt(Scalar(1) - (Scalar(2) * alpha - Scalar(1)) * r2);
        const Scalar norm2 = alpha * sqrt2 + Scalar(1) - alpha;
        const Scalar mz = (Scalar(1) - xi2Sq * r2) / norm2;
        const Scalar mz2 = mz * mz;
        const Scalar norm1 = mz2 + r2;
        const Scalar sqrt1 = sqrt(mz2 + (Scalar(1) - xi1Sq) * r2);
        const Scalar k = (mz * xi + sqrt1) / norm1;
        p3d.setZero();
        p3d[0] = k * mx;
        p3d[1] = k * my;
        p3d[2] = k * mz - xi;
        if constexpr (!std::is_same_v<DerivedJ2D, std::nullptr_t> || !std::is_same_v<DerivedJparam, std::nullptr_t>) {
            const Scalar norm2Sq = norm2 * norm2;
            const Scalar norm1Sq = norm1 * norm1;
            const Scalar dMzDR2 = (Scalar(0.5) * alpha - xi2Sq) * (r2 * xi2Sq - Scalar(1)) / (sqrt2 * norm2Sq) - xi2Sq / norm2;
            const Scalar dMzDMx = 2 * mx * dMzDR2;
            const Scalar dMzDMy = 2 * my * dMzDR2;
            const Scalar dKDMz = (norm1 * (xi * sqrt1 + mz) - 2 * mz * (mz * xi + sqrt1) * sqrt1) / (norm1Sq * sqrt1);
            const Scalar dKDR2 =
                (xi * dMzDR2 + Scalar(0.5) / sqrt1 * (Scalar(2) * mz * dMzDR2 + Scalar(1) - xi1Sq)) / norm1 -
                (mz * xi + sqrt1) * (Scalar(2) * mz * dMzDR2 + Scalar(1)) / norm1Sq;
            const Scalar dKDMx = dKDR2 * 2 * mx;
            const Scalar dKDMy = dKDR2 * 2 * my;
            constexpr int SIZE_3D = DerivedPoint3D::SizeAtCompileTime;
            Eigen::Matrix<Scalar, SIZE_3D, 1> c0, c1;
            c0.setZero();
            c0[0] = (mx * dKDMx + k);
            c0[1] = my * dKDMx;
            c0[2] = (mz * dKDMx + k * dMzDMx);
            c0 /= fx;
            c1.setZero();
            c1[0] = mx * dKDMy;
            c1[1] = (my * dKDMy + k);
            c1[2] = (mz * dKDMy + k * dMzDMy);
            c1 /= fy;
            if constexpr (!std::is_same_v<DerivedJ2D, std::nullptr_t>) {
                ensure(dP3dDProj, "VIO check failed");
                dP3dDProj->col(0) = c0;
                dP3dDProj->col(1) = c1;
            } else {
                (void)(dP3dDProj);
            }
            if constexpr (!std::is_same_v<DerivedJparam, std::nullptr_t>) {
                ensure(dP3dDParam, "VIO check failed");
                const Scalar dKDXi1 = (mz * sqrt1 - xi * r2) / (sqrt1 * norm1);
                const Scalar dMzDXi2 = ((Scalar(1) - r2 * xi2Sq) * (r2 * alpha / sqrt2 - sqrt2 + Scalar(1)) / norm2 -
                                              Scalar(2) * r2 * alpha) /
                    norm2;
                const Scalar dKDXi2 = dKDMz * dMzDXi2;
                dP3dDParam->setZero();
                (*dP3dDParam).col(0) = -c0 * mx;
                (*dP3dDParam).col(1) = -c1 * my;
                (*dP3dDParam).col(2) = -c0;
                (*dP3dDParam).col(3) = -c1;
                (*dP3dDParam)(0, 4) = mx * dKDXi1;
                (*dP3dDParam)(1, 4) = my * dKDXi1;
                (*dP3dDParam)(2, 4) = mz * dKDXi1 - 1;
                (*dP3dDParam)(0, 5) = mx * dKDXi2;
                (*dP3dDParam)(1, 5) = my * dKDXi2;
                (*dP3dDParam)(2, 5) = mz * dKDXi2 + k * dMzDXi2;
            } else {
                (void)(dP3dDParam);
                (void)(dKDMz);
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
        param_[4] = 0;
        param_[5] = 0.5;
    }

    void operator+=(const VecN& inc) {
        param_ += inc;
        param_[4] = std::clamp(param_[4], Scalar(-1), Scalar(1));
        param_[5] = std::clamp(param_[5], Scalar(0), Scalar(1));
    }
    const VecN& getParam() const { return param_; }
    static std::vector<DoubleSphereCamera> getTestProjections() {
        std::vector<DoubleSphereCamera> res;
        VecN vec1;
        vec1 << 0.5 * 805, 0.5 * 800, 505, 509, 0.5 * -0.150694, 0.5 * 1.48785;
        res.emplace_back(vec1);
        return res;
    }

private:
    VecN param_;
};

} // arc
