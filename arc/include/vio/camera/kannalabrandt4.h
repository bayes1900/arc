#pragma once

#include <base/base.h>

#include <vio/camera/traits.h>

#include <base/liegroup.h>

namespace arc {

using std::cos;
using std::sin;
using std::sqrt;

template <typename Scalar = double>
class KannalaBrandtCamera4 {
public:
    static constexpr int N = 8;
    using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
    using Vec4 = Eigen::Matrix<Scalar, 4, 1>;
    using VecN = Eigen::Matrix<Scalar, N, 1>;
    using Mat24 = Eigen::Matrix<Scalar, 2, 4>;
    using Mat2N = Eigen::Matrix<Scalar, 2, N>;
    using Mat42 = Eigen::Matrix<Scalar, 4, 2>;
    using Mat4N = Eigen::Matrix<Scalar, 4, N>;
    KannalaBrandtCamera4() { param_.setZero(); }
    explicit KannalaBrandtCamera4(const VecN& p) { param_ = p; }
    static std::string getName() { return "kb4"; }

    template <class Scalar2>
    KannalaBrandtCamera4<Scalar2> cast() const {
        return KannalaBrandtCamera4<Scalar2>(param_.template cast<Scalar2>());
    }

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
        const Scalar& k1 = param_[4];
        const Scalar& k2 = param_[5];
        const Scalar& k3 = param_[6];
        const Scalar& k4 = param_[7];
        const Scalar& x = p3dEval[0];
        const Scalar& y = p3dEval[1];
        const Scalar& z = p3dEval[2];
        const Scalar r2 = x * x + y * y;
        const Scalar r = sqrt(r2);
        bool isValid = true;
        if (r > Sophus::Constants<Scalar>::epsilonSqrt()) {
            const Scalar theta = atan2(r, z);
            const Scalar theta2 = theta * theta;
            Scalar rTheta = k4 * theta2;
            rTheta += k3;
            rTheta *= theta2;
            rTheta += k2;
            rTheta *= theta2;
            rTheta += k1;
            rTheta *= theta2;
            rTheta += 1;
            rTheta *= theta;
            const Scalar mx = x * rTheta / r;
            const Scalar my = y * rTheta / r;
            proj[0] = fx * mx + cx;
            proj[1] = fy * my + cy;
            if constexpr (!std::is_same_v<DerivedJ3D, std::nullptr_t>) {
                ensure(dProjDP3d, "VIO check failed");
                const Scalar dRDX = x / r;
                const Scalar dRDY = y / r;
                const Scalar tmp = (z * z + r2);
                const Scalar dThetaDX = dRDX * z / tmp;
                const Scalar dThetaDY = dRDY * z / tmp;
                const Scalar dThetaDZ = -r / tmp;
                Scalar dRThetaDTheta = Scalar(9) * k4 * theta2;
                dRThetaDTheta += Scalar(7) * k3;
                dRThetaDTheta *= theta2;
                dRThetaDTheta += Scalar(5) * k2;
                dRThetaDTheta *= theta2;
                dRThetaDTheta += Scalar(3) * k1;
                dRThetaDTheta *= theta2;
                dRThetaDTheta += Scalar(1);
                dProjDP3d->setZero();
                (*dProjDP3d)(0, 0) = fx * (rTheta * r + x * r * dRThetaDTheta * dThetaDX - x * x * rTheta / r) / r2;
                (*dProjDP3d)(1, 0) = fy * y * (dRThetaDTheta * dThetaDX * r - x * rTheta / r) / r2;
                (*dProjDP3d)(0, 1) = fx * x * (dRThetaDTheta * dThetaDY * r - y * rTheta / r) / r2;
                (*dProjDP3d)(1, 1) = fy * (rTheta * r + y * r * dRThetaDTheta * dThetaDY - y * y * rTheta / r) / r2;
                (*dProjDP3d)(0, 2) = fx * x * dRThetaDTheta * dThetaDZ / r;
                (*dProjDP3d)(1, 2) = fy * y * dRThetaDTheta * dThetaDZ / r;
            } else {
                (void)(dProjDP3d);
            }
            if constexpr (!std::is_same_v<DerivedJparam, std::nullptr_t>) {
                ensure(dProjDParam, "VIO check failed");
                (*dProjDParam).setZero();
                (*dProjDParam)(0, 0) = mx;
                (*dProjDParam)(0, 2) = Scalar(1);
                (*dProjDParam)(1, 1) = my;
                (*dProjDParam)(1, 3) = Scalar(1);
                (*dProjDParam)(0, 4) = fx * x * theta * theta2 / r;
                (*dProjDParam)(1, 4) = fy * y * theta * theta2 / r;
                dProjDParam->col(5) = dProjDParam->col(4) * theta2;
                dProjDParam->col(6) = dProjDParam->col(5) * theta2;
                dProjDParam->col(7) = dProjDParam->col(6) * theta2;
            } else {
                (void)(dProjDParam);
            }
        } else {
            if (z < Sophus::Constants<Scalar>::epsilonSqrt()) {
                isValid = false;
            }
            proj[0] = fx * x / z + cx;
            proj[1] = fy * y / z + cy;
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
        }
        return isValid;
    }

    template <int ITER>
    Scalar solveTheta(const Scalar& rTheta, Scalar& dFuncDTheta) const {
        const Scalar& k1 = param_[4];
        const Scalar& k2 = param_[5];
        const Scalar& k3 = param_[6];
        const Scalar& k4 = param_[7];
        Scalar theta = rTheta;
        for (int i = ITER; i > 0; i--) {
            Scalar theta2 = theta * theta;
            Scalar func = k4 * theta2;
            func += k3;
            func *= theta2;
            func += k2;
            func *= theta2;
            func += k1;
            func *= theta2;
            func += Scalar(1);
            func *= theta;
            dFuncDTheta = Scalar(9) * k4 * theta2;
            dFuncDTheta += Scalar(7) * k3;
            dFuncDTheta *= theta2;
            dFuncDTheta += Scalar(5) * k2;
            dFuncDTheta *= theta2;
            dFuncDTheta += Scalar(3) * k1;
            dFuncDTheta *= theta2;
            dFuncDTheta += Scalar(1);
            theta += (rTheta - func) / dFuncDTheta;
        }
        return theta;
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
        Scalar theta(0);
        Scalar sinTheta(0);
        Scalar cosTheta(1);
        Scalar thetad = sqrt(mx * mx + my * my);
        Scalar scaling(1);
        Scalar dFuncDTheta(0);
        if (thetad > Sophus::Constants<Scalar>::epsilonSqrt()) {
            theta = solveTheta<3>(thetad, dFuncDTheta);
            sinTheta = sin(theta);
            cosTheta = cos(theta);
            scaling = sinTheta / thetad;
        }
        p3d.setZero();
        p3d[0] = mx * scaling;
        p3d[1] = my * scaling;
        p3d[2] = cosTheta;
        if constexpr (!std::is_same_v<DerivedJ2D, std::nullptr_t> || !std::is_same_v<DerivedJparam, std::nullptr_t>) {
            Scalar dThetadDMx = Scalar(0);
            Scalar dThetadDMy = Scalar(0);
            Scalar dScalingDThetad = Scalar(0);
            Scalar dCosDThetad = Scalar(0);
            Scalar dScalingDK1 = Scalar(0);
            Scalar dCosDK1 = Scalar(0);
            Scalar theta2 = Scalar(0);
            if (thetad > Sophus::Constants<Scalar>::epsilonSqrt()) {
                dThetadDMx = mx / thetad;
                dThetadDMy = my / thetad;
                theta2 = theta * theta;
                dScalingDThetad = (thetad * cosTheta / dFuncDTheta - sinTheta) / (thetad * thetad);
                dCosDThetad = sinTheta / dFuncDTheta;
                dScalingDK1 = -cosTheta * theta * theta2 / (dFuncDTheta * thetad);
                dCosDK1 = dCosDThetad * theta * theta2;
            }
            const Scalar dRes0DMx = scaling + mx * dScalingDThetad * dThetadDMx;
            const Scalar dRes0DMy = mx * dScalingDThetad * dThetadDMy;
            const Scalar dRes1DMx = my * dScalingDThetad * dThetadDMx;
            const Scalar dRes1DMy = scaling + my * dScalingDThetad * dThetadDMy;
            const Scalar dRes2DMx = -dCosDThetad * dThetadDMx;
            const Scalar dRes2DMy = -dCosDThetad * dThetadDMy;
            constexpr int SIZE_3D = DerivedPoint3D::SizeAtCompileTime;
            Eigen::Matrix<Scalar, SIZE_3D, 1> c0, c1;
            c0.setZero();
            c0(0) = dRes0DMx / fx;
            c0(1) = dRes1DMx / fx;
            c0(2) = dRes2DMx / fx;
            c1.setZero();
            c1(0) = dRes0DMy / fy;
            c1(1) = dRes1DMy / fy;
            c1(2) = dRes2DMy / fy;
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
                dP3dDParam->col(2) = -c0;
                dP3dDParam->col(3) = -c1;
                dP3dDParam->col(0) = -c0 * mx;
                dP3dDParam->col(1) = -c1 * my;
                (*dP3dDParam)(0, 4) = mx * dScalingDK1;
                (*dP3dDParam)(1, 4) = my * dScalingDK1;
                (*dP3dDParam)(2, 4) = dCosDK1;
                dP3dDParam->col(5) = dP3dDParam->col(4) * theta2;
                dP3dDParam->col(6) = dP3dDParam->col(5) * theta2;
                dP3dDParam->col(7) = dP3dDParam->col(6) * theta2;
            } else {
                (void)(dP3dDParam);
                (void)(dScalingDK1);
                (void)(dCosDK1);
            }
        } else {
            (void)(dP3dDProj);
            (void)(dP3dDParam);
        }
        return true;
    }

    void operator+=(const VecN& inc) { param_ += inc; }
    const VecN& getParam() const { return param_; }
    void setFromInit(const Vec4& init) {
        param_[0] = init[0];
        param_[1] = init[1];
        param_[2] = init[2];
        param_[3] = init[3];
        param_[4] = 0;
        param_[5] = 0;
        param_[6] = 0;
        param_[7] = 0;
    }
    static std::vector<KannalaBrandtCamera4> getTestProjections() {
        std::vector<KannalaBrandtCamera4> res;
        VecN vec1;
        vec1 << 379.045, 379.008, 505.512, 509.969, 0.00693023, -0.0013828, -0.000272596, -0.000452646;
        res.emplace_back(vec1);
        return res;
    }

private:
    VecN param_;
};

} // arc
