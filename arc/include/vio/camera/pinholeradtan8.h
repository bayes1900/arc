#pragma once

#include <base/base.h>

#include <vio/camera/traits.h>

#include <base/liegroup.h>

#define SN1                                                                                                            \
    Scalar {                                                                                                           \
        -1                                                                                                             \
    }
#define S0                                                                                                             \
    Scalar {                                                                                                           \
        0                                                                                                              \
    }
#define S1                                                                                                             \
    Scalar {                                                                                                           \
        1                                                                                                              \
    }
#define S2                                                                                                             \
    Scalar {                                                                                                           \
        2                                                                                                              \
    }
#define S3                                                                                                             \
    Scalar {                                                                                                           \
        3                                                                                                              \
    }

namespace arc {

using std::abs;
using std::max;
using std::sqrt;

template <typename Scalar = double>
class PinholeRadtan8Camera {
public:
    static constexpr int N = 12;
    using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
    using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
    using Vec4 = Eigen::Matrix<Scalar, 4, 1>;
    using VecN = Eigen::Matrix<Scalar, N, 1>;
    using Mat22 = Eigen::Matrix<Scalar, 2, 2>;
    using Mat24 = Eigen::Matrix<Scalar, 2, 4>;
    using Mat2N = Eigen::Matrix<Scalar, 2, N>;
    using Mat42 = Eigen::Matrix<Scalar, 4, 2>;
    using Mat4N = Eigen::Matrix<Scalar, 4, N>;
    PinholeRadtan8Camera() {
        param_.setZero();
        rpmax_ = S0;
    }
    explicit PinholeRadtan8Camera(const VecN& p, Scalar rpmax = SN1) {
        param_ = p;
        rpmax_ = rpmax == SN1 ? computeRpmax() : rpmax;
    }

    template <class Scalar2>
    PinholeRadtan8Camera<Scalar2> cast() const {
        return PinholeRadtan8Camera<Scalar2>(param_.template cast<Scalar2>());
    }
    static std::string getName() { return "pinhole-radtan8"; }
    Scalar computeRpmax() {
        Scalar rpmaxBackup = rpmax_;
        rpmax_ = S0;
        const int MAX_ITERS{1000};
        const Scalar STEP_SIZE{0.1};
        const Scalar MIN_REL_STEP{0.0001};
        const Scalar NUDGE{0.1};
        const Scalar CORNER_BOUND_SCALE{1.5};
        const Scalar RPMAX_SCALE{0.85};
        const Scalar& fx = param_[0];
        const Scalar& fy = param_[1];
        const Scalar& cx = param_[2];
        const Scalar& cy = param_[3];
        Vec2 almostCentralPixel{NUDGE * fx + cx, NUDGE * fy + cy};
        Vec3 acpUproj;
        bool unprojectSuccess = unproject(almostCentralPixel, acpUproj);
        ensure(unprojectSuccess, "VIO check failed");
        acpUproj /= acpUproj.z();
        Vec2 guess{acpUproj.x(), acpUproj.y()};
        auto rpp2 = [this](Vec2 xy) {
            Vec2 xypp;
            distort(xy, xypp);
            return xypp.squaredNorm();
        };
        auto numericRpp2Jacobian = [&rpp2](Vec2 xy) {
            const Scalar eps = Sophus::Constants<Scalar>::epsilonSqrt();
            const auto sign = [](Scalar n) { return n < S0 ? SN1 : S1; };
            const Scalar hx = eps * sign(xy.x()) * max(S1, xy.x());
            const Scalar hy = eps * sign(xy.y()) * max(S1, xy.y());
            const auto& f = rpp2;
            const Scalar fXy = f(xy);
            const Scalar dfDx = (f(xy + Vec2{hx, S0}) - fXy) / hx;
            const Scalar dfDy = (f(xy + Vec2{S0, hy}) - fXy) / hy;
            return Vec2{dfDx, dfDy};
        };
        auto rp2 = [this](Vec2 uv) {
            Vec3 xyz;
            unproject(uv, xyz);
            xyz /= xyz.z();
            return xyz.x() * xyz.x() + xyz.y() * xyz.y();
        };
        const Scalar w = S2 * cx;
        const Scalar h = S2 * cy;
        const std::vector<Vec2> corners = {{S0, S0}, {w, S0}, {S0, h}, {w, h}};
        Scalar cornersMaxrp2{-1};
        for (const Vec2& uv : corners) {
            const Scalar cornerRp2 = rp2(uv);
            if (cornerRp2 > cornersMaxrp2) {
                cornersMaxrp2 = cornerRp2;
            }
        }
        const Scalar domainBound = cornersMaxrp2 * CORNER_BOUND_SCALE;
        Vec2 x = guess;
        Scalar rpp2X = rpp2(x);
        bool diverged = false;
        for (int i = 1; i < MAX_ITERS; ++ i) {
            x += STEP_SIZE * numericRpp2Jacobian(x);
            const Scalar rp2X = x.squaredNorm();
            if (rp2X > domainBound) {
                diverged = true;
                break;
            }
            const Scalar oldRpp2X = rpp2X;
            rpp2X = rpp2(x);
            if (abs(rpp2X - oldRpp2X) < MIN_REL_STEP * oldRpp2X) {
                break;
            }
        }
        const Scalar rpmax = diverged ? S0 : RPMAX_SCALE * x.norm();
        rpmax_ = rpmaxBackup;
        return rpmax;
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
        const Scalar& p1 = param_[6];
        const Scalar& p2 = param_[7];
        const Scalar& k3 = param_[8];
        const Scalar& k4 = param_[9];
        const Scalar& k5 = param_[10];
        const Scalar& k6 = param_[11];
        const Scalar& x = p3dEval[0];
        const Scalar& y = p3dEval[1];
        const Scalar& z = p3dEval[2];
        const Scalar xp = x / z;
        const Scalar yp = y / z;
        const Scalar rp2 = xp * xp + yp * yp;
        const Scalar cdist = (S1 + rp2 * (k1 + rp2 * (k2 + rp2 * k3))) / (S1 + rp2 * (k4 + rp2 * (k5 + rp2 * k6)));
        const Scalar deltaX = S2 * p1 * xp * yp + p2 * (rp2 + S2 * xp * xp);
        const Scalar deltaY = S2 * p2 * xp * yp + p1 * (rp2 + S2 * yp * yp);
        const Scalar xpp = xp * cdist + deltaX;
        const Scalar ypp = yp * cdist + deltaY;
        const Scalar u = fx * xpp + cx;
        const Scalar v = fy * ypp + cy;
        proj[0] = u;
        proj[1] = v;
        bool positiveZ = z >= Sophus::Constants<Scalar>::epsilonSqrt();
        bool inInjectiveArea = rpmax_ == S0 ? true : rp2 <= rpmax_ * rpmax_;
        bool isValid = positiveZ && inInjectiveArea;
        if constexpr (!std::is_same_v<DerivedJ3D, std::nullptr_t>) {
            ensure(dProjDP3d, "VIO check failed");
            dProjDP3d->setZero();
            const Scalar v0 = p1 * y;
            const Scalar v1 = p2 * x;
            const Scalar v2 = z * z * z * z * z * z;
            const Scalar v3 = x * x;
            const Scalar v4 = y * y;
            const Scalar v5 = v3 + v4;
            const Scalar v6 = z * z * z * z;
            const Scalar v7 = z * z;
            const Scalar v8 = k5 * v7;
            const Scalar v9 = k6 * v5;
            const Scalar v10 = k4 * v6 + v5 * (v8 + v9);
            const Scalar v11 = v10 * v5 + v2;
            const Scalar v12 = v11 * v11;
            const Scalar v13 = S2 * v12;
            const Scalar v14 = k2 * v7;
            const Scalar v15 = k3 * v5;
            const Scalar v16 = k1 * v6 + v5 * (v14 + v15);
            const Scalar v17 = v16 * v5 + v2;
            const Scalar v18 = v17 * z * (v10 + v5 * (v8 + S2 * v9));
            const Scalar v19 = S2 * v18;
            const Scalar v20 = v16 + v5 * (v14 + S2 * v15);
            const Scalar v21 = S2 * v20;
            const Scalar v22 = v11 * z;
            const Scalar v23 = S1 / v7;
            const Scalar v24 = S1 / v12;
            const Scalar v25 = fx * v24;
            const Scalar v26 = v23 * v25;
            const Scalar v27 = p2 * y;
            const Scalar v28 = x * y;
            const Scalar v29 = S2 * v12 * (p1 * x + v27) - S2 * v18 * v28 + S2 * v20 * v22 * v28;
            const Scalar v30 = S1 / (z * z * z);
            const Scalar v31 = S2 * x;
            const Scalar v32 = v22 * (v17 + v21 * v5);
            const Scalar v33 = fy * v24;
            const Scalar v34 = v23 * v33;
            const Scalar duDx = v26 * (v13 * (v0 + S3 * v1) - v19 * v3 + v22 * (v17 + v21 * v3));
            const Scalar duDy = v26 * v29;
            const Scalar duDz = -v25 * v30 * (v13 * (p2 * (S3 * v3 + v4) + v0 * v31) - v18 * v31 * v5 + v32 * x);
            const Scalar dvDx = v29 * v34;
            const Scalar dvDy = v34 * (v13 * (S3 * v0 + v1) - v19 * v4 + v22 * (v17 + v21 * v4));
            const Scalar dvDz = -v30 * v33 * (v13 * (p1 * (v3 + S3 * v4) + v27 * v31) - v19 * v5 * y + v32 * y);
            (*dProjDP3d)(0, 0) = duDx;
            (*dProjDP3d)(0, 1) = duDy;
            (*dProjDP3d)(0, 2) = duDz;
            (*dProjDP3d)(1, 0) = dvDx;
            (*dProjDP3d)(1, 1) = dvDy;
            (*dProjDP3d)(1, 2) = dvDz;
        } else {
            (void)(dProjDP3d);
        }
        if constexpr (!std::is_same_v<DerivedJparam, std::nullptr_t>) {
            ensure(dProjDParam, "VIO check failed");
            dProjDParam->setZero();
            const Scalar w0 = z * z * z * z * z * z;
            const Scalar w1 = x * x;
            const Scalar w2 = y * y;
            const Scalar w3 = w1 + w2;
            const Scalar w4 = z * z * z * z;
            const Scalar w5 = z * z;
            const Scalar w6 = w0 + w3 * (k1 * w4 + w3 * (k2 * w5 + k3 * w3));
            const Scalar w7 = w6 * z;
            const Scalar w8 = w7 * x;
            const Scalar w9 = S2 * x * y;
            const Scalar w10 = S3 * w1 + w2;
            const Scalar w11 = w0 + w3 * (k4 * w4 + w3 * (k5 * w5 + k6 * w3));
            const Scalar w12 = S1 / w5;
            const Scalar w13 = S1 / w11;
            const Scalar w14 = w12 * w13;
            const Scalar w15 = w3 * (z * z * z);
            const Scalar w16 = fx * x;
            const Scalar w17 = w13 * w16;
            const Scalar w18 = w3 * w3;
            const Scalar w19 = w18 * z;
            const Scalar w20 = fx * w12;
            const Scalar w21 = (w3 * w3 * w3) * S1 / z;
            const Scalar w22 = S1 / (w11 * w11);
            const Scalar w23 = w22 * w6;
            const Scalar w24 = w16 * w23;
            const Scalar w25 = w18 * w22;
            const Scalar w26 = w7 * y;
            const Scalar w27 = w1 + S3 * w2;
            const Scalar w28 = fy * y;
            const Scalar w29 = w13 * w28;
            const Scalar w30 = fy * w12;
            const Scalar w31 = w23 * w28;
            const Scalar duFx = w14 * (w11 * (p1 * w9 + p2 * w10) + w8);
            const Scalar duFy = S0;
            const Scalar duCx = S1;
            const Scalar duCy = S0;
            const Scalar duK1 = w15 * w17;
            const Scalar duK2 = w17 * w19;
            const Scalar duP1 = w20 * w9;
            const Scalar duP2 = w10 * w20;
            const Scalar duK3 = w17 * w21;
            const Scalar duK4 = -w15 * w24;
            const Scalar duK5 = -fx * w25 * w8;
            const Scalar duK6 = -w21 * w24;
            const Scalar dvFx = S0;
            const Scalar dvFy = w14 * (w11 * (p1 * w27 + p2 * w9) + w26);
            const Scalar dvCx = S0;
            const Scalar dvCy = S1;
            const Scalar dvK1 = w15 * w29;
            const Scalar dvK2 = w19 * w29;
            const Scalar dvP1 = w27 * w30;
            const Scalar dvP2 = w30 * w9;
            const Scalar dvK3 = w21 * w29;
            const Scalar dvK4 = -w15 * w31;
            const Scalar dvK5 = -fy * w25 * w26;
            const Scalar dvK6 = -w21 * w31;
            (*dProjDParam)(0, 0) = duFx;
            (*dProjDParam)(0, 1) = duFy;
            (*dProjDParam)(0, 2) = duCx;
            (*dProjDParam)(0, 3) = duCy;
            (*dProjDParam)(0, 4) = duK1;
            (*dProjDParam)(0, 5) = duK2;
            (*dProjDParam)(0, 6) = duP1;
            (*dProjDParam)(0, 7) = duP2;
            (*dProjDParam)(0, 8) = duK3;
            (*dProjDParam)(0, 9) = duK4;
            (*dProjDParam)(0, 10) = duK5;
            (*dProjDParam)(0, 11) = duK6;
            (*dProjDParam)(1, 0) = dvFx;
            (*dProjDParam)(1, 1) = dvFy;
            (*dProjDParam)(1, 2) = dvCx;
            (*dProjDParam)(1, 3) = dvCy;
            (*dProjDParam)(1, 4) = dvK1;
            (*dProjDParam)(1, 5) = dvK2;
            (*dProjDParam)(1, 6) = dvP1;
            (*dProjDParam)(1, 7) = dvP2;
            (*dProjDParam)(1, 8) = dvK3;
            (*dProjDParam)(1, 9) = dvK4;
            (*dProjDParam)(1, 10) = dvK5;
            (*dProjDParam)(1, 11) = dvK6;
        } else {
            (void)(dProjDParam);
        }
        return isValid;
    }

    template <class DerivedJundist = std::nullptr_t>
    void distort(const Vec2& undist, Vec2& dist, DerivedJundist dDistDUndist = nullptr) const {
        const Scalar& k1 = param_[4];
        const Scalar& k2 = param_[5];
        const Scalar& p1 = param_[6];
        const Scalar& p2 = param_[7];
        const Scalar& k3 = param_[8];
        const Scalar& k4 = param_[9];
        const Scalar& k5 = param_[10];
        const Scalar& k6 = param_[11];
        const Scalar xp = undist.x();
        const Scalar yp = undist.y();
        const Scalar rp2 = xp * xp + yp * yp;
        const Scalar cdist = (S1 + rp2 * (k1 + rp2 * (k2 + rp2 * k3))) / (S1 + rp2 * (k4 + rp2 * (k5 + rp2 * k6)));
        const Scalar deltaX = S2 * p1 * xp * yp + p2 * (rp2 + S2 * xp * xp);
        const Scalar deltaY = S2 * p2 * xp * yp + p1 * (rp2 + S2 * yp * yp);
        const Scalar xpp = xp * cdist + deltaX;
        const Scalar ypp = yp * cdist + deltaY;
        dist.x() = xpp;
        dist.y() = ypp;
        if constexpr (!std::is_same_v<DerivedJundist, std::nullptr_t>) {
            ensure(dDistDUndist, "VIO check failed");
            const Scalar v0 = xp * xp;
            const Scalar v1 = yp * yp;
            const Scalar v2 = v0 + v1;
            const Scalar v3 = k6 * v2;
            const Scalar v4 = k4 + v2 * (k5 + v3);
            const Scalar v5 = v2 * v4 + S1;
            const Scalar v6 = v5 * v5;
            const Scalar v7 = S1 / v6;
            const Scalar v8 = p1 * yp;
            const Scalar v9 = p2 * xp;
            const Scalar v10 = S2 * v6;
            const Scalar v11 = k3 * v2;
            const Scalar v12 = k1 + v2 * (k2 + v11);
            const Scalar v13 = v12 * v2 + S1;
            const Scalar v14 = v13 * (v2 * (k5 + S2 * v3) + v4);
            const Scalar v15 = S2 * v14;
            const Scalar v16 = v12 + v2 * (k2 + S2 * v11);
            const Scalar v17 = S2 * v16;
            const Scalar v18 = xp * yp;
            const Scalar v19 = S2 * v7 * (-v14 * v18 + v16 * v18 * v5 + v6 * (p1 * xp + p2 * yp));
            const Scalar dxppDxp = v7 * (-v0 * v15 + v10 * (v8 + S3 * v9) + v5 * (v0 * v17 + v13));
            const Scalar dxppDyp = v19;
            const Scalar dyppDxp = v19;
            const Scalar dyppDyp = v7 * (-v1 * v15 + v10 * (S3 * v8 + v9) + v5 * (v1 * v17 + v13));
            (*dDistDUndist)(0, 0) = dxppDxp;
            (*dDistDUndist)(0, 1) = dxppDyp;
            (*dDistDUndist)(1, 0) = dyppDxp;
            (*dDistDUndist)(1, 1) = dyppDyp;
        } else {
            (void)(dDistDUndist);
        }
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
        const Scalar& u = projEval[0];
        const Scalar& v = projEval[1];
        const Scalar x0 = (u - cx) / fx;
        const Scalar y0 = (v - cy) / fy;
#if 1
        Vec2 dist{x0, y0};
        Vec2 undist{dist};
        const Scalar EPS = Sophus::Constants<Scalar>::epsilonSqrt();
        constexpr int N = 5;
        for (int i = 0; i < N; ++ i) {
            Mat22 J{};
            Vec2 fundist{};
            distort(undist, fundist, &J);
            Vec2 residual = fundist - dist;
            undist -= J.inverse() * residual;
            if (residual.norm() < EPS) {
                break;
            }
        }
        const Scalar xp = undist.x();
        const Scalar yp = undist.y();
#else
        constexpr int N = 100;
        Scalar xp = x0;
        Scalar yp = y0;
        for (int i = 0; i < N; ++ i) {
            const Scalar rp2 = xp * xp + yp * yp;
            const Scalar icdist = (S1 + rp2 * (k4 + rp2 * (k5 + rp2 * k6))) / (S1 + rp2 * (k1 + rp2 * (k2 + rp2 * k3)));
            if (icdist <= S0) {
                return false;
            }
            const Scalar deltaX = S2 * p1 * xp * yp + p2 * (rp2 + S2 * xp * xp);
            const Scalar deltaY = S2 * p2 * xp * yp + p1 * (rp2 + S2 * yp * yp);
            xp = (x0 - deltaX) * icdist;
            yp = (y0 - deltaY) * icdist;
        }
#endif
        const Scalar normInv = S1 / sqrt(xp * xp + yp * yp + S1);
        p3d.setZero();
        p3d[0] = xp * normInv;
        p3d[1] = yp * normInv;
        p3d[2] = normInv;
        if constexpr (!std::is_same_v<DerivedJ2D, std::nullptr_t> || !std::is_same_v<DerivedJparam, std::nullptr_t>) {
            throw std::logic_error("unreachable");
        }
        (void)(dP3dDProj);
        (void)(dP3dDParam);
        const Scalar rp2 = xp * xp + yp * yp;
        bool inInjectiveArea = rpmax_ == S0 ? true : rp2 <= rpmax_ * rpmax_;
        bool isValid = inInjectiveArea;
        return isValid;
    }
    void setFromInit(const Vec4& init) {
        param_.setZero();
        param_[0] = init[0];
        param_[1] = init[1];
        param_[2] = init[2];
        param_[3] = init[3];
        rpmax_ = S0;
    }

    void operator+=(const VecN& inc) {
        param_ += inc;
        rpmax_ = computeRpmax();
    }
    const VecN& getParam() const {
        return param_;
    }
    Scalar getRpmax() const {
        return rpmax_;
    }
    static std::vector<PinholeRadtan8Camera> getTestProjections() {
        std::vector<PinholeRadtan8Camera> res;
        VecN vec1{};
        vec1 << 269.0600776672363, 269.1679859161377, 324.3333053588867, 245.22674560546875, 0.6257319450378418,
            0.46612036228179932, -0.00018502399325370789, -4.2882973502855748e-5, 0.0041795829311013222,
            0.89431935548782349, 0.54253977537155151, 0.0662121474742889;
        res.emplace_back(vec1);
        return res;
    }
    static std::vector<Eigen::Vector2i> getTestResolutions() {
        std::vector<Eigen::Vector2i> res;
        res.emplace_back(640, 480);
        return res;
    }

private:
    VecN param_;
    Scalar rpmax_;
};

} // arc

#undef SN1
#undef S0
#undef S1
#undef S2
#undef S3
