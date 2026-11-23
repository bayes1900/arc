#pragma once

#include <vio/camera/stereographic.h>
#include <vio/estimator/landmarks.h>

namespace arc {

inline Sophus::SE3d computeRelPose(
    const Sophus::SE3d& T_w_i_h,
    const Sophus::SE3d& T_i_c_h,
    const Sophus::SE3d& T_w_i_t,
    const Sophus::SE3d& T_i_c_t,
    Sophus::Matrix6d* dRelDH = nullptr,
    Sophus::Matrix6d* dRelDT = nullptr
) {
    Sophus::SE3d tmp2 = (T_i_c_t).inverse();
    Sophus::SE3d T_t_i_h_i;
    T_t_i_h_i.so3() = T_w_i_t.so3().inverse() * T_w_i_h.so3();
    T_t_i_h_i.translation() = T_w_i_t.so3().inverse() * (T_w_i_h.translation() - T_w_i_t.translation());
    Sophus::SE3d tmp = tmp2 * T_t_i_h_i;
    Sophus::SE3d res = tmp * T_i_c_h;
    if (dRelDH) {
        Sophus::Matrix3d R = T_w_i_h.so3().inverse().matrix();
        Sophus::Matrix6d RR;
        RR.setZero();
        RR.topLeftCorner<3, 3>() = R;
        RR.bottomRightCorner<3, 3>() = R;
        *dRelDH = tmp.Adj() * RR;
    }
    if (dRelDT) {
        Sophus::Matrix3d R = T_w_i_t.so3().inverse().matrix();
        Sophus::Matrix6d RR;
        RR.setZero();
        RR.topLeftCorner<3, 3>() = R;
        RR.bottomRightCorner<3, 3>() = R;
        *dRelDT = -tmp2.Adj() * RR;
    }
    return res;
}

template <class CamT>
inline bool linearizePoint(
    const Eigen::Vector2d& kptObs,
    const Keypoint& kptPos,
    const Eigen::Matrix4d& T_t_h,
    const CamT& cam,
    Eigen::Vector2d& res,
    Eigen::Matrix<double, 2, 6>* dResDXi = nullptr,
    Eigen::Matrix<double, 2, 3>* dResDP = nullptr,
    Eigen::Vector4d* proj = nullptr
) {
    Eigen::Matrix<double, 4, 2> Jup;
    Eigen::Vector4d pH3d;
    pH3d = StereographicParam::unproject(kptPos.direction, &Jup);
    pH3d[3] = kptPos.invDist;
    const Eigen::Vector4d pT3d = T_t_h * pH3d;
    Eigen::Matrix<double, 2, 4> Jp;
    bool valid = cam.project(pT3d, res, &Jp);
    valid &= res.array().isFinite().all();
    if (!valid) {
        return false;
    }
    if (proj) {
        proj->template head<2>() = res;
        (*proj)[2] = pT3d[3] / pT3d.head<3>().norm();
    }
    res -= kptObs;
    if (dResDXi) {
        Eigen::Matrix<double, 4, 6> dPointDXi;
        dPointDXi.topLeftCorner<3, 3>() = Eigen::Matrix3d::Identity() * kptPos.invDist;
        dPointDXi.topRightCorner<3, 3>() = -Sophus::SO3d::hat(pT3d.head<3>());
        dPointDXi.row(3).setZero();
        *dResDXi = Jp * dPointDXi;
    }
    if (dResDP) {
        Eigen::Matrix<double, 4, 3> Jpp;
        Jpp.setZero();
        Jpp.block<3, 2>(0, 0) = T_t_h.topLeftCorner<3, 4>() * Jup;
        Jpp.col(2) = T_t_h.col(3);
        *dResDP = Jp * Jpp;
    }
    return true;
}

} // arc
