#pragma once

#include <Eigen/Dense>
#include <sophus/se2.hpp>

#include <vio/image/image.h>
#include <vio/opticalflow/base.h>
#include <vio/opticalflow/pattern.h>

namespace arc {

template <typename Pattern>
struct OpticalFlowPatch : public OpticalFlowTypes {
    static constexpr int PATTERN_SIZE = Pattern::PATTERN_SIZE;
    using Vector2i = Eigen::Matrix<int, 2, 1>;
    using VectorP = Eigen::Matrix<Scalar, PATTERN_SIZE, 1>;
    using Matrix2P = Eigen::Matrix<Scalar, 2, PATTERN_SIZE>;
    using MatrixP2 = Eigen::Matrix<Scalar, PATTERN_SIZE, 2>;
    using MatrixP3 = Eigen::Matrix<Scalar, PATTERN_SIZE, 3>;
    using Matrix3P = Eigen::Matrix<Scalar, 3, PATTERN_SIZE>;
    using MatrixP4 = Eigen::Matrix<Scalar, PATTERN_SIZE, 4>;
    using Matrix2Pi = Eigen::Matrix<int, 2, PATTERN_SIZE>;
    static const Matrix2P PATTERN2;
    OpticalFlowPatch() = default;
    OpticalFlowPatch(const Image<const uint16_t>& img, const Vector2& pos) { setFromImage(img, pos); }

    template <typename ImgT>
    static void setData(
        const ImgT& img, const Vector2& pos, Scalar& mean, VectorP& data, const Sophus::SE2<Scalar>* se2 = nullptr) {
        int numValidPoints = 0;
        Scalar sum = 0;
        for (int i = 0; i < PATTERN_SIZE; ++ i) {
            Vector2 p;
            if (se2) {
                p = pos + (*se2) * PATTERN2.col(i);
            } else {
                p = pos + PATTERN2.col(i);
            };
            if (img.InBounds(p, 2)) {
                Scalar val = img.template interp<Scalar>(p);
                data[i] = val;
                sum += val;
                ++ numValidPoints;
            } else {
                data[i] = -1;
            }
        }
        mean = sum / numValidPoints;
        data /= mean;
    }

    template <typename ImgT>
    static void setDataJacSe2(const ImgT& img, const Vector2& pos, Scalar& mean, VectorP& data, MatrixP3& J_se2) {
        int numValidPoints = 0;
        Scalar sum = 0;
        Vector3 gradSumSe2(0, 0, 0);
        Eigen::Matrix<Scalar, 2, 3> Jw_se2;
        Jw_se2.topLeftCorner<2, 2>().setIdentity();
        for (int i = 0; i < PATTERN_SIZE; ++ i) {
            Vector2 p = pos + PATTERN2.col(i);
            Jw_se2(0, 2) = -PATTERN2(1, i);
            Jw_se2(1, 2) = PATTERN2(0, i);
            if (img.inBounds(p, 2)) {
                Vector3 valGrad = img.template interpGrad<Scalar>(p);
                data[i] = valGrad[0];
                sum += valGrad[0];
                J_se2.row(i) = valGrad.tail<2>().transpose() * Jw_se2;
                gradSumSe2 += J_se2.row(i);
                ++ numValidPoints;
            } else {
                data[i] = -1;
            }
        }
        mean = sum / numValidPoints;
        const Scalar meanInv = numValidPoints / sum;
        for (int i = 0; i < PATTERN_SIZE; ++ i) {
            if (data[i] >= 0) {
                J_se2.row(i) -= gradSumSe2.transpose() * data[i] / sum;
                data[i] *= meanInv;
            } else {
                J_se2.row(i).setZero();
            }
        }
        J_se2 *= meanInv;
    }
    void setFromImage(const Image<const uint16_t>& img, const Vector2& pos) {
        this->pos = pos;
        MatrixP3 J_se2;
        setDataJacSe2(img, pos, mean, data, J_se2);
        Matrix3 H_se2 = J_se2.transpose() * J_se2;
        Matrix3 H_se2_inv;
        H_se2_inv.setIdentity();
        H_se2.ldlt().solveInPlace(H_se2_inv);
        hSe2InvJSe2T = H_se2_inv * J_se2.transpose();
        valid = mean > std::numeric_limits<Scalar>::epsilon() && hSe2InvJSe2T.array().isFinite().all() && data.array().isFinite().all();
    }
    bool residual(
        const Image<const uint16_t>& img, const Matrix2P& transformedPattern, VectorP& residual) const {
        Scalar sum = 0;
        int numValidPoints = 0;
        for (int i = 0; i < PATTERN_SIZE; ++ i) {
            if (img.inBounds(transformedPattern.col(i), 2)) {
                residual[i] = img.interp<Scalar>(transformedPattern.col(i));
                sum += residual[i];
                ++ numValidPoints;
            } else {
                residual[i] = -1;
            }
        }
        if (sum < std::numeric_limits<Scalar>::epsilon()) {
            residual.setZero();
            return false;
        }
        int numResiduals = 0;
        for (int i = 0; i < PATTERN_SIZE; ++ i) {
            if (residual[i] >= 0 && data[i] >= 0) {
                const Scalar val = residual[i];
                residual[i] = numValidPoints * val / sum - data[i];
                ++ numResiduals;
            } else {
                residual[i] = 0;
            }
        }
        return numResiduals > PATTERN_SIZE / 2;
    }

    Vector2 pos = Vector2::Zero();
    VectorP data = VectorP::Zero();

    Matrix3P hSe2InvJSe2T = Matrix3P::Zero();
    Scalar mean = 0;
    bool valid = false;
};

template <typename Pattern>
const typename OpticalFlowPatch<Pattern>::Matrix2P OpticalFlowPatch<Pattern>::PATTERN2 = Pattern::PATTERN2;

} // arc
