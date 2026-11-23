#pragma once

#include <base/base.h>

namespace arc {

class MargHelper {
public:
    static void marginalizeHelperSqToSq(
        MatXd& abs_H,
        VecXd& absB,
        const std::set<int>& idxToKeep,
        const std::set<int>& idxToMarg,
        MatXd& marg_H,
        VecXd& margB
    ) {
        int keepSize = idxToKeep.size();
        int margSize = idxToMarg.size();
        ensure(keepSize + margSize == abs_H.cols(), "VIO check failed");
        Eigen::Matrix<int, Eigen::Dynamic, 1> indices(idxToKeep.size() + idxToMarg.size());
        {
            auto it = idxToKeep.begin();
            for (size_t i = 0; i < idxToKeep.size(); ++ i) {
                indices[i] = *it;
                ++ it;
            }
        }
        {
            auto it = idxToMarg.begin();
            for (size_t i = 0; i < idxToMarg.size(); ++ i) {
                indices[idxToKeep.size() + i] = *it;
                ++ it;
            }
        }
        const Eigen::PermutationWrapper<Eigen::Matrix<int, Eigen::Dynamic, 1>> p(indices);
        const Eigen::PermutationMatrix<Eigen::Dynamic> pt = p.transpose();
        absB.applyOnTheLeft(pt);
        abs_H.applyOnTheLeft(pt);
        abs_H.applyOnTheRight(p);
        auto H_mm_decomposition = abs_H.bottomRightCorner(margSize, margSize).completeOrthogonalDecomposition();
        MatXd H_mm_inv = H_mm_decomposition.pseudoInverse();
        abs_H.topRightCorner(keepSize, margSize) *= H_mm_inv;
        marg_H = abs_H.topLeftCorner(keepSize, keepSize);
        margB = absB.head(keepSize);
        marg_H -= abs_H.topRightCorner(keepSize, margSize) * abs_H.bottomLeftCorner(margSize, keepSize);
        margB -= abs_H.topRightCorner(keepSize, margSize) * absB.tail(margSize);
        abs_H.resize(0, 0);
        absB.resize(0);
    }
    static void marginalizeHelperSqToSqrt(
        MatXd& abs_H,
        VecXd& absB,
        const std::set<int>& idxToKeep,
        const std::set<int>& idxToMarg,
        MatXd& marg_sqrt_H,
        VecXd& margSqrtB
    ) {
        int keepSize = idxToKeep.size();
        int margSize = idxToMarg.size();
        ensure(keepSize + margSize == abs_H.cols(), "VIO check failed");
        Eigen::Matrix<int, Eigen::Dynamic, 1> indices(idxToKeep.size() + idxToMarg.size());
        {
            auto it = idxToKeep.begin();
            for (size_t i = 0; i < idxToKeep.size(); ++ i) {
                indices[i] = *it;
                ++ it;
            }
        }
        {
            auto it = idxToMarg.begin();
            for (size_t i = 0; i < idxToMarg.size(); ++ i) {
                indices[idxToKeep.size() + i] = *it;
                ++ it;
            }
        }
        const Eigen::PermutationWrapper<Eigen::Matrix<int, Eigen::Dynamic, 1>> p(indices);
        const Eigen::PermutationMatrix<Eigen::Dynamic> pt = p.transpose();
        absB.applyOnTheLeft(pt);
        abs_H.applyOnTheLeft(pt);
        abs_H.applyOnTheRight(p);
        auto H_mm_decomposition = abs_H.bottomRightCorner(margSize, margSize).completeOrthogonalDecomposition();
        MatXd H_mm_inv = H_mm_decomposition.pseudoInverse();
        abs_H.topRightCorner(keepSize, margSize) *= H_mm_inv;
        MatXd marg_H;
        VecXd margB;
        marg_H = abs_H.topLeftCorner(keepSize, keepSize);
        margB = absB.head(keepSize);
        marg_H -= abs_H.topRightCorner(keepSize, margSize) * abs_H.bottomLeftCorner(margSize, keepSize);
        margB -= abs_H.topRightCorner(keepSize, margSize) * absB.tail(margSize);
        Eigen::LDLT<Eigen::Ref<MatXd>> ldlt(marg_H);
        VecXd D_sqrt = ldlt.vectorD().array().max(0).sqrt().matrix();
        marg_sqrt_H.setIdentity(keepSize, keepSize);
        marg_sqrt_H = ldlt.transpositionsP() * marg_sqrt_H;
        marg_sqrt_H = ldlt.matrixU() * marg_sqrt_H;
        marg_sqrt_H = D_sqrt.asDiagonal() * marg_sqrt_H;
        margSqrtB = ldlt.transpositionsP() * margB;
        ldlt.matrixL().solveInPlace(margSqrtB);
        for (int i = 0; i < margSqrtB.size(); ++ i) {
            if (D_sqrt(i) > std::sqrt(std::numeric_limits<double>::min()))
                margSqrtB(i) /= D_sqrt(i);
            else
                margSqrtB(i) = 0;
        }
        abs_H.resize(0, 0);
        absB.resize(0);
    }
    static void marginalizeHelperSqrtToSqrt(
        MatXd& Q2Jp,
        VecXd& Q2r,
        const std::set<int>& idxToKeep,
        const std::set<int>& idxToMarg,
        MatXd& marg_sqrt_H,
        VecXd& margSqrtB
    ) {
        Eigen::Index keepSize = idxToKeep.size();
        Eigen::Index margSize = idxToMarg.size();
        ensure(keepSize + margSize == Q2Jp.cols(), "VIO check failed");
        ensure(Q2Jp.rows() == Q2r.rows(), "VIO check failed");
        Eigen::Matrix<int, Eigen::Dynamic, 1> indices(idxToKeep.size() + idxToMarg.size());
        {
            auto it = idxToMarg.begin();
            for (size_t i = 0; i < idxToMarg.size(); ++ i) {
                indices[i] = *it;
                ++ it;
            }
        }
        {
            auto it = idxToKeep.begin();
            for (size_t i = 0; i < idxToKeep.size(); ++ i) {
                indices[idxToMarg.size() + i] = *it;
                ++ it;
            }
        }
        const Eigen::PermutationWrapper<Eigen::Matrix<int, Eigen::Dynamic, 1>> p(indices);
        Q2Jp.applyOnTheRight(p);
        Eigen::Index margRank = 0;
        Eigen::Index totalRank = 0;
        {
            const double rankThreshold = std::sqrt(std::numeric_limits<double>::epsilon());
            const Eigen::Index rows = Q2Jp.rows();
            const Eigen::Index cols = Q2Jp.cols();
            VecXd tempVector;
            tempVector.resize(cols + 1);
            double* tempData = tempVector.data();
            for (Eigen::Index k = 0; k < cols && totalRank < rows; ++ k) {
                Eigen::Index remainingRows = rows - totalRank;
                Eigen::Index remainingCols = cols - k - 1;
                double beta;
                double hCoeff;
                Q2Jp.col(k).tail(remainingRows).makeHouseholderInPlace(hCoeff, beta);
                if (std::abs(beta) > rankThreshold) {
                    Q2Jp.coeffRef(totalRank, k) = beta;
                    Q2Jp.bottomRightCorner(remainingRows, remainingCols)
                        .applyHouseholderOnTheLeft(Q2Jp.col(k).tail(remainingRows - 1), hCoeff, tempData + k + 1);
                    Q2r.tail(remainingRows)
                        .applyHouseholderOnTheLeft(Q2Jp.col(k).tail(remainingRows - 1), hCoeff, tempData + cols);
                    ++ totalRank;
                } else {
                    Q2Jp.coeffRef(totalRank, k) = 0;
                }
                Q2Jp.col(k).tail(remainingRows - 1).setZero();
                if (k == margSize - 1) {
                    margRank = totalRank;
                }
            }
        }
        Eigen::Index keepValidRows = std::max(totalRank - margRank, Eigen::Index(1));
        marg_sqrt_H = Q2Jp.block(margRank, margSize, keepValidRows, keepSize);
        margSqrtB = Q2r.segment(margRank, keepValidRows);
        Q2Jp.resize(0, 0);
        Q2r.resize(0);
    }
};

} // arc
