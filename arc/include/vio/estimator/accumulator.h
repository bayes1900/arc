#pragma once

#include <Eigen/Dense>

#include <base/base.h>

namespace arc {

class DenseAccumulator {
public:
    template <int ROWS, int COLS, typename Derived>
    void addH(int i, int j, const Eigen::MatrixBase<Derived>& data) {
        h_.block<ROWS, COLS>(i, j) += data;
    }

    template <int ROWS, typename Derived>
    void addB(int i, const Eigen::MatrixBase<Derived>& data) {
        b_.segment<ROWS>(i) += data;
    }
    VecXd solve(const VecXd* diagonal) const {
        if (diagonal == nullptr) {
            return h_.ldlt().solve(b_);
        } else {
            MatXd HH = h_;
            HH.diagonal() += *diagonal;
            return HH.ldlt().solve(b_);
        }
    }
    void reset(int optSize) {
        if (optSize <= 0) {
            throw std::invalid_argument("invalid accumulator size");
        }
        h_.setZero(optSize, optSize);
        b_.setZero(optSize);
    }
    void join(const DenseAccumulator& other) {
        h_ += other.h_;
        b_ += other.b_;
    }
    const MatXd& getH() const { return h_; }
    const VecXd& getB() const { return b_; }
    MatXd& getH() { return h_; }
    VecXd& getB() { return b_; }

private:
    MatXd h_;
    VecXd b_;
};

} // arc
