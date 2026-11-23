#pragma once

#include <base/base.h>
#include <base/manifold/math.h>
#include <base/manifold/so3.h>
#include <omp.h>

namespace arc {

template <typename State, int ProcessNoiseDof, typename Input>
class ManifoldEsekf {
public:
    static constexpr int STATE_DOF = State::DOF;
    static constexpr int STATE_DIM = State::DIM;
    using Scalar = typename State::Scalar;
    using StateVector = Eigen::Matrix<Scalar, STATE_DOF, 1>;
    using Covariance = Eigen::Matrix<Scalar, STATE_DOF, STATE_DOF>;
    using ProcessNoiseCovariance = Eigen::Matrix<Scalar, ProcessNoiseDof, ProcessNoiseDof>;
    using ProcessModel = std::function<Eigen::Matrix<Scalar, STATE_DIM, 1>(State&, const Input&)>;
    using ProcessMatrix1 = std::function<Eigen::Matrix<Scalar, STATE_DIM, STATE_DOF>(State&, const Input&)>;
    using ProcessMatrix2 = std::function<Eigen::Matrix<Scalar, STATE_DIM, ProcessNoiseDof>(State&, const Input&)>;

    struct MeasurementData {
        bool valid{true};
        bool converge{true};
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> h;
        Eigen::Matrix<Scalar, Eigen::Dynamic, 12> hX;
    };
    using MeasurementModel = std::function<void(State&, MeasurementData&)>;

    ManifoldEsekf()
        : state_()
        , covariance_(Covariance::Identity())
    {
        state_.buildStateIndex();
        limits_.fill(Scalar(0.001));
    }

    void initProcessModel(ProcessModel model, ProcessMatrix1 matrix1, ProcessMatrix2 matrix2) {
        processModel_ = std::move(model);
        processMatrix1_ = std::move(matrix1);
        processMatrix2_ = std::move(matrix2);
    }

    void initMeasurementModel(MeasurementModel model, int maxIterations, Scalar limit = Scalar(0.001)) {
        measurementModel_ = std::move(model);
        maxIterations_ = maxIterations;
        limits_.fill(limit);
        state_.buildStateIndex();
    }

    void predict(Scalar dt, const ProcessNoiseCovariance& processNoise, const Input& input) {
        const Eigen::Matrix<Scalar, STATE_DIM, 1> f = processModel_(state_, input);
        const Eigen::Matrix<Scalar, STATE_DIM, STATE_DOF> fX = processMatrix1_(state_, input);
        const Eigen::Matrix<Scalar, STATE_DIM, ProcessNoiseDof> fW = processMatrix2_(state_, input);

        State stateBefore = state_;
        state_.oPlus(f, dt);

        Covariance fX1 = Covariance::Identity();
        Covariance fXFinal = Covariance::Zero();
        Eigen::Matrix<Scalar, STATE_DOF, ProcessNoiseDof> fWFinal =
            Eigen::Matrix<Scalar, STATE_DOF, ProcessNoiseDof>::Zero();

        for (const auto& entry : state_.vectorState()) {
            fXFinal.block(entry.index, 0, entry.dof, STATE_DOF) =
                fX.block(entry.dimension, 0, entry.dof, STATE_DOF);
            fWFinal.block(entry.index, 0, entry.dof, ProcessNoiseDof) =
                fW.block(entry.dimension, 0, entry.dof, ProcessNoiseDof);
        }

        for (const auto& entry : state_.so3State()) {
            Eigen::Matrix<Scalar, 3, 1> segment;
            for (int i = 0; i < 3; ++ i) {
                segment(i) = -f(entry.dimension + i) * dt;
            }
            // FAST_LIO uses scalar_type(1/2), which is integer zero. Keep the
            // identity rotation for numerical parity with upstream.
            const manifold::ManifoldSo3<Scalar> rotation =
                manifold::ManifoldSo3<Scalar>::exp(segment, Scalar(0));
            fX1.template block<3, 3>(entry.index, entry.index) = rotation.toRotationMatrix();
            const Eigen::Matrix<Scalar, 3, 3> a = manifold::aMatrix(segment);
            for (int i = 0; i < STATE_DOF; ++ i) {
                fXFinal.template block<3, 1>(entry.index, i) =
                    a * fX.template block<3, 1>(entry.dimension, i);
            }
            for (int i = 0; i < ProcessNoiseDof; ++ i) {
                fWFinal.template block<3, 1>(entry.index, i) =
                    a * fW.template block<3, 1>(entry.dimension, i);
            }
        }

        for (const auto& entry : state_.s2State()) {
            Eigen::Matrix<Scalar, 3, 1> segment;
            for (int i = 0; i < 3; ++ i) {
                segment(i) = f(entry.dimension + i) * dt;
            }
            // Same upstream integer-division behavior as the SO3 branch.
            const manifold::ManifoldSo3<Scalar> rotation =
                manifold::ManifoldSo3<Scalar>::exp(segment, Scalar(0));
            Eigen::Matrix<Scalar, 2, 3> nx;
            Eigen::Matrix<Scalar, 3, 2> mx;
            const Eigen::Matrix<Scalar, 2, 1> zero = Eigen::Matrix<Scalar, 2, 1>::Zero();
            state_.s2NxYy(nx, entry.index);
            stateBefore.s2Mx(mx, zero, entry.index);
            fX1.template block<2, 2>(entry.index, entry.index) =
                nx * rotation.toRotationMatrix() * mx;

            Eigen::Matrix<Scalar, 3, 3> beforeHat;
            stateBefore.s2Hat(beforeHat, entry.index);
            const Eigen::Matrix<Scalar, 2, 3> residual =
                -nx * rotation.toRotationMatrix() * beforeHat
                    * manifold::aMatrix(segment).transpose();
            for (int i = 0; i < STATE_DOF; ++ i) {
                fXFinal.template block<2, 1>(entry.index, i) =
                    residual * fX.template block<3, 1>(entry.dimension, i);
            }
            for (int i = 0; i < ProcessNoiseDof; ++ i) {
                fWFinal.template block<2, 1>(entry.index, i) =
                    residual * fW.template block<3, 1>(entry.dimension, i);
            }
        }

        fX1 += fXFinal * dt;
        covariance_ = fX1 * covariance_ * fX1.transpose()
            + (dt * fWFinal) * processNoise * (dt * fWFinal).transpose();
    }

    void update(Scalar measurementVariance, Scalar& solveTime) {
        MeasurementData measurement;
        State statePropagated = state_;
        Covariance covariancePropagated = covariance_;
        int dofMeasurement = 0;
        int convergenceCount = 0;

        StateVector kH = StateVector::Zero();
        Covariance kX = Covariance::Zero();
        StateVector dxNew = StateVector::Zero();

        for (int iteration = -1; iteration < maxIterations_; ++ iteration) {
            measurement.valid = true;
            measurementModel_(state_, measurement);
            if (!measurement.valid) {
                continue;
            }

            const Eigen::Matrix<Scalar, Eigen::Dynamic, 12> hX = measurement.hX;
            const double solveStart = omp_get_wtime();
            dofMeasurement = static_cast<int>(hX.rows());

            StateVector dx;
            state_.boxMinus(dx, statePropagated);
            dxNew = dx;
            covariance_ = covariancePropagated;

            for (const auto& entry : state_.so3State()) {
                Eigen::Matrix<Scalar, 3, 1> segment;
                for (int i = 0; i < 3; ++ i) {
                    segment(i) = dx(entry.index + i);
                }
                const Eigen::Matrix<Scalar, 3, 3> reparameterization =
                    manifold::aMatrix(segment).transpose();
                dxNew.template block<3, 1>(entry.index, 0) =
                    reparameterization * dxNew.template block<3, 1>(entry.index, 0);
                for (int i = 0; i < STATE_DOF; ++ i) {
                    covariance_.template block<3, 1>(entry.index, i) =
                        reparameterization * covariance_.template block<3, 1>(entry.index, i);
                }
                for (int i = 0; i < STATE_DOF; ++ i) {
                    covariance_.template block<1, 3>(i, entry.index) =
                        covariance_.template block<1, 3>(i, entry.index)
                            * reparameterization.transpose();
                }
            }

            for (const auto& entry : state_.s2State()) {
                Eigen::Matrix<Scalar, 2, 1> segment;
                for (int i = 0; i < 2; ++ i) {
                    segment(i) = dx(entry.index + i);
                }
                Eigen::Matrix<Scalar, 2, 3> nx;
                Eigen::Matrix<Scalar, 3, 2> mx;
                state_.s2NxYy(nx, entry.index);
                statePropagated.s2Mx(mx, segment, entry.index);
                const Eigen::Matrix<Scalar, 2, 2> reparameterization = nx * mx;
                dxNew.template block<2, 1>(entry.index, 0) =
                    reparameterization * dxNew.template block<2, 1>(entry.index, 0);
                for (int i = 0; i < STATE_DOF; ++ i) {
                    covariance_.template block<2, 1>(entry.index, i) =
                        reparameterization * covariance_.template block<2, 1>(entry.index, i);
                }
                for (int i = 0; i < STATE_DOF; ++ i) {
                    covariance_.template block<1, 2>(i, entry.index) =
                        covariance_.template block<1, 2>(i, entry.index)
                            * reparameterization.transpose();
                }
            }

            if (STATE_DOF > dofMeasurement) {
                Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> hXCurrent =
                    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero(
                        dofMeasurement, STATE_DOF);
                hXCurrent.topLeftCorner(dofMeasurement, 12) = hX;
                const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> gain =
                    covariance_ * hXCurrent.transpose()
                    * (hXCurrent * covariance_ * hXCurrent.transpose() / measurementVariance
                        + Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Identity(
                            dofMeasurement, dofMeasurement)).inverse()
                    / measurementVariance;
                kH = gain * measurement.h;
                kX = gain * hXCurrent;
            } else {
                Covariance pTemp = (covariance_ / measurementVariance).inverse();
                const Eigen::Matrix<Scalar, 12, 12> hth = hX.transpose() * hX;
                pTemp.template block<12, 12>(0, 0) += hth;
                const Covariance pInv = pTemp.inverse();
                kH = pInv.template block<STATE_DOF, 12>(0, 0)
                    * hX.transpose() * measurement.h;
                kX.setZero();
                kX.template block<STATE_DOF, 12>(0, 0) =
                    pInv.template block<STATE_DOF, 12>(0, 0) * hth;
            }

            const StateVector dxCorrection = kH + (kX - Covariance::Identity()) * dxNew;
            state_.boxPlus(dxCorrection);

            measurement.converge = true;
            for (int i = 0; i < STATE_DOF; ++ i) {
                if (std::fabs(dxCorrection[i]) > limits_[i]) {
                    measurement.converge = false;
                    break;
                }
            }
            if (measurement.converge) {
                ++ convergenceCount;
            }
            if (convergenceCount == 0 && iteration == maxIterations_ - 2) {
                measurement.converge = true;
            }

            if (convergenceCount > 1 || iteration == maxIterations_ - 1) {
                Covariance l = covariance_;
                for (const auto& entry : state_.so3State()) {
                    Eigen::Matrix<Scalar, 3, 1> segment;
                    for (int i = 0; i < 3; ++ i) {
                        segment(i) = dxCorrection(entry.index + i);
                    }
                    const Eigen::Matrix<Scalar, 3, 3> reparameterization =
                        manifold::aMatrix(segment).transpose();
                    for (int i = 0; i < STATE_DOF; ++ i) {
                        l.template block<3, 1>(entry.index, i) =
                            reparameterization * covariance_.template block<3, 1>(entry.index, i);
                    }
                    for (int i = 0; i < 12; ++ i) {
                        kX.template block<3, 1>(entry.index, i) =
                            reparameterization * kX.template block<3, 1>(entry.index, i);
                    }
                    for (int i = 0; i < STATE_DOF; ++ i) {
                        l.template block<1, 3>(i, entry.index) =
                            l.template block<1, 3>(i, entry.index)
                                * reparameterization.transpose();
                        covariance_.template block<1, 3>(i, entry.index) =
                            covariance_.template block<1, 3>(i, entry.index)
                                * reparameterization.transpose();
                    }
                }

                for (const auto& entry : state_.s2State()) {
                    Eigen::Matrix<Scalar, 2, 1> segment;
                    for (int i = 0; i < 2; ++ i) {
                        segment(i) = dxCorrection(entry.index + i);
                    }
                    Eigen::Matrix<Scalar, 2, 3> nx;
                    Eigen::Matrix<Scalar, 3, 2> mx;
                    state_.s2NxYy(nx, entry.index);
                    statePropagated.s2Mx(mx, segment, entry.index);
                    const Eigen::Matrix<Scalar, 2, 2> reparameterization = nx * mx;
                    for (int i = 0; i < STATE_DOF; ++ i) {
                        l.template block<2, 1>(entry.index, i) =
                            reparameterization * covariance_.template block<2, 1>(entry.index, i);
                    }
                    for (int i = 0; i < 12; ++ i) {
                        kX.template block<2, 1>(entry.index, i) =
                            reparameterization * kX.template block<2, 1>(entry.index, i);
                    }
                    for (int i = 0; i < STATE_DOF; ++ i) {
                        l.template block<1, 2>(i, entry.index) =
                            l.template block<1, 2>(i, entry.index)
                                * reparameterization.transpose();
                        covariance_.template block<1, 2>(i, entry.index) =
                            covariance_.template block<1, 2>(i, entry.index)
                                * reparameterization.transpose();
                    }
                }

                covariance_ = l
                    - kX.template block<STATE_DOF, 12>(0, 0)
                        * covariance_.template block<12, STATE_DOF>(0, 0);
                solveTime += omp_get_wtime() - solveStart;
                return;
            }
            solveTime += omp_get_wtime() - solveStart;
        }
    }

    const State& state() const {
        return state_;
    }

    const Covariance& covariance() const {
        return covariance_;
    }

    void setState(const State& state) {
        state_ = state;
        state_.buildStateIndex();
    }

    void setCovariance(const Covariance& covariance) {
        covariance_ = covariance;
    }

private:
    State state_;
    Covariance covariance_;
    ProcessModel processModel_;
    ProcessMatrix1 processMatrix1_;
    ProcessMatrix2 processMatrix2_;
    MeasurementModel measurementModel_;
    int maxIterations_{0};
    std::array<Scalar, STATE_DOF> limits_{};
};

} // arc
