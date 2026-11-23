#pragma once

#include <base/base.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vio/estimator/accumulator.h>
#include <vio/estimator/landmarks.h>
#include <vio/linearization/blockdiagonal.h>

namespace arc {

struct RelPoseLin {
    Mat4d tTH;
    Mat6d dRelDH;
    Mat6d dRelDT;
};

class LandmarkBlock {
public:
    struct Options {
        bool useHouseholder = true;
        bool useValidProjectionsOnly = true;
        double huberParameter = 0;
        double obsStdDev = 1;
        double jacobiScalingEps = 1e-6;
    };
    enum State { UNINITIALIZED = 0, ALLOCATED, NUMERICAL_FAILURE, LINEARIZED, MARGINALIZED };
    using RowMatX = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using RowMat3 = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;
    virtual ~LandmarkBlock() {}
    virtual bool isNumericalFailure() const = 0;
    virtual void allocateLandmark(
        Keypoint& lm,
        const std::unordered_map<std::pair<TimeCamId, TimeCamId>, RelPoseLin>& relativePoseLin,
        const Calibration<double>& calib,
        const AbsOrderMap& aom,
        const Options& options,
        const std::map<TimeCamId, size_t>* relOrder = nullptr
    ) = 0;
    virtual double linearizeLandmark() = 0;
    virtual void performQR() = 0;
    virtual void setLandmarkDamping(double lambda) = 0;
    virtual void backSubstitute(const VecXd& poseInc, double& lDiff) = 0;
    virtual void addQ2JpTQ2JpMultX(VecXd& res, const VecXd& xPose) const = 0;
    virtual void addQ2JpTQ2r(VecXd& res) const = 0;
    virtual void addQ2JpTQ2JpBlockdiag(BlockDiagonalAccumulator& accu) const = 0;
    virtual void scaleJlCols() = 0;
    virtual void scaleJpCols(const VecXd& jacobianScaling) = 0;
    virtual void printStorage(const std::string& filename) const = 0;
    virtual State getState() const = 0;
    virtual size_t numReducedCams() const = 0;
    virtual void getDenseQ2JpQ2r(MatXd& Q2Jp, VecXd& Q2r, size_t startIdx) const = 0;
    virtual void getDenseQ2JpQ2rRel(MatXd& Q2Jp, VecXd& Q2r, size_t startIdx, const std::map<TimeCamId, size_t>& relOrder) const = 0;
    virtual void addDenseHB(DenseAccumulator& accum) const = 0;
    virtual void addDenseHB(MatXd& H, VecXd& b) const = 0;
    virtual void addDenseHBRel(MatXd& H_rel, VecXd& bRel, const std::map<TimeCamId, size_t>& relOrder) const = 0;
    virtual const Eigen::PermutationMatrix<Eigen::Dynamic>& getRelPermutation() const = 0;
    virtual Eigen::PermutationMatrix<Eigen::Dynamic> computeRelPermutation(const std::map<TimeCamId, size_t>& relOrder) const = 0;
    virtual void addDenseHBRel2(MatXd& H_rel, VecXd& bRel) const = 0;
    virtual TimeCamId getHostKf() const = 0;
    virtual size_t numQ2rows() const = 0;
    static std::unique_ptr<LandmarkBlock> createLandmarkBlock();
};

} // arc
