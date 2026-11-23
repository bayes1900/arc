#pragma once

#include <base/base.h>

#include <sophus/se2.hpp>

#include <vio/vioconfig.h>

#include <vio/image/image.h>
#include <vio/viocalibration.h>
#include <vio/camera/stereographic.h>

#include <tbb/concurrent_queue.h>

namespace arc {

using KeypointId = size_t;

struct OpticalFlowTypes {
    using Scalar = float;
    using Vector2 = Eigen::Matrix<Scalar, 2, 1>;
    using Vector2T = Eigen::Matrix<Scalar, 1, 2>;
    using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
    using Matrix2 = Eigen::Matrix<Scalar, 2, 2>;
    using Matrix3 = Eigen::Matrix<Scalar, 3, 3>;
    using Vector4 = Eigen::Matrix<Scalar, 4, 1>;
    using Matrix4 = Eigen::Matrix<Scalar, 4, 4>;
    using SE2 = Sophus::SE2<Scalar>;
};

struct OpticalFlowInput {
    using Ptr = std::shared_ptr<OpticalFlowInput>;
    int64_t tNs;
    std::vector<ImageData> imgData;
};

struct OpticalFlowResult {
    using Ptr = std::shared_ptr<OpticalFlowResult>;
    int64_t tNs;
    std::vector<std::map<KeypointId, Eigen::AffineCompact2f>> observations;
    std::vector<std::map<KeypointId, size_t>> pyramidLevels;
    OpticalFlowInput::Ptr inputImages;
};

class OpticalFlowBase : public OpticalFlowTypes {
public:
    using Ptr = std::shared_ptr<OpticalFlowBase>;
    virtual ~OpticalFlowBase() = default;
    tbb::concurrent_bounded_queue<OpticalFlowInput::Ptr> inputQueue;
    tbb::concurrent_bounded_queue<OpticalFlowResult::Ptr>* outputQueue = nullptr;
    Eigen::MatrixXf patchCoord;
};

} // arc
