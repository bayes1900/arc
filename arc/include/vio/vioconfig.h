#pragma once

#include <base/base.h>

namespace arc {

enum class LinearizationType { ABS_QR, ABS_SC, REL_SC };

struct VioConfig {
    std::string opticalFlowType{"frame_to_frame"};
    int opticalFlowDetectionGridSize{50};
    float opticalFlowMaxRecoveredDist2{0.09f};
    int opticalFlowPattern{51};
    int opticalFlowMaxIterations{5};
    int opticalFlowLevels{3};
    float opticalFlowEpipolarError{0.005f};
    int opticalFlowSkipFrames{1};
    LinearizationType vioLinearizationType{LinearizationType::ABS_QR};
    bool vioSqrtMarg{true};
    int vioMaxStates{3};
    int vioMaxKfs{7};
    int vioMinFramesAfterKf{5};
    float vioNewKfKeypointsThresh{0.7f};
    bool vioDebug{false};
    bool vioExtendedLogging{false};
    int vioMaxIterations{7};
    double vioObsStdDev{0.5};
    double vioObsHuberThresh{1.0};
    double vioMinTriangulationDist{0.05};
    bool vioEnforceRealtime{false};
    double vioLmLambdaInitial{1e-4};
    double vioLmLambdaMin{1e-6};
    double vioLmLambdaMax{1e2};
    double vioInitPoseWeight{1e8};
    double vioInitBaWeight{1e1};
    double vioInitBgWeight{1e2};
    bool vioMargLostLandmarks{true};
    double vioKfMargFeatureRatio{0.1};
};

} // arc
