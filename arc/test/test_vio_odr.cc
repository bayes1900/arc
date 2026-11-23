#include <vio/vio.h>

int vioOdrHelper() {
    arc::VioConfig config;
    arc::Calibration<double> calibration;
    arc::LandmarkDatabase lmdb;
    arc::DenseAccumulator accumulator;
    accumulator.reset(6);
    return config.vioMaxStates + static_cast<int>(calibration.imuUpdateRate) - 200 + 39 +
        static_cast<int>(lmdb.numLandmarks()) + static_cast<int>(accumulator.getH().rows()) - 6;
}
