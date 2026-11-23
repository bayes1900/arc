#pragma once

#include <vio/opticalflow/base.h>
#include <vio/opticalflow/f2f.h>
#include <vio/opticalflow/msf2f.h>
#include <vio/opticalflow/patchflow.h>

namespace arc {

class OpticalFlowFactory {
public:
    static OpticalFlowBase::Ptr getOpticalFlow(const VioConfig& config, const Calibration<double>& cam);
};

inline OpticalFlowBase::Ptr OpticalFlowFactory::getOpticalFlow(
    const VioConfig& config, const Calibration<double>& cam) {
    OpticalFlowBase::Ptr res;
    if (config.opticalFlowType == "patch") {
        switch (config.opticalFlowPattern) {
            case 24: {
                res.reset(new PatchOpticalFlow<Pattern24>(config, cam));
                break;
            }
            case 52: {
                res.reset(new PatchOpticalFlow<Pattern52>(config, cam));
                break;
            }
            case 51: {
                res.reset(new PatchOpticalFlow<Pattern51>(config, cam));
                break;
            }
            case 50: {
                res.reset(new PatchOpticalFlow<Pattern50>(config, cam));
                break;
            }
            default: {
                std::cerr << "config.optical_flow_pattern " << config.opticalFlowPattern << " is not supported."
                          << std::endl;
                std::abort();
            }
        }
    }
    if (config.opticalFlowType == "frame_to_frame") {
        switch (config.opticalFlowPattern) {
            case 24: {
                res.reset(new FrameToFrameOpticalFlow<Pattern24>(config, cam));
                break;
            }
            case 52: {
                res.reset(new FrameToFrameOpticalFlow<Pattern52>(config, cam));
                break;
            }
            case 51: {
                res.reset(new FrameToFrameOpticalFlow<Pattern51>(config, cam));
                break;
            }
            case 50: {
                res.reset(new FrameToFrameOpticalFlow<Pattern50>(config, cam));
                break;
            }
            default: {
                std::cerr << "config.optical_flow_pattern " << config.opticalFlowPattern << " is not supported."
                          << std::endl;
                std::abort();
            }
        }
    }
    if (config.opticalFlowType == "multiscale_frame_to_frame") {
        switch (config.opticalFlowPattern) {
            case 24: {
                res.reset(new MultiscaleFrameToFrameOpticalFlow<Pattern24>(config, cam));
                break;
            }
            case 52: {
                res.reset(new MultiscaleFrameToFrameOpticalFlow<Pattern52>(config, cam));
                break;
            }
            case 51: {
                res.reset(new MultiscaleFrameToFrameOpticalFlow<Pattern51>(config, cam));
                break;
            }
            case 50: {
                res.reset(new MultiscaleFrameToFrameOpticalFlow<Pattern50>(config, cam));
                break;
            }
            default: {
                std::cerr << "config.optical_flow_pattern " << config.opticalFlowPattern << " is not supported."
                          << std::endl;
                std::abort();
            }
        }
    }
    return res;
}

} // arc
