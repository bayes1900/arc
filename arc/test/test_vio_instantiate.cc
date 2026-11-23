#include <vio/vio.h>

namespace arc {

template struct Calibration<double>;
template class GenericCamera<double>;
template class PinholeCamera<double>;
template class DoubleSphereCamera<double>;
template class KannalaBrandtCamera4<double>;
template class ExtendedUnifiedCamera<double>;
template class UnifiedCamera<double>;
template class PinholeRadtan8Camera<double>;
template class FrameToFrameOpticalFlow<Pattern50>;
template class FrameToFrameOpticalFlow<Pattern51>;
template class FrameToFrameOpticalFlow<Pattern52>;
template class FrameToFrameOpticalFlow<Pattern24>;
template class PatchOpticalFlow<Pattern50>;
template class PatchOpticalFlow<Pattern51>;
template class PatchOpticalFlow<Pattern52>;
template class PatchOpticalFlow<Pattern24>;
template class MultiscaleFrameToFrameOpticalFlow<Pattern50>;
template class MultiscaleFrameToFrameOpticalFlow<Pattern51>;
template class MultiscaleFrameToFrameOpticalFlow<Pattern52>;
template class MultiscaleFrameToFrameOpticalFlow<Pattern24>;
} // arc

int main() {
    return 0;
}
