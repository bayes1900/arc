#pragma once

#include <lio/ekf.h>
#include <lio/state.h>

namespace arc {

using LioEkf = ManifoldEsekf<LioState, 12, LioInput>;

} // arc
