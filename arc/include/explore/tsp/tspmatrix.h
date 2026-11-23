#pragma once

namespace arc {

struct TspMatrixCosts {
    static constexpr double DEPOT_TO_DRONE{-1000.0};
    static constexpr double DRONE_TO_DEPOT{1000.0};
    static constexpr double DRONE_TO_DRONE{10000.0};
    static constexpr double GRID_LINK{1000.0};
    static constexpr double SELF_LOOP{1000.0};
};

} // arc
