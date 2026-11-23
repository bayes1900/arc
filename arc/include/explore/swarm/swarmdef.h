#pragma once

#include <base/iarc.h>

namespace arc {

struct DroneState {
    int droneId{0};
    std::vector<int> assignedGridIds;
    Eigen::Vector3d pos{Eigen::Vector3d::Zero()};
    Eigen::Vector3d vel{Eigen::Vector3d::Zero()};
    double yaw{0.0};
    double stamp{0.0};
    double lastPairOptAttemptTime{0.0};
};

struct PairOptRequest {
    int fromDroneId{0};
    int toDroneId{0};
    double stamp{0.0};
    std::vector<int> selfGridIds;
    std::vector<int> otherGridIds;
};

struct PairOptResponse {
    int fromDroneId{0};
    int toDroneId{0};
    double stamp{0.0};
    int status{0};
};

struct PairOptStatus {
    static constexpr int ACCEPTED = 1;
    static constexpr int REJECTED = 2;
};

struct SwarmParams {
    double droneStateInterval{0.04};
    double pairOptSelectionInterval{0.05};
    double pairOptAttemptInterval{0.1};
    double pairOptCooldownInterval{0.5};
    double droneStateTimeout{0.2};
    double pairOptResponseTimeout{1.0};
    double pairOptCostMargin{0.1};
    int repeatSendCount{10};
};

} // arc