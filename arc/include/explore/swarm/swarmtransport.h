#pragma once

#include <base/iarc.h>
#include <explore/swarm/swarmdef.h>

#include <functional>

namespace arc {

class SwarmTransport {
public:
    virtual ~SwarmTransport() = default;
    virtual void sendDroneState(const DroneState& state) = 0;
    virtual void sendPairOpt(const PairOptRequest& request) = 0;
    virtual void sendPairOptResponse(const PairOptResponse& response) = 0;
};

class InMemorySwarmTransport : public SwarmTransport {
public:
    using DroneStateHandler = std::function<void(const DroneState&)>;
    using PairOptHandler = std::function<void(const PairOptRequest&)>;
    using PairOptResponseHandler = std::function<void(const PairOptResponse&)>;

    void setDroneStateHandler(DroneStateHandler handler) {
        droneStateHandler_ = std::move(handler);
    }
    void setPairOptHandler(PairOptHandler handler) {
        pairOptHandler_ = std::move(handler);
    }
    void setPairOptResponseHandler(PairOptResponseHandler handler) {
        pairOptResponseHandler_ = std::move(handler);
    }
    void sendDroneState(const DroneState& state) override {
        if (droneStateHandler_) {
            droneStateHandler_(state);
        }
    }
    void sendPairOpt(const PairOptRequest& request) override {
        if (pairOptHandler_) {
            pairOptHandler_(request);
        }
    }
    void sendPairOptResponse(const PairOptResponse& response) override {
        if (pairOptResponseHandler_) {
            pairOptResponseHandler_(response);
        }
    }

private:
    DroneStateHandler droneStateHandler_;
    PairOptHandler pairOptHandler_;
    PairOptResponseHandler pairOptResponseHandler_;
};

} // arc