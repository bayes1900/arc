#pragma once

#include <base/iarc.h>
#include <explore/swarm/swarmdef.h>

#include <unordered_set>

namespace arc {

class SwarmState {
public:
    SwarmState(int droneId, int droneNum)
        : droneId_(droneId)
        , droneNum_(droneNum)
        , states_(droneNum)
        , recentInteractTime_(droneNum, 0.0)
    {
        if (droneId_ < 1 || droneId_ > droneNum_) {
            throw std::invalid_argument("invalid drone id / drone number");
        }
        for (int i = 0; i < droneNum_; ++i) {
            states_[i].droneId = i + 1;
        }
    }
    bool updateState(const DroneState& state) {
        if (state.droneId < 1 || state.droneId > droneNum_) {
            return false;
        }
        DroneState& current = states_[state.droneId - 1];
        if (current.stamp > state.stamp + 1e-6) {
            return false;
        }
        current = state;
        return true;
    }
    DroneState& state(int droneId) {
        return states_.at(static_cast<size_t>(droneId - 1));
    }
    const DroneState& state(int droneId) const {
        return states_.at(static_cast<size_t>(droneId - 1));
    }
    std::vector<DroneState>& states() {
        return states_;
    }
    const std::vector<DroneState>& states() const {
        return states_;
    }
    int droneId() const {
        return droneId_;
    }
    int droneNum() const {
        return droneNum_;
    }
    double recentInteractTime(int droneId) const {
        return recentInteractTime_.at(static_cast<size_t>(droneId - 1));
    }
    void setRecentInteractTime(int droneId, double time) {
        recentInteractTime_.at(static_cast<size_t>(droneId - 1)) = time;
    }
    void findUnallocated(const std::vector<int>& active, std::vector<int>& missed) const {
        missed.clear();
        std::unordered_set<int> allocated;
        for (const DroneState& state : states_) {
            for (int id : state.assignedGridIds) {
                allocated.insert(id);
            }
        }
        for (int id : active) {
            if (allocated.find(id) == allocated.end()) {
                missed.push_back(id);
            }
        }
    }

private:
    int droneId_;
    int droneNum_;
    std::vector<DroneState> states_;
    std::vector<double> recentInteractTime_;
};

} // arc