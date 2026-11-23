#pragma once

#include <base/base.h>

#include <vio/estimator/state.h>

namespace arc {

struct KeypointObservation {
    int kptId;
    Eigen::Matrix<double, 2, 1> pos;
};

struct Keypoint {
    using ObsMap = std::map<TimeCamId, Vec2d>;
    using MapIter = typename ObsMap::iterator;
    Vec2d direction;
    double invDist;
    TimeCamId hostKfId;
    ObsMap obs;
    void backup() {
        backupDirection_ = direction;
        backupInvDist_ = invDist;
    }
    void restore() {
        direction = backupDirection_;
        invDist = backupInvDist_;
    }

private:
    Vec2d backupDirection_;
    double backupInvDist_;
};

class LandmarkDatabase {
public:
    void addLandmark(KeypointId lmId, const Keypoint& pos) {
        auto& kpt = kpts_[lmId];
        kpt.direction = pos.direction;
        kpt.invDist = pos.invDist;
        kpt.hostKfId = pos.hostKfId;
    }
    void removeFrame(const int64_t& frame) {
        for (auto it = kpts_.begin(); it != kpts_.end();) {
            for (auto it2 = it->second.obs.begin(); it2 != it->second.obs.end();) {
                if (it2->first.frameId == frame)
                    it2 = removeLandmarkObservationHelper_(it, it2);
                else
                    ++ it2;
            }
            if (it->second.obs.size() < MIN_NUM_OBS) {
                it = removeLandmarkHelper_(it);
            } else {
                ++ it;
            }
        }
    }
    void removeKeyframes(
        const std::set<int64_t>& kfsToMarg,
        const std::set<int64_t>& posesToMarg,
        const std::set<int64_t>& statesToMargAll
    ) {
        for (auto it = kpts_.begin(); it != kpts_.end();) {
            if (kfsToMarg.count(it->second.hostKfId.frameId) > 0) {
                it = removeLandmarkHelper_(it);
            } else {
                for (auto it2 = it->second.obs.begin(); it2 != it->second.obs.end();) {
                    int64_t fid = it2->first.frameId;
                    if (posesToMarg.count(fid) > 0 || statesToMargAll.count(fid) > 0 ||
                        kfsToMarg.count(fid) > 0)
                        it2 = removeLandmarkObservationHelper_(it, it2);
                    else
                        ++ it2;
                }
                if (it->second.obs.size() < MIN_NUM_OBS) {
                    it = removeLandmarkHelper_(it);
                } else {
                    ++ it;
                }
            }
        }
    }
    void addObservation(const TimeCamId& tcidTarget, const KeypointObservation& o) {
        auto it = kpts_.find(o.kptId);
        ensure(it != kpts_.end(), "VIO check failed");
        it->second.obs[tcidTarget] = o.pos;
        observations_[it->second.hostKfId][tcidTarget].insert(it->first);
    }
    Keypoint& getLandmark(KeypointId lmId) { return kpts_.at(lmId); }
    const Keypoint& getLandmark(KeypointId lmId) const { return kpts_.at(lmId); }
    std::vector<TimeCamId> getHostKfs() const {
        std::vector<TimeCamId> res;
        for (const auto& kv : observations_)
            res.emplace_back(kv.first);
        return res;
    }
    std::vector<const Keypoint*> getLandmarksForHost(const TimeCamId& tcid) const {
        std::vector<const Keypoint*> res;
        for (const auto& [k, obs] : observations_.at(tcid))
            for (const auto& v : obs)
                res.emplace_back(&kpts_.at(v));
        return res;
    }
    const std::unordered_map<TimeCamId, std::map<TimeCamId, std::set<KeypointId>>>& getObservations() const {
        return observations_;
    }
    const std::unordered_map<KeypointId, Keypoint>& getLandmarks() const { return kpts_; }
    bool landmarkExists(int lmId) const { return kpts_.count(lmId) > 0; }
    size_t numLandmarks() const { return kpts_.size(); }
    int numObservations() const {
        int numObservations = 0;
        for (const auto& [_, valMap] : observations_) {
            for (const auto& [_, val] : valMap) {
                numObservations += val.size();
            }
        }
        return numObservations;
    }
    int numObservations(KeypointId lmId) const { return kpts_.at(lmId).obs.size(); }
    void removeLandmark(KeypointId lmId) {
        auto it = kpts_.find(lmId);
        if (it != kpts_.end())
            removeLandmarkHelper_(it);
    }
    void removeObservations(KeypointId lmId, const std::set<TimeCamId>& obs) {
        auto it = kpts_.find(lmId);
        ensure(it != kpts_.end(), "VIO check failed");
        for (auto it2 = it->second.obs.begin(); it2 != it->second.obs.end();) {
            if (obs.count(it2->first) > 0) {
                it2 = removeLandmarkObservationHelper_(it, it2);
            } else
                ++ it2;
        }
        if (it->second.obs.size() < MIN_NUM_OBS) {
            removeLandmarkHelper_(it);
        }
    }
    void backup() {
        for (auto& kv : kpts_)
            kv.second.backup();
    }
    void restore() {
        for (auto& kv : kpts_)
            kv.second.restore();
    }

private:
    using MapIter = typename std::unordered_map<KeypointId, Keypoint>::iterator;
    MapIter removeLandmarkHelper_(MapIter it) {
        auto hostIt = observations_.find(it->second.hostKfId);
        for (const auto& [k, v] : it->second.obs) {
            auto targetIt = hostIt->second.find(k);
            targetIt->second.erase(it->first);
            if (targetIt->second.empty())
                hostIt->second.erase(targetIt);
        }
        if (hostIt->second.empty())
            observations_.erase(hostIt);
        return kpts_.erase(it);
    }
    Keypoint::MapIter removeLandmarkObservationHelper_(MapIter it, Keypoint::MapIter it2) {
        auto hostIt = observations_.find(it->second.hostKfId);
        auto targetIt = hostIt->second.find(it2->first);
        targetIt->second.erase(it->first);
        if (targetIt->second.empty())
            hostIt->second.erase(targetIt);
        if (hostIt->second.empty())
            observations_.erase(hostIt);
        return it->second.obs.erase(it2);
    }

    std::unordered_map<KeypointId, Keypoint> kpts_;
    std::unordered_map<TimeCamId, std::map<TimeCamId, std::set<KeypointId>>> observations_;
    static constexpr int MIN_NUM_OBS = 2;
};

} // arc
