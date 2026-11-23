#pragma once

#include <base/base.h>
#include <sophus/se3.hpp>

namespace arc {

struct TimeCamId {
    TimeCamId()
        : frameId(0)
        , camId(0)
    {}
    TimeCamId(const int64_t& frameId, const size_t& camId)
        : frameId(frameId)
        , camId(camId)
    {}
    int64_t frameId;
    size_t camId;
};

inline std::ostream& operator<<(std::ostream& os, const TimeCamId& tcid) {
    os << tcid.frameId << "_" << tcid.camId;
    return os;
}

inline bool operator<(const TimeCamId& o1, const TimeCamId& o2) {
    if (o1.frameId == o2.frameId) {
        return o1.camId < o2.camId;
    }
    return o1.frameId < o2.frameId;
}

inline bool operator==(const TimeCamId& o1, const TimeCamId& o2) {
    return o1.frameId == o2.frameId && o1.camId == o2.camId;
}

inline bool operator!=(const TimeCamId& o1, const TimeCamId& o2) {
    return o1.frameId != o2.frameId || o1.camId != o2.camId;
}

struct KeypointsData {
    std::vector<Eigen::Vector2d> corners;
    std::vector<double> cornerAngles;
    std::vector<std::bitset<256>> cornerDescriptors;
};

} // arc

namespace std {

template <>
struct hash<arc::TimeCamId> {
    size_t operator()(const arc::TimeCamId& x) const {
        size_t seed = 0;
        arc::hashCombine(seed, x.frameId);
        arc::hashCombine(seed, x.camId);
        return seed;
    }
};

template <>
struct hash<std::pair<arc::TimeCamId, arc::TimeCamId>> {
    size_t operator()(const std::pair<arc::TimeCamId, arc::TimeCamId>& x) const {
        size_t seed = 0;
        arc::hashCombine(seed, x.first.frameId);
        arc::hashCombine(seed, x.first.camId);
        arc::hashCombine(seed, x.second.frameId);
        arc::hashCombine(seed, x.second.camId);
        return seed;
    }
};

} // std
