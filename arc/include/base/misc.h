#pragma once

#include <cmath>
#include <cstddef>
#include <functional>
#include <type_traits>
#include <vector>
#include <Eigen/Dense>

namespace arc {

template <class T>
struct DependentFalse : std::false_type {};

template <class T>
inline void hashCombine(size_t& seed, const T& value) {
    if constexpr (sizeof(size_t) == 4) {
        seed ^= std::hash<T>{}(value) + 0x9e3779b9U + (seed << 6) + (seed >> 2);
    } else if constexpr (sizeof(size_t) == 8) {
        seed ^= std::hash<T>{}(value) + 0x9e3779b97f4a7c15LLU + (seed << 12) + (seed >> 4);
    } else {
        static_assert(DependentFalse<T>::value, "hashCombine not implemented");
    }
}

namespace misc {

inline int signum(int value) {
    return value == 0 ? 0 : (value < 0 ? -1 : 1);
}

inline double mod(double value, double modulus) {
    return std::fmod(std::fmod(value, modulus) + modulus, modulus);
}

inline double intbound(double s, double ds) {
    if (ds < 0.0) {
        return intbound(-s, -ds);
    }
    s = mod(s, 1.0);
    return (1.0 - s) / ds;
}

inline int floor(double v) {
    int i = static_cast<int>(v);
    return v < i ? i - 1 : i;
}

inline double logit(double v) {
    return log(v / (1.0 - v));
}

inline double normalizeAngle(double angle) {
    constexpr double pi = 3.14159265358979323846;
    while (angle > pi) {
        angle -= 2.0 * pi;
    }
    while (angle < -pi) {
        angle += 2.0 * pi;
    }
    return angle;
}

inline double pathLength(const std::vector<Eigen::Vector3d>& path) {
    double length = 0.0;
    for (size_t i = 0; i + 1 < path.size(); ++ i) {
        length += (path[i + 1] - path[i]).norm();
    }
    return length;
}

inline bool boxesOverlap(const Eigen::Vector3d& min1, const Eigen::Vector3d& max1, const Eigen::Vector3d& min2, const Eigen::Vector3d& max2) {
    return (min1.array() <= max2.array() + 1e-3).all() && (max1.array() >= min2.array() - 1e-3).all();
}

struct Vector3iHash {
    size_t operator()(const Eigen::Vector3i& index) const {
        size_t seed = 0;
        for (int i = 0; i < 3; ++ i) {
            seed ^= std::hash<int>()(index(i)) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

} // misc
} // arc
