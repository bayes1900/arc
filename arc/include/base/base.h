#pragma once

#include <map>
#include <set>
#include <string>
#include <vector>
#include <array>
#include <queue>
#include <cmath>
#include <limits>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <utility>
#include <cstdint>
#include <chrono>
#include <bitset>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <functional>
#include <memory>
#include <cstring>
#include <stdexcept>
#include <optional>
#include <thread>
#include <cstddef>
#include <filesystem>
#include <mutex>
#include <atomic>
#include <Eigen/Eigen>
#include <Eigen/Core>
#include <base/misc.h>

namespace arc {

constexpr double __pi__ = 3.14159265358979323846;

using Vec2d = Eigen::Vector2d;
using Vec3d = Eigen::Vector3d;
using Vec4d = Eigen::Vector4d;
using Vec6d = Eigen::Matrix<double, 6, 1>;
using Vec9d = Eigen::Matrix<double, 9, 1>;
using Vec15d = Eigen::Matrix<double, 15, 1>;
using VecXd = Eigen::VectorXd;

using Mat2d = Eigen::Matrix2d;
using Mat3d = Eigen::Matrix3d;
using Mat4d = Eigen::Matrix4d;
using Mat6d = Eigen::Matrix<double, 6, 6>;
using MatXd = Eigen::MatrixXd;

using Mat24d = Eigen::Matrix<double, 2, 4>;
using Mat42d = Eigen::Matrix<double, 4, 2>;
using Mat36d = Eigen::Matrix<double, 3, 6>;
using Mat63d = Eigen::Matrix<double, 6, 3>;

inline void ensure(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

} // arc
