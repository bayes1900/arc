#pragma once

#include <Eigen/Dense>

namespace arc {

struct Pattern24 {
    static constexpr float PATTERN_RAW[][2] = {{-1, 5},
        {1, 5},
        {-3, 3},
        {-1, 3},
        {1, 3},
        {3, 3},
        {-5, 1},
        {-3, 1},
        {-1, 1},
        {1, 1},
        {3, 1},
        {5, 1},
        {-5, -1},
        {-3, -1},
        {-1, -1},
        {1, -1},
        {3, -1},
        {5, -1},
        {-3, -3},
        {-1, -3},
        {1, -3},
        {3, -3},
        {-1, -5},
        {1, -5}
    };

    static constexpr int PATTERN_SIZE = sizeof(PATTERN_RAW) / (2 * sizeof(float));
    using Matrix2P = Eigen::Matrix<float, 2, PATTERN_SIZE>;
    static const Matrix2P PATTERN2;
};

inline const Pattern24::Matrix2P Pattern24::PATTERN2 = Eigen::Map<Pattern24::Matrix2P>(
    (float*)Pattern24::PATTERN_RAW);

struct Pattern52 {
    static constexpr float PATTERN_RAW[][2] = {{-3, 7},
        {-1, 7},
        {1, 7},
        {3, 7},
        {-5, 5},
        {-3, 5},
        {-1, 5},
        {1, 5},
        {3, 5},
        {5, 5},
        {-7, 3},
        {-5, 3},
        {-3, 3},
        {-1, 3},
        {1, 3},
        {3, 3},
        {5, 3},
        {7, 3},
        {-7, 1},
        {-5, 1},
        {-3, 1},
        {-1, 1},
        {1, 1},
        {3, 1},
        {5, 1},
        {7, 1},
        {-7, -1},
        {-5, -1},
        {-3, -1},
        {-1, -1},
        {1, -1},
        {3, -1},
        {5, -1},
        {7, -1},
        {-7, -3},
        {-5, -3},
        {-3, -3},
        {-1, -3},
        {1, -3},
        {3, -3},
        {5, -3},
        {7, -3},
        {-5, -5},
        {-3, -5},
        {-1, -5},
        {1, -5},
        {3, -5},
        {5, -5},
        {-3, -7},
        {-1, -7},
        {1, -7},
        {3, -7}
    };

    static constexpr int PATTERN_SIZE = sizeof(PATTERN_RAW) / (2 * sizeof(float));
    using Matrix2P = Eigen::Matrix<float, 2, PATTERN_SIZE>;
    static const Matrix2P PATTERN2;
};

inline const Pattern52::Matrix2P Pattern52::PATTERN2 = Eigen::Map<Pattern52::Matrix2P>(
    (float*)Pattern52::PATTERN_RAW);

struct Pattern51 {
    static constexpr int PATTERN_SIZE = Pattern52::PATTERN_SIZE;
    using Matrix2P = Eigen::Matrix<float, 2, PATTERN_SIZE>;
    static const Matrix2P PATTERN2;
};
inline const Pattern51::Matrix2P Pattern51::PATTERN2 = 0.5f * Pattern52::PATTERN2;

struct Pattern50 {
    static constexpr int PATTERN_SIZE = Pattern52::PATTERN_SIZE;
    using Matrix2P = Eigen::Matrix<float, 2, PATTERN_SIZE>;
    static const Matrix2P PATTERN2;
};
inline const Pattern50::Matrix2P Pattern50::PATTERN2 = 0.75f * Pattern52::PATTERN2;

} // arc
