#pragma once

#include <unordered_map>
#include <Eigen/Dense>
#include <base/base.h>

namespace arc {

using IndexedBlocks = std::unordered_map<size_t, Eigen::MatrixXd>;
inline void scaleJacobians(
    IndexedBlocks& blockDiagonal,
    size_t numBlocks,
    size_t blockSize,
    const Eigen::VectorXd& scalingVector
) {
    ensure(numBlocks * blockSize == static_cast<size_t>(scalingVector.size()), "VIO check failed");
    for (auto& [camIdx, block] : blockDiagonal) {
        auto D = scalingVector.segment(blockSize * camIdx, blockSize).asDiagonal();
        block = D * block * D;
    }
}
inline void addDiagonal(
    IndexedBlocks& blockDiagonal,
    size_t numBlocks,
    size_t blockSize,
    const Eigen::VectorXd& diagonal
) {
    ensure(numBlocks * blockSize == static_cast<size_t>(diagonal.size()), "VIO check failed");
    for (size_t idx = 0; idx < numBlocks; ++ idx) {
        auto [it, isNew] = blockDiagonal.try_emplace(idx);
        if (isNew) {
            it->second = diagonal.segment(blockSize * idx, blockSize).asDiagonal();
        } else {
            it->second += diagonal.segment(blockSize * idx, blockSize).asDiagonal();
        }
    }
}

class BlockDiagonalAccumulator {
public:
    void add(size_t idx, MatXd&& block) {
        auto [it, isNew] = blockDiagonal.try_emplace(idx);
        if (isNew) {
            it->second = std::move(block);
        } else {
            it->second += block;
        }
    }
    void addDiag(size_t numBlocks, size_t blockSize, const VecXd& diagonal) {
        addDiagonal(blockDiagonal, numBlocks, blockSize, diagonal);
    }
    void join(BlockDiagonalAccumulator& b) {
        for (auto& [k, v] : b.blockDiagonal) {
            auto [it, isNew] = blockDiagonal.try_emplace(k);
            if (isNew) {
                it->second = std::move(v);
            } else {
                it->second += v;
            }
        }
    }

private:
    IndexedBlocks blockDiagonal;
};

} // arc
