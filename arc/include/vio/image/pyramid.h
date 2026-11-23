#pragma once

#include <vio/image/image.h>

namespace arc {

template <typename T, class Allocator = DefaultImageAllocator<T>>
class ManagedImagePyr {
public:
    using PixelType = T;
    using Ptr = std::shared_ptr<ManagedImagePyr<T, Allocator>>;
    ManagedImagePyr() {}
    ManagedImagePyr(const ManagedImage<T>& other, size_t numLevels) { setFromImage(other, numLevels); }
    void setFromImage(const ManagedImage<T>& other, size_t numLevels) {
        origW_ = other.w;
        image_.reinitialise(other.w + other.w / 2, other.h);
        image_.fill(0);
        lvlInternal(0).copyFrom(other);
        for (size_t i = 0; i < numLevels; ++ i) {
            const Image<const T> l = lvl(i);
            Image<T> lp1 = lvlInternal(i + 1);
            subsample(l, lp1);
        }
    }
    static inline int border101(int x, int h) { return h - 1 - std::abs(h - 1 - x); }
    static void subsample(const Image<const T>& img, Image<T>& imgSub) {
        static_assert(std::is_same<T, uint16_t>::value || std::is_same<T, uint8_t>::value);
        constexpr int KERNEL[5] = {1, 4, 6, 4, 1};
        ManagedImage<int> tmp(imgSub.h, img.w);
        {
            for (int r = 0; r < int(imgSub.h); ++ r) {
                const T* rowMinus2 = img.rowPtr(std::abs(2 * r - 2));
                const T* rowMinus1 = img.rowPtr(std::abs(2 * r - 1));
                const T* row = img.rowPtr(2 * r);
                const T* rowPlus1 = img.rowPtr(border101(2 * r + 1, img.h));
                const T* rowPlus2 = img.rowPtr(border101(2 * r + 2, img.h));
                for (int c = 0; c < int(img.w); ++ c) {
                    tmp(r, c) = KERNEL[0] * int(rowMinus2[c]) + KERNEL[1] * int(rowMinus1[c]) + KERNEL[2] * int(row[c]) +
                        KERNEL[3] * int(rowPlus1[c]) + KERNEL[4] * int(rowPlus2[c]);
                }
            }
        }
        {
            for (int c = 0; c < int(imgSub.w); ++ c) {
                const int* rowMinus2 = tmp.rowPtr(std::abs(2 * c - 2));
                const int* rowMinus1 = tmp.rowPtr(std::abs(2 * c - 1));
                const int* row = tmp.rowPtr(2 * c);
                const int* rowPlus1 = tmp.rowPtr(border101(2 * c + 1, tmp.h));
                const int* rowPlus2 = tmp.rowPtr(border101(2 * c + 2, tmp.h));
                for (int r = 0; r < int(tmp.w); ++ r) {
                    int valInt = KERNEL[0] * rowMinus2[r] + KERNEL[1] * rowMinus1[r] + KERNEL[2] * row[r] +
                        KERNEL[3] * rowPlus1[r] + KERNEL[4] * rowPlus2[r];
                    T val = ((valInt + (1 << 7)) >> 8);
                    imgSub(c, r) = val;
                }
            }
        }
    }
    const Image<const T> lvl(size_t lvl) const {
        size_t x = (lvl == 0) ? 0 : origW_;
        size_t y = (lvl <= 1) ? 0 : (image_.h - (image_.h >> (lvl - 1)));
        size_t width = (origW_ >> lvl);
        size_t height = (image_.h >> lvl);
        return image_.subImage(x, y, width, height);
    }
    const Image<const T> mipmap() const { return image_.SubImage(0, 0, image_.w, image_.h); }

    template <typename S>
    Eigen::Matrix<S, 2, 1> lvlOffset(size_t lvl) {
        size_t x = (lvl == 0) ? 0 : origW_;
        size_t y = (lvl <= 1) ? 0 : (image_.h - (image_.h >> (lvl - 1)));
        return Eigen::Matrix<S, 2, 1>(x, y);
    }

protected:
    Image<T> lvlInternal(size_t lvl) {
        size_t x = (lvl == 0) ? 0 : origW_;
        size_t y = (lvl <= 1) ? 0 : (image_.h - (image_.h >> (lvl - 1)));
        size_t width = (origW_ >> lvl);
        size_t height = (image_.h >> lvl);
        return image_.subImage(x, y, width, height);
    }

    size_t origW_;
    ManagedImage<T> image_;
};

} // arc
