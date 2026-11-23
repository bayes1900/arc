#pragma once

#include <env/mapdef.h>

namespace arc {

class Esdf {
public:
    Esdf(const MapParams& mp, MapData& md)
        : mp_(mp)
        , md_(md)
        , tmpBuffer1_(md.numVoxels, 0.0)
        , tmpBuffer2_(md.numVoxels, 0.0)
        , vScratch_(mp.voxelShape.maxCoeff())
        , zScratch_(mp.voxelShape.maxCoeff() + 1)
    {}
    void update() {
        const int minX = md_.localBoundMin[0];
        const int maxX = md_.localBoundMax[0];
        const int minY = md_.localBoundMin[1];
        const int maxY = md_.localBoundMax[1];
        const int minZ = md_.localBoundMin[2];
        const int maxZ = md_.localBoundMax[2];
        if (minX > maxX || minY > maxY || minZ > maxZ) {
            return;
        }
        const double maxSquared = static_cast<double>(mp_.voxelShape.squaredNorm());
        for (int x = minX; x <= maxX; ++ x) {
            for (int y = minY; y <= maxY; ++ y) {
                fill1d_(
                    [&](int z) {
                        return md_.occupied[mp_.getVoxelIdx(x, y, z)] ? 0.0 : maxSquared;
                    },
                    [&](int z, double val) {
                        tmpBuffer1_[mp_.getVoxelIdx(x, y, z)] = val;
                    },
                    minZ, maxZ);
            }
        }
        for (int x = minX; x <= maxX; ++ x) {
            for (int z = minZ; z <= maxZ; ++ z) {
                fill1d_(
                    [&](int y) {
                        return tmpBuffer1_[mp_.getVoxelIdx(x, y, z)];
                    },
                    [&](int y, double val) {
                        tmpBuffer2_[mp_.getVoxelIdx(x, y, z)] = val;
                    },
                    minY, maxY);
            }
        }
        for (int y = minY; y <= maxY; ++ y) {
            for (int z = minZ; z <= maxZ; ++ z) {
                fill1d_(
                    [&](int x) {
                        return tmpBuffer2_[mp_.getVoxelIdx(x, y, z)];
                    },
                    [&](int x, double val) {
                        md_.distance[mp_.getVoxelIdx(x, y, z)] = mp_.resolution * std::sqrt(std::max(0.0, val));
                    },
                    minX, maxX);
            }
        }
    }

private:
    template<typename Fget, typename Fset> void fill1d_(Fget fget, Fset fset, int start, int end) {
        std::vector<int>& v = vScratch_;
        std::vector<double>& z = zScratch_;
        int k = start;
        v[start] = start;
        z[start] = -std::numeric_limits<double>::max();
        z[start + 1] = std::numeric_limits<double>::max();
        for (int q = start + 1; q <= end; ++ q) {
            ++ k;
            double s;
            do {
                -- k;
                const double q2 = static_cast<double>(q) * q;
                const double vk2 = static_cast<double>(v[k]) * v[k];
                s = ((fget(q) + q2) - (fget(v[k]) + vk2)) /
                    (2.0 * q - 2.0 * v[k]);
            } while (s <= z[k]);

            ++ k;
            v[k] = q;
            z[k] = s;
            z[k + 1] = std::numeric_limits<double>::max();
        }
        k = start;
        for (int q = start; q <= end; ++ q) {
            while (z[k + 1] < q) {
                ++ k;
            }
            const double diff = static_cast<double>(q - v[k]);
            const double val = diff * diff + fget(v[k]);
            fset(q, val);
        }
    }

private:
    const MapParams& mp_;
    MapData& md_;
    std::vector<double> tmpBuffer1_;
    std::vector<double> tmpBuffer2_;
    std::vector<int> vScratch_;
    std::vector<double> zScratch_;
};

} // arc
