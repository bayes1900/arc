#pragma once

#include <base/iarc.h>
#include <env/mapdef.h>

namespace arc {

class RayCaster {
public:
    RayCaster(const MapParams& mp)
        : mp_(mp)
    {}
    bool input(const Eigen::Vector3d& start, const Eigen::Vector3d& end) {
        return inputNormalized((start - mp_.origin) * mp_.resolutionInv, end);
    }
    bool inputNormalized(const Eigen::Vector3d& startNormalized, const Eigen::Vector3d& end) {
        start_ = startNormalized;
        end_ = (end - mp_.origin) * mp_.resolutionInv;

        x_ = misc::floor(start_.x());
        y_ = misc::floor(start_.y());
        z_ = misc::floor(start_.z());
        endX_ = misc::floor(end_.x());
        endY_ = misc::floor(end_.y());
        endZ_ = misc::floor(end_.z());
        dx_ = endX_ - x_;
        dy_ = endY_ - y_;
        dz_ = endZ_ - z_;
        stepX_ = misc::signum(dx_);
        stepY_ = misc::signum(dy_);
        stepZ_ = misc::signum(dz_);

        const double inf = std::numeric_limits<double>::infinity();
        tMaxX_ = dx_ == 0 ? inf : misc::intbound(start_.x(), dx_);
        tMaxY_ = dy_ == 0 ? inf : misc::intbound(start_.y(), dy_);
        tMaxZ_ = dz_ == 0 ? inf : misc::intbound(start_.z(), dz_);
        tDeltaX_ = dx_ == 0 ? inf : static_cast<double>(stepX_) / dx_;
        tDeltaY_ = dy_ == 0 ? inf : static_cast<double>(stepY_) / dy_;
        tDeltaZ_ = dz_ == 0 ? inf : static_cast<double>(stepZ_) / dz_;

        return !(dx_ == 0 && dy_ == 0 && dz_ == 0);
    }
    bool nextId(Eigen::Vector3i& idx) {
        idx = Eigen::Vector3i(x_, y_, z_);
        if (x_ == endX_ && y_ == endY_ && z_ == endZ_) {
            return false;
        }
        advance_();
        return true;
    }

private:
    void advance_() {
        if (tMaxX_ < tMaxY_) {
            if (tMaxX_ < tMaxZ_) {
                x_ += stepX_;
                tMaxX_ += tDeltaX_;
            } else {
                z_ += stepZ_;
                tMaxZ_ += tDeltaZ_;
            }
        } else {
            if (tMaxY_ < tMaxZ_) {
                y_ += stepY_;
                tMaxY_ += tDeltaY_;
            } else {
                z_ += stepZ_;
                tMaxZ_ += tDeltaZ_;
            }
        }
    }

private:
    const MapParams& mp_;

private:
    Eigen::Vector3d start_;
    Eigen::Vector3d end_;
    int x_{0};
    int y_{0};
    int z_{0};
    int endX_{0};
    int endY_{0};
    int endZ_{0};
    int dx_{0};
    int dy_{0};
    int dz_{0};
    int stepX_{0};
    int stepY_{0};
    int stepZ_{0};
    double tMaxX_{0.0};
    double tMaxY_{0.0};
    double tMaxZ_{0.0};
    double tDeltaX_{0.0};
    double tDeltaY_{0.0};
    double tDeltaZ_{0.0};
};

} // arc
