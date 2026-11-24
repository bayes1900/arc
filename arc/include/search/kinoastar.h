#pragma once

#include <base/iarc.h>
#include <env/sdfmap.h>

namespace arc {

struct KinoAstarOptions {
    double resolution{0.1};
    double maxTau{0.8};
    double initMaxTau{1.0};

    double maxVel{1.5};
    double velMargin{0.0};
    double maxAcc{1.0};

    double timeWeight{10.0};
    double horizon{5.0};
    double lambdaHeu{10.0};

    double collisionCheckStep{0.1};
    double maxSearchTime{1.0};
    double pathSampleInterval{0.01};

    size_t maxNodes{100000};
};

struct KinoAstarSearchStatus {
    static constexpr uint8_t DONE     = 0;
    static constexpr uint8_t HORIZON  = 1;
    static constexpr uint8_t NEAR_END = 2;
    static constexpr uint8_t NOPATH   = 3;
    static constexpr uint8_t TIMEOUT  = 4;
    static constexpr uint8_t OUTSIDE  = 5;
    static constexpr uint8_t NOMEM    = 6;
};

struct KinoAstarNodeState {
    static constexpr uint8_t INIT  = 0;
    static constexpr uint8_t OPEN  = 1;
    static constexpr uint8_t CLOSE = 2;
};

struct KinoAstarNode {
    uint8_t nodeState{KinoAstarNodeState::INIT};
    double gScore{0.0};
    double fScore{0.0};
    double duration{0.0};
    Eigen::Vector3i index{Eigen::Vector3i::Zero()};
    Eigen::Vector3d input{Eigen::Vector3d::Zero()};
    Vec6d state{Vec6d::Zero()};
    KinoAstarNode* parent{nullptr};
};

struct KinoAstarNodeCompare {
    bool operator()(const KinoAstarNode* lhs, const KinoAstarNode* rhs) const {
        return lhs->fScore > rhs->fScore;
    }
};

class KinoAstar {
public:
    KinoAstar(SdfMap& sdfmap, const KinoAstarOptions& options)
        : sdfmap_(sdfmap)
        , mp_(sdfmap.params())
        , md_(sdfmap.data())
        , options_(options)
        , boxMin_(sdfmap.params().boxMin)
        , boxMax_(sdfmap.params().boxMax)
    {
        if (options_.resolution <= 1e-6) {
            throw std::invalid_argument("KinoAstar resolution must be positive");
        }
        if (options_.maxTau <= 1e-6 || options_.initMaxTau <= 1e-6) {
            throw std::invalid_argument("KinoAstar time steps must be positive");
        }
        if (options_.maxVel + options_.velMargin <= 1e-6) {
            throw std::invalid_argument("KinoAstar max velocity must be positive");
        }
        if (options_.maxAcc <= 1e-6) {
            throw std::invalid_argument("KinoAstar max acceleration must be positive");
        }
        resolutionInv_ = 1.0 / options_.resolution;
        maxVel_ = options_.maxVel + options_.velMargin;
    }
    ~KinoAstar() {
        for (KinoAstarNode* node : nodePool_) {
            delete node;
        }
    }
    KinoAstar(const KinoAstar&) = delete;
    KinoAstar& operator=(const KinoAstar&) = delete;

public:
    void reset() {
        pathNodes_.clear();
        expandedNodes_.clear();
        isShotSucc_ = false;
        tShot_ = 0.0;
        coefShot_.setZero();
        endVel_.setZero();

        std::priority_queue<KinoAstarNode*, std::vector<KinoAstarNode*>, KinoAstarNodeCompare> emptyOpenSet;
        openSet_.swap(emptyOpenSet);
        for (size_t i = 0; i < nodeNum_; ++ i) {
            nodePool_[i]->parent = nullptr;
            nodePool_[i]->nodeState = KinoAstarNodeState::INIT;
        }
        nodeNum_ = 0;
        iterNum_ = 0;
    }
    std::vector<Eigen::Vector3d> getKinoTraj(double deltaT) const {
        std::vector<Eigen::Vector3d> stateList;
        if (pathNodes_.empty()) {
            return stateList;
        }
        if (deltaT <= 1e-6) {
            deltaT = options_.pathSampleInterval;
        }
        auto append = [&](const Eigen::Vector3d& point) {
            if (stateList.empty() || (stateList.back() - point).norm() > 1e-6) {
                stateList.push_back(point);
            }
        };
        for (size_t i = 0; i + 1 < pathNodes_.size(); ++ i) {
            const KinoAstarNode& node = *pathNodes_[i + 1];
            const Vec6d& state0 = pathNodes_[i]->state;
            const int steps = std::max(1, static_cast<int>(std::ceil(node.duration / deltaT)));
            for (int k = 0; k <= steps; ++ k) {
                const double t = node.duration * static_cast<double>(k) / static_cast<double>(steps);
                Vec6d state;
                stateTransit_(state0, state, node.input, t);
                append(state.head<3>());
            }
        }
        if (pathNodes_.size() == 1) {
            append(pathNodes_.front()->state.head<3>());
        }
        if (isShotSucc_) {
            const int steps = std::max(1, static_cast<int>(std::ceil(tShot_ / deltaT)));
            for (int k = 1; k <= steps; ++ k) {
                const double t = tShot_ * static_cast<double>(k) / static_cast<double>(steps);
                append(evaluateShot_(t));
            }
        }
        return stateList;
    }
    void getSamples(double& ts, std::vector<Eigen::Vector3d>& pointSet, std::vector<Eigen::Vector3d>& boundaryDerivatives) const {
        pointSet.clear();
        boundaryDerivatives.clear();
        if (pathNodes_.empty()) {
            return;
        }
        double searchDuration = 0.0;
        for (size_t i = 1; i < pathNodes_.size(); ++ i) {
            searchDuration += pathNodes_[i]->duration;
        }
        const double totalDuration = searchDuration + (isShotSucc_ ? tShot_ : 0.0);
        if (totalDuration <= 1e-6) {
            return;
        }
        Eigen::Vector3d endVel;
        Eigen::Vector3d endAcc;
        if (isShotSucc_) {
            endVel = endVel_;
            endAcc = 2.0 * coefShot_.col(2) + 6.0 * coefShot_.col(3) * tShot_;
        } else {
            endVel = pathNodes_.back()->state.tail<3>();
            endAcc = pathNodes_.back()->input;
        }
        Eigen::Vector3d startAcc;
        if (pathNodes_.size() > 1) {
            startAcc = pathNodes_[1]->input;
        } else if (isShotSucc_) {
            startAcc = 2.0 * coefShot_.col(2);
        } else {
            startAcc = startAcc_;
        }
        if (ts <= 1e-6) {
            ts = options_.pathSampleInterval;
        }
        int segNum = static_cast<int>(std::floor(totalDuration / ts));
        segNum = std::max(8, segNum);
        ts = totalDuration / static_cast<double>(segNum);

        for (int k = 0; k <= segNum; ++ k) {
            const double t = (k == segNum) ? totalDuration : ts * static_cast<double>(k);
            if (t <= searchDuration + 1e-9 && pathNodes_.size() > 1) {
                double tAccum = 0.0;
                bool sampled = false;
                for (size_t i = 1; i < pathNodes_.size(); ++ i) {
                    const double duration = pathNodes_[i]->duration;
                    if (t <= tAccum + duration + 1e-9) {
                        Vec6d state;
                        stateTransit_(pathNodes_[i - 1]->state, state, pathNodes_[i]->input, t - tAccum);
                        pointSet.push_back(state.head<3>());
                        sampled = true;
                        break;
                    }
                    tAccum += duration;
                }
                if (!sampled) {
                    pointSet.push_back(pathNodes_.back()->state.head<3>());
                }
            } else if (isShotSucc_) {
                pointSet.push_back(evaluateShot_(std::max(0.0, t - searchDuration)));
            } else {
                pointSet.push_back(pathNodes_.back()->state.head<3>());
            }
        }
        boundaryDerivatives.push_back(startVel_);
        boundaryDerivatives.push_back(endVel);
        boundaryDerivatives.push_back(startAcc);
        boundaryDerivatives.push_back(endAcc);
    }

public:
    // The caller should retry with init=false when this returns NOPATH. This
    // mirrors the original RACER two-stage search.
    uint8_t search(
        const Eigen::Vector3d& start,
        const Eigen::Vector3d& startVel,
        const Eigen::Vector3d& startAcc,
        const Eigen::Vector3d& end,
        const Eigen::Vector3d& endVel,
        bool init=true,
        bool optimistic=true
    ) {
        if (!isInBox_(end) ||
            !mp_.isInMap(start) || !mp_.isInMap(end)) {
            return KinoAstarSearchStatus::OUTSIDE;
        }
        reset();
        const auto searchStart = std::chrono::steady_clock::now();
        startVel_ = startVel;
        startAcc_ = startAcc;
        optimistic_ = optimistic;

        KinoAstarNode* currentNode = releaseNode_();
        if (currentNode == nullptr) {
            return KinoAstarSearchStatus::NOMEM;
        }
        currentNode->parent = nullptr;
        currentNode->state.head<3>() = start;
        currentNode->state.tail<3>() = startVel;
        currentNode->index = mp_.p2i(start, resolutionInv_);
        currentNode->gScore = 0.0;
        double initialShotTime = 0.0;
        currentNode->fScore = options_.lambdaHeu * estimateHeuristic_(currentNode->state, endState_(end, endVel), initialShotTime);
        currentNode->nodeState = KinoAstarNodeState::OPEN;

        const Vec6d endState = endState_(end, endVel);
        const Eigen::Vector3i endIndex = mp_.p2i(end, resolutionInv_);
        openSet_.push(currentNode);
        expandedNodes_[currentNode->index] = currentNode;

        bool firstSearch = init;
        const int tolerance = std::max(1, static_cast<int>(std::ceil(1.0 / options_.resolution)));

        while (!openSet_.empty()) {
            currentNode = openSet_.top();
            const bool reachHorizon = (currentNode->state.head<3>() - start).norm() >= options_.horizon;
            const bool nearEnd = std::abs(currentNode->index(0) - endIndex(0)) <= tolerance &&
                std::abs(currentNode->index(1) - endIndex(1)) <= tolerance &&
                std::abs(currentNode->index(2) - endIndex(2)) <= tolerance;
            if (reachHorizon) {
                retrievePath_(currentNode);
                double shotTime = 0.0;
                estimateHeuristic_(currentNode->state, endState, shotTime);
                computeShotTraj_(currentNode->state, endState, shotTime);
                return isShotSucc_ ? KinoAstarSearchStatus::DONE : KinoAstarSearchStatus::HORIZON;
            }
            if (nearEnd) {
                double shotTime = 0.0;
                estimateHeuristic_(currentNode->state, endState, shotTime);
                computeShotTraj_(currentNode->state, endState, shotTime);
                if (isShotSucc_) {
                    retrievePath_(currentNode);
                    return KinoAstarSearchStatus::DONE;
                }
                if (currentNode->index == endIndex) {
                    retrievePath_(currentNode);
                    return KinoAstarSearchStatus::NEAR_END;
                }
            }
            if (options_.maxSearchTime > 0.0) {
                const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - searchStart).count();
                if (elapsed > options_.maxSearchTime) {
                    return KinoAstarSearchStatus::TIMEOUT;
                }
            }
            openSet_.pop();
            currentNode->nodeState = KinoAstarNodeState::CLOSE;
            ++ iterNum_;

            const Vec6d currentState = currentNode->state;
            std::vector<Eigen::Vector3d> inputs;
            std::vector<double> durations;
            if (firstSearch) {
                inputs.push_back(startAcc_);
                const double step = options_.initMaxTau / 20.0;
                for (double tau = step; tau <= options_.initMaxTau + 1e-3; tau += step) {
                    durations.push_back(std::min(tau, options_.initMaxTau));
                }
                if (durations.empty()) {
                    durations.push_back(options_.initMaxTau);
                }
                firstSearch = false;
            } else {
                const double accelerationStep = options_.maxAcc * 0.5;
                for (double ax = -options_.maxAcc; ax <= options_.maxAcc + 1e-3; ax += accelerationStep) {
                    for (double ay = -options_.maxAcc; ay <= options_.maxAcc + 1e-3; ay += accelerationStep) {
                        for (double az = -options_.maxAcc; az <= options_.maxAcc + 1e-3; az += accelerationStep) {
                            inputs.emplace_back(ax, ay, az);
                        }
                    }
                }
                durations.push_back(options_.maxTau);
            }

            std::vector<KinoAstarNode*> tmpExpandNodes;
            for (const Eigen::Vector3d& input : inputs) {
                for (const double duration : durations) {
                    Vec6d proState;
                    stateTransit_(currentState, proState, input, duration);
                    const Eigen::Vector3d proPos = proState.head<3>();
                    if (!isInBox_(proPos) || !mp_.isInMap(proPos)) {
                        continue;
                    }
                    const Eigen::Vector3i proIndex = mp_.p2i(proPos, resolutionInv_);
                    const auto expandedIter = expandedNodes_.find(proIndex);
                    KinoAstarNode* proNode =
                        expandedIter == expandedNodes_.end() ? nullptr : expandedIter->second;
                    if (proNode != nullptr && proNode->nodeState == KinoAstarNodeState::CLOSE) {
                        continue;
                    }
                    const Eigen::Vector3d proVel = proState.tail<3>();
                    if (std::abs(proVel(0)) > maxVel_ ||
                        std::abs(proVel(1)) > maxVel_ ||
                        std::abs(proVel(2)) > maxVel_) {
                        continue;
                    }
                    if (proIndex == currentNode->index) {
                        continue;
                    }
                    if (!isTransitionSafe_(currentState, input, duration, optimistic)) {
                        continue;
                    }
                    double shotTime = 0.0;
                    const double newGScore = (input.squaredNorm() + options_.timeWeight) * duration + currentNode->gScore;
                    const double newFScore = newGScore + options_.lambdaHeu * estimateHeuristic_(proState, endState, shotTime);
                    bool prune = false;
                    for (KinoAstarNode* expandedNode : tmpExpandNodes) {
                        if ((proIndex - expandedNode->index).norm() == 0) {
                            prune = true;
                            if (newFScore < expandedNode->fScore) {
                                expandedNode->fScore = newFScore;
                                expandedNode->gScore = newGScore;
                                expandedNode->state = proState;
                                expandedNode->input = input;
                                expandedNode->duration = duration;
                            }
                            break;
                        }
                    }
                    if (prune) {
                        continue;
                    }
                    if (proNode == nullptr) {
                        proNode = releaseNode_();
                        if (proNode == nullptr) {
                            return KinoAstarSearchStatus::NOMEM;
                        }
                        proNode->index = proIndex;
                        proNode->state = proState;
                        proNode->fScore = newFScore;
                        proNode->gScore = newGScore;
                        proNode->input = input;
                        proNode->duration = duration;
                        proNode->parent = currentNode;
                        proNode->nodeState = KinoAstarNodeState::OPEN;
                        openSet_.push(proNode);
                        expandedNodes_[proIndex] = proNode;
                        tmpExpandNodes.push_back(proNode);
                    } else if (proNode->nodeState == KinoAstarNodeState::OPEN) {
                        if (newGScore < proNode->gScore) {
                            proNode->state = proState;
                            proNode->fScore = newFScore;
                            proNode->gScore = newGScore;
                            proNode->input = input;
                            proNode->duration = duration;
                            proNode->parent = currentNode;
                        }
                    }
                }
            }
        }
        return KinoAstarSearchStatus::NOPATH;
    }

private:
    KinoAstarNode* releaseNode_() {
        if (nodeNum_ >= options_.maxNodes) {
            return nullptr;
        }
        if (nodeNum_ >= nodePool_.size()) {
            nodePool_.push_back(new KinoAstarNode);
        }
        KinoAstarNode* node = nodePool_[nodeNum_ ++];
        node->parent = nullptr;
        node->nodeState = KinoAstarNodeState::INIT;
        return node;
    }
    bool isInBox_(const Eigen::Vector3d& pos) const {
        return (pos.array() > boxMin_.array()).all() && (pos.array() < boxMax_.array()).all();
    }
    bool isVoxelFree_(const Eigen::Vector3d& pos, bool optimistic) const {
        if (!isInBox_(pos) || !mp_.isInMap(pos)) {
            return false;
        }
        if (md_.getInflateOccupancy(pos) != VoxelInflateOccupancyCode::FREE) {
            return false;
        }
        if (!optimistic) {
            const uint8_t occupancy = md_.getOccupancy(pos);
            if (occupancy == VoxelOccupancyCode::UNKNOWN ||
                occupancy == VoxelOccupancyCode::OUTSIDE) {
                return false;
            }
        }
        return true;
    }
    bool isTransitionSafe_(
        const Vec6d& state0,
        const Eigen::Vector3d& input,
        double duration,
        bool optimistic
    ) const {
        const double distance = state0.tail<3>().norm() * duration + 0.5 * input.norm() * duration * duration;
        const int checks = std::max(
            1,
            static_cast<int>(std::ceil(distance / std::max(options_.collisionCheckStep, 1e-3)))
        );
        for (int k = 1; k <= checks; ++ k) {
            const double t = duration * static_cast<double>(k) / static_cast<double>(checks);
            Vec6d state;
            stateTransit_(state0, state, input, t);
            if (!isVoxelFree_(state.head<3>(), optimistic)) {
                return false;
            }
        }
        return true;
    }
    void retrievePath_(KinoAstarNode* endNode) {
        pathNodes_.clear();
        for (KinoAstarNode* node = endNode; node != nullptr; node = node->parent) {
            pathNodes_.push_back(node);
        }
        std::reverse(pathNodes_.begin(), pathNodes_.end());
    }
    Vec6d endState_(const Eigen::Vector3d& end, const Eigen::Vector3d& endVel) const {
        Vec6d state;
        state.head<3>() = end;
        state.tail<3>() = endVel;
        return state;
    }
    void stateTransit_(
        const Vec6d& state0,
        Vec6d& state1,
        const Eigen::Vector3d& input,
        double duration
    ) const {
        state1.head<3>() = state0.head<3>() + duration * state0.tail<3>() + 0.5 * duration * duration * input;
        state1.tail<3>() = state0.tail<3>() + duration * input;
    }
    Eigen::Vector3d evaluateShot_(double t) const {
        Eigen::Vector4d time;
        time << 1.0, t, t * t, t * t * t;
        return coefShot_ * time;
    }
    bool computeShotTraj_(const Vec6d& state1, const Vec6d& state2, double timeToGoal) {
        if (timeToGoal <= 1e-4) {
            return false;
        }
        const Eigen::Vector3d p0 = state1.head<3>();
        const Eigen::Vector3d dp = state2.head<3>() - p0;
        const Eigen::Vector3d v0 = state1.tail<3>();
        const Eigen::Vector3d v1 = state2.tail<3>();
        const Eigen::Vector3d dv = v1 - v0;
        const double tD = timeToGoal;

        Eigen::Matrix<double, 3, 4> coef;
        endVel_ = v1;
        const Eigen::Vector3d a = 1.0 / 6.0 * (-12.0 / (tD * tD * tD) * (dp - v0 * tD) + 6.0 / (tD * tD) * dv);
        const Eigen::Vector3d b = 0.5 * (6.0 / (tD * tD) * (dp - v0 * tD) - 2.0 / tD * dv);
        coef.col(0) = p0;
        coef.col(1) = v0;
        coef.col(2) = b;
        coef.col(3) = a;

        const int checks = std::max(
            10,
            static_cast<int>(std::ceil(dp.norm() / std::max(options_.collisionCheckStep, 1e-3)))
        );
        for (int k = 1; k <= checks; ++ k) {
            const double t = tD * static_cast<double>(k) / static_cast<double>(checks);
            if (!isVoxelFree_(evaluateShotWithCoef_(coef, t), optimistic_)) {
                return false;
            }
        }
        coefShot_ = coef;
        tShot_ = tD;
        isShotSucc_ = true;
        return true;
    }
    static Eigen::Vector3d evaluateShotWithCoef_(const Eigen::Matrix<double, 3, 4>& coef, double t) {
        Eigen::Vector4d time;
        time << 1.0, t, t * t, t * t * t;
        return coef * time;
    }
    double estimateHeuristic_(const Vec6d& x1, const Vec6d& x2, double& optimalTime) const {
        const Eigen::Vector3d dp = x2.head<3>() - x1.head<3>();
        const Eigen::Vector3d v0 = x1.tail<3>();
        const Eigen::Vector3d v1 = x2.tail<3>();

        const double c1 = -36.0 * dp.dot(dp);
        const double c2 = 24.0 * (v0 + v1).dot(dp);
        const double c3 = -4.0 * (v0.dot(v0) + v0.dot(v1) + v1.dot(v1));
        const double c4 = 0.0;
        const double c5 = std::max(options_.timeWeight, 1e-6);
        std::vector<double> times = quartic_(c5, c4, c3, c2, c1);

        const double vMax = std::max(maxVel_ * 0.5, 1e-3);
        const double tBar = (x1.head<3>() - x2.head<3>()).lpNorm<Eigen::Infinity>() / vMax;
        times.push_back(tBar);

        double cost = std::numeric_limits<double>::max();
        double tBest = std::max(tBar, 1e-4);
        for (const double t : times) {
            if (!std::isfinite(t) || t < tBar || t <= 1e-6) {
                continue;
            }
            const double c = -c1 / (3.0 * t * t * t) - c2 / (2.0 * t * t) - c3 / t + c5 * t;
            if (std::isfinite(c) && c < cost) {
                cost = c;
                tBest = t;
            }
        }
        if (!std::isfinite(cost)) {
            const double t = std::max(tBar, 1e-4);
            cost = -c1 / (3.0 * t * t * t) - c2 / (2.0 * t * t) - c3 / t + c5 * t;
            tBest = t;
        }
        optimalTime = tBest;
        return (1.0 + tieBreaker_) * cost;
    }
    static std::vector<double> cubic_(double a, double b, double c, double d) {
        std::vector<double> times;
        if (std::abs(a) < 1e-12) {
            return times;
        }
        const double a2 = b / a;
        const double a1 = c / a;
        const double a0 = d / a;
        const double q = (3.0 * a1 - a2 * a2) / 9.0;
        const double r = (9.0 * a1 * a2 - 27.0 * a0 - 2.0 * a2 * a2 * a2) / 54.0;
        const double disc = q * q * q + r * r;
        if (disc > 0.0) {
            const double s = std::cbrt(r + std::sqrt(disc));
            const double t = std::cbrt(r - std::sqrt(disc));
            times.push_back(-a2 / 3.0 + s + t);
        } else if (disc == 0.0) {
            const double s = std::cbrt(r);
            times.push_back(-a2 / 3.0 + s + s);
            times.push_back(-a2 / 3.0 - s);
        } else {
            const double theta = std::acos(r / std::sqrt(-q * q * q));
            times.push_back(2.0 * std::sqrt(-q) * std::cos(theta / 3.0) - a2 / 3.0);
            times.push_back(2.0 * std::sqrt(-q) * std::cos((theta + 2.0 * __pi__) / 3.0) - a2 / 3.0);
            times.push_back(2.0 * std::sqrt(-q) * std::cos((theta + 4.0 * __pi__) / 3.0) - a2 / 3.0);
        }
        return times;
    }
    static std::vector<double> quartic_(double a, double b, double c, double d, double e) {
        std::vector<double> times;
        if (std::abs(a) < 1e-12) {
            return times;
        }
        const double a3 = b / a;
        const double a2 = c / a;
        const double a1 = d / a;
        const double a0 = e / a;
        const std::vector<double> ys = cubic_(1.0, -a2, a1 * a3 - 4.0 * a0, 4.0 * a2 * a0 - a1 * a1 - a3 * a3 * a0);
        if (ys.empty()) {
            return times;
        }
        const double y1 = ys.front();
        const double r = a3 * a3 / 4.0 - a2 + y1;
        if (r < 0.0) {
            return times;
        }
        const double root = std::sqrt(r);
        double dValue = 0.0;
        double eValue = 0.0;
        if (root != 0.0) {
            const double term = 0.25 * (4.0 * a3 * a2 - 8.0 * a1 - a3 * a3 * a3) / root;
            const double base = 0.75 * a3 * a3 - root * root - 2.0 * a2;
            if (base + term < 0.0 || base - term < 0.0) {
                return times;
            }
            dValue = std::sqrt(base + term);
            eValue = std::sqrt(base - term);
        } else {
            const double rootTerm = y1 * y1 - 4.0 * a0;
            if (rootTerm < 0.0) {
                return times;
            }
            const double base = 0.75 * a3 * a3 - 2.0 * a2;
            if (base + 2.0 * std::sqrt(rootTerm) < 0.0 ||
                base - 2.0 * std::sqrt(rootTerm) < 0.0) {
                return times;
            }
            dValue = std::sqrt(base + 2.0 * std::sqrt(rootTerm));
            eValue = std::sqrt(base - 2.0 * std::sqrt(rootTerm));
        }
        if (std::isfinite(dValue)) {
            times.push_back(-a3 / 4.0 + root / 2.0 + dValue / 2.0);
            times.push_back(-a3 / 4.0 + root / 2.0 - dValue / 2.0);
        }
        if (std::isfinite(eValue)) {
            times.push_back(-a3 / 4.0 - root / 2.0 + eValue / 2.0);
            times.push_back(-a3 / 4.0 - root / 2.0 - eValue / 2.0);
        }
        return times;
    }

private:
    SdfMap& sdfmap_;
    const MapParams& mp_;
    const MapData& md_;
    KinoAstarOptions options_;
    Eigen::Vector3d boxMin_;
    Eigen::Vector3d boxMax_;
    double resolutionInv_{10.0};
    double maxVel_{1.5};
    double tieBreaker_{1.0 + 1e-4};
    bool optimistic_{true};
    size_t nodeNum_{0};
    int iterNum_{0};

private:
    std::vector<KinoAstarNode*> nodePool_;
    std::priority_queue<KinoAstarNode*, std::vector<KinoAstarNode*>, KinoAstarNodeCompare> openSet_;
    std::unordered_map<Eigen::Vector3i, KinoAstarNode*, misc::Vector3iHash> expandedNodes_;
    std::vector<KinoAstarNode*> pathNodes_;
    Eigen::Vector3d startVel_{Eigen::Vector3d::Zero()};
    Eigen::Vector3d startAcc_{Eigen::Vector3d::Zero()};
    Eigen::Vector3d endVel_{Eigen::Vector3d::Zero()};
    bool isShotSucc_{false};
    Eigen::Matrix<double, 3, 4> coefShot_{Eigen::Matrix<double, 3, 4>::Zero()};
    double tShot_{0.0};
};

} // arc
