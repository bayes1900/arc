#pragma once

#include <base/base.h>

#include <variant>

namespace arc {

class Timer {
public:
    using Clock = std::chrono::high_resolution_clock;

    Timer()
        : start_(Clock::now()) {}
    double elapsed() const { return std::chrono::duration<double>(Clock::now() - start_).count(); }
    double reset() {
        const auto now = Clock::now();
        const double elapsed = std::chrono::duration<double>(now - start_).count();
        start_ = now;
        return elapsed;
    }

private:
    std::chrono::time_point<Clock> start_;
};
inline double variance(const Eigen::VectorXd& vec) {
    const Eigen::VectorXd centered = vec.array() - vec.mean();
    return centered.squaredNorm() / static_cast<double>(vec.size());
}

namespace {

template <class... Ts>
struct Overload : Ts... {
    using Ts::operator()...;
};

template <class... Ts>
Overload(Ts...) -> Overload<Ts...>;

}

class ExecutionStats {
public:
    struct Meta {
        Meta& setFormat(const std::string& s) {
            formatString = s;
            return *this;
        }
        void setMeta(const Meta& other) { formatString = other.formatString; }

        std::variant<std::vector<double>, std::vector<Eigen::VectorXd>> data;
        std::string formatString;
    };
    Meta& add(const std::string& name, double value) {
        auto [it, newItem] = stats_.try_emplace(name);
        if (newItem) {
            order_.push_back(name);
            it->second.data = std::vector<double>();
        }
        std::get<std::vector<double>>(it->second.data).push_back(value);
        return it->second;
    }
    Meta& add(const std::string& name, const Eigen::VectorXd& value) {
        auto [it, newItem] = stats_.try_emplace(name);
        if (newItem) {
            order_.push_back(name);
            it->second.data = std::vector<Eigen::VectorXd>();
        }
        std::get<std::vector<Eigen::VectorXd>>(it->second.data).push_back(value);
        return it->second;
    }
    Meta& add(const std::string& name, const Eigen::VectorXf& value) {
        Eigen::VectorXd x = value.cast<double>();
        return add(name, x);
    }
    void mergeAll(const ExecutionStats& other) {
        for (const auto& name : other.order_) {
            const auto& meta = other.stats_.at(name);
            std::visit(
                [&](auto& data) {
                    for (auto v : data) {
                        add(name, v);
                    }
                },
                meta.data);
            stats_.at(name).setMeta(meta);
        }
    }
    void mergeSums(const ExecutionStats& other) {
        for (const auto& name : other.order_) {
            const auto& meta = other.stats_.at(name);
            std::visit(Overload{[&](const std::vector<double>& data) {
                                    Eigen::Map<const Eigen::VectorXd> map(data.data(), data.size());
                                    add(name, map.sum());
                                },
                           [&](const std::vector<Eigen::VectorXd>& data) { (void)data; }},
                meta.data);
            stats_.at(name).setMeta(meta);
        }
    }
    void print() const {
        for (const auto& name : order_) {
            const auto& meta = stats_.at(name);
            std::visit(Overload{[&](const std::vector<double>& data) {
                                    Eigen::Map<const Eigen::VectorXd> map(data.data(), data.size());
                                    Eigen::VectorXd vec = map;
                                    if (meta.formatString == "ms") {
                                        vec *= 1000;
                                    }
                                    int count = vec.size();
                                    double mean = vec.mean();
                                    double stddev = std::sqrt(variance(vec));
                                    double max = vec.maxCoeff();
                                    double min = vec.minCoeff();
                                    if (meta.formatString == "count") {
                                        std::cout << name << " (" << count << "): " << mean << "+-" << stddev << " ["
                                                  << min << ", " << max << "]\n";
                                    } else if (meta.formatString != "none") {
                                        std::cout << name << " (" << count << "): " << mean << "+-" << stddev << " ["
                                                  << min << ", " << max << "]\n";
                                    }
                                },
                           [&](const std::vector<Eigen::VectorXd>& data) {
                               int count = data.size();
                               std::cout << name << " (" << count << ")\n";
                           }},
                meta.data);
        }
    }

private:
    std::unordered_map<std::string, Meta> stats_;
    std::vector<std::string> order_;
};

} // arc
