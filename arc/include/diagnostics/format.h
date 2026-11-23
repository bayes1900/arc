#pragma once

#include <base/iarc.h>

#include <iomanip>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace arc {

class DiagFormat {
public:
    static std::string number(double value, int precision = 3) {
        std::ostringstream stream;
        stream << std::fixed << std::setprecision(precision) << value;
        return stream.str();
    }
    static std::string vectorParen(const Eigen::Vector3d& value, int precision = 2) {
        std::ostringstream stream;
        stream << std::fixed << std::setprecision(precision)
            << "(" << value(0) << "," << value(1) << "," << value(2) << ")";
        return stream.str();
    }
};

class DiagDetail {
public:
    DiagDetail& add(const std::string& key, const std::string& value) {
        items_.emplace_back(key, value);
        return *this;
    }
    DiagDetail& add(const std::string& key, const char* value) {
        items_.emplace_back(key, value ? value : "");
        return *this;
    }
    DiagDetail& add(const std::string& key, double value) {
        items_.emplace_back(key, DiagFormat::number(value));
        return *this;
    }
    DiagDetail& add(const std::string& key, int value) {
        items_.emplace_back(key, std::to_string(value));
        return *this;
    }
    DiagDetail& add(const std::string& key, size_t value) {
        items_.emplace_back(key, std::to_string(value));
        return *this;
    }
    std::string str() const {
        std::ostringstream stream;
        for (size_t i = 0; i < items_.size(); ++ i) {
            if (i > 0) {
                stream << " ";
            }
            stream << items_[i].first << "=" << items_[i].second;
        }
        return stream.str();
    }

private:
    std::vector<std::pair<std::string, std::string>> items_;
};

} // arc
