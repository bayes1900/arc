#pragma once

#include <string>
#include <rclcpp/rclcpp.hpp>

namespace arc {
namespace misc {

template<typename T> T getParams(rclcpp::Node::SharedPtr node, const std::string& name, const T& defaultValue) {
    if (node->has_parameter(name)) {
        return node->get_parameter(name).get_value<T>();
    }
    return node->declare_parameter<T>(name, defaultValue);
}

template<typename T> T getParams(rclcpp::Node::SharedPtr node, const std::string& name) {
    if (node->has_parameter(name)) {
        return node->get_parameter(name).get_value<T>();
    }
    return node->declare_parameter<T>(name);
}

} // misc
} // arc
