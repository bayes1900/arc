#pragma once

#include <base/base.h>
#include <lio/liotypes.h>
#include <yaml-cpp/yaml.h>

namespace arc {

struct LioConfig {
    LidarType lidarType{LidarType::Generic};
    int scanLine{16};
    int scanRate{10};
    int pointFilterNum{1};
    int timestampUnit{2};
    double blind{0.01};
    double blindSq{0.01 * 0.01};

    int maxIterations{4};
    double filterSizeSurf{0.5};
    double filterSizeMap{0.5};
    double cubeSideLength{200.0};
    double detRange{300.0};

    double accCov{0.1};
    double gyrCov{0.1};
    double biasAccCov{0.0001};
    double biasGyrCov{0.0001};
    bool extrinsicEstimate{false};
    Vec3d extrinsicT{Vec3d::Zero()};
    Mat3d extrinsicR{Mat3d::Identity()};

    double timeOffsetLidarToImu{0.0};

    static LioConfig loadConfig(const std::string& path) {
        return configFromYaml_(YAML::LoadFile(path));
    }

private:
    static double yamlOrAny_(const YAML::Node& node, const char* key, double defaultValue) {
        if (!node || !node[key]) {
            return defaultValue;
        }
        try {
            return node[key].as<double>();
        } catch (const YAML::Exception&) {
            return defaultValue;
        }
    }
    static int yamlOrAny_(const YAML::Node& node, const char* key, int defaultValue) {
        if (!node || !node[key]) {
            return defaultValue;
        }
        try {
            return node[key].as<int>();
        } catch (const YAML::Exception&) {
            return defaultValue;
        }
    }
    static bool yamlOrAny_(const YAML::Node& node, const char* key, bool defaultValue) {
        if (!node || !node[key]) {
            return defaultValue;
        }
        try {
            return node[key].as<bool>();
        } catch (const YAML::Exception&) {
            return defaultValue;
        }
    }
    static LidarType lidarTypeFrom_(const YAML::Node& node, LidarType defaultValue) {
        if (!node) {
            return defaultValue;
        }
        try {
            const int value = node.as<int>();
            switch (value) {
                case 1: return LidarType::Avia;
                case 2: return LidarType::Generic;
                case 3: return LidarType::Ouster;
                case 4: return LidarType::Sim;
                default: return defaultValue;
            }
        } catch (const YAML::Exception&) {
        }
        const std::string value = node.as<std::string>();
        if (value == "avia" || value == "livox") {
            return LidarType::Avia;
        }
        if (value == "ouster" || value == "ouster64") {
            return LidarType::Ouster;
        }
        if (value == "sim" || value == "marsim") {
            return LidarType::Sim;
        }
        return LidarType::Generic;
    }
    static Vec3d vec3From_(const YAML::Node& node, const Vec3d& defaultValue) {
        if (!node) {
            return defaultValue;
        }
        try {
            const std::vector<double> values = node.as<std::vector<double>>();
            if (values.size() >= 3) {
                return Vec3d(values[0], values[1], values[2]);
            }
        } catch (const YAML::Exception&) {
        }
        return defaultValue;
    }
    static Mat3d mat3From_(const YAML::Node& node, const Mat3d& defaultValue) {
        if (!node) {
            return defaultValue;
        }
        try {
            const std::vector<double> values = node.as<std::vector<double>>();
            if (values.size() >= 9) {
                Mat3d matrix;
                matrix << values[0], values[1], values[2],
                    values[3], values[4], values[5],
                    values[6], values[7], values[8];
                return matrix;
            }
        } catch (const YAML::Exception&) {
        }
        return defaultValue;
    }
    static LioConfig configFromYaml_(const YAML::Node& root) {
        const YAML::Node node = root["lio"] ? root["lio"] : root;
        const YAML::Node mapping = node["mapping"] ? node["mapping"] : node;

        LioConfig config;

        config.lidarType = lidarTypeFrom_(node["lidarType"], config.lidarType);
        config.scanLine = yamlOrAny_(node, "scanLine", config.scanLine);
        config.scanRate = yamlOrAny_(node, "scanRate", config.scanRate);
        config.pointFilterNum = yamlOrAny_(node, "pointFilterNum", config.pointFilterNum);
        config.timestampUnit = yamlOrAny_(node, "timestampUnit", config.timestampUnit);
        config.blind = yamlOrAny_(node, "blind", config.blind);
        config.blindSq = config.blind * config.blind;

        config.maxIterations = yamlOrAny_(node, "maxIterations", config.maxIterations);
        config.filterSizeSurf = yamlOrAny_(node, "filterSizeSurf", config.filterSizeSurf);
        config.filterSizeMap = yamlOrAny_(node, "filterSizeMap", config.filterSizeMap);
        config.cubeSideLength = yamlOrAny_(node, "cubeSideLength", config.cubeSideLength);
        config.detRange = yamlOrAny_(mapping, "detRange", config.detRange);

        config.accCov = yamlOrAny_(mapping, "accCov", config.accCov);
        config.gyrCov = yamlOrAny_(mapping, "gyrCov", config.gyrCov);
        config.biasAccCov = yamlOrAny_(mapping, "biasAccCov", config.biasAccCov);
        config.biasGyrCov = yamlOrAny_(mapping, "biasGyrCov", config.biasGyrCov);
        config.extrinsicEstimate = yamlOrAny_(mapping, "extrinsicEstimate", config.extrinsicEstimate);
        config.extrinsicT = vec3From_(mapping["extrinsicT"], config.extrinsicT);
        config.extrinsicR = mat3From_(mapping["extrinsicR"], config.extrinsicR);

        config.timeOffsetLidarToImu = yamlOrAny_(node, "timeOffsetLidarToImu", config.timeOffsetLidarToImu);

        return config;
    }
};

} // arc
