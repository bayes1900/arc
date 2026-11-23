#pragma once

#include <base/base.h>
#include <lio/liotypes.h>

#include <yaml-cpp/yaml.h>

namespace arc {

struct LioConfig {
    static LioConfig loadConfig(const std::string& path) {
        return configFromYaml_(YAML::LoadFile(path));
    }

    LidarType lidarType{LidarType::Generic};
    int scanLine{16};
    int scanRate{10};
    int pointFilterNum{1};
    int timestampUnit{2};
    double blind{0.01};
    bool featureExtractEnable{false};

    int maxIterations{4};
    double filterSizeSurf{0.5};
    double filterSizeMap{0.5};
    double cubeSideLength{200.0};
    double detRange{300.0};
    double fovDegree{180.0};

    double accCov{0.1};
    double gyrCov{0.1};
    double biasAccCov{0.0001};
    double biasGyrCov{0.0001};
    bool extrinsicEstimate{false};
    Vec3d extrinsicT{Vec3d::Zero()};
    Mat3d extrinsicR{Mat3d::Identity()};

    bool timeSyncEnable{false};
    double timeOffsetLidarToImu{0.0};

    bool publishPath{false};
    bool publishScan{true};
    bool publishDense{true};
    bool publishScanBody{false};
    int publishFramePeriod{20};

private:
    static bool has_(const YAML::Node& node, const char* key) {
        return node && node[key];
    }

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
        const YAML::Node preprocess = node["preprocess"] ? node["preprocess"] : YAML::Node();
        const YAML::Node mapping = node["mapping"] ? node["mapping"] : node;

        LioConfig config;

        const YAML::Node lidarTypeNode = has_(node, "lidarType") ? node["lidarType"] : preprocess["lidar_type"];
        config.lidarType = lidarTypeFrom_(lidarTypeNode, config.lidarType);
        config.scanLine = yamlOrAny_(node, "scanLine",
            yamlOrAny_(preprocess, "scan_line", config.scanLine));
        config.scanRate = yamlOrAny_(node, "scanRate",
            yamlOrAny_(preprocess, "scan_rate", config.scanRate));
        config.pointFilterNum = yamlOrAny_(node, "pointFilterNum",
            yamlOrAny_(preprocess, "point_filter_num", config.pointFilterNum));
        config.timestampUnit = yamlOrAny_(node, "timestampUnit",
            yamlOrAny_(preprocess, "timestamp_unit", config.timestampUnit));
        config.blind = yamlOrAny_(node, "blind",
            yamlOrAny_(preprocess, "blind", config.blind));
        config.featureExtractEnable = yamlOrAny_(node, "featureExtractEnable",
            yamlOrAny_(node, "feature_extract_enable",
                yamlOrAny_(preprocess, "feature_extract_enable", config.featureExtractEnable)));

        config.maxIterations = yamlOrAny_(node, "maxIterations",
            yamlOrAny_(node, "max_iteration", config.maxIterations));
        config.filterSizeSurf = yamlOrAny_(node, "filterSizeSurf",
            yamlOrAny_(node, "filter_size_surf", config.filterSizeSurf));
        config.filterSizeMap = yamlOrAny_(node, "filterSizeMap",
            yamlOrAny_(node, "filter_size_map", config.filterSizeMap));
        config.cubeSideLength = yamlOrAny_(node, "cubeSideLength",
            yamlOrAny_(node, "cube_side_length", config.cubeSideLength));
        config.detRange = yamlOrAny_(mapping, "detRange",
            yamlOrAny_(mapping, "det_range", config.detRange));
        config.fovDegree = yamlOrAny_(mapping, "fovDegree",
            yamlOrAny_(mapping, "fov_degree", config.fovDegree));

        config.accCov = yamlOrAny_(mapping, "accCov",
            yamlOrAny_(mapping, "acc_cov", config.accCov));
        config.gyrCov = yamlOrAny_(mapping, "gyrCov",
            yamlOrAny_(mapping, "gyr_cov", config.gyrCov));
        config.biasAccCov = yamlOrAny_(mapping, "biasAccCov",
            yamlOrAny_(mapping, "b_acc_cov", config.biasAccCov));
        config.biasGyrCov = yamlOrAny_(mapping, "biasGyrCov",
            yamlOrAny_(mapping, "b_gyr_cov", config.biasGyrCov));
        config.extrinsicEstimate = yamlOrAny_(mapping, "extrinsicEstimate",
            yamlOrAny_(mapping, "extrinsic_est_en", config.extrinsicEstimate));
        config.extrinsicT = vec3From_(
            has_(mapping, "extrinsicT") ? mapping["extrinsicT"] : mapping["extrinsic_T"],
            config.extrinsicT);
        config.extrinsicR = mat3From_(
            has_(mapping, "extrinsicR") ? mapping["extrinsicR"] : mapping["extrinsic_R"],
            config.extrinsicR);

        config.timeSyncEnable = yamlOrAny_(node, "timeSyncEnable",
            yamlOrAny_(node, "time_sync_en", config.timeSyncEnable));
        config.timeOffsetLidarToImu = yamlOrAny_(node, "timeOffsetLidarToImu",
            yamlOrAny_(node, "time_offset_lidar_to_imu", config.timeOffsetLidarToImu));

        config.publishPath = yamlOrAny_(node, "publishPath",
            yamlOrAny_(node, "path_en", config.publishPath));
        config.publishScan = yamlOrAny_(node, "publishScan",
            yamlOrAny_(node, "scan_publish_en", config.publishScan));
        config.publishDense = yamlOrAny_(node, "publishDense",
            yamlOrAny_(node, "dense_publish_en", config.publishDense));
        config.publishScanBody = yamlOrAny_(node, "publishScanBody",
            yamlOrAny_(node, "scan_bodyframe_pub_en", config.publishScanBody));
        config.publishFramePeriod = yamlOrAny_(node, "publishFramePeriod",
            yamlOrAny_(node, "publish_frame_period", config.publishFramePeriod));

        ensure(!config.featureExtractEnable,
            "feature_extract_enable is not ported; use the raw-point FAST-LIO2 path");
        return config;
    }
};

} // arc
