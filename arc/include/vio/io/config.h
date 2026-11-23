#pragma once

#include <base/base.h>

#include <nlohmann/json.hpp>
#include <yaml-cpp/yaml.h>

#include <vio/vio.h>

namespace arc {

class VioIo {
public:
    static VioConfig loadConfig(const std::string& path) {
        if (hasSuffix_(path, ".json")) {
            return configFromJson_(loadJson_(path));
        }
        return configFromYaml_(YAML::LoadFile(path));
    }
    static Calibration<double> loadCalibration(const std::string& path) {
        if (hasSuffix_(path, ".json")) {
            return calibrationFromJson_(loadJson_(path));
        }
        return calibrationFromYaml_(YAML::LoadFile(path));
    }

private:
    static bool hasSuffix_(const std::string& path, const std::string& suffix) {
        return path.size() >= suffix.size() && path.compare(path.size() - suffix.size(), suffix.size(), suffix) == 0;
    }
    static nlohmann::json loadJson_(const std::string& path) {
        std::ifstream stream(path);
        if (!stream.is_open()) {
            throw std::runtime_error("cannot open VIO json: " + path);
        }
        nlohmann::json root;
        stream >> root;
        return root;
    }
    static const nlohmann::json& unwrap_(const nlohmann::json& root) {
        if (root.contains("value0")) {
            return root.at("value0");
        }
        return root;
    }
    template <typename T>
    static T jsonOr_(const nlohmann::json& node, const char* key, const T& value) {
        return node.contains(key) ? node.at(key).get<T>() : value;
    }
    static VioConfig configFromJson_(const nlohmann::json& root) {
        const nlohmann::json& node = unwrap_(root);
        VioConfig config;
        config.opticalFlowType = jsonOr_(node, "config.optical_flow_type", jsonOr_(node, "opticalFlowType", config.opticalFlowType));
        config.opticalFlowDetectionGridSize = jsonOr_(node,
            "config.optical_flow_detection_grid_size",
            jsonOr_(node, "opticalFlowDetectionGridSize", config.opticalFlowDetectionGridSize));
        config.opticalFlowMaxRecoveredDist2 = jsonOr_(node,
            "config.optical_flow_max_recovered_dist2",
            jsonOr_(node, "opticalFlowMaxRecoveredDist2", config.opticalFlowMaxRecoveredDist2));
        config.opticalFlowPattern = jsonOr_(node, "config.optical_flow_pattern", jsonOr_(node, "opticalFlowPattern", config.opticalFlowPattern));
        config.opticalFlowMaxIterations = jsonOr_(node,
            "config.optical_flow_max_iterations",
            jsonOr_(node, "opticalFlowMaxIterations", config.opticalFlowMaxIterations));
        config.opticalFlowLevels = jsonOr_(node, "config.optical_flow_levels", jsonOr_(node, "opticalFlowLevels", config.opticalFlowLevels));
        config.opticalFlowEpipolarError = jsonOr_(node,
            "config.optical_flow_epipolar_error",
            jsonOr_(node, "opticalFlowEpipolarError", config.opticalFlowEpipolarError));
        config.opticalFlowSkipFrames = jsonOr_(node,
            "config.optical_flow_skip_frames",
            jsonOr_(node, "opticalFlowSkipFrames", config.opticalFlowSkipFrames));
        const std::string linearization = jsonOr_(node, "config.vio_linearization_type", jsonOr_(node, "vioLinearizationType", std::string("ABS_QR")));
        if (linearization == "ABS_QR") {
            config.vioLinearizationType = LinearizationType::ABS_QR;
        } else if (linearization == "ABS_SC") {
            config.vioLinearizationType = LinearizationType::ABS_SC;
        } else if (linearization == "REL_SC") {
            config.vioLinearizationType = LinearizationType::REL_SC;
        } else {
            throw std::runtime_error("unknown linearization type: " + linearization);
        }
        config.vioSqrtMarg = jsonOr_(node, "config.vio_sqrt_marg", jsonOr_(node, "vioSqrtMarg", config.vioSqrtMarg));
        config.vioMaxStates = jsonOr_(node, "config.vio_max_states", jsonOr_(node, "vioMaxStates", config.vioMaxStates));
        config.vioMaxKfs = jsonOr_(node, "config.vio_max_kfs", jsonOr_(node, "vioMaxKfs", config.vioMaxKfs));
        config.vioMinFramesAfterKf = jsonOr_(
            node, "config.vio_min_frames_after_kf", jsonOr_(node, "vioMinFramesAfterKf", config.vioMinFramesAfterKf));
        config.vioNewKfKeypointsThresh = jsonOr_(node,
            "config.vio_new_kf_keypoints_thresh",
            jsonOr_(node, "vioNewKfKeypointsThresh", config.vioNewKfKeypointsThresh));
        config.vioDebug = jsonOr_(node, "config.vio_debug", jsonOr_(node, "vioDebug", config.vioDebug));
        config.vioExtendedLogging = jsonOr_(node, "config.vio_extended_logging", jsonOr_(node, "vioExtendedLogging", config.vioExtendedLogging));
        config.vioMaxIterations = jsonOr_(node, "config.vio_max_iterations", jsonOr_(node, "vioMaxIterations", config.vioMaxIterations));
        config.vioObsStdDev = jsonOr_(node, "config.vio_obs_std_dev", jsonOr_(node, "vioObsStdDev", config.vioObsStdDev));
        config.vioObsHuberThresh = jsonOr_(node, "config.vio_obs_huber_thresh", jsonOr_(node, "vioObsHuberThresh", config.vioObsHuberThresh));
        config.vioMinTriangulationDist = jsonOr_(node,
            "config.vio_min_triangulation_dist",
            jsonOr_(node, "vioMinTriangulationDist", config.vioMinTriangulationDist));
        config.vioEnforceRealtime = jsonOr_(node, "config.vio_enforce_realtime", jsonOr_(node, "vioEnforceRealtime", config.vioEnforceRealtime));
        config.vioLmLambdaInitial = jsonOr_(node, "config.vio_lm_lambda_initial", jsonOr_(node, "vioLmLambdaInitial", config.vioLmLambdaInitial));
        config.vioLmLambdaMin = jsonOr_(node, "config.vio_lm_lambda_min", jsonOr_(node, "vioLmLambdaMin", config.vioLmLambdaMin));
        config.vioLmLambdaMax = jsonOr_(node, "config.vio_lm_lambda_max", jsonOr_(node, "vioLmLambdaMax", config.vioLmLambdaMax));
        config.vioInitPoseWeight = jsonOr_(node, "config.vio_init_pose_weight", jsonOr_(node, "vioInitPoseWeight", config.vioInitPoseWeight));
        config.vioInitBaWeight = jsonOr_(node, "config.vio_init_ba_weight", jsonOr_(node, "vioInitBaWeight", config.vioInitBaWeight));
        config.vioInitBgWeight = jsonOr_(node, "config.vio_init_bg_weight", jsonOr_(node, "vioInitBgWeight", config.vioInitBgWeight));
        config.vioMargLostLandmarks = jsonOr_(
            node, "config.vio_marg_lost_landmarks", jsonOr_(node, "vioMargLostLandmarks", config.vioMargLostLandmarks));
        config.vioKfMargFeatureRatio = jsonOr_(node,
            "config.vio_kf_marg_feature_ratio",
            jsonOr_(node, "vioKfMargFeatureRatio", config.vioKfMargFeatureRatio));
        return config;
    }
    static VioConfig configFromYaml_(const YAML::Node& root) {
        const YAML::Node node = root["vio"] ? root["vio"] : root;
        VioConfig config;
        config.opticalFlowType = yamlOr_<std::string>(node, "opticalFlowType", config.opticalFlowType);
        config.opticalFlowDetectionGridSize = yamlOr_<int>(node, "opticalFlowDetectionGridSize", config.opticalFlowDetectionGridSize);
        config.opticalFlowMaxRecoveredDist2 = yamlOr_<float>(node, "opticalFlowMaxRecoveredDist2", config.opticalFlowMaxRecoveredDist2);
        config.opticalFlowPattern = yamlOr_<int>(node, "opticalFlowPattern", config.opticalFlowPattern);
        config.opticalFlowMaxIterations = yamlOr_<int>(node, "opticalFlowMaxIterations", config.opticalFlowMaxIterations);
        config.opticalFlowLevels = yamlOr_<int>(node, "opticalFlowLevels", config.opticalFlowLevels);
        config.opticalFlowEpipolarError = yamlOr_<float>(node, "opticalFlowEpipolarError", config.opticalFlowEpipolarError);
        config.opticalFlowSkipFrames = yamlOr_<int>(node, "opticalFlowSkipFrames", config.opticalFlowSkipFrames);
        const std::string linearization = yamlOr_<std::string>(node, "vioLinearizationType", "ABS_QR");
        if (linearization == "ABS_QR") {
            config.vioLinearizationType = LinearizationType::ABS_QR;
        } else if (linearization == "ABS_SC") {
            config.vioLinearizationType = LinearizationType::ABS_SC;
        } else if (linearization == "REL_SC") {
            config.vioLinearizationType = LinearizationType::REL_SC;
        } else {
            throw std::runtime_error("unknown linearization type: " + linearization);
        }
        config.vioSqrtMarg = yamlOr_<bool>(node, "vioSqrtMarg", config.vioSqrtMarg);
        config.vioMaxStates = yamlOr_<int>(node, "vioMaxStates", config.vioMaxStates);
        config.vioMaxKfs = yamlOr_<int>(node, "vioMaxKfs", config.vioMaxKfs);
        config.vioMinFramesAfterKf = yamlOr_<int>(node, "vioMinFramesAfterKf", config.vioMinFramesAfterKf);
        config.vioNewKfKeypointsThresh = yamlOr_<float>(node, "vioNewKfKeypointsThresh", config.vioNewKfKeypointsThresh);
        config.vioDebug = yamlOr_<bool>(node, "vioDebug", config.vioDebug);
        config.vioExtendedLogging = yamlOr_<bool>(node, "vioExtendedLogging", config.vioExtendedLogging);
        config.vioMaxIterations = yamlOr_<int>(node, "vioMaxIterations", config.vioMaxIterations);
        config.vioObsStdDev = yamlOr_<double>(node, "vioObsStdDev", config.vioObsStdDev);
        config.vioObsHuberThresh = yamlOr_<double>(node, "vioObsHuberThresh", config.vioObsHuberThresh);
        config.vioMinTriangulationDist = yamlOr_<double>(node, "vioMinTriangulationDist", config.vioMinTriangulationDist);
        config.vioEnforceRealtime = yamlOr_<bool>(node, "vioEnforceRealtime", config.vioEnforceRealtime);
        config.vioLmLambdaInitial = yamlOr_<double>(node, "vioLmLambdaInitial", config.vioLmLambdaInitial);
        config.vioLmLambdaMin = yamlOr_<double>(node, "vioLmLambdaMin", config.vioLmLambdaMin);
        config.vioLmLambdaMax = yamlOr_<double>(node, "vioLmLambdaMax", config.vioLmLambdaMax);
        config.vioInitPoseWeight = yamlOr_<double>(node, "vioInitPoseWeight", config.vioInitPoseWeight);
        config.vioInitBaWeight = yamlOr_<double>(node, "vioInitBaWeight", config.vioInitBaWeight);
        config.vioInitBgWeight = yamlOr_<double>(node, "vioInitBgWeight", config.vioInitBgWeight);
        config.vioMargLostLandmarks = yamlOr_<bool>(node, "vioMargLostLandmarks", config.vioMargLostLandmarks);
        config.vioKfMargFeatureRatio = yamlOr_<double>(node, "vioKfMargFeatureRatio", config.vioKfMargFeatureRatio);
        return config;
    }

    template <typename T>
    static T yamlOr_(const YAML::Node& node, const char* key, const T& value) {
        return node[key] ? node[key].as<T>() : value;
    }
    static Calibration<double> calibrationFromJson_(const nlohmann::json& root) {
        const nlohmann::json& node = unwrap_(root);
        Calibration<double> calibration;
        for (const auto& value : node.at("T_imu_cam")) {
            const Eigen::Vector3d translation(
                value.at("px").get<double>(), value.at("py").get<double>(), value.at("pz").get<double>());
            const Eigen::Quaterniond rotation(value.at("qw").get<double>(),
                value.at("qx").get<double>(),
                value.at("qy").get<double>(),
                value.at("qz").get<double>());
            calibration.tIC.emplace_back(rotation, translation);
        }
        for (const auto& value : node.at("intrinsics")) {
            const std::string type = value.at("camera_type").get<std::string>();
            const nlohmann::json& params = value.at("intrinsics");
            calibration.intrinsics.push_back(makeCamera_(type, paramsOrder_(type, params)));
        }
        for (const auto& value : node.at("resolution")) {
            calibration.resolution.emplace_back(value.at(0).get<int>(), value.at(1).get<int>());
        }
        calibration.calibAccelBias.getParam() = vectorFromJson_<9>(node.at("calib_accel_bias"));
        calibration.calibGyroBias.getParam() = vectorFromJson_<12>(node.at("calib_gyro_bias"));
        calibration.imuUpdateRate = jsonOr_<double>(node, "imu_update_rate", calibration.imuUpdateRate);
        calibration.accelNoiseStd = vectorFromJson_<3>(node.at("accel_noise_std"));
        calibration.gyroNoiseStd = vectorFromJson_<3>(node.at("gyro_noise_std"));
        calibration.accelBiasStd = vectorFromJson_<3>(node.at("accel_bias_std"));
        calibration.gyroBiasStd = vectorFromJson_<3>(node.at("gyro_bias_std"));
        calibration.camTimeOffsetNs = jsonOr_<int64_t>(node, "cam_time_offset_ns", calibration.camTimeOffsetNs);
        return calibration;
    }
    static Calibration<double> calibrationFromYaml_(const YAML::Node& root) {
        Calibration<double> calibration;
        for (const auto& camera : root["cameras"]) {
            const std::string type = camera["type"].as<std::string>();
            std::vector<double> values;
            for (const auto& value : camera["intrinsics"]) {
                values.push_back(value.as<double>());
            }
            calibration.intrinsics.push_back(makeCamera_(type, values));
            const auto resolution = camera["resolution"];
            calibration.resolution.emplace_back(resolution[0].as<int>(), resolution[1].as<int>());
            const auto matrix = camera["camToImu"];
            Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
            for (size_t row = 0; row < 4; ++ row) {
                for (size_t col = 0; col < 4; ++ col) {
                    transform(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col)) = matrix[row][col].as<double>();
                }
            }
            calibration.tIC.emplace_back(Sophus::SE3d(transform.block<3, 3>(0, 0), transform.block<3, 1>(0, 3)));
        }
        const YAML::Node imu = root["imu"];
        if (imu) {
            calibration.imuUpdateRate = yamlOr_<double>(imu, "rate", calibration.imuUpdateRate);
            calibration.camTimeOffsetNs = yamlOr_<int64_t>(imu, "timeOffsetNs", calibration.camTimeOffsetNs);
            calibration.accelNoiseStd = vectorFromYaml_<3>(imu["accelNoiseStd"], calibration.accelNoiseStd);
            calibration.gyroNoiseStd = vectorFromYaml_<3>(imu["gyroNoiseStd"], calibration.gyroNoiseStd);
            calibration.accelBiasStd = vectorFromYaml_<3>(imu["accelBiasStd"], calibration.accelBiasStd);
            calibration.gyroBiasStd = vectorFromYaml_<3>(imu["gyroBiasStd"], calibration.gyroBiasStd);
            if (imu["accelBias"]) {
                calibration.calibAccelBias.getParam() = vectorFromYaml_<9>(imu["accelBias"], calibration.calibAccelBias.getParam());
            }
            if (imu["gyroBias"]) {
                calibration.calibGyroBias.getParam() = vectorFromYaml_<12>(imu["gyroBias"], calibration.calibGyroBias.getParam());
            }
        }
        return calibration;
    }
    static std::vector<double> paramsOrder_(const std::string& type, const nlohmann::json& params) {
        auto get = [&](const char* key) { return params.at(key).get<double>(); };
        if (type == "pinhole") {
            return {get("fx"), get("fy"), get("cx"), get("cy")};
        }
        if (type == "ds") {
            return {get("fx"), get("fy"), get("cx"), get("cy"), get("xi"), get("alpha")};
        }
        if (type == "eucm") {
            return {get("fx"), get("fy"), get("cx"), get("cy"), get("alpha"), get("beta")};
        }
        if (type == "kb4") {
            return {get("fx"), get("fy"), get("cx"), get("cy"), get("k1"), get("k2"), get("k3"), get("k4")};
        }
        if (type == "ucm") {
            return {get("fx"), get("fy"), get("cx"), get("cy"), get("alpha")};
        }
        if (type == "pinhole-radtan8") {
            return {get("fx"),
                get("fy"),
                get("cx"),
                get("cy"),
                get("k1"),
                get("k2"),
                get("p1"),
                get("p2"),
                get("k3"),
                get("k4"),
                get("k5"),
                get("k6")};
        }
        throw std::runtime_error("unsupported camera type: " + type);
    }
    static GenericCamera<double> makeCamera_(const std::string& type, const std::vector<double>& values) {
        GenericCamera<double> camera;
        if (type == "pinhole") {
            camera.variant = PinholeCamera<double>(toEigen_<4>(values));
        } else if (type == "ds") {
            camera.variant = DoubleSphereCamera<double>(toEigen_<6>(values));
        } else if (type == "eucm") {
            camera.variant = ExtendedUnifiedCamera<double>(toEigen_<6>(values));
        } else if (type == "kb4") {
            camera.variant = KannalaBrandtCamera4<double>(toEigen_<8>(values));
        } else if (type == "ucm") {
            camera.variant = UnifiedCamera<double>(toEigen_<5>(values));
        } else if (type == "pinhole-radtan8") {
            camera.variant = PinholeRadtan8Camera<double>(toEigen_<12>(values));
        } else {
            throw std::runtime_error("unsupported camera type: " + type);
        }
        return camera;
    }
    template <int N>
    static Eigen::Matrix<double, N, 1> toEigen_(const std::vector<double>& values) {
        if (values.size() != static_cast<size_t>(N)) {
            throw std::runtime_error("invalid camera intrinsics size");
        }
        Eigen::Matrix<double, N, 1> result;
        for (int i = 0; i < N; ++ i) {
            result(i) = values[static_cast<size_t>(i)];
        }
        return result;
    }
    template <int N>
    static Eigen::Matrix<double, N, 1> vectorFromJson_(const nlohmann::json& node) {
        if (node.is_number()) {
            Eigen::Matrix<double, N, 1> result;
            result.setConstant(node.get<double>());
            return result;
        }
        if (node.size() != static_cast<size_t>(N)) {
            throw std::runtime_error("invalid calibration vector size");
        }
        Eigen::Matrix<double, N, 1> result;
        for (int i = 0; i < N; ++ i) {
            result(i) = node.at(static_cast<size_t>(i)).get<double>();
        }
        return result;
    }
    template <int N>
    static Eigen::Matrix<double, N, 1> vectorFromYaml_(
        const YAML::Node& node, const Eigen::Matrix<double, N, 1>& fallback) {
        if (!node) {
            return fallback;
        }
        if (node.IsScalar()) {
            Eigen::Matrix<double, N, 1> result;
            result.setConstant(node.as<double>());
            return result;
        }
        if (node.size() != static_cast<size_t>(N)) {
            throw std::runtime_error("invalid calibration vector size");
        }
        Eigen::Matrix<double, N, 1> result;
        for (int i = 0; i < N; ++ i) {
            result(i) = node[static_cast<size_t>(i)].as<double>();
        }
        return result;
    }
};

} // arc
