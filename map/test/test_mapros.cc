#include <gtest/gtest.h>

#include <memory>

#include <rclcpp/rclcpp.hpp>

#include <mapros.h>

namespace {

class MapRosTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        if (!rclcpp::ok()) {
            rclcpp::init(0, nullptr);
        }
    }

    static void TearDownTestSuite() {
        if (rclcpp::ok()) {
            rclcpp::shutdown();
        }
    }

    void SetUp() override {
        rclcpp::NodeOptions options;
        options.parameter_overrides({
            rclcpp::Parameter("sdfmap.xlen", 4.0),
            rclcpp::Parameter("sdfmap.ylen", 4.0),
            rclcpp::Parameter("sdfmap.zlen", 3.0),
            rclcpp::Parameter("sdfmap.resolution", 0.2),
            rclcpp::Parameter("sdfmap.inflation", 0.2),
            rclcpp::Parameter("sdfmap.maxRayLength", 4.5),
            rclcpp::Parameter("sdfmap.phit", 0.65),
            rclcpp::Parameter("sdfmap.pmiss", 0.35),
            rclcpp::Parameter("sdfmap.pmin", 0.12),
            rclcpp::Parameter("sdfmap.pmax", 0.90),
            rclcpp::Parameter("sdfmap.pocc", 0.80),
        });
        node_ = std::make_shared<rclcpp::Node>("test_mapros", options);
        mapRos_ = std::make_unique<arc::MapRos>(node_);
    }

    void TearDown() override {
        mapRos_.reset();
        node_.reset();
    }

    rclcpp::Node::SharedPtr node_;
    std::unique_ptr<arc::MapRos> mapRos_;
};

TEST_F(MapRosTest, CreatesSubscriptionsAndPublishers) {
    const auto topics = node_->get_topic_names_and_types();
    EXPECT_NE(topics.find("/map_ros/depth"), topics.end());
    EXPECT_NE(topics.find("/map_ros/cloud"), topics.end());
    EXPECT_NE(topics.find("/map_ros/pose"), topics.end());
    EXPECT_NE(topics.find("/sdf_map/occupancy_local"), topics.end());
    EXPECT_NE(topics.find("/sdf_map/occupancy_all"), topics.end());
    EXPECT_NE(topics.find("/sdf_map/esdf"), topics.end());
    EXPECT_NE(topics.find("/sdf_map/update_range"), topics.end());
}

} // namespace
