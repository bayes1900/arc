#include <gtest/gtest.h>

#include <chrono>

#include <base/iarc.h>
#include <vis/planningvisualization.h>

namespace {

class PlanningVisualizationTest : public ::testing::Test {
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
        node_ = std::make_shared<rclcpp::Node>("test_planning_visualization");
        visualization_ = std::make_unique<arc::PlanningVisualization>(node_);
    }

    void TearDown() override {
        visualization_.reset();
        node_.reset();
    }

    rclcpp::Node::SharedPtr node_;
    std::unique_ptr<arc::PlanningVisualization> visualization_;
};

TEST_F(PlanningVisualizationTest, ColorMapEndpointsAndAlpha) {
    const Eigen::Vector4d red = arc::PlanningVisualization::getColor(0.0, 0.5);
    EXPECT_NEAR(red(0), 1.0, 1e-9);
    EXPECT_NEAR(red(1), 0.0, 1e-9);
    EXPECT_NEAR(red(2), 0.0, 1e-9);
    EXPECT_NEAR(red(3), 0.5, 1e-9);

    const Eigen::Vector4d magenta = arc::PlanningVisualization::getColor(1.0 / 6.0);
    EXPECT_NEAR(magenta(0), 1.0, 1e-9);
    EXPECT_NEAR(magenta(1), 0.0, 1e-9);
    EXPECT_NEAR(magenta(2), 1.0, 1e-9);

    const Eigen::Vector4d wrapped = arc::PlanningVisualization::getColor(1.0);
    EXPECT_NEAR(wrapped(0), 1.0, 1e-9);
    EXPECT_NEAR(wrapped(1), 0.0, 1e-9);
}

TEST_F(PlanningVisualizationTest, FillsMarkerGeometry) {
    visualization_msgs::msg::Marker marker;
    visualization_->fillBasicInfo(
        marker,
        Eigen::Vector3d(0.1, 0.2, 0.3),
        Eigen::Vector4d(1, 0, 0, 0.8),
        "test",
        7,
        visualization_msgs::msg::Marker::LINE_LIST
    );
    EXPECT_EQ(marker.header.frame_id, "world");
    EXPECT_EQ(marker.ns, "test");
    EXPECT_EQ(marker.id, 7);
    EXPECT_EQ(marker.type, visualization_msgs::msg::Marker::LINE_LIST);
    EXPECT_NEAR(marker.scale.x, 0.1, 1e-9);
    EXPECT_NEAR(marker.color.a, 0.8, 1e-6);

    const std::vector<Eigen::Vector3d> list = {
        Eigen::Vector3d(0, 0, 0),
        Eigen::Vector3d(1, 2, 3)
    };
    visualization_->fillGeometryInfo(marker, list);
    ASSERT_EQ(marker.points.size(), 2u);
    EXPECT_NEAR(marker.points[1].x, 1.0, 1e-9);
    EXPECT_NEAR(marker.points[1].y, 2.0, 1e-9);
    EXPECT_NEAR(marker.points[1].z, 3.0, 1e-9);

    visualization_msgs::msg::Marker lineMarker;
    visualization_->fillBasicInfo(
        lineMarker,
        Eigen::Vector3d(0.05, 0.05, 0.05),
        Eigen::Vector4d(0, 1, 0, 1),
        "line",
        1,
        visualization_msgs::msg::Marker::LINE_LIST
    );
    visualization_->fillGeometryInfo(lineMarker, list, list);
    EXPECT_EQ(lineMarker.points.size(), 4u);
}

TEST_F(PlanningVisualizationTest, CreatesPublicTopics) {
    auto topics = node_->get_topic_names_and_types();
    for (int attempt = 0; attempt < 100; ++attempt) {
        if (topics.find("/planning_vis/trajectory") != topics.end() &&
            topics.find("/planning_vis/viewpoints") != topics.end()) {
            break;
        }
        rclcpp::sleep_for(std::chrono::milliseconds(10));
        topics = node_->get_topic_names_and_types();
    }
    EXPECT_NE(topics.find("/planning_vis/trajectory"), topics.end());
    EXPECT_NE(topics.find("/planning_vis/viewpoints"), topics.end());
}

TEST_F(PlanningVisualizationTest, DrawCallsDoNotThrow) {
    const std::vector<Eigen::Vector3d> points = {
        Eigen::Vector3d(0, 0, 0),
        Eigen::Vector3d(1, 0, 0),
        Eigen::Vector3d(1, 1, 0)
    };
    EXPECT_NO_THROW(visualization_->drawSpheres(
        points, 0.1, Eigen::Vector4d(1, 0, 0, 1), "test", 0,
        arc::PlanningVisualization::VIEWPOINT));
    EXPECT_NO_THROW(visualization_->drawLines(
        points, 0.05, Eigen::Vector4d(0, 1, 0, 1), "test", 0,
        arc::PlanningVisualization::TRAJECTORY));
    EXPECT_NO_THROW(visualization_->drawCubes(
        points, 0.1, Eigen::Vector4d(0, 0, 1, 1), "test", 0,
        arc::PlanningVisualization::FRONTIER));
    EXPECT_NO_THROW(visualization_->drawBox(
        Eigen::Vector3d::Zero(), Eigen::Vector3d(1, 1, 1),
        Eigen::Vector4d(1, 1, 0, 1), "test", 0,
        arc::PlanningVisualization::TOPO_PATH));
    EXPECT_NO_THROW(visualization_->drawText(
        Eigen::Vector3d::Zero(), "demo", 0.2,
        Eigen::Vector4d(1, 1, 1, 1), "test", 0,
        arc::PlanningVisualization::YAW));
}

} // namespace
