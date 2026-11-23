#include "lidarrender.h"

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("lidar_render_node");
    arc::LidarRenderNode lidarRender(node);
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
