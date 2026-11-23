#include <rendernode.h>

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("pcl_render_node");
    arc::RenderNode renderNode(node);
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
