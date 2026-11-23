#include <odompub.h>

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("odom_pub");
    arc::OdometerPub odomPub(node);
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
