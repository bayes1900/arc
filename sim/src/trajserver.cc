#include <trajserver.h>

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("traj_server");
    arc::TrajectoryServer trajectoryServer(node);
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
