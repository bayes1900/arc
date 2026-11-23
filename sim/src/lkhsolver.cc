#include <lkhsolvernode.h>

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("lkh_solver");
    arc::LkhSolverNode solver(node);
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
