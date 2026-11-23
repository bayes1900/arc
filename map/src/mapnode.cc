#include <base/iarc.h>
#include <mapros.h>


int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("mapros");
    arc::MapRos mapRos(node);
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
