#include <mappub.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

namespace arc {

///////////////////////////////////////////////////////////////////////////////////////////
// MapPub
///////////////////////////////////////////////////////////////////////////////////////////

MapPub::MapPub(rclcpp::Node::SharedPtr node)
    : node_(std::move(node))
{
    pcdPath_ = node_->declare_parameter<std::string>("pcdPath");
    addGround_ = node_->declare_parameter<bool>("addGround", true);
}

bool MapPub::initialize() {
    if (!loadMap_()) {
        return false;
    }
    mapPub_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/map_generator/global_cloud",
        rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local()
    );
    mapMsg_.header.stamp = node_->now();
    mapPub_->publish(mapMsg_);
    return true;
}

///////////////////////////////////////////////////////////////////////////////////////////
// private
///////////////////////////////////////////////////////////////////////////////////////////

bool MapPub::loadMap_() {
    if (pcdPath_.empty()) {
        RCLCPP_ERROR(node_->get_logger(), "pcd path is empty");
        return false;
    }
    pcl::PointCloud<pcl::PointXYZ> cloud;
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcdPath_, cloud) != 0) {
        RCLCPP_ERROR(node_->get_logger(), "Failed to read file: %s", pcdPath_.c_str());
        return false;
    }
    if (addGround_) {
        for (double x = -7.0; x <= 7.0; x += 0.1) {
            for (double y = -15.0; y <= 15.0; y += 0.1) {
                cloud.push_back(pcl::PointXYZ(x, y, 0.0));
            }
        }
    }
    pcl::toROSMsg(cloud, mapMsg_);
    mapMsg_.header.frame_id = "world";
    return true;
}

} // arc

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("map_pub");
    arc::MapPub mapPub(node);
    if (!mapPub.initialize()) {
        rclcpp::shutdown();
        return 1;
    }
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
