#pragma once

#include <base/iarc.h>
#include <base/timestats.h>
#include <env/depthprojector.h>
#include <env/sdfmap.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <opencv2/core.hpp>
#if __has_include(<cv_bridge/cv_bridge.hpp>)
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif
#include <pcl_conversions/pcl_conversions.h>

namespace arc {

class MapRos {
public:
    MapRos(rclcpp::Node::SharedPtr node);

public:
    void depthPoseCallback(const sensor_msgs::msg::Image::ConstSharedPtr& img, const geometry_msgs::msg::PoseStamped::ConstSharedPtr& pose);
    void cloudPoseCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg, const geometry_msgs::msg::PoseStamped::ConstSharedPtr& pose);
    void visualizationTimerCallback_();
    void esdfTimerCallback();

private:
    void publishDepth_();
    void publishMapAll_();
    void publishMapLocal_();
    void publishUnknown_();
    void publishESDF_();
    void publishUpdateRange_();
    void publishCloud_(const pcl::PointCloud<pcl::PointXYZ>& cloud, const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr& publisher);

private:
    bool inVisualizationHeight_(double z) const;
    void init_();

private:
    rclcpp::Node::SharedPtr node_;
    SdfMap sdfmap_;
    DepthProjector depthProjector_;
    TimeStats fuseStats_;
    TimeStats esdfStats_;
    std::string frameId_;

private:
    double esdfSliceHeight_;
    double visualizationTruncateHeight_;
    double visualizationTruncateLow_;
    bool showAllMap_;
    bool showUnknown_;
    bool showEsdf_;
    bool showInflate_;
    bool showOccTime_;
    bool showEsdfTime_;

private:
    using SyncPolicyImagePose = message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, geometry_msgs::msg::PoseStamped>;
    using SyncPolicyCloudPose = message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::PointCloud2, geometry_msgs::msg::PoseStamped>;

private:
    std::unique_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> depthSub_;
    std::unique_ptr<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>> cloudSub_;
    std::unique_ptr<message_filters::Subscriber<geometry_msgs::msg::PoseStamped>> poseSub_;
    std::unique_ptr<message_filters::Synchronizer<SyncPolicyImagePose>> syncImagePose_;
    std::unique_ptr<message_filters::Synchronizer<SyncPolicyCloudPose>> syncCloudPose_;

private:
    rclcpp::TimerBase::SharedPtr visualizationTimer_;
    rclcpp::TimerBase::SharedPtr esdfTimer_;

private:
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr depthPub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr mapAllPub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr mapLocalPub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr mapLocalInflatePub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr unknownPub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr esdfPub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr updateRangePub_;
};

} // arc
