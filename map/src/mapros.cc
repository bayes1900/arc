#include <mapros.h>

namespace arc {

///////////////////////////////////////////////////////////////////////////////////////////
// MapRos
///////////////////////////////////////////////////////////////////////////////////////////

MapRos::MapRos(rclcpp::Node::SharedPtr node)
    : node_(node)
    , sdfmap_(node)
    , fuseStats_("Fusion", false)
    , esdfStats_("ESDF", false)
{
    init_();
}

///////////////////////////////////////////////////////////////////////////////////////////
// public - callbacks
///////////////////////////////////////////////////////////////////////////////////////////

void MapRos::depthPoseCallback(const sensor_msgs::msg::Image::ConstSharedPtr& img, const geometry_msgs::msg::PoseStamped::ConstSharedPtr& pose) {
    const Eigen::Vector3d cameraPos(pose->pose.position.x, pose->pose.position.y, pose->pose.position.z);
    if (!sdfmap_.params().isInMap(cameraPos)) {
        return;
    }
    cv_bridge::CvImagePtr cvPtr;
    try {
        cvPtr = cv_bridge::toCvCopy(img, img->encoding);
    } catch (const cv_bridge::Exception& e) {
        RCLCPP_ERROR(node_->get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }
    if (img->encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
        cvPtr->image.convertTo(cvPtr->image, CV_16UC1, depthProjector_.scalingFactor());
    }
    if (cvPtr->image.type() != CV_16UC1) {
        RCLCPP_ERROR(node_->get_logger(), "Unsupported depth image type: %d", cvPtr->image.type());
        return;
    }
    const Eigen::Quaterniond cameraQ(
        pose->pose.orientation.w,
        pose->pose.orientation.x,
        pose->pose.orientation.y,
        pose->pose.orientation.z
    );
    const Eigen::Matrix3d cameraR = cameraQ.toRotationMatrix();
    depthProjector_.project(cvPtr->image, cameraPos, cameraR);
    publishDepth_();

    const auto t1 = node_->now();
    sdfmap_.updateFramePoints(depthProjector_.cloud(), depthProjector_.size(), cameraPos);
    const auto t2 = node_->now();
    fuseStats_.record((t2 - t1).seconds());
}

void MapRos::cloudPoseCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg, const geometry_msgs::msg::PoseStamped::ConstSharedPtr& pose) {
    const Eigen::Vector3d cameraPos(pose->pose.position.x, pose->pose.position.y, pose->pose.position.z);
    if (!sdfmap_.params().isInMap(cameraPos)) {
        return;
    }
    pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::fromROSMsg(*msg, cloud);

    const auto t1 = node_->now();
    sdfmap_.updateFramePoints(cloud, cloud.points.size(), cameraPos);
    const auto t2 = node_->now();
    fuseStats_.record((t2 - t1).seconds());
}

void MapRos::visualizationTimerCallback_() {
    publishMapLocal_();
    if (showAllMap_) {
        publishMapAll_();
    }
    if (showUnknown_) {
        publishUnknown_();
    }
    if (showEsdf_) {
        publishESDF_();
    }
    publishUpdateRange_();
}

void MapRos::esdfTimerCallback() {
    if (!sdfmap_.needsESDFUpdate()) {
        return;
    }
    const auto t1 = node_->now();
    sdfmap_.updateESDF();
    const auto t2 = node_->now();

    esdfStats_.record((t2 - t1).seconds());
}

///////////////////////////////////////////////////////////////////////////////////////////
// private - visualization
///////////////////////////////////////////////////////////////////////////////////////////

void MapRos::publishDepth_() {
    publishCloud_(depthProjector_.cloud(), depthPub_);
}

void MapRos::publishMapAll_() {
    const MapParams& mp = sdfmap_.params();
    const MapData& md = sdfmap_.data();
    if ((md.allMax.array() < md.allMin.array()).any()) {
        pcl::PointCloud<pcl::PointXYZ> empty;
        publishCloud_(empty, mapAllPub_);
        return;
    }
    Eigen::Vector3i minIdx = mp.p2i(md.allMin);
    Eigen::Vector3i maxIdx = mp.p2i(md.allMax);
    mp.boundIndex(minIdx);
    mp.boundIndex(maxIdx);
    pcl::PointCloud<pcl::PointXYZ> cloud;

    for (int x = minIdx[0]; x <= maxIdx[0]; ++ x) {
        for (int y = minIdx[1]; y <= maxIdx[1]; ++ y) {
            for (int z = minIdx[2]; z <= maxIdx[2]; ++ z) {
                Eigen::Vector3i idx(x, y, z);
                if (md.getOccupancy(idx) != VoxelOccupancyCode::OCCUPIED) {
                    continue;
                }
                const Eigen::Vector3d pos = mp.i2p(idx);
                if (!inVisualizationHeight_(pos.z())) {
                    continue;
                }
                cloud.push_back(pcl::PointXYZ(pos.x(), pos.y(), pos.z()));
            }
        }
    }
    cloud.width = cloud.points.size();
    cloud.height = 1;
    cloud.is_dense = true;
    publishCloud_(cloud, mapAllPub_);
}

void MapRos::publishMapLocal_() {
    const MapParams& mp = sdfmap_.params();
    const MapData& md = sdfmap_.data();
    pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::PointCloud<pcl::PointXYZ> inflateCloud;

    for (int x = md.localBoundMin[0]; x <= md.localBoundMax[0]; ++ x) {
        for (int y = md.localBoundMin[1]; y <= md.localBoundMax[1]; ++ y) {
            for (int z = md.localBoundMin[2]; z <= md.localBoundMax[2]; ++ z) {
                Eigen::Vector3i idx(x, y, z);
                const Eigen::Vector3d pos = mp.i2p(idx);
                if (!inVisualizationHeight_(pos.z())) {
                    continue;
                }

                if (md.getOccupancy(idx) == VoxelOccupancyCode::OCCUPIED) {
                    cloud.push_back(pcl::PointXYZ(pos.x(), pos.y(), pos.z()));
                } else if (showInflate_ &&
                           md.getInflateOccupancy(idx) == VoxelInflateOccupancyCode::OCCUPIED) {
                    inflateCloud.push_back(pcl::PointXYZ(pos.x(), pos.y(), pos.z()));
                }
            }
        }
    }
    cloud.width = cloud.points.size();
    cloud.height = 1;
    cloud.is_dense = true;
    publishCloud_(cloud, mapLocalPub_);

    inflateCloud.width = inflateCloud.points.size();
    inflateCloud.height = 1;
    inflateCloud.is_dense = true;
    publishCloud_(inflateCloud, mapLocalInflatePub_);
}

void MapRos::publishUnknown_() {
    const MapParams& mp = sdfmap_.params();
    const MapData& md = sdfmap_.data();
    if ((md.allMax.array() < md.allMin.array()).any()) {
        pcl::PointCloud<pcl::PointXYZ> empty;
        publishCloud_(empty, unknownPub_);
        return;
    }
    Eigen::Vector3i minIdx = mp.p2i(md.allMin);
    Eigen::Vector3i maxIdx = mp.p2i(md.allMax);
    mp.boundIndex(minIdx);
    mp.boundIndex(maxIdx);
    pcl::PointCloud<pcl::PointXYZ> cloud;

    for (int x = minIdx[0]; x <= maxIdx[0]; ++ x) {
        for (int y = minIdx[1]; y <= maxIdx[1]; ++ y) {
            for (int z = minIdx[2]; z <= maxIdx[2]; ++ z) {
                Eigen::Vector3i idx(x, y, z);
                if (md.getOccupancy(idx) != VoxelOccupancyCode::UNKNOWN) {
                    continue;
                }
                const Eigen::Vector3d pos = mp.i2p(idx);
                if (!inVisualizationHeight_(pos.z())) {
                    continue;
                }
                cloud.push_back(pcl::PointXYZ(pos.x(), pos.y(), pos.z()));
            }
        }
    }
    cloud.width = cloud.points.size();
    cloud.height = 1;
    cloud.is_dense = true;
    publishCloud_(cloud, unknownPub_);
}

void MapRos::publishESDF_() {
    const MapParams& mp = sdfmap_.params();
    const MapData& md = sdfmap_.data();
    const double minDist = 0.0;
    const double maxDist = 3.0;
    pcl::PointCloud<pcl::PointXYZI> cloud;

    for (int x = md.localBoundMin[0]; x <= md.localBoundMax[0]; ++ x) {
        for (int y = md.localBoundMin[1]; y <= md.localBoundMax[1]; ++ y) {
            Eigen::Vector3d pos = mp.i2p(Eigen::Vector3i(x, y, 0));
            pos.z() = esdfSliceHeight_;
            if (!mp.isInMap(pos)) {
                continue;
            }
            double dist = md.getDistance(pos);
            if (dist < 0.0) {
                continue;
            }
            dist = std::min(std::max(dist, minDist), maxDist);
            pcl::PointXYZI pt;
            pt.x = pos.x();
            pt.y = pos.y();
            pt.z = esdfSliceHeight_;
            pt.intensity = (dist - minDist) / (maxDist - minDist);
            cloud.push_back(pt);
        }
    }
    cloud.width = cloud.points.size();
    cloud.height = 1;
    cloud.is_dense = true;

    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(cloud, msg);
    msg.header.stamp = node_->now();
    msg.header.frame_id = frameId_;
    esdfPub_->publish(msg);
}

void MapRos::publishUpdateRange_() {
    const MapParams& mp = sdfmap_.params();
    const MapData& md = sdfmap_.data();
    const Eigen::Vector3d minPos = mp.i2p(md.localBoundMin);
    const Eigen::Vector3d maxPos = mp.i2p(md.localBoundMax);

    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = frameId_;
    marker.header.stamp = node_->now();
    marker.type = visualization_msgs::msg::Marker::CUBE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.id = 0;

    marker.pose.position.x = 0.5 * (minPos.x() + maxPos.x());
    marker.pose.position.y = 0.5 * (minPos.y() + maxPos.y());
    marker.pose.position.z = 0.5 * (minPos.z() + maxPos.z());

    marker.scale.x = maxPos.x() - minPos.x();
    marker.scale.y = maxPos.y() - minPos.y();
    marker.scale.z = maxPos.z() - minPos.z();

    marker.color.a = 0.3;
    marker.color.r = 1.0;
    marker.color.g = 0.0;
    marker.color.b = 0.0;

    marker.pose.orientation.w = 1.0;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;

    updateRangePub_->publish(marker);
}

void MapRos::publishCloud_(
    const pcl::PointCloud<pcl::PointXYZ>& cloud,
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr& publisher) {
    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(cloud, msg);
    msg.header.stamp = node_->now();
    msg.header.frame_id = frameId_;
    publisher->publish(msg);
}

bool MapRos::inVisualizationHeight_(double z) const {
    return z <= visualizationTruncateHeight_ && z >= visualizationTruncateLow_;
}

///////////////////////////////////////////////////////////////////////////////////////////
// private - init
///////////////////////////////////////////////////////////////////////////////////////////

void MapRos::init_() {
    DepthProjectorParams projectorParams;
    projectorParams.fx = node_->declare_parameter<double>("mapros.fx", 387.229);
    projectorParams.fy = node_->declare_parameter<double>("mapros.fy", 387.229);
    projectorParams.cx = node_->declare_parameter<double>("mapros.cx", 321.046);
    projectorParams.cy = node_->declare_parameter<double>("mapros.cy", 243.449);
    projectorParams.minDepth = node_->declare_parameter<double>("mapros.depth_filter_mindist", 0.2);
    projectorParams.maxDepth = node_->declare_parameter<double>("mapros.depth_filter_maxdist", 4.6);
    projectorParams.margin = node_->declare_parameter<int>("mapros.depth_filter_margin", 2);
    projectorParams.scalingFactor = node_->declare_parameter<double>("mapros.k_depth_scaling_factor", 1000.0);
    projectorParams.skipPixel = node_->declare_parameter<int>("mapros.skip_pixel", 2);
    depthProjector_.init(projectorParams);

    frameId_ = node_->declare_parameter<std::string>("mapros.frame_id", "world");
    esdfSliceHeight_ = node_->declare_parameter<double>("mapros.esdf_slice_height", 0.5);
    visualizationTruncateHeight_ = node_->declare_parameter<double>("mapros.visualization_truncate_height", 10.09);
    visualizationTruncateLow_ = node_->declare_parameter<double>("mapros.visualization_truncate_low", -2.0);
    showAllMap_ = node_->declare_parameter<bool>("mapros.show_all_map", true);
    showUnknown_ = node_->declare_parameter<bool>("mapros.show_unknown", false);
    showEsdf_ = node_->declare_parameter<bool>("mapros.show_esdf", false);
    showInflate_ = node_->declare_parameter<bool>("mapros.show_inflate", false);
    showOccTime_ = node_->declare_parameter<bool>("mapros.show_occ_time", false);
    showEsdfTime_ = node_->declare_parameter<bool>("mapros.show_esdf_time", false);

    fuseStats_.setEnabled(showOccTime_);
    esdfStats_.setEnabled(showEsdfTime_);
    fuseStats_.setLogger(node_->get_logger());
    esdfStats_.setLogger(node_->get_logger());

    depthSub_ = std::make_unique<message_filters::Subscriber<sensor_msgs::msg::Image>>(
        node_.get(), "/map_ros/depth", rclcpp::QoS(50).get_rmw_qos_profile()
    );
    cloudSub_ = std::make_unique<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>>(
        node_.get(), "/map_ros/cloud", rclcpp::QoS(50).get_rmw_qos_profile()
    );
    poseSub_ = std::make_unique<message_filters::Subscriber<geometry_msgs::msg::PoseStamped>>(
        node_.get(), "/map_ros/pose", rclcpp::QoS(25).get_rmw_qos_profile()
    );

    syncImagePose_ = std::make_unique<message_filters::Synchronizer<SyncPolicyImagePose>>(
        SyncPolicyImagePose(100), *depthSub_, *poseSub_
    );
    // message_filters invokes the callback with nine arguments in Humble.
    syncImagePose_->registerCallback(
        [this](const sensor_msgs::msg::Image::ConstSharedPtr& img,
               const geometry_msgs::msg::PoseStamped::ConstSharedPtr& pose,
               const auto&...
        ) {
            depthPoseCallback(img, pose);
        }
    );
    syncCloudPose_ = std::make_unique<message_filters::Synchronizer<SyncPolicyCloudPose>>(
        SyncPolicyCloudPose(100), *cloudSub_, *poseSub_
    );
    syncCloudPose_->registerCallback(
        [this](const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg,
               const geometry_msgs::msg::PoseStamped::ConstSharedPtr& pose,
               const auto&...
        ) {
            cloudPoseCallback(msg, pose);
        }
    );

    visualizationTimer_ = node_->create_wall_timer(
        std::chrono::milliseconds(200),
        [this]() { visualizationTimerCallback_(); }
    );
    esdfTimer_ = node_->create_wall_timer(
        std::chrono::milliseconds(50),
        [this]() { esdfTimerCallback(); }
    );

    depthPub_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>("/sdf_map/depth_cloud", 10);
    mapAllPub_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>("/sdf_map/occupancy_all", 10);
    mapLocalPub_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>("/sdf_map/occupancy_local", 10);
    mapLocalInflatePub_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>("/sdf_map/occupancy_local_inflate", 10);
    unknownPub_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>("/sdf_map/unknown", 10);
    esdfPub_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>("/sdf_map/esdf", 10);
    updateRangePub_ = node_->create_publisher<visualization_msgs::msg::Marker>("/sdf_map/update_range", 10);
}

} // arc
