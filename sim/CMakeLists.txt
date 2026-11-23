cmake_minimum_required(VERSION 3.16)
project(arc_sim)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(arcmsgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(message_filters REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED CONFIG)
find_package(NLopt REQUIRED)
find_package(arc REQUIRED)

add_executable(map_pub
  src/mappub.cc
)

target_include_directories(map_pub PRIVATE include)
target_include_directories(map_pub SYSTEM PRIVATE ${PCL_INCLUDE_DIRS})

ament_target_dependencies(map_pub
  rclcpp
  sensor_msgs
  pcl_conversions
)

target_link_libraries(map_pub arc::arc ${PCL_LIBRARIES})

add_executable(odom_pub
  src/odompub.cc
)

target_include_directories(odom_pub PRIVATE include)

ament_target_dependencies(odom_pub
  rclcpp
  arcmsgs
  nav_msgs
)

target_link_libraries(odom_pub arc::arc)

add_executable(pcl_render_node
  src/rendernode.cc
)

target_include_directories(pcl_render_node PRIVATE include)
target_include_directories(pcl_render_node SYSTEM PRIVATE
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

ament_target_dependencies(pcl_render_node
  rclcpp
  sensor_msgs
  nav_msgs
  geometry_msgs
  pcl_conversions
  cv_bridge
)

target_link_libraries(pcl_render_node arc::arc ${PCL_LIBRARIES} ${OpenCV_LIBS})

add_executable(exploration_node
  src/explorationnode.cc
)

target_include_directories(exploration_node PRIVATE include)
target_include_directories(exploration_node SYSTEM PRIVATE
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

ament_target_dependencies(exploration_node
  rclcpp
  sensor_msgs
  nav_msgs
  geometry_msgs
  visualization_msgs
  std_msgs
  arcmsgs
  pcl_conversions
  cv_bridge
  message_filters
)

target_link_libraries(exploration_node arc::arc ${PCL_LIBRARIES} ${OpenCV_LIBS})

add_executable(vio_node
  src/vionode.cc
)

target_include_directories(vio_node PRIVATE include)

target_link_libraries(vio_node
  arc::arc_vio
)

ament_target_dependencies(vio_node
  rclcpp
  sensor_msgs
  nav_msgs
  geometry_msgs
  tf2_ros
  cv_bridge
)

add_executable(lio_node
  src/lionode.cc
)

target_include_directories(lio_node PRIVATE include)

target_link_libraries(lio_node
  arc::arc_lio
)

ament_target_dependencies(lio_node
  rclcpp
  sensor_msgs
  nav_msgs
  geometry_msgs
  tf2_ros
  pcl_conversions
)

add_executable(lidar_render_node
  src/lidarrender.cc
)

target_include_directories(lidar_render_node PRIVATE include)
target_include_directories(lidar_render_node SYSTEM PRIVATE
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(lidar_render_node arc::arc ${PCL_LIBRARIES})

ament_target_dependencies(lidar_render_node
  rclcpp
  sensor_msgs
  nav_msgs
  pcl_conversions
)

add_executable(dronevisual_pub
  src/dronevisualpub.cc
)

target_include_directories(dronevisual_pub PRIVATE include)

ament_target_dependencies(dronevisual_pub
  rclcpp
  nav_msgs
  geometry_msgs
  visualization_msgs
)

target_link_libraries(dronevisual_pub arc::arc)

add_executable(lkh_solver
  src/lkhsolver.cc
)

target_include_directories(lkh_solver PRIVATE include)

ament_target_dependencies(lkh_solver
  rclcpp
  arcmsgs
)

target_link_libraries(lkh_solver arc::arc)

add_executable(traj_server
  src/trajserver.cc
)

target_include_directories(traj_server PRIVATE include)
target_include_directories(traj_server SYSTEM PRIVATE
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

ament_target_dependencies(traj_server
  rclcpp
  nav_msgs
  geometry_msgs
  visualization_msgs
  sensor_msgs
  arcmsgs
  pcl_conversions
)

target_link_libraries(traj_server arc::arc ${PCL_LIBRARIES})

install(TARGETS map_pub odom_pub pcl_render_node exploration_node dronevisual_pub lkh_solver traj_server vio_node lio_node lidar_render_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY resource
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
