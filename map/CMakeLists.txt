cmake_minimum_required(VERSION 3.16)
project(arc_map)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT Python3_EXECUTABLE AND EXISTS "/usr/bin/python3")
  set(Python3_EXECUTABLE "/usr/bin/python3")
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED COMPONENTS common)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED CONFIG)
find_package(NLopt REQUIRED)
find_package(arc REQUIRED)

add_library(arc_map SHARED src/mapros.cc)
add_library(arc_map::arc_map ALIAS arc_map)

target_include_directories(arc_map PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_include_directories(arc_map SYSTEM PUBLIC
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(arc_map PUBLIC
  arc::arc
  cv_bridge::cv_bridge
  ${PCL_COMMON_LIBRARY}
  opencv_core
)

ament_target_dependencies(arc_map PUBLIC
  rclcpp
  sensor_msgs
  geometry_msgs
  visualization_msgs
  message_filters
  pcl_conversions
)

add_executable(map_node src/mapnode.cc)
target_link_libraries(map_node arc_map)
ament_target_dependencies(map_node rclcpp)

if(APPLE)
  set_target_properties(map_node PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "@loader_path;@loader_path/.."
  )
endif()

install(TARGETS arc_map map_node
  EXPORT export_arc_map
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(EXPORT export_arc_map
  NAMESPACE arc_map::
  DESTINATION share/${PROJECT_NAME}/cmake
)

ament_export_targets(export_arc_map HAS_LIBRARY_TARGET)
ament_export_dependencies(
  arc
  rclcpp
  sensor_msgs
  geometry_msgs
  visualization_msgs
  message_filters
  pcl_conversions
  cv_bridge
  Eigen3
  PCL
  OpenCV
)
ament_export_include_directories(include)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_mapros test/test_mapros.cc)
  if(TARGET test_mapros)
    target_link_libraries(test_mapros arc_map)
    ament_target_dependencies(test_mapros rclcpp)
  endif()
endif()

ament_package()
