cmake_minimum_required(VERSION 3.16)
project(arc)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT Python3_EXECUTABLE AND EXISTS "/usr/bin/python3")
  set(Python3_EXECUTABLE "/usr/bin/python3")
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(PCL REQUIRED COMPONENTS common filters)
find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED CONFIG)
find_package(NLopt REQUIRED)
find_package(TBB REQUIRED)
find_package(fmt REQUIRED)
find_path(SOPHUS_INCLUDE_DIR sophus/se3.hpp PATHS /opt/ros/humble/include REQUIRED)

file(GLOB LKH_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/third_party/lkh/src/*.c)
add_library(arc_lkh STATIC
  ${LKH_SOURCES}
  third_party/lkh/lkh_mtsp_interface.cpp
)
target_include_directories(arc_lkh PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/lkh/include
)
target_compile_options(arc_lkh PRIVATE -w -fcommon)
target_link_libraries(arc_lkh PRIVATE m)

add_library(arc INTERFACE)
add_library(arc::arc ALIAS arc)

target_include_directories(arc INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_include_directories(arc SYSTEM INTERFACE
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${NLOPT_INCLUDE_DIRS}
)

target_link_libraries(arc INTERFACE
  arc_lkh
  rclcpp::rclcpp
  Eigen3::Eigen
  ${PCL_COMMON_LIBRARY}
  opencv_core
  geometry_msgs::geometry_msgs
  visualization_msgs::visualization_msgs
  ${NLOPT_LIBRARIES}
)

add_library(arc_vio INTERFACE)
add_library(arc::vio ALIAS arc_vio)

target_include_directories(arc_vio INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_compile_definitions(arc_vio INTERFACE EIGEN_DONT_PARALLELIZE)

target_include_directories(arc_vio SYSTEM INTERFACE
  ${SOPHUS_INCLUDE_DIR}
)

target_link_libraries(arc_vio INTERFACE
  Eigen3::Eigen
  fmt::fmt
  TBB::tbb
  opencv_core
  opencv_imgproc
  opencv_features2d
  nlohmann_json::nlohmann_json
  yaml-cpp
)

add_library(arc_lio INTERFACE)
add_library(arc::lio ALIAS arc_lio)

target_include_directories(arc_lio INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_compile_definitions(arc_lio INTERFACE EIGEN_DONT_PARALLELIZE)

target_include_directories(arc_lio SYSTEM INTERFACE
  ${PCL_INCLUDE_DIRS}
  ${SOPHUS_INCLUDE_DIR}
)

target_link_libraries(arc_lio INTERFACE
  pthread
  Eigen3::Eigen
  ${PCL_COMMON_LIBRARY}
  ${PCL_FILTERS_LIBRARY}
  OpenMP::OpenMP_CXX
  fmt::fmt
  yaml-cpp
)

install(DIRECTORY include/
  DESTINATION include
)

install(TARGETS arc arc_vio arc_lio
  EXPORT export_arc
)

install(TARGETS arc_lkh
  EXPORT export_arc
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(EXPORT export_arc
  NAMESPACE arc::
  DESTINATION share/${PROJECT_NAME}/cmake
)

ament_export_targets(export_arc)
ament_export_dependencies(
  rclcpp
  geometry_msgs
  visualization_msgs
  Eigen3
  PCL
  OpenCV
  NLopt
  TBB
  OpenMP
  fmt
  nlohmann_json
  yaml-cpp
)
ament_export_include_directories(include)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_planning test/test_planning.cc)
  if(TARGET test_planning)
    target_link_libraries(test_planning arc)
    ament_target_dependencies(test_planning rclcpp)
  endif()

  ament_add_gtest(test_bsplineopt test/test_bsplineopt.cc)
  if(TARGET test_bsplineopt)
    target_link_libraries(test_bsplineopt arc)
    ament_target_dependencies(test_bsplineopt rclcpp)
  endif()

  ament_add_gtest(test_graph_search test/test_graph_search.cc)
  if(TARGET test_graph_search)
    target_link_libraries(test_graph_search arc)
    ament_target_dependencies(test_graph_search rclcpp)
  endif()

  ament_add_gtest(test_planning_visualization test/test_planning_visualization.cc)
  if(TARGET test_planning_visualization)
    target_link_libraries(test_planning_visualization arc)
    ament_target_dependencies(test_planning_visualization rclcpp visualization_msgs)
  endif()

  ament_add_gtest(test_fastplanner test/test_fastplanner.cc)
  if(TARGET test_fastplanner)
    target_link_libraries(test_fastplanner arc)
    ament_target_dependencies(test_fastplanner rclcpp)
  endif()

  ament_add_gtest(test_frontiermanager test/test_frontiermanager.cc)
  if(TARGET test_frontiermanager)
    target_link_libraries(test_frontiermanager arc)
    ament_target_dependencies(test_frontiermanager rclcpp)
  endif()

  ament_add_gtest(test_uniformgrid test/test_uniformgrid.cc)
  if(TARGET test_uniformgrid)
    target_link_libraries(test_uniformgrid arc)
    ament_target_dependencies(test_uniformgrid rclcpp)
  endif()

  ament_add_gtest(test_hgrid test/test_hgrid.cc)
  if(TARGET test_hgrid)
    target_link_libraries(test_hgrid arc)
    ament_target_dependencies(test_hgrid rclcpp)
  endif()

  ament_add_gtest(test_multimapmanager test/test_multimapmanager.cc)
  if(TARGET test_multimapmanager)
    target_link_libraries(test_multimapmanager arc)
    ament_target_dependencies(test_multimapmanager rclcpp)
  endif()

  ament_add_gtest(test_gridfusion test/test_gridfusion.cc)
  if(TARGET test_gridfusion)
    target_link_libraries(test_gridfusion arc)
    ament_target_dependencies(test_gridfusion rclcpp)
  endif()

  ament_add_gtest(test_tsp test/test_tsp.cc)
  if(TARGET test_tsp)
    target_link_libraries(test_tsp arc)
    ament_target_dependencies(test_tsp rclcpp)
  endif()

  ament_add_gtest(test_exploration test/test_exploration.cc)
  if(TARGET test_exploration)
    target_link_libraries(test_exploration arc)
    ament_target_dependencies(test_exploration rclcpp)
  endif()

  ament_add_gtest(test_swarm test/test_swarm.cc)
  if(TARGET test_swarm)
    target_link_libraries(test_swarm arc)
    ament_target_dependencies(test_swarm rclcpp)
  endif()

  ament_add_gtest(test_diagnostics test/test_diagnostics.cc)
  if(TARGET test_diagnostics)
    target_link_libraries(test_diagnostics arc)
    ament_target_dependencies(test_diagnostics rclcpp)
  endif()

  ament_add_gtest(test_vio test/test_vio.cc test/test_vio_odr.cc)
  if(TARGET test_vio)
    target_link_libraries(test_vio arc_vio)
    target_compile_options(test_vio PRIVATE -O0 -g0)
  endif()

  ament_add_gtest(test_lio test/test_lio.cc)
  if(TARGET test_lio)
    target_link_libraries(test_lio arc_lio)
    target_compile_options(test_lio PRIVATE -O0 -g0)
  endif()

  ament_add_gtest(test_lio_odr test/test_lio_odr.cc test/test_lio_odr2.cc)
  if(TARGET test_lio_odr)
    target_link_libraries(test_lio_odr arc_lio)
    target_compile_options(test_lio_odr PRIVATE -O0 -g0)
  endif()
endif()

option(ARC_VIO_FULL_INSTANTIATION "Compile remaining VIO template instantiations" OFF)
if(ARC_VIO_FULL_INSTANTIATION)
  add_executable(test_vio_instantiate test/test_vio_instantiate.cc)
  target_link_libraries(test_vio_instantiate arc_vio)
  target_compile_options(test_vio_instantiate PRIVATE -O0 -g0)
endif()

ament_package()
