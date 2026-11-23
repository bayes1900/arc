cmake_minimum_required(VERSION 3.8)
project(arcmsgs)

if(NOT Python3_EXECUTABLE AND EXISTS "/usr/bin/python3")
  set(Python3_EXECUTABLE "/usr/bin/python3")
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/PositionCommand.msg"
  "msg/DroneState.msg"
  "msg/PairOpt.msg"
  "msg/PairOptResponse.msg"
  "msg/IdxList.msg"
  "msg/ChunkStamps.msg"
  "msg/ChunkData.msg"
  "msg/GridTour.msg"
  "msg/SwarmTrajectory.msg"
  "msg/ExplorationEvent.msg"
  "msg/ExplorationMetrics.msg"
  "msg/TrajectoryMetrics.msg"
  "srv/SolveTsp.srv"
  DEPENDENCIES std_msgs geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
