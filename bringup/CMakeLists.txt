cmake_minimum_required(VERSION 3.16)
project(arc_bringup)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS scripts/run_map_pub.sh
  DESTINATION share/${PROJECT_NAME}/scripts
)

ament_package()
