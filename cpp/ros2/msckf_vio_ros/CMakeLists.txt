cmake_minimum_required(VERSION 3.16)
project(msckf_vio_ros LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)

# The VIO library from the repository (cpp/), built as a subproject: the node
# runs exactly the code that main_euroc runs.
set(MSCKF_BUILD_APPS OFF CACHE BOOL "" FORCE)
set(MSCKF_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../.. msckf_core)

add_executable(vio_node src/vio_node.cpp)
target_link_libraries(vio_node msckf)
target_compile_options(vio_node PRIVATE -Wall -Wextra -Wpedantic)
ament_target_dependencies(vio_node builtin_interfaces rclcpp sensor_msgs nav_msgs geometry_msgs tf2_ros)

install(TARGETS vio_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

ament_package()
