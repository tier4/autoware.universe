cmake_minimum_required(VERSION 3.5)
project(auto_parking_planner)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Werror)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

ament_auto_add_library(auto_parking_planner_node SHARED
  src/auto_parking_planner.cpp
  src/utils.cpp
  src/preparation.cpp
  src/circular_route_planning.cpp
  src/preparking_route_planning.cpp
)

rclcpp_components_register_node(auto_parking_planner_node
    PLUGIN "auto_parking_planner::AutoParkingPlanner"
  EXECUTABLE auto_parking_planner
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package(
  INSTALL_TO_SHARE
)
