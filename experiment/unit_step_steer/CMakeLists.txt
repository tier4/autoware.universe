cmake_minimum_required(VERSION 3.14)
project(unit_step_steer)

find_package(autoware_cmake REQUIRED)
autoware_package()

find_package(autoware_adapi_v1_msgs REQUIRED)

add_executable(unit_step_steer src/step_unit_pub.cpp)
ament_target_dependencies(unit_step_steer
  rclcpp
  autoware_auto_control_msgs
  autoware_adapi_v1_msgs
)

install(TARGETS
  unit_step_steer
  DESTINATION lib/${PROJECT_NAME}
)

ament_auto_package()
