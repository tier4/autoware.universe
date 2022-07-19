cmake_minimum_required(VERSION 3.14)
project(autoware_ad_api_msgs)

find_package(autoware_cmake REQUIRED)
autoware_package()

rosidl_generate_interfaces(${PROJECT_NAME}
  srv/InterfaceVersion.srv
  common/msg/ResponseStatus.msg
  system/srv/MRMOperation.srv
  system/msg/MRMState.msg
  system/msg/MRMBehaviorStatus.msg
  DEPENDENCIES
    builtin_interfaces
    std_msgs
    geometry_msgs
)

ament_auto_package()
