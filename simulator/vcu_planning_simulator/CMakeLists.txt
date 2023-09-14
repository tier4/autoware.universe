cmake_minimum_required(VERSION 3.14)
project(vcu_planning_simulator)

find_package(autoware_cmake REQUIRED)
autoware_package()

# Component
ament_auto_add_library(${PROJECT_NAME} SHARED
  include/vcu_planning_simulator/vcu_planning_simulator_core.hpp
  include/vcu_planning_simulator/visibility_control.hpp
  src/vcu_planning_simulator/vcu_planning_simulator_core.cpp
  src/vcu_planning_simulator/vehicle_model/sim_model_interface.cpp
  src/vcu_planning_simulator/vehicle_model/sim_model_ideal_steer_vel.cpp
  src/vcu_planning_simulator/vehicle_model/sim_model_delay_steer_vel.cpp
)
target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC ${tf2_INCLUDE_DIRS})

target_compile_options(${PROJECT_NAME} PRIVATE -Wno-old-style-cast) # RCLCPP_ERROR_THROTTLE() has built-in old-style casts.

# Node executable
rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN "simulation::vcu_planning_simulator::VcuPlanningSimulator"
  EXECUTABLE ${PROJECT_NAME}_exe
)

if(BUILD_TESTING)
  ament_add_ros_isolated_gtest(test_vcu_planning_simulator
    test/test_vcu_planning_simulator.cpp
    TIMEOUT 120
  )

  target_link_libraries(test_vcu_planning_simulator
    ${PROJECT_NAME}
  )
endif()

ament_auto_package(INSTALL_TO_SHARE param launch config)
