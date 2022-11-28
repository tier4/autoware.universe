cmake_minimum_required(VERSION 3.10)
project(system_identification)

# require that dependencies from package.xml be available
find_package(ament_cmake_auto REQUIRED)
find_package(rosidl_default_generators REQUIRED)

ament_auto_find_build_dependencies(REQUIRED
        ${${PROJECT_NAME}_BUILD_DEPENDS}
        ${${PROJECT_NAME}_BUILDTOOL_DEPENDS})

set(CMAKE_CXX_STANDARD 17)

## ----------------- Library headers and sources -------------------------------
set(SYSID_LIB system_identification_lib)
set(SYSID_LIB_HEADERS
        include/input_library/prbs_tap_bits.hpp
        include/input_library/input_lib.hpp)

set(SYSID_LIB_SOURCE
        src/input_library/prbs_tap_bits.cpp
        src/input_library/input_lib.cpp)

ament_auto_add_library(${SYSID_LIB} SHARED
        ${SYSID_LIB_SOURCE}
        ${SYSID_LIB_HEADERS})

target_compile_options(${SYSID_LIB} PUBLIC -Wno-error=old-style-cast)

## ------------------- NODE ----------------------------------------------------

set(SYSID_NODE system_identification_node)
ament_auto_add_library(${SYSID_NODE} SHARED
        include/sysid_node.hpp
        src/sysid_node.cpp)

target_link_libraries(${SYSID_NODE}
        ${SYSID_LIB})

##------------------- TYPE SUPPORT ---------------------------------------------
set(MSG_FILES
        "msg/SysIDSteeringVars.idl")

rosidl_generate_interfaces(${PROJECT_NAME}
        ${MSG_FILES}
        ADD_LINTER_TESTS
        DEPENDENCIES builtin_interfaces)

if (${rosidl_cmake_VERSION} VERSION_LESS 2.5.0)
    rosidl_target_interfaces(${SYSID_NODE}
            ${PROJECT_NAME} "rosidl_typesupport_cpp")

    rosidl_target_interfaces(${SYSID_LIB}
            ${PROJECT_NAME} "rosidl_typesupport_cpp")

else ()
    rosidl_get_typesupport_target(
            cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
    target_link_libraries(${SYSID_NODE} "${cpp_typesupport_target}")
    target_link_libraries(${SYSID_LIB} "${cpp_typesupport_target}")

endif ()

##------------- REGISTER NODE ---------------------------------------------------
rclcpp_components_register_node(${SYSID_NODE}
        PLUGIN "sysid::SystemIdentificationNode"
        EXECUTABLE ${SYSID_NODE}_exe)

##----------------- Library Tests------------------------------------------------
if (BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    ament_lint_auto_find_test_dependencies()
    find_package(ament_cmake_gtest REQUIRED)
endif ()

#ament package generation and installing
ament_auto_package(INSTALL_TO_SHARE
        launch
        param)
