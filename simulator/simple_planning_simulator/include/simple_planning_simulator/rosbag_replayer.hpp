// Copyright 2023 The Autoware Foundation.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SIMPLE_PLANNING_SIMULATOR__ROSBAG_REPLAYER_HPP_
#define SIMPLE_PLANNING_SIMULATOR__ROSBAG_REPLAYER_HPP_


#include <ament_index_cpp/get_package_share_directory.hpp>
#include <nlohmann/json.hpp>

#include <autoware_auto_perception_msgs/msg/predicted_objects.hpp>
#include <autoware_auto_control_msgs/msg/ackermann_control_command.hpp>
#include <autoware_auto_system_msgs/msg/autoware_state.hpp>
#include <autoware_auto_system_msgs/msg/float32_multi_array_diagnostic.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <tier4_debug_msgs/msg/float32_stamped.hpp>
#include <tier4_external_api_msgs/srv/engage.hpp>
#include <tier4_planning_msgs/msg/velocity_limit.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <rosbag2_cpp/reader.hpp>

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>


class Config
{
public:
  Config(std::string name = "no27")
  {
    std::string directory_path =
      ament_index_cpp::get_package_share_directory("simple_planning_simulator");
    std::ifstream ifs(directory_path + "/config.json");
    if (!ifs.is_open()) {
      std::cerr << "cannot open config file" << std::endl;
      exit(1);
    }
    std::string json_str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    using json = nlohmann::json;
    auto configs = json::parse(json_str);

    auto config = configs[name];

    velocity_limit_mps = config["velocity_limit"].get<double>() / 3.6;

    start_line_left_x = config["bag_start_line"]["left"][0].get<double>();
    start_line_left_y = config["bag_start_line"]["left"][1].get<double>();
    start_line_right_x = config["bag_start_line"]["right"][0].get<double>();
    start_line_right_y = config["bag_start_line"]["right"][1].get<double>();

    forward_vehicle_uuid.push_back(config["vehicle_uuid"][0].get<uint8_t>());
    forward_vehicle_uuid.push_back(config["vehicle_uuid"][1].get<uint8_t>());

    rosbag_start_time =
      config["bag_start_time"].get<double>() + config["start_offset"].get<double>();

    // wheelbase: 2.75m, front overhang: 0.8m
    baselink_to_front = 3.55;

    rosbag_directory = configs["rosbag_directory"].get<std::string>();
    rosbag_path = rosbag_directory + "/" + config["bag_name"].get<std::string>();
  }

  float velocity_limit_mps;
  float start_line_left_x;
  float start_line_left_y;
  float start_line_right_x;
  float start_line_right_y;
  std::vector<uint8_t> forward_vehicle_uuid;
  float rosbag_start_time;
  float baselink_to_front;
  std::filesystem::path rosbag_path;
  std::string rosbag_directory;
};

class RealRosbagReplayer : public rclcpp::Node
{
public:
  RealRosbagReplayer() : Node("real_rosbag_replayer") {}

private:
  rclcpp::Publisher<autoware_auto_perception_msgs::msg::PredictedObjects>::SharedPtr
    perception_publisher;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_publisher;
  rclcpp::Publisher<autoware_auto_control_msgs::msg::AckermannControlCommand>::SharedPtr
    control_cmd_publisher;

  // publisher for debug
  rclcpp::Publisher<autoware_auto_system_msgs::msg::Float32MultiArrayDiagnostic>::SharedPtr
    control_debug_publisher;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_publisher;

  // publisher for analyzer
  rclcpp::Publisher<tier4_debug_msgs::msg::Float32Stamped>::SharedPtr distance_publisher;
  rclcpp::Publisher<tier4_debug_msgs::msg::Float32Stamped>::SharedPtr ttc_publisher;
};


#endif  // SIMPLE_PLANNING_SIMULATOR__ROSBAG_REPLAYER_HPP_
