// Copyright 2024 Autoware Foundation
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

#ifndef NODE_HPP_
#define NODE_HPP_

#include "planner_manager.hpp"

#include <autoware_motion_velocity_planner_common/planner_data.hpp>
#include <autoware_motion_velocity_planner_node/srv/load_plugin.hpp>
#include <autoware_motion_velocity_planner_node/srv/unload_plugin.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tier4_autoware_utils/ros/logger_level_configure.hpp>
#include <tier4_autoware_utils/ros/published_time_publisher.hpp>

#include <autoware_auto_mapping_msgs/msg/had_map_bin.hpp>
#include <autoware_auto_perception_msgs/msg/predicted_objects.hpp>
#include <autoware_auto_planning_msgs/msg/trajectory.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace autoware::motion_velocity_planner
{
using autoware_auto_mapping_msgs::msg::HADMapBin;
using autoware_auto_planning_msgs::msg::Trajectory;
using autoware_motion_velocity_planner_node::srv::LoadPlugin;
using autoware_motion_velocity_planner_node::srv::UnloadPlugin;
using TrajectoryPoints = std::vector<autoware_auto_planning_msgs::msg::TrajectoryPoint>;

class MotionVelocityPlannerNode : public rclcpp::Node
{
public:
  explicit MotionVelocityPlannerNode(const rclcpp::NodeOptions & node_options);

private:
  // tf
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  // subscriber
  rclcpp::Subscription<autoware_auto_planning_msgs::msg::Trajectory>::SharedPtr sub_trajectory_;
  rclcpp::Subscription<autoware_auto_perception_msgs::msg::PredictedObjects>::SharedPtr
    sub_predicted_objects_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_no_ground_pointcloud_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_vehicle_odometry_;
  rclcpp::Subscription<geometry_msgs::msg::AccelWithCovarianceStamped>::SharedPtr sub_acceleration_;
  rclcpp::Subscription<autoware_auto_mapping_msgs::msg::HADMapBin>::SharedPtr sub_lanelet_map_;
  rclcpp::Subscription<autoware_perception_msgs::msg::TrafficSignalArray>::SharedPtr
    sub_traffic_signals_;
  rclcpp::Subscription<tier4_v2x_msgs::msg::VirtualTrafficLightStateArray>::SharedPtr
    sub_virtual_traffic_light_states_;
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr sub_occupancy_grid_;

  void on_trajectory(
    const autoware_auto_planning_msgs::msg::Trajectory::ConstSharedPtr input_trajectory_msg);
  void on_predicted_objects(
    const autoware_auto_perception_msgs::msg::PredictedObjects::ConstSharedPtr msg);
  void on_no_ground_pointcloud(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg);
  void on_odometry(const nav_msgs::msg::Odometry::ConstSharedPtr msg);
  void on_acceleration(const geometry_msgs::msg::AccelWithCovarianceStamped::ConstSharedPtr msg);
  void on_lanelet_map(const autoware_auto_mapping_msgs::msg::HADMapBin::ConstSharedPtr msg);
  void on_traffic_signals(
    const autoware_perception_msgs::msg::TrafficSignalArray::ConstSharedPtr msg);
  void on_virtual_traffic_light_states(
    const tier4_v2x_msgs::msg::VirtualTrafficLightStateArray::ConstSharedPtr msg);
  void on_occupancy_grid(const nav_msgs::msg::OccupancyGrid::ConstSharedPtr msg);

  // publishers
  rclcpp::Publisher<autoware_auto_planning_msgs::msg::Trajectory>::SharedPtr trajectory_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr debug_viz_pub_;
  rclcpp::Publisher<autoware_adapi_v1_msgs::msg::VelocityFactorArray>::SharedPtr
    velocity_factor_publisher_;

  //  parameters
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr set_param_callback_;
  bool smooth_velocity_before_planning_{};
  /// @brief set parameters of the velocity smoother
  void set_velocity_smoother_params();

  // members
  PlannerData planner_data_;
  MotionVelocityPlannerManager planner_manager_;
  HADMapBin::ConstSharedPtr map_ptr_{nullptr};
  bool has_received_map_ = false;

  rclcpp::Service<LoadPlugin>::SharedPtr srv_load_plugin_;
  rclcpp::Service<UnloadPlugin>::SharedPtr srv_unload_plugin_;
  void on_unload_plugin(
    const UnloadPlugin::Request::SharedPtr request,
    const UnloadPlugin::Response::SharedPtr response);
  void on_load_plugin(
    const LoadPlugin::Request::SharedPtr request, const LoadPlugin::Response::SharedPtr response);
  rcl_interfaces::msg::SetParametersResult on_set_param(
    const std::vector<rclcpp::Parameter> & parameters);

  // mutex for planner_data_
  std::mutex mutex_;

  // function
  bool is_data_ready() const;
  void insert_stop(
    autoware_auto_planning_msgs::msg::Trajectory & trajectory,
    const geometry_msgs::msg::Point & stop_point) const;
  void insert_slowdown(
    autoware_auto_planning_msgs::msg::Trajectory & trajectory,
    const autoware::motion_velocity_planner::SlowdownInterval & slowdown_interval) const;
  autoware::motion_velocity_planner::TrajectoryPoints smooth_trajectory(
    const autoware::motion_velocity_planner::TrajectoryPoints & trajectory_points,
    const autoware::motion_velocity_planner::PlannerData & planner_data) const;
  autoware_auto_planning_msgs::msg::Trajectory generate_trajectory(
    autoware::motion_velocity_planner::TrajectoryPoints input_trajectory_points);

  std::unique_ptr<tier4_autoware_utils::LoggerLevelConfigure> logger_configure_;

  std::unique_ptr<tier4_autoware_utils::PublishedTimePublisher> published_time_publisher_;
};
}  // namespace autoware::motion_velocity_planner

#endif  // NODE_HPP_
