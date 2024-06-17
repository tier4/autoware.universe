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

#ifndef AUTOWARE_MOTION_VELOCITY_PLANNER_COMMON__PLANNER_DATA_HPP_
#define AUTOWARE_MOTION_VELOCITY_PLANNER_COMMON__PLANNER_DATA_HPP_

#include <motion_velocity_smoother/smoother/smoother_base.hpp>
#include <route_handler/route_handler.hpp>
#include <vehicle_info_util/vehicle_info_util.hpp>

#include <autoware_auto_mapping_msgs/msg/had_map_bin.hpp>
#include <autoware_auto_perception_msgs/msg/predicted_objects.hpp>
#include <autoware_perception_msgs/msg/traffic_signal.hpp>
#include <autoware_perception_msgs/msg/traffic_signal_array.hpp>
#include <geometry_msgs/msg/accel_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/header.hpp>
#include <tier4_api_msgs/msg/crosswalk_status.hpp>
#include <tier4_api_msgs/msg/intersection_status.hpp>
#include <tier4_planning_msgs/msg/velocity_limit.hpp>
#include <tier4_v2x_msgs/msg/virtual_traffic_light_state_array.hpp>

#include <lanelet2_core/Forward.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <algorithm>
#include <deque>
#include <map>
#include <memory>
#include <optional>
#include <vector>

namespace autoware::motion_velocity_planner
{
struct TrafficSignalStamped
{
  builtin_interfaces::msg::Time stamp;
  autoware_perception_msgs::msg::TrafficSignal signal;
};
struct PlannerData
{
  explicit PlannerData(rclcpp::Node & node)
  : vehicle_info_(vehicle_info_util::VehicleInfoUtil(node).getVehicleInfo())
  {
  }

  // msgs from callbacks that are used for data-ready
  geometry_msgs::msg::PoseStamped::ConstSharedPtr current_odometry;
  geometry_msgs::msg::TwistStamped::ConstSharedPtr current_velocity;
  geometry_msgs::msg::AccelWithCovarianceStamped::ConstSharedPtr current_acceleration;
  autoware_auto_perception_msgs::msg::PredictedObjects::ConstSharedPtr predicted_objects;
  pcl::PointCloud<pcl::PointXYZ>::ConstPtr no_ground_pointcloud;
  nav_msgs::msg::OccupancyGrid::ConstSharedPtr occupancy_grid;
  std::shared_ptr<route_handler::RouteHandler> route_handler;

  // nearest search
  double ego_nearest_dist_threshold;
  double ego_nearest_yaw_threshold;

  // other internal data
  // traffic_light_id_map_raw is the raw observation, while traffic_light_id_map_keep_last keeps the
  // last observed infomation for UNKNOWN
  std::map<lanelet::Id, TrafficSignalStamped> traffic_light_id_map_raw_;
  std::map<lanelet::Id, TrafficSignalStamped> traffic_light_id_map_last_observed_;
  std::optional<tier4_planning_msgs::msg::VelocityLimit> external_velocity_limit;
  tier4_v2x_msgs::msg::VirtualTrafficLightStateArray::ConstSharedPtr virtual_traffic_light_states;

  // velocity smoother
  std::shared_ptr<motion_velocity_smoother::SmootherBase> velocity_smoother_{};
  // parameters
  vehicle_info_util::VehicleInfo vehicle_info_;

  /**
   *@fn
   *@brief queries the traffic signal information of given Id. if keep_last_observation = true,
   *recent UNKNOWN observation is overwritten as the last non-UNKNOWN observation
   */
  std::optional<TrafficSignalStamped> get_traffic_signal(
    const lanelet::Id id, const bool keep_last_observation = false) const
  {
    const auto & traffic_light_id_map =
      keep_last_observation ? traffic_light_id_map_last_observed_ : traffic_light_id_map_raw_;
    if (traffic_light_id_map.count(id) == 0) {
      return std::nullopt;
    }
    return std::make_optional<TrafficSignalStamped>(traffic_light_id_map.at(id));
  }
};
}  // namespace autoware::motion_velocity_planner

#endif  // AUTOWARE_MOTION_VELOCITY_PLANNER_COMMON__PLANNER_DATA_HPP_
