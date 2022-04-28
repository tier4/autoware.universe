// Copyright 2022 Tier IV, Inc. All rights reserved.
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

#ifndef AUTO_PARKING_PLANNER_HPP_
#define AUTO_PARKING_PLANNER_HPP_

#include "autoware_parking_srvs/srv/freespace_plan.hpp"
#include "autoware_parking_srvs/srv/parking_mission_plan.hpp"
#include "rclcpp/rclcpp.hpp"
#include "route_handler/route_handler.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"

#include "autoware_auto_mapping_msgs/msg/had_map_bin.hpp"
#include "autoware_auto_planning_msgs/msg/detail/had_map_route__struct.hpp"
#include "autoware_auto_planning_msgs/msg/had_map_route.hpp"
#include "autoware_auto_planning_msgs/msg/trajectory.hpp"
#include "autoware_auto_system_msgs/msg/autoware_state.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"

#include <boost/optional.hpp>
#include <boost/optional/optional.hpp>

#include <deque>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace auto_parking_planner
{

using autoware_auto_mapping_msgs::msg::HADMapBin;
using autoware_auto_mapping_msgs::msg::HADMapSegment;
using autoware_auto_planning_msgs::msg::HADMapRoute;
using autoware_auto_planning_msgs::msg::Trajectory;
using autoware_auto_planning_msgs::msg::TrajectoryPoint;
using autoware_auto_system_msgs::msg::AutowareState;
using autoware_parking_srvs::srv::ParkingMissionPlan;

using geometry_msgs::msg::Pose;
using geometry_msgs::msg::PoseStamped;
using geometry_msgs::msg::TwistStamped;

// some util functions
bool containLanelet(const lanelet::ConstPolygon3d & polygon, const lanelet::ConstLanelet & llt);
bool containPolygon(
  const lanelet::ConstPolygon3d & polygon, const lanelet::ConstPolygon3d & polygon2);

struct AutoParkingConfig
{
  bool check_goal_only;
  double lookahead_length;
  double reedsshepp_threashold_length;
  double euclid_threashold_length;
  double reedsshepp_radius;
  double freespace_plan_timeout;
};

struct SubscribedMessages
{
  autoware_auto_mapping_msgs::msg::HADMapBin::ConstSharedPtr map_ptr;
  autoware_auto_system_msgs::msg::AutowareState::ConstSharedPtr state_ptr;
  geometry_msgs::msg::TwistStamped::ConstSharedPtr twist_ptr_;
  autoware_auto_planning_msgs::msg::Trajectory::ConstSharedPtr traj_ptr_;
};

struct PlanningResult
{
  bool success;
  std::string next_phase;
  HADMapRoute route;
  std::string message;
};

struct CircularPlanCache
{
  std::deque<lanelet::ConstLanelets> path_seq;
  lanelet::ConstLanelets current_path;
};

enum class ParkingLaneletType : int { ENTRANCE, EXIT, NORMAL };

struct ParkingMapInfo
{
  lanelet::ConstPolygon3d focus_region;
  lanelet::LaneletMapPtr lanelet_map_ptr;
  lanelet::traffic_rules::TrafficRulesPtr traffic_rules_ptr;
  lanelet::routing::RoutingGraphPtr routing_graph_ptr;
  lanelet::ConstLanelets road_llts;
  std::vector<Pose> parking_poses;
  std::map<size_t, ParkingLaneletType> llt_type_table;
};

class AutoParkingPlanner : public rclcpp::Node
{
public:
  explicit AutoParkingPlanner(const rclcpp::NodeOptions & node_options);

  // Node stuff
  rclcpp::CallbackGroup::SharedPtr cb_group_;
  rclcpp::CallbackGroup::SharedPtr cb_group_nested_;

  rclcpp::Subscription<autoware_auto_system_msgs::msg::AutowareState>::SharedPtr state_subscriber_;
  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr twist_subscriber_;
  rclcpp::Subscription<autoware_auto_mapping_msgs::msg::HADMapBin>::SharedPtr map_subscriber_;
  rclcpp::Subscription<autoware_auto_planning_msgs::msg::Trajectory>::SharedPtr traj_subscriber_;
  rclcpp::Service<autoware_parking_srvs::srv::ParkingMissionPlan>::SharedPtr srv_parking_mission_;
  rclcpp::Client<autoware_parking_srvs::srv::FreespacePlan>::SharedPtr freespaceplane_client_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  AutoParkingConfig config_;
  SubscribedMessages sub_msgs_;
  boost::optional<ParkingMapInfo> parking_map_info_;

  std::string base_link_frame_;
  std::string map_frame_;

  mutable CircularPlanCache circular_plan_cache_;
  mutable std::vector<Pose> feasible_parking_goal_poses_;

  boost::optional<std::string> previous_phase_;
  boost::optional<HADMapRoute> previous_route_;

  void mapCallback(const autoware_auto_mapping_msgs::msg::HADMapBin::ConstSharedPtr msg);
  void stateCallback(const autoware_auto_system_msgs::msg::AutowareState::ConstSharedPtr msg);
  void twistCallback(const geometry_msgs::msg::TwistStamped::ConstSharedPtr msg);
  void trajCallback(const autoware_auto_planning_msgs::msg::Trajectory::ConstSharedPtr msg);

  bool transformPose(
    const PoseStamped & input_pose, PoseStamped * output_pose,
    const std::string target_frame) const;
  PoseStamped getEgoVehiclePose() const;

  bool parkingMissionPlanCallback(
    const std::shared_ptr<rmw_request_id_t> request_header,
    const std::shared_ptr<autoware_parking_srvs::srv::ParkingMissionPlan::Request> request,
    std::shared_ptr<autoware_parking_srvs::srv::ParkingMissionPlan::Response> response);

  bool waitUntilPreviousRouteFinished() const;

  std::vector<Pose> askFeasibleGoalIndex(
    const Pose & start, const std::vector<Pose> & goal_poses) const;
  void prepare();
  PlanningResult planCircularRoute() const;    // except circular_plan_cache_
  PlanningResult planPreparkingRoute() const;  // except TODO(HiroIshida): what is excepted
  PlanningResult planParkingRoute() const;     // except TODO(HiroIshida): what is excepted
};

}  // namespace auto_parking_planner

#endif  // AUTO_PARKING_PLANNER_HPP_
