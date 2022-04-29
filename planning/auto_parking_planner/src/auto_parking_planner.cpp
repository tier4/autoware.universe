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

#include "auto_parking_planner.hpp"

#include "tf2/utils.h"

#include <rclcpp/executors.hpp>

#include <chrono>
#include <future>
#include <stdexcept>

namespace auto_parking_planner
{

AutoParkingPlanner::AutoParkingPlanner(const rclcpp::NodeOptions & node_options)
: rclcpp::Node("auto_parking_planner_node", node_options),
  tf_buffer_(get_clock()),
  tf_listener_(tf_buffer_),
  previous_phase_(boost::none)
{
  map_frame_ = declare_parameter("map_frame", "map");
  base_link_frame_ = declare_parameter("base_link_frame", "base_link");

  {  // set node config
    config_.check_goal_only = declare_parameter("check_only_goal", true);
    config_.lookahead_length = declare_parameter("lookahead_length", 4.0);
    config_.lookahead_length_min = declare_parameter("lookahead_length_min", 2.0);
    config_.lookahead_length_max = declare_parameter("lookahead_length_max", 10.0);
    config_.reedsshepp_threashold_length = declare_parameter("reedsshepp_threashold_length", 6.0);
    config_.euclid_threashold_length = declare_parameter("euclid_threashold_length", 2.0);
    config_.reedsshepp_radius = declare_parameter("reedsshepp_radius", 5.0);
    config_.freespace_plan_timeout = declare_parameter("freespace_plan_timeout", 1.0);
  }

  cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  cb_group_nested_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  using std::placeholders::_1;
  using std::placeholders::_2;
  using std::placeholders::_3;
  map_subscriber_ = create_subscription<autoware_auto_mapping_msgs::msg::HADMapBin>(
    "~/input/vector_map", rclcpp::QoS{10}.transient_local(),
    std::bind(&AutoParkingPlanner::mapCallback, this, _1));
  state_subscriber_ = create_subscription<autoware_auto_system_msgs::msg::AutowareState>(
    "~/input/state", rclcpp::QoS{1}, std::bind(&AutoParkingPlanner::stateCallback, this, _1));

  velocity_subscriber_ = this->create_subscription<autoware_auto_vehicle_msgs::msg::VelocityReport>(
    "~/input/velocity_report", 10,
    std::bind(&AutoParkingPlanner::velocityCallback, this, std::placeholders::_1));

  traj_subscriber_ = create_subscription<Trajectory>(
    "~/input/trajectory", rclcpp::QoS{1}, std::bind(&AutoParkingPlanner::trajCallback, this, _1));

  srv_parking_mission_ = this->create_service<autoware_parking_srvs::srv::ParkingMissionPlan>(
    "/service/plan_parking_mission",
    std::bind(&AutoParkingPlanner::parkingMissionPlanCallback, this, _1, _2, _3),
    rmw_qos_profile_services_default, cb_group_);

  freespaceplane_client_ = this->create_client<autoware_parking_srvs::srv::FreespacePlan>(
    "/planning/scenario_planning/parking/freespace_planner/service/freespace_plan",
    rmw_qos_profile_services_default, cb_group_nested_);

  rclcpp::QoS durable_qos{1};
  durable_qos.transient_local();
  lookahead_pose_publisher_ = create_publisher<PoseStamped>("debug/lookahead_pose", durable_qos);
}

void AutoParkingPlanner::mapCallback(
  const autoware_auto_mapping_msgs::msg::HADMapBin::ConstSharedPtr msg)
{
  sub_msgs_.map_ptr = msg;
}

void AutoParkingPlanner::stateCallback(
  const autoware_auto_system_msgs::msg::AutowareState::ConstSharedPtr msg)
{
  sub_msgs_.state_ptr = msg;
}

void AutoParkingPlanner::trajCallback(
  const autoware_auto_planning_msgs::msg::Trajectory::ConstSharedPtr msg)
{
  sub_msgs_.traj_ptr_ = msg;
}

void AutoParkingPlanner::velocityCallback(
  const autoware_auto_vehicle_msgs::msg::VelocityReport::ConstSharedPtr msg)
{
  sub_msgs_.velocity_ptr_ = msg;
}

bool AutoParkingPlanner::transformPose(
  const geometry_msgs::msg::PoseStamped & input_pose, geometry_msgs::msg::PoseStamped * output_pose,
  const std::string target_frame) const
{
  geometry_msgs::msg::TransformStamped transform;
  try {
    transform =
      tf_buffer_.lookupTransform(target_frame, input_pose.header.frame_id, tf2::TimePointZero);
    tf2::doTransform(input_pose, *output_pose, transform);
    return true;
  } catch (tf2::TransformException & ex) {
    RCLCPP_WARN(get_logger(), "%s", ex.what());
    return false;
  }
}

geometry_msgs::msg::PoseStamped AutoParkingPlanner::getEgoVehiclePose() const
{
  geometry_msgs::msg::PoseStamped base_link_origin;
  base_link_origin.header.frame_id = base_link_frame_;
  base_link_origin.pose.position.x = 0;
  base_link_origin.pose.position.y = 0;
  base_link_origin.pose.position.z = 0;
  base_link_origin.pose.orientation.x = 0;
  base_link_origin.pose.orientation.y = 0;
  base_link_origin.pose.orientation.z = 0;
  base_link_origin.pose.orientation.w = 1;

  geometry_msgs::msg::PoseStamped ego_vehicle_pose;
  transformPose(base_link_origin, &ego_vehicle_pose, map_frame_);
  return ego_vehicle_pose;
}

bool AutoParkingPlanner::parkingMissionPlanCallback(
  const std::shared_ptr<rmw_request_id_t> request_header,
  const std::shared_ptr<autoware_parking_srvs::srv::ParkingMissionPlan::Request> request,
  std::shared_ptr<autoware_parking_srvs::srv::ParkingMissionPlan::Response> response)
{
  (void)request_header;
  (void)response;
  boost::optional<HADMapRoute> route = boost::none;

  RCLCPP_INFO_STREAM(
    get_logger(), "reciedved ParkingMissionPlan srv request: type " << request->type);

  if (parking_map_info_ == boost::none) {
    prepare();
  }

  PlanningResult result;
  if (request->type == request->CIRCULAR) {
    waitUntilPreviousRouteFinished();
    result = planCircularRoute();
  } else if (request->type == request->PREPARKING) {
    result = planPreparkingRoute();
  } else if (request->type == request->PARKING) {
    waitUntilPreviousRouteFinished();
    result = planParkingRoute();
  } else {
    throw std::logic_error("not implemented yet");
  }

  response->next_type = result.next_phase;
  response->route = result.route;
  response->success = result.success;

  previous_phase_ = result.next_phase;
  previous_route_ = result.route;
  return true;
}

bool AutoParkingPlanner::waitUntilPreviousRouteFinished() const
{
  RCLCPP_INFO_STREAM(get_logger(), "waiting for preivous route finished...");
  if (!previous_route_) {
    return true;
  }

  if (!sub_msgs_.state_ptr) {
    rclcpp::sleep_for(std::chrono::milliseconds(500));
    if (!sub_msgs_.state_ptr) {
      return false;
    }
  }

  while (true) {
    RCLCPP_INFO_STREAM(get_logger(), "waiting now...");
    const double dist_error =
      tier4_autoware_utils::calcDistance2d(getEgoVehiclePose(), previous_route_->goal_pose);
    RCLCPP_INFO_STREAM(get_logger(), "dist error:" << dist_error);
    RCLCPP_INFO_STREAM(get_logger(), "aw state:" << sub_msgs_.state_ptr->state);
    rclcpp::sleep_for(std::chrono::milliseconds(300));
    if (sub_msgs_.state_ptr->state == AutowareState::WAITING_FOR_ROUTE) {
      if (dist_error < 1.5) {
        RCLCPP_INFO_STREAM(get_logger(), "preivous route finished!");
        return true;
      }
    }
  }
}

std::vector<Pose> AutoParkingPlanner::askFeasibleGoalIndex(
  const Pose & start, const std::vector<Pose> & goal_poses) const
{
  auto freespace_plan_req = std::make_shared<autoware_parking_srvs::srv::FreespacePlan::Request>();

  for (const auto & goal : goal_poses) {
    if (!config_.check_goal_only) {
      PoseStamped start_pose;
      start_pose.header.frame_id = map_frame_;
      start_pose.pose = start;
      freespace_plan_req->start_poses.push_back(start_pose);
    }

    PoseStamped goal_pose;
    goal_pose.header.frame_id = map_frame_;
    goal_pose.pose = goal;
    freespace_plan_req->goal_poses.push_back(goal_pose);
  }

  const auto f = freespaceplane_client_->async_send_request(freespace_plan_req);
  if (std::future_status::ready != f.wait_for(std::chrono::seconds(10))) {
    RCLCPP_FATAL_STREAM(get_logger(), "took to long time to obtain freespace planning result");
  }

  const auto & result = f.get();
  RCLCPP_INFO_STREAM(get_logger(), "Obtained fresult from the freespace planning server.");

  std::vector<Pose> feasible_goal_poses;
  for (size_t idx = 0; idx < result->successes.size(); idx++) {
    if (result->successes.at(idx)) {
      feasible_goal_poses.push_back(goal_poses.at(idx));
    }
  }
  return feasible_goal_poses;
}

}  // namespace auto_parking_planner

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(auto_parking_planner::AutoParkingPlanner)
