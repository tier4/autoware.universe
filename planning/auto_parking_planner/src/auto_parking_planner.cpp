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

#include <future>
#include <stdexcept>

namespace auto_parking_planner
{

AutoParkingPlanner::AutoParkingPlanner(const rclcpp::NodeOptions & node_options)
: rclcpp::Node("auto_parking_planner_node", node_options),
  tf_buffer_(get_clock()),
  tf_listener_(tf_buffer_),
  previous_mode_(boost::none)
{
  map_frame_ = declare_parameter("map_frame", "map");
  base_link_frame_ = declare_parameter("base_link_frame", "base_link");

  {  // set node config
    config_.lookahead_length = declare_parameter("lookahead_length", 4.0);
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

  twist_subscriber_ = create_subscription<geometry_msgs::msg::TwistStamped>(
    "~/input/twist", rclcpp::QoS{1}, std::bind(&AutoParkingPlanner::twistCallback, this, _1));

  traj_subscriber_ = create_subscription<Trajectory>(
    "~/input/trajectory", rclcpp::QoS{1}, std::bind(&AutoParkingPlanner::trajCallback, this, _1));

  srv_parking_mission_ = this->create_service<autoware_parking_srvs::srv::ParkingMissionPlan>(
    "/service/plan_parking_mission",
    std::bind(&AutoParkingPlanner::parkingMissionPlanCallback, this, _1, _2, _3),
    rmw_qos_profile_services_default);

  freespaceplane_client_ = this->create_client<autoware_parking_srvs::srv::FreespacePlan>(
    "/planning/scenario_planning/parking/freespace_planner/service/freespace_plan",
    rmw_qos_profile_services_default, cb_group_nested_);
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

void AutoParkingPlanner::twistCallback(const geometry_msgs::msg::TwistStamped::ConstSharedPtr msg)
{
  sub_msgs_.twist_ptr_ = msg;
}

void AutoParkingPlanner::trajCallback(
  const autoware_auto_planning_msgs::msg::Trajectory::ConstSharedPtr msg)
{
  sub_msgs_.traj_ptr_ = msg;
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

  /*
  std::string next_plan_type;
  if (request->type == request->CIRCULAR) {
    if (previous_mode_ != request->PARKING) {
      waitForPreviousRouteFinished();
    }
    next_plan_type = planCircularRoute(route);
  } else if (request->type == request->PREPARKING) {
    // Don't have to waitForPreviousRouteFinished
    next_plan_type = planPreparkingRoute(route);
  } else if (request->type == request->PARKING) {
    waitForPreviousRouteFinished();
    next_plan_type = planParkingRoute(route);
  } else {
    RCLCPP_WARN(get_logger(), "type field seemes to be invaid value.");
    next_plan_type = request->END;
  }
  const bool prohibit_publish = (route == boost::none);

  // create debug goal pose
  if (!prohibit_publish) {
    auto debug_pose = PoseStamped();
    debug_pose.header.stamp = this->now();
    debug_pose.header.frame_id = map_frame_;
    debug_pose.pose = route.get().goal_pose;
    debug_goal_pose_publisher_->publish(debug_pose);
  }

  previous_mode_ = request->type;
  previous_route_ = route;
  if (!prohibit_publish) {
    response->route = route.get();
  }

  response->next_type = next_plan_type;
  response->prohibit_publish = prohibit_publish;
  */
  const auto result = planCircularRoute();
  response->next_type = result.next_phase;
  response->route = result.route;
  response->success = result.success;
  return true;
}

/*
AutoParkingPlanner::AutoParkingPlanner(const rclcpp::NodeOptions & node_options)
: rclcpp::Node("auto_parking_planner_node", node_options),
  tf_buffer_(get_clock()),
  tf_listener_(tf_buffer_),
  previous_mode_(boost::none),
  previous_route_(boost::none)
{
  map_frame_ = declare_parameter("map_frame", "map");
  base_link_frame_ = declare_parameter("base_link_frame", "base_link");

  {  // set node config
    config_.lookahead_length = declare_parameter("lookahead_length", 4.0);
    config_.reedsshepp_threashold_length = declare_parameter("reedsshepp_threashold_length", 6.0);
    config_.euclid_threashold_length = declare_parameter("euclid_threashold_length", 2.0);
    config_.reedsshepp_radius = declare_parameter("reedsshepp_radius", 5.0);
  }

  rclcpp::QoS durable_qos{1};
  durable_qos.transient_local();
  route_publisher_ =
    create_publisher<autoware_auto_planning_msgs::msg::HADMapRoute>("~/output/route", durable_qos);
  debug_goal_pose_publisher_ =
    create_publisher<PoseStamped>("/ishida/preparking/start_pose", durable_qos);
  debug_target_poses_publisher_ =
    create_publisher<PoseArray>("/ishida/preparking/goal_poses", durable_qos);
  debug_target_poses_publisher2_ =
    create_publisher<PoseArray>("/ishida/parking/goal_poses", durable_qos);
  debug_trajectory_publisher_ =
    create_publisher<Trajectory>("/ishida/debug_trajectory", durable_qos);

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

  twist_subscriber_ = create_subscription<geometry_msgs::msg::TwistStamped>(
    "~/input/twist", rclcpp::QoS{1}, std::bind(&AutoParkingPlanner::twistCallback, this, _1));

  traj_subscriber_ = create_subscription<Trajectory>(
    "~/input/trajectory", rclcpp::QoS{1}, std::bind(&AutoParkingPlanner::trajCallback, this, _1));

  srv_parking_mission_ = this->create_service<autoware_parking_srvs::srv::ParkingMissionPlan>(
    "service/plan_parking_mission",
    std::bind(&AutoParkingPlanner::parkingMissionPlanCallback, this, _1, _2, _3),
    rmw_qos_profile_services_default, cb_group_);

  freespaceplane_client_ = this->create_client<autoware_parking_srvs::srv::FreespacePlan>(
    "/planning/scenario_planning/parking/freespace_planner/service/freespace_plan",
    rmw_qos_profile_services_default, cb_group_nested_);

  freespace_plan_timeout_ = 1.0;
}

bool AutoParkingPlanner::transformPose(
  const geometry_msgs::msg::PoseStamped & input_pose, geometry_msgs::msg::PoseStamped * output_pose,
  const std::string target_frame)
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

geometry_msgs::msg::PoseStamped AutoParkingPlanner::getEgoVehiclePose()
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
  //  transform base_link frame origin to map_frame to get vehicle positions
  transformPose(base_link_origin, &ego_vehicle_pose, map_frame_);
  return ego_vehicle_pose;
}

void AutoParkingPlanner::mapCallback(
  const autoware_auto_mapping_msgs::msg::HADMapBin::ConstSharedPtr msg)
{
  RCLCPP_INFO(get_logger(), "[ishida] new map set");
  lanelet_map_ptr_ = std::make_shared<lanelet::LaneletMap>();
  lanelet::utils::conversion::fromBinMsg(
    *msg, lanelet_map_ptr_, &traffic_rules_ptr_, &routing_graph_ptr_);
  lanelet::ConstLanelets all_lanelets = lanelet::utils::query::laneletLayer(lanelet_map_ptr_);
  road_lanelets_ = lanelet::utils::query::roadLanelets(all_lanelets);
}

void AutoParkingPlanner::trajCallback(const Trajectory::ConstSharedPtr msg)
{
  // check trajectory validity
  // it is probably the problem inside lane driving, but Motion planning node
  // publishes strange trajectory once before the valid trajectory
  // To avoid subscribing the strange one, we will filter it as bellow
  const Pose terminal_traj_point = msg->points.back().pose;
  const Pose terminal_llt_center = computeLaneletCenterPose(circling_path_.back());
  const auto dist_err =
    tier4_autoware_utils::calcDistance2d(terminal_traj_point, terminal_llt_center);
  if (dist_err > 3.0) {
    RCLCPP_INFO_STREAM(
      get_logger(), "trajectory is subscribed "
                      << "but the obtained trajectory's terminal point is far away "
                      << "from the circular route's terminal point, we will ignore this msg");
    return;
  }

  this->trajectory_ptr_ = msg;
}

bool AutoParkingPlanner::reset()
{
  if (!lanelet_map_ptr_) {
    RCLCPP_WARN(get_logger(), "reset failed as lanelet map is not subscribed yet.");
    return false;
  }
  const auto all_parking_lots = lanelet::utils::query::getAllParkingLots(lanelet_map_ptr_);
  const auto nearest_parking_lot = all_parking_lots[0];  // TODO(HiroIshida) temp
  extractSubGraphInfo(nearest_parking_lot);
  parking_target_poses_ = findParkingTargetPoses();
  trajectory_ptr_ = nullptr;
  return true;
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

  const bool is_initilized = (sub_graph_ptr_ != nullptr);
  if (!is_initilized) {
    const bool success = reset();
    if (!success) {
      RCLCPP_INFO_STREAM(rclcpp::get_logger("ishida_debug"), "cannot reset subgraph");
      return false;
    }
  }

  std::string next_plan_type;
  if (request->type == request->CIRCULAR) {
    if (previous_mode_ != request->PARKING) {
      waitForPreviousRouteFinished();
    }
    next_plan_type = planCircularRoute(route);
  } else if (request->type == request->PREPARKING) {
    // Don't have to waitForPreviousRouteFinished
    next_plan_type = planPreparkingRoute(route);
  } else if (request->type == request->PARKING) {
    waitForPreviousRouteFinished();
    next_plan_type = planParkingRoute(route);
  } else {
    RCLCPP_WARN(get_logger(), "type field seemes to be invaid value.");
    next_plan_type = request->END;
  }
  RCLCPP_INFO_STREAM(get_logger(), "processed");

  const bool prohibit_publish = (route == boost::none);

  // create debug goal pose
  if (!prohibit_publish) {
    auto debug_pose = PoseStamped();
    debug_pose.header.stamp = this->now();
    debug_pose.header.frame_id = map_frame_;
    debug_pose.pose = route.get().goal_pose;
    debug_goal_pose_publisher_->publish(debug_pose);
  }

  previous_mode_ = request->type;
  previous_route_ = route;
  if (!prohibit_publish) {
    response->route = route.get();
  }
  response->next_type = next_plan_type;
  response->prohibit_publish = prohibit_publish;
  return true;
}

*/
}  // namespace auto_parking_planner

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(auto_parking_planner::AutoParkingPlanner)
