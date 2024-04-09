// Copyright 2019 Autoware Foundation
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

#include "mission_planner.hpp"

#include <lanelet2_extension/utility/message_conversion.hpp>
#include <lanelet2_extension/utility/query.hpp>
#include <lanelet2_extension/utility/route_checker.hpp>
#include <lanelet2_extension/utility/utilities.hpp>

#include <autoware_adapi_v1_msgs/srv/set_route.hpp>
#include <autoware_adapi_v1_msgs/srv/set_route_points.hpp>
#include <autoware_auto_mapping_msgs/msg/had_map_bin.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <lanelet2_core/geometry/LineString.h>

#include <algorithm>
#include <array>
#include <random>
#include <utility>

namespace
{

using autoware_adapi_v1_msgs::msg::RoutePrimitive;
using autoware_adapi_v1_msgs::msg::RouteSegment;
using autoware_planning_msgs::msg::LaneletPrimitive;
using autoware_planning_msgs::msg::LaneletSegment;

LaneletPrimitive convert(const RoutePrimitive & in)
{
  LaneletPrimitive out;
  out.id = in.id;
  out.primitive_type = in.type;
  return out;
}

LaneletSegment convert(const RouteSegment & in)
{
  LaneletSegment out;
  out.primitives.reserve(in.alternatives.size() + 1);
  out.primitives.push_back(convert(in.preferred));
  for (const auto & primitive : in.alternatives) {
    out.primitives.push_back(convert(primitive));
  }
  out.preferred_primitive = convert(in.preferred);
  return out;
}

std::array<uint8_t, 16> generate_random_id()
{
  static std::independent_bits_engine<std::mt19937, 8, uint8_t> engine(std::random_device{}());
  std::array<uint8_t, 16> id;
  std::generate(id.begin(), id.end(), std::ref(engine));
  return id;
}

}  // namespace

namespace mission_planner
{

MissionPlanner::MissionPlanner(const rclcpp::NodeOptions & options)
: Node("mission_planner", options),
  arrival_checker_(this),
  plugin_loader_("mission_planner", "mission_planner::PlannerPlugin"),
  tf_buffer_(get_clock()),
  tf_listener_(tf_buffer_),
  odometry_(nullptr),
  map_ptr_(nullptr),
  reroute_availability_(nullptr),
  normal_route_(nullptr),
  mrm_route_(nullptr)
{
  map_frame_ = declare_parameter<std::string>("map_frame");
  reroute_time_threshold_ = declare_parameter<double>("reroute_time_threshold");
  minimum_reroute_length_ = declare_parameter<double>("minimum_reroute_length");

  planner_ = plugin_loader_.createSharedInstance("mission_planner::lanelet2::DefaultPlanner");
  planner_->initialize(this);

  sub_odometry_ = create_subscription<Odometry>(
    "/localization/kinematic_state", rclcpp::QoS(1),
    std::bind(&MissionPlanner::on_odometry, this, std::placeholders::_1));
  sub_reroute_availability_ = create_subscription<RerouteAvailability>(
    "/planning/scenario_planning/lane_driving/behavior_planning/behavior_path_planner/output/"
    "is_reroute_available",
    rclcpp::QoS(1),
    std::bind(&MissionPlanner::on_reroute_availability, this, std::placeholders::_1));

  const auto durable_qos = rclcpp::QoS(1).transient_local();
  sub_vector_map_ = create_subscription<HADMapBin>(
    "input/vector_map", durable_qos,
    std::bind(&MissionPlanner::on_map, this, std::placeholders::_1));
  sub_modified_goal_ = create_subscription<PoseWithUuidStamped>(
    "input/modified_goal", durable_qos,
    std::bind(&MissionPlanner::on_modified_goal, this, std::placeholders::_1));

  pub_marker_ = create_publisher<MarkerArray>("debug/route_marker", durable_qos);

  const auto adaptor = component_interface_utils::NodeAdaptor(this);
  adaptor.init_pub(pub_state_);
  adaptor.init_pub(pub_route_);
  adaptor.init_pub(pub_normal_route_);
  adaptor.init_pub(pub_mrm_route_);
  adaptor.init_srv(srv_clear_route_, this, &MissionPlanner::on_clear_route);
  adaptor.init_srv(srv_set_route_, this, &MissionPlanner::on_set_route);
  adaptor.init_srv(srv_set_route_points_, this, &MissionPlanner::on_set_route_points);
  adaptor.init_srv(srv_change_route_, this, &MissionPlanner::on_change_route);
  adaptor.init_srv(srv_change_route_points_, this, &MissionPlanner::on_change_route_points);
  adaptor.init_srv(srv_set_mrm_route_, this, &MissionPlanner::on_set_mrm_route);
  adaptor.init_srv(srv_clear_mrm_route_, this, &MissionPlanner::on_clear_mrm_route);

  // Route state will be published when the node gets ready for route api after initialization,
  // otherwise the mission planner rejects the request for the API.
  data_check_timer_ = create_wall_timer(
    std::chrono::milliseconds(100), std::bind(&MissionPlanner::checkInitialization, this));

  logger_configure_ = std::make_unique<tier4_autoware_utils::LoggerLevelConfigure>(this);
}

void MissionPlanner::checkInitialization()
{
  if (!planner_->ready()) {
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 5000, "waiting lanelet map... Route API is not ready.");
    return;
  }
  if (!odometry_) {
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 5000, "waiting odometry... Route API is not ready.");
    return;
  }

  // All data is ready. Now API is available.
  RCLCPP_INFO(get_logger(), "Route API is ready.");
  change_state(RouteState::Message::UNSET);
  data_check_timer_->cancel();  // stop timer callback
}

void MissionPlanner::on_odometry(const Odometry::ConstSharedPtr msg)
{
  odometry_ = msg;

  // NOTE: Do not check in the changing state as goal may change.
  if (state_.state == RouteState::Message::SET) {
    PoseStamped pose;
    pose.header = odometry_->header;
    pose.pose = odometry_->pose.pose;
    if (arrival_checker_.is_arrived(pose)) {
      change_state(RouteState::Message::ARRIVED);
    }
  }
}

void MissionPlanner::on_reroute_availability(const RerouteAvailability::ConstSharedPtr msg)
{
  reroute_availability_ = msg;
}

void MissionPlanner::on_map(const HADMapBin::ConstSharedPtr msg)
{
  map_ptr_ = msg;
  lanelet_map_ptr_ = std::make_shared<lanelet::LaneletMap>();
  lanelet::utils::conversion::fromBinMsg(*map_ptr_, lanelet_map_ptr_);
}

PoseStamped MissionPlanner::transform_pose(const PoseStamped & input)
{
  PoseStamped output;
  geometry_msgs::msg::TransformStamped transform;
  try {
    transform = tf_buffer_.lookupTransform(map_frame_, input.header.frame_id, tf2::TimePointZero);
    tf2::doTransform(input, output, transform);
    return output;
  } catch (tf2::TransformException & error) {
    throw component_interface_utils::TransformError(error.what());
  }
}

void MissionPlanner::clear_route()
{
  arrival_checker_.set_goal();
  planner_->clearRoute();
  normal_route_ = nullptr;
  // TODO(Takagi, Isamu): publish an empty route here
}

void MissionPlanner::clear_mrm_route()
{
  mrm_route_ = nullptr;
}

void MissionPlanner::change_route(const LaneletRoute & route)
{
  PoseWithUuidStamped goal;
  goal.header = route.header;
  goal.pose = route.goal_pose;
  goal.uuid = route.uuid;

  arrival_checker_.set_goal(goal);
  pub_route_->publish(route);
  pub_normal_route_->publish(route);
  pub_marker_->publish(planner_->visualize(route));
  planner_->updateRoute(route);

  // update normal route
  normal_route_ = std::make_shared<LaneletRoute>(route);
}

void MissionPlanner::change_mrm_route(const LaneletRoute & route)
{
  PoseWithUuidStamped goal;
  goal.header = route.header;
  goal.pose = route.goal_pose;
  goal.uuid = route.uuid;

  arrival_checker_.set_goal(goal);
  pub_route_->publish(route);
  pub_mrm_route_->publish(route);
  pub_marker_->publish(planner_->visualize(route));
  planner_->updateRoute(route);

  // update emergency route
  mrm_route_ = std::make_shared<LaneletRoute>(route);
}

LaneletRoute MissionPlanner::create_route(
  const std_msgs::msg::Header & header,
  const std::vector<autoware_adapi_v1_msgs::msg::RouteSegment> & route_segments,
  const geometry_msgs::msg::Pose & goal_pose, const bool allow_goal_modification)
{
  PoseStamped goal_pose_stamped;
  goal_pose_stamped.header = header;
  goal_pose_stamped.pose = goal_pose;

  // Convert route.
  LaneletRoute route;
  route.start_pose = odometry_->pose.pose;
  route.goal_pose = transform_pose(goal_pose_stamped).pose;
  for (const auto & segment : route_segments) {
    route.segments.push_back(convert(segment));
  }
  route.header.stamp = header.stamp;
  route.header.frame_id = map_frame_;
  route.uuid.uuid = generate_random_id();
  route.allow_modification = allow_goal_modification;

  return route;
}

LaneletRoute MissionPlanner::create_route(
  const std_msgs::msg::Header & header, const std::vector<geometry_msgs::msg::Pose> & waypoints,
  const geometry_msgs::msg::Pose & goal_pose, const bool allow_goal_modification)
{
  // Use temporary pose stamped for transform.
  PoseStamped pose;
  pose.header = header;

  // Convert route points.
  PlannerPlugin::RoutePoints points;
  points.push_back(odometry_->pose.pose);
  for (const auto & waypoint : waypoints) {
    pose.pose = waypoint;
    points.push_back(transform_pose(pose).pose);
  }
  pose.pose = goal_pose;
  points.push_back(transform_pose(pose).pose);

  // Plan route.
  LaneletRoute route = planner_->plan(points);
  route.header.stamp = header.stamp;
  route.header.frame_id = map_frame_;
  route.uuid.uuid = generate_random_id();
  route.allow_modification = allow_goal_modification;

  return route;
}

LaneletRoute MissionPlanner::create_route(const SetRoute::Service::Request::SharedPtr req)
{
  const auto & header = req->header;
  const auto & route_segments = req->segments;
  const auto & goal_pose = req->goal;
  const auto & allow_goal_modification = req->option.allow_goal_modification;

  return create_route(header, route_segments, goal_pose, allow_goal_modification);
}

LaneletRoute MissionPlanner::create_route(const SetRoutePoints::Service::Request::SharedPtr req)
{
  const auto & header = req->header;
  const auto & waypoints = req->waypoints;
  const auto & goal_pose = req->goal;
  const auto & allow_goal_modification = req->option.allow_goal_modification;

  return create_route(header, waypoints, goal_pose, allow_goal_modification);
}

void MissionPlanner::change_state(RouteState::Message::_state_type state)
{
  state_.stamp = now();
  state_.state = state;
  pub_state_->publish(state_);
}

// NOTE: The route interface should be mutually exclusive by callback group.
void MissionPlanner::on_clear_route(
  const ClearRoute::Service::Request::SharedPtr, const ClearRoute::Service::Response::SharedPtr res)
{
  clear_route();
  change_state(RouteState::Message::UNSET);
  res->status.success = true;
}

// NOTE: The route interface should be mutually exclusive by callback group.
void MissionPlanner::on_set_route(
  const SetRoute::Service::Request::SharedPtr req, const SetRoute::Service::Response::SharedPtr res)
{
  using ResponseCode = autoware_adapi_v1_msgs::srv::SetRoute::Response;

  if (state_.state != RouteState::Message::UNSET) {
    throw component_interface_utils::ServiceException(
      ResponseCode::ERROR_ROUTE_EXISTS, "The route is already set.");
  }
  if (!planner_->ready()) {
    throw component_interface_utils::ServiceException(
      ResponseCode::ERROR_PLANNER_UNREADY, "The planner is not ready.");
  }
  if (!odometry_) {
    throw component_interface_utils::ServiceException(
      ResponseCode::ERROR_PLANNER_UNREADY, "The vehicle pose is not received.");
  }
  if (mrm_route_) {
    throw component_interface_utils::ServiceException(
      ResponseCode::ERROR_INVALID_STATE, "Cannot reroute in the emergency state.");
  }

  // Convert request to a new route.
  const auto route = create_route(req);

  // Check planned routes
  if (route.segments.empty()) {
    throw component_interface_utils::ServiceException(
      ResponseCode::ERROR_PLANNER_FAILED, "The planned route is empty.");
  }

  // Update route.
  change_route(route);
  change_state(RouteState::Message::SET);
  res->status.success = true;
}

// NOTE: The route interface should be mutually exclusive by callback group.
void MissionPlanner::on_set_route_points(
  const SetRoutePoints::Service::Request::SharedPtr req,
  const SetRoutePoints::Service::Response::SharedPtr res)
{
  using ResponseCode = autoware_adapi_v1_msgs::srv::SetRoutePoints::Response;

  if (state_.state != RouteState::Message::UNSET) {
    throw component_interface_utils::ServiceException(
      ResponseCode::ERROR_ROUTE_EXISTS, "The route is already set.");
  }
  if (!planner_->ready()) {
    throw component_interface_utils::ServiceException(
      ResponseCode::ERROR_PLANNER_UNREADY, "The planner is not ready.");
  }
  if (!odometry_) {
    throw component_interface_utils::ServiceException(
      ResponseCode::ERROR_PLANNER_UNREADY, "The vehicle pose is not received.");
  }
  if (mrm_route_) {
    throw component_interface_utils::ServiceException(
      ResponseCode::ERROR_INVALID_STATE, "Cannot reroute in the emergency state.");
  }

  // Plan route.
  const auto route = create_route(req);

  // Check planned routes
  if (route.segments.empty()) {
    throw component_interface_utils::ServiceException(
      ResponseCode::ERROR_PLANNER_FAILED, "The planned route is empty.");
  }

  // Update route.
  change_route(route);
  change_state(RouteState::Message::SET);
  res->status.success = true;
}

// NOTE: The route interface should be mutually exclusive by callback group.
void MissionPlanner::on_set_mrm_route(
  const SetMrmRoute::Service::Request::SharedPtr req,
  const SetMrmRoute::Service::Response::SharedPtr res)
{
  using ResponseCode = autoware_adapi_v1_msgs::srv::SetRoutePoints::Response;

  if (!planner_->ready()) {
    throw component_interface_utils::ServiceException(
      ResponseCode::ERROR_PLANNER_UNREADY, "The planner is not ready.");
  }
  if (!odometry_) {
    throw component_interface_utils::ServiceException(
      ResponseCode::ERROR_PLANNER_UNREADY, "The vehicle pose is not received.");
  }
  if (reroute_availability_ && !reroute_availability_->availability) {
    throw component_interface_utils::ServiceException(
      ResponseCode::ERROR_INVALID_STATE, "Cannot reroute as the planner is not in lane following.");
  }

  const auto prev_state = state_.state;
  change_state(RouteState::Message::CHANGING);

  // Plan route.
  const auto new_route = create_route(req);

  if (new_route.segments.empty()) {
    change_state(prev_state);
    throw component_interface_utils::ServiceException(
      ResponseCode::ERROR_PLANNER_FAILED, "Failed to plan a new route.");
  }

  // check route safety
  // step1. if in mrm state, check with mrm route
  if (mrm_route_) {
    if (check_reroute_safety(*mrm_route_, new_route)) {
      // success to reroute
      change_mrm_route(new_route);
      res->status.success = true;
    } else {
      // failed to reroute
      change_mrm_route(*mrm_route_);
      res->status.success = false;
    }
    change_state(RouteState::Message::SET);
    RCLCPP_INFO(get_logger(), "Route is successfully changed with the modified goal");
    return;
  }

  if (!normal_route_) {
    // if it does not set normal route, just use the new planned route
    change_mrm_route(new_route);
    change_state(RouteState::Message::SET);
    res->status.success = true;
    RCLCPP_INFO(get_logger(), "MRM route is successfully changed with the modified goal");
    return;
  }

  // step2. if not in mrm state, check with normal route
  if (check_reroute_safety(*normal_route_, new_route)) {
    // success to reroute
    change_mrm_route(new_route);
    res->status.success = true;
  } else {
    // Failed to reroute
    change_route(*normal_route_);
    res->status.success = false;
  }
  change_state(RouteState::Message::SET);
}

// NOTE: The route interface should be mutually exclusive by callback group.
void MissionPlanner::on_clear_mrm_route(
  const ClearMrmRoute::Service::Request::SharedPtr,
  const ClearMrmRoute::Service::Response::SharedPtr res)
{
  using ResponseCode = autoware_adapi_v1_msgs::srv::SetRoutePoints::Response;

  if (!planner_->ready()) {
    change_state(RouteState::Message::SET);
    throw component_interface_utils::ServiceException(
      ResponseCode::ERROR_PLANNER_UNREADY, "The planner is not ready.");
  }
  if (!odometry_) {
    throw component_interface_utils::ServiceException(
      ResponseCode::ERROR_PLANNER_UNREADY, "The vehicle pose is not received.");
  }
  if (!mrm_route_) {
    throw component_interface_utils::NoEffectWarning("MRM route is not set");
  }
  if (
    state_.state == RouteState::Message::SET && reroute_availability_ &&
    !reroute_availability_->availability) {
    throw component_interface_utils::ServiceException(
      ResponseCode::ERROR_INVALID_STATE,
      "Cannot clear MRM route as the planner is not lane following before arriving at the goal.");
  }

  change_state(RouteState::Message::CHANGING);

  if (!normal_route_) {
    clear_mrm_route();
    change_state(RouteState::Message::UNSET);
    res->status.success = true;
    return;
  }

  // check route safety
  if (check_reroute_safety(*mrm_route_, *normal_route_)) {
    clear_mrm_route();
    change_route(*normal_route_);
    change_state(RouteState::Message::SET);
    res->status.success = true;
    return;
  }

  // reroute with normal goal
  const std::vector<geometry_msgs::msg::Pose> empty_waypoints;
  const auto new_route = create_route(
    odometry_->header, empty_waypoints, normal_route_->goal_pose,
    normal_route_->allow_modification);

  // check new route safety
  if (new_route.segments.empty() || !check_reroute_safety(*mrm_route_, new_route)) {
    // failed to create a new route
    RCLCPP_ERROR(get_logger(), "Reroute with normal goal failed.");
    change_mrm_route(*mrm_route_);
    change_state(RouteState::Message::SET);
    res->status.success = false;
  } else {
    clear_mrm_route();
    change_route(new_route);
    change_state(RouteState::Message::SET);
    res->status.success = true;
  }
}

void MissionPlanner::on_modified_goal(const ModifiedGoal::Message::ConstSharedPtr msg)
{
  RCLCPP_INFO(get_logger(), "Received modified goal.");

  if (state_.state != RouteState::Message::SET) {
    RCLCPP_ERROR(get_logger(), "The route hasn't set yet. Cannot reroute.");
    return;
  }
  if (!planner_->ready()) {
    RCLCPP_ERROR(get_logger(), "The planner is not ready.");
    return;
  }
  if (!odometry_) {
    RCLCPP_ERROR(get_logger(), "The vehicle pose is not received.");
    return;
  }
  if (!normal_route_) {
    RCLCPP_ERROR(get_logger(), "Normal route has not set yet.");
    return;
  }

  if (mrm_route_ && mrm_route_->uuid == msg->uuid) {
    // set to changing state
    change_state(RouteState::Message::CHANGING);

    const std::vector<geometry_msgs::msg::Pose> empty_waypoints;
    auto new_route =
      create_route(msg->header, empty_waypoints, msg->pose, mrm_route_->allow_modification);
    // create_route generate new uuid, so set the original uuid again to keep that.
    new_route.uuid = msg->uuid;
    if (new_route.segments.empty()) {
      change_mrm_route(*mrm_route_);
      change_state(RouteState::Message::SET);
      RCLCPP_ERROR(get_logger(), "The planned MRM route is empty.");
      return;
    }

    change_mrm_route(new_route);
    change_state(RouteState::Message::SET);
    RCLCPP_INFO(get_logger(), "Changed the MRM route with the modified goal");
    return;
  }

  if (normal_route_->uuid == msg->uuid) {
    // set to changing state
    change_state(RouteState::Message::CHANGING);

    const std::vector<geometry_msgs::msg::Pose> empty_waypoints;
    auto new_route =
      create_route(msg->header, empty_waypoints, msg->pose, normal_route_->allow_modification);
    // create_route generate new uuid, so set the original uuid again to keep that.
    new_route.uuid = msg->uuid;
    if (new_route.segments.empty()) {
      change_route(*normal_route_);
      change_state(RouteState::Message::SET);
      RCLCPP_ERROR(get_logger(), "The planned route is empty.");
      return;
    }

    change_route(new_route);
    change_state(RouteState::Message::SET);
    RCLCPP_INFO(get_logger(), "Changed the route with the modified goal");
    return;
  }

  RCLCPP_ERROR(get_logger(), "Goal uuid is incorrect.");
}

void MissionPlanner::on_change_route(
  const SetRoute::Service::Request::SharedPtr req, const SetRoute::Service::Response::SharedPtr res)
{
  using ResponseCode = autoware_adapi_v1_msgs::srv::SetRoute::Response;

  if (state_.state != RouteState::Message::SET) {
    throw component_interface_utils::ServiceException(
      ResponseCode::ERROR_INVALID_STATE, "The route hasn't set yet. Cannot reroute.");
  }
  if (!planner_->ready()) {
    throw component_interface_utils::ServiceException(
      ResponseCode::ERROR_PLANNER_UNREADY, "The planner is not ready.");
  }
  if (!odometry_) {
    throw component_interface_utils::ServiceException(
      ResponseCode::ERROR_PLANNER_UNREADY, "The vehicle pose is not received.");
  }
  if (!normal_route_) {
    throw component_interface_utils::ServiceException(
      ResponseCode::ERROR_PLANNER_UNREADY, "Normal route is not set.");
  }
  if (mrm_route_) {
    throw component_interface_utils::ServiceException(
      ResponseCode::ERROR_INVALID_STATE, "Cannot reroute in the emergency state.");
  }
  if (reroute_availability_ && !reroute_availability_->availability) {
    throw component_interface_utils::ServiceException(
      ResponseCode::ERROR_INVALID_STATE, "Cannot reroute as the planner is not in lane following.");
  }

  // set to changing state
  change_state(RouteState::Message::CHANGING);

  // Convert request to a new route.
  const auto new_route = create_route(req);

  // Check planned routes
  if (new_route.segments.empty()) {
    change_route(*normal_route_);
    change_state(RouteState::Message::SET);
    res->status.success = false;
    throw component_interface_utils::ServiceException(
      ResponseCode::ERROR_PLANNER_FAILED, "The planned route is empty.");
  }

  // check route safety
  if (check_reroute_safety(*normal_route_, new_route)) {
    // success to reroute
    change_route(new_route);
    res->status.success = true;
    change_state(RouteState::Message::SET);
  } else {
    // failed to reroute
    change_route(*normal_route_);
    res->status.success = false;
    change_state(RouteState::Message::SET);
    throw component_interface_utils::ServiceException(
      ResponseCode::ERROR_REROUTE_FAILED, "New route is not safe. Reroute failed.");
  }
}

// NOTE: The route interface should be mutually exclusive by callback group.
void MissionPlanner::on_change_route_points(
  const SetRoutePoints::Service::Request::SharedPtr req,
  const SetRoutePoints::Service::Response::SharedPtr res)
{
  using ResponseCode = autoware_adapi_v1_msgs::srv::SetRoutePoints::Response;

  if (state_.state != RouteState::Message::SET) {
    throw component_interface_utils::ServiceException(
      ResponseCode::ERROR_INVALID_STATE, "The route hasn't set yet. Cannot reroute.");
  }
  if (!planner_->ready()) {
    throw component_interface_utils::ServiceException(
      ResponseCode::ERROR_PLANNER_UNREADY, "The planner is not ready.");
  }
  if (!odometry_) {
    throw component_interface_utils::ServiceException(
      ResponseCode::ERROR_PLANNER_UNREADY, "The vehicle pose is not received.");
  }
  if (!normal_route_) {
    throw component_interface_utils::ServiceException(
      ResponseCode::ERROR_PLANNER_UNREADY, "Normal route is not set.");
  }
  if (mrm_route_) {
    throw component_interface_utils::ServiceException(
      ResponseCode::ERROR_INVALID_STATE, "Cannot reroute in the emergency state.");
  }
  if (reroute_availability_ && !reroute_availability_->availability) {
    throw component_interface_utils::ServiceException(
      ResponseCode::ERROR_INVALID_STATE, "Cannot reroute as the planner is not in lane following.");
  }

  change_state(RouteState::Message::CHANGING);

  // Plan route.
  const auto new_route = create_route(req);

  // Check planned routes
  if (new_route.segments.empty()) {
    change_state(RouteState::Message::SET);
    change_route(*normal_route_);
    res->status.success = false;
    throw component_interface_utils::ServiceException(
      ResponseCode::ERROR_PLANNER_FAILED, "The planned route is empty.");
  }

  // check route safety
  if (check_reroute_safety(*normal_route_, new_route)) {
    // success to reroute
    change_route(new_route);
    res->status.success = true;
    change_state(RouteState::Message::SET);
  } else {
    // failed to reroute
    change_route(*normal_route_);
    res->status.success = false;
    change_state(RouteState::Message::SET);
    throw component_interface_utils::ServiceException(
      ResponseCode::ERROR_REROUTE_FAILED, "New route is not safe. Reroute failed.");
  }
}

bool MissionPlanner::check_reroute_safety(
  const LaneletRoute & original_route, const LaneletRoute & target_route)
{
  if (
    original_route.segments.empty() || target_route.segments.empty() || !map_ptr_ ||
    !lanelet_map_ptr_ || !odometry_) {
    RCLCPP_ERROR(get_logger(), "Check reroute safety failed. Route, map or odometry is not set.");
    return false;
  }

  const auto current_velocity = odometry_->twist.twist.linear.x;

  // if vehicle is stopped, do not check safety
  if (current_velocity < 0.01) {
    return true;
  }

  auto hasSamePrimitives = [](
                             const std::vector<LaneletPrimitive> & original_primitives,
                             const std::vector<LaneletPrimitive> & target_primitives) {
    if (original_primitives.size() != target_primitives.size()) {
      return false;
    }

    for (const auto & primitive : original_primitives) {
      const auto has_same = [&](const auto & p) { return p.id == primitive.id; };
      const bool is_same =
        std::find_if(target_primitives.begin(), target_primitives.end(), has_same) !=
        target_primitives.end();
      if (!is_same) {
        return false;
      }
    }
    return true;
  };

  // =============================================================================================
  // NOTE: the target route is calculated while ego is driving on the original route, so basically
  // the first lane of the target route should be in the original route lanelets. So the common
  // segment interval matches the beginning of the target route. The exception is that if ego is
  // on an intersection lanelet, getClosestLanelet() may not return the same lanelet which exists
  // in the original route. In that case the common segment interval does not match the beginning of
  // the target lanelet
  // =============================================================================================
  const auto start_idx_opt =
    std::invoke([&]() -> std::optional<std::pair<size_t /* original */, size_t /* target */>> {
      for (size_t i = 0; i < original_route.segments.size(); ++i) {
        const auto & original_segment = original_route.segments.at(i).primitives;
        for (size_t j = 0; j < target_route.segments.size(); ++j) {
          const auto & target_segment = target_route.segments.at(j).primitives;
          if (hasSamePrimitives(original_segment, target_segment)) {
            return std::make_pair(i, j);
          }
        }
      }
      return std::nullopt;
    });
  if (!start_idx_opt.has_value()) {
    RCLCPP_ERROR(
      get_logger(), "Check reroute safety failed. Cannot find the start index of the route.");
    return false;
  }
  const auto [start_idx_original, start_idx_target] = start_idx_opt.value();

  // find last idx that matches the target primitives
  size_t end_idx_original = start_idx_original;
  size_t end_idx_target = start_idx_target;
  for (size_t i = 1; i < target_route.segments.size() - start_idx_target; ++i) {
    if (start_idx_original + i > original_route.segments.size() - 1) {
      break;
    }

    const auto & original_primitives =
      original_route.segments.at(start_idx_original + i).primitives;
    const auto & target_primitives = target_route.segments.at(start_idx_target + i).primitives;
    if (!hasSamePrimitives(original_primitives, target_primitives)) {
      break;
    }
    end_idx_original = start_idx_original + i;
    end_idx_target = start_idx_target + i;
  }

  // at the very first transition from main/MRM to MRM/main, the requested route from the
  // route_selector may not begin from ego current lane (because route_selector requests the
  // previous route once, and then replan)
  const bool ego_is_on_first_target_section = std::any_of(
    target_route.segments.front().primitives.begin(),
    target_route.segments.front().primitives.end(), [&](const auto & primitive) {
      const auto lanelet = lanelet_map_ptr_->laneletLayer.get(primitive.id);
      return lanelet::utils::isInLanelet(target_route.start_pose, lanelet);
    });
  if (!ego_is_on_first_target_section) {
    RCLCPP_ERROR(
      get_logger(),
      "Check reroute safety failed. Ego is not on the first section of target route.");
    return false;
  }

  // if the front of target route is not the front of common segment, it is expected that the front
  // of the target route is conflicting with another lane which is equal to original
  // route[start_idx_original-1]
  double accumulated_length = 0.0;

  if (start_idx_target != 0 && start_idx_original > 1) {
    // compute distance from the current pose to the beginning of the common segment
    const auto current_pose = target_route.start_pose;
    const auto primitives = original_route.segments.at(start_idx_original - 1).primitives;
    lanelet::ConstLanelets start_lanelets;
    for (const auto & primitive : primitives) {
      const auto lanelet = lanelet_map_ptr_->laneletLayer.get(primitive.id);
      start_lanelets.push_back(lanelet);
    }
    // closest lanelet in start lanelets
    lanelet::ConstLanelet closest_lanelet;
    if (!lanelet::utils::query::getClosestLanelet(start_lanelets, current_pose, &closest_lanelet)) {
      RCLCPP_ERROR(get_logger(), "Check reroute safety failed. Cannot find the closest lanelet.");
      return false;
    }

    const auto & centerline_2d = lanelet::utils::to2D(closest_lanelet.centerline());
    const auto lanelet_point = lanelet::utils::conversion::toLaneletPoint(current_pose.position);
    const auto arc_coordinates = lanelet::geometry::toArcCoordinates(
      centerline_2d, lanelet::utils::to2D(lanelet_point).basicPoint());
    const double dist_to_current_pose = arc_coordinates.length;
    const double lanelet_length = lanelet::utils::getLaneletLength2d(closest_lanelet);
    accumulated_length = lanelet_length - dist_to_current_pose;
  } else {
    // compute distance from the current pose to the end of the current lanelet
    const auto current_pose = target_route.start_pose;
    const auto primitives = original_route.segments.at(start_idx_original).primitives;
    lanelet::ConstLanelets start_lanelets;
    for (const auto & primitive : primitives) {
      const auto lanelet = lanelet_map_ptr_->laneletLayer.get(primitive.id);
      start_lanelets.push_back(lanelet);
    }
    // closest lanelet in start lanelets
    lanelet::ConstLanelet closest_lanelet;
    if (!lanelet::utils::query::getClosestLanelet(start_lanelets, current_pose, &closest_lanelet)) {
      RCLCPP_ERROR(get_logger(), "Check reroute safety failed. Cannot find the closest lanelet.");
      return false;
    }

    const auto & centerline_2d = lanelet::utils::to2D(closest_lanelet.centerline());
    const auto lanelet_point = lanelet::utils::conversion::toLaneletPoint(current_pose.position);
    const auto arc_coordinates = lanelet::geometry::toArcCoordinates(
      centerline_2d, lanelet::utils::to2D(lanelet_point).basicPoint());
    const double dist_to_current_pose = arc_coordinates.length;
    const double lanelet_length = lanelet::utils::getLaneletLength2d(closest_lanelet);
    accumulated_length = lanelet_length - dist_to_current_pose;
  }

  // compute distance from the start_idx+1 to end_idx
  for (size_t i = start_idx_original + 1; i <= end_idx_original; ++i) {
    const auto primitives = original_route.segments.at(i).primitives;
    if (primitives.empty()) {
      break;
    }

    std::vector<double> lanelets_length(primitives.size());
    for (size_t primitive_idx = 0; primitive_idx < primitives.size(); ++primitive_idx) {
      const auto & primitive = primitives.at(primitive_idx);
      const auto & lanelet = lanelet_map_ptr_->laneletLayer.get(primitive.id);
      lanelets_length.at(primitive_idx) = (lanelet::utils::getLaneletLength2d(lanelet));
    }
    accumulated_length += *std::min_element(lanelets_length.begin(), lanelets_length.end());
  }

  // check if the goal is inside of the target terminal lanelet
  const auto & target_end_primitives = target_route.segments.at(end_idx_target).primitives;
  const auto & target_goal = target_route.goal_pose;
  for (const auto & target_end_primitive : target_end_primitives) {
    const auto lanelet = lanelet_map_ptr_->laneletLayer.get(target_end_primitive.id);
    if (lanelet::utils::isInLanelet(target_goal, lanelet)) {
      const auto target_goal_position =
        lanelet::utils::conversion::toLaneletPoint(target_goal.position);
      const double dist_to_goal = lanelet::geometry::toArcCoordinates(
                                    lanelet::utils::to2D(lanelet.centerline()),
                                    lanelet::utils::to2D(target_goal_position).basicPoint())
                                    .length;
      const double target_lanelet_length = lanelet::utils::getLaneletLength2d(lanelet);
      // NOTE: `accumulated_length` here contains the length of the entire target_end_primitive, so
      // the remaining distance from the goal to the end of the target_end_primitive needs to be
      // subtracted.
      const double remaining_dist = target_lanelet_length - dist_to_goal;
      accumulated_length = std::max(accumulated_length - remaining_dist, 0.0);
      break;
    }
  }

  // check safety
  const double safety_length =
    std::max(current_velocity * reroute_time_threshold_, minimum_reroute_length_);
  if (accumulated_length > safety_length) {
    return true;
  }

  RCLCPP_WARN(
    get_logger(),
    "Length of lane where original and B target (= %f) is less than safety length (= %f), so "
    "reroute is not safe.",
    accumulated_length, safety_length);
  return false;
}
}  // namespace mission_planner

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(mission_planner::MissionPlanner)
