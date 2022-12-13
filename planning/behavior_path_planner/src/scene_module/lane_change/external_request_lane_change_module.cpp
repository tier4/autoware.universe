// Copyright 2022 TIER IV, Inc.
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

#include "behavior_path_planner/scene_module/lane_change/external_request_lane_change_module.hpp"

#include "behavior_path_planner/path_utilities.hpp"
#include "behavior_path_planner/scene_module/lane_change/util.hpp"
#include "behavior_path_planner/scene_module/scene_module_interface.hpp"
#include "behavior_path_planner/scene_module/scene_module_visitor.hpp"
#include "behavior_path_planner/turn_signal_decider.hpp"
#include "behavior_path_planner/utilities.hpp"

#include <lanelet2_extension/utility/message_conversion.hpp>
#include <lanelet2_extension/utility/utilities.hpp>
#include <tier4_autoware_utils/tier4_autoware_utils.hpp>

#include "tier4_planning_msgs/msg/detail/lane_change_debug_msg_array__struct.hpp"
#include <autoware_auto_perception_msgs/msg/object_classification.hpp>
#include <autoware_auto_vehicle_msgs/msg/turn_indicators_command.hpp>

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>
namespace behavior_path_planner
{
std::string getTopicName(const ExternalRequestLaneChangeModule::Direction & direction)
{
  const std::string direction_name =
    direction == ExternalRequestLaneChangeModule::Direction::RIGHT ? "right" : "left";
  return "ext_request_lane_change_" + direction_name;
}

ExternalRequestLaneChangeModule::ExternalRequestLaneChangeModule(
  const std::string & name, rclcpp::Node & node, std::shared_ptr<LaneChangeParameters> parameters,
  const Direction & direction)
: SceneModuleInterface{name, node}, parameters_{std::move(parameters)}, direction_{direction}
{
  rtc_interface_ptr_ = std::make_shared<RTCInterface>(&node, getTopicName(direction));
}

BehaviorModuleOutput ExternalRequestLaneChangeModule::run()
{
  RCLCPP_DEBUG(getLogger(), "Was waiting approval, and now approved. Do plan().");
  current_state_ = BT::NodeStatus::RUNNING;
  auto output = plan();

  if (!isSafe()) {
    current_state_ = BT::NodeStatus::SUCCESS;  // for breaking loop
    return output;
  }

  const auto current_pose = getEgoPose();
  const double start_distance = motion_utils::calcSignedArcLength(
    output.path->points, current_pose.position,
    status_.lane_change_path.shift_point.start.position);
  waitApproval(start_distance);

  return output;
}

void ExternalRequestLaneChangeModule::onEntry()
{
  RCLCPP_DEBUG(getLogger(), "LANE_CHANGE onEntry");
  current_state_ = BT::NodeStatus::SUCCESS;
  updateLaneChangeStatus();
}

void ExternalRequestLaneChangeModule::onExit()
{
  resetParameters();
  current_state_ = BT::NodeStatus::SUCCESS;
  RCLCPP_DEBUG(getLogger(), "LANE_CHANGE onExit");
}

bool ExternalRequestLaneChangeModule::isExecutionRequested() const
{
  if (current_state_ == BT::NodeStatus::RUNNING) {
    return true;
  }

  const auto current_lanes = util::getCurrentLanes(planner_data_);

  const auto lane_change_lanes = getLaneChangeLanes(current_lanes, lane_change_lane_length_);
  LaneChangePath selected_path;
  const auto [found_valid_path, found_safe_path] =
    getSafePath(lane_change_lanes, check_distance_, selected_path);

  return found_valid_path;
}

bool ExternalRequestLaneChangeModule::isExecutionReady() const
{
  if (current_state_ == BT::NodeStatus::RUNNING) {
    return true;
  }

  const auto current_lanes = util::getCurrentLanes(planner_data_);

  const auto lane_change_lanes = getLaneChangeLanes(current_lanes, lane_change_lane_length_);
  LaneChangePath selected_path;
  const auto [found_valid_path, found_safe_path] =
    getSafePath(lane_change_lanes, check_distance_, selected_path);

  return found_safe_path;
}

BT::NodeStatus ExternalRequestLaneChangeModule::updateState()
{
  RCLCPP_DEBUG(getLogger(), "LANE_CHANGE updateState");
  if (!isSafe()) {
    current_state_ = BT::NodeStatus::SUCCESS;
    return current_state_;
  }

  if (isAbortConditionSatisfied()) {
    if (isNearEndOfLane() && isCurrentSpeedLow()) {
      current_state_ = BT::NodeStatus::RUNNING;
      return current_state_;
    }
    // cancel lane change path
    current_state_ = BT::NodeStatus::FAILURE;
    return current_state_;
  }

  if (hasFinishedLaneChange()) {
    current_state_ = BT::NodeStatus::SUCCESS;
    return current_state_;
  }
  current_state_ = BT::NodeStatus::RUNNING;
  return current_state_;
}

BehaviorModuleOutput ExternalRequestLaneChangeModule::plan()
{
  resetPathCandidate();

  auto path = status_.lane_change_path.path;

  generateExtendedDrivableArea(path);

  if (isAbortConditionSatisfied()) {
    if (isNearEndOfLane() && isCurrentSpeedLow()) {
      const auto stop_point = util::insertStopPoint(0.1, &path);
    }
  }

  BehaviorModuleOutput output;
  output.path = std::make_shared<PathWithLaneId>(path);
  updateOutputTurnSignal(output);

  return output;
}

CandidateOutput ExternalRequestLaneChangeModule::planCandidate() const
{
  CandidateOutput output;

  // Get lane change lanes
  const auto current_lanes = util::getCurrentLanes(planner_data_);
  const auto lane_change_lanes = getLaneChangeLanes(current_lanes, lane_change_lane_length_);

  LaneChangePath selected_path;
  [[maybe_unused]] const auto [found_valid_path, found_safe_path] =
    getSafePath(lane_change_lanes, check_distance_, selected_path);
  selected_path.path.header = planner_data_->route_handler->getRouteHeader();

  if (selected_path.path.points.empty()) {
    return output;
  }

  const auto & start_idx = selected_path.shift_point.start_idx;
  const auto & end_idx = selected_path.shift_point.end_idx;

  output.path_candidate = selected_path.path;
  output.lateral_shift = selected_path.shifted_path.shift_length.at(end_idx) -
                         selected_path.shifted_path.shift_length.at(start_idx);
  output.distance_to_path_change = motion_utils::calcSignedArcLength(
    selected_path.path.points, getEgoPose().position, selected_path.shift_point.start.position);

  // updateSteeringFactorPtr(output, selected_path);
  return output;
}

BehaviorModuleOutput ExternalRequestLaneChangeModule::planWaitingApproval()
{
  BehaviorModuleOutput out;
  out.path = std::make_shared<PathWithLaneId>(getReferencePath());
  const auto candidate = planCandidate();
  path_candidate_ = std::make_shared<PathWithLaneId>(candidate.path_candidate);
  waitApproval(candidate.distance_to_path_change);
  return out;
}

void ExternalRequestLaneChangeModule::updateLaneChangeStatus()
{
  status_.current_lanes = util::getCurrentLanes(planner_data_);
  status_.lane_change_lanes = getLaneChangeLanes(status_.current_lanes, lane_change_lane_length_);

  // Find lane change path
  LaneChangePath selected_path;
  [[maybe_unused]] const auto [found_valid_path, found_safe_path] =
    getSafePath(status_.lane_change_lanes, check_distance_, selected_path);

  // Update status
  status_.is_safe = found_safe_path;
  status_.lane_change_path = selected_path;
  status_.lane_follow_lane_ids = util::getIds(status_.current_lanes);
  status_.lane_change_lane_ids = util::getIds(status_.lane_change_lanes);

  const auto arclength_start =
    lanelet::utils::getArcCoordinates(status_.lane_change_lanes, getEgoPose());
  status_.start_distance = arclength_start.length;
  status_.lane_change_path.path.header = getRouteHeader();
}

PathWithLaneId ExternalRequestLaneChangeModule::getReferencePath() const
{
  PathWithLaneId reference_path;

  const auto & route_handler = planner_data_->route_handler;
  const auto current_pose = getEgoPose();
  const auto & common_parameters = planner_data_->parameters;

  // Set header
  reference_path.header = getRouteHeader();

  const auto current_lanes = util::getCurrentLanes(planner_data_);

  if (current_lanes.empty()) {
    return reference_path;
  }

  if (reference_path.points.empty()) {
    reference_path = util::getCenterLinePath(
      *route_handler, current_lanes, current_pose, common_parameters.backward_path_length,
      common_parameters.forward_path_length, common_parameters);
  }

  const int num_lane_change =
    std::abs(route_handler->getNumLaneToPreferredLane(current_lanes.back()));
  double optional_lengths{0.0};
  const auto isInIntersection = util::checkLaneIsInIntersection(
    *route_handler, reference_path, current_lanes, common_parameters, optional_lengths);
  if (isInIntersection) {
    reference_path = util::getCenterLinePath(
      *route_handler, current_lanes, current_pose, common_parameters.backward_path_length,
      common_parameters.forward_path_length, common_parameters, optional_lengths);
  }
  const double & buffer =
    common_parameters.backward_length_buffer_for_end_of_lane;  // buffer for min_lane_change_length
  const double lane_change_buffer =
    num_lane_change * (common_parameters.minimum_lane_change_length + buffer) + optional_lengths;

  reference_path = util::setDecelerationVelocity(
    *route_handler, reference_path, current_lanes, parameters_->lane_change_prepare_duration,
    lane_change_buffer);

  // const auto drivable_lanes = util::generateDrivableLanes(current_lanes);
  // const auto expanded_lanes = util::expandLanelets(
  //   drivable_lanes, parameters_->drivable_area_left_bound_offset,
  //   parameters_->drivable_area_right_bound_offset);

  reference_path.drivable_area = util::generateDrivableArea(
    reference_path, current_lanes, common_parameters.drivable_area_resolution,
    common_parameters.vehicle_length, planner_data_);

  return reference_path;
}

lanelet::ConstLanelets ExternalRequestLaneChangeModule::getLaneChangeLanes(
  const lanelet::ConstLanelets & current_lanes, const double lane_change_lane_length) const
{
  lanelet::ConstLanelets lane_change_lanes;
  const auto & route_handler = planner_data_->route_handler;
  const auto minimum_lane_change_length = planner_data_->parameters.minimum_lane_change_length;
  const auto lane_change_prepare_duration = parameters_->lane_change_prepare_duration;
  const auto current_pose = getEgoPose();
  const auto current_twist = getEgoTwist();

  if (current_lanes.empty()) {
    return lane_change_lanes;
  }

  // Get lane change lanes
  lanelet::ConstLanelet current_lane;
  lanelet::utils::query::getClosestLanelet(current_lanes, current_pose, &current_lane);
  const double lane_change_prepare_length =
    std::max(current_twist.linear.x * lane_change_prepare_duration, minimum_lane_change_length);
  lanelet::ConstLanelets current_check_lanes =
    route_handler->getLaneletSequence(current_lane, current_pose, 0.0, lane_change_prepare_length);
  lanelet::ConstLanelet lane_change_lane;

  auto getLaneChangeTargetExceptPreferredLane = [this, &route_handler](
                                                  const lanelet::ConstLanelets & lanelets,
                                                  lanelet::ConstLanelet * target_lanelet) {
    return direction_ == Direction::RIGHT
             ? route_handler->getRightLaneChangeTargetExceptPreferredLane(lanelets, target_lanelet)
             : route_handler->getLeftLaneChangeTargetExceptPreferredLane(lanelets, target_lanelet);
  };

  if (getLaneChangeTargetExceptPreferredLane(current_check_lanes, &lane_change_lane)) {
    lane_change_lanes = route_handler->getLaneletSequence(
      lane_change_lane, current_pose, lane_change_lane_length, lane_change_lane_length);
  } else {
    lane_change_lanes.clear();
  }

  return lane_change_lanes;
}

std::pair<bool, bool> ExternalRequestLaneChangeModule::getSafePath(
  const lanelet::ConstLanelets & lane_change_lanes, const double check_distance,
  LaneChangePath & safe_path) const
{
  const auto & route_handler = planner_data_->route_handler;
  const auto current_pose = getEgoPose();
  const auto current_twist = getEgoTwist();
  const auto & common_parameters = planner_data_->parameters;

  const auto current_lanes = util::getCurrentLanes(planner_data_);

  if (!lane_change_lanes.empty()) {
    // find candidate paths
    const auto lane_change_paths = lane_change_utils::getLaneChangePaths(
      *route_handler, current_lanes, lane_change_lanes, current_pose, current_twist,
      common_parameters, *parameters_);

    // get lanes used for detection
    lanelet::ConstLanelets check_lanes;
    if (!lane_change_paths.empty()) {
      const auto & longest_path = lane_change_paths.front();
      // we want to see check_distance [m] behind vehicle so add lane changing length
      const double check_distance_with_path =
        check_distance + longest_path.preparation_length + longest_path.lane_change_length;
      check_lanes = route_handler->getCheckTargetLanesFromPath(
        longest_path.path, lane_change_lanes, check_distance_with_path);
    }

    // select valid path
    const LaneChangePaths valid_paths = lane_change_utils::selectValidPaths(
      lane_change_paths, current_lanes, check_lanes, *route_handler, current_pose,
      route_handler->getGoalPose(),
      common_parameters.minimum_lane_change_length +
        common_parameters.backward_length_buffer_for_end_of_lane +
        parameters_->lane_change_finish_judge_buffer);

    if (valid_paths.empty()) {
      return std::make_pair(false, false);
    }
    debug_valid_path_ = valid_paths;

    // select safe path
    const bool found_safe_path = lane_change_utils::selectSafePath(
      valid_paths, current_lanes, check_lanes, planner_data_->dynamic_object, current_pose,
      current_twist, common_parameters, *parameters_, &safe_path, object_debug_);

    if (parameters_->publish_debug_marker) {
      setObjectDebugVisualization();
    } else {
      debug_marker_.markers.clear();
    }

    return std::make_pair(true, found_safe_path);
  }

  return std::make_pair(false, false);
}

bool ExternalRequestLaneChangeModule::isSafe() const { return status_.is_safe; }

bool ExternalRequestLaneChangeModule::isNearEndOfLane() const
{
  const auto & current_pose = getEgoPose();
  const auto minimum_lane_change_length = planner_data_->parameters.minimum_lane_change_length;
  const auto end_of_lane_buffer = planner_data_->parameters.backward_length_buffer_for_end_of_lane;
  const double threshold = end_of_lane_buffer + minimum_lane_change_length;

  return std::max(0.0, util::getDistanceToEndOfLane(current_pose, status_.current_lanes)) <
         threshold;
}

bool ExternalRequestLaneChangeModule::isCurrentSpeedLow() const
{
  constexpr double threshold_ms = 10.0 * 1000 / 3600;
  return util::l2Norm(getEgoTwist().linear) < threshold_ms;
}

bool ExternalRequestLaneChangeModule::isAbortConditionSatisfied() const
{
  const auto & route_handler = planner_data_->route_handler;
  const auto current_pose = getEgoPose();
  const auto current_twist = getEgoTwist();
  const auto & dynamic_objects = planner_data_->dynamic_object;
  const auto & common_parameters = planner_data_->parameters;

  const auto & current_lanes = status_.current_lanes;

  // check abort enable flag
  if (!parameters_->enable_abort_lane_change) {
    return false;
  }

  if (!is_activated_) {
    return false;
  }

  // find closest lanelet in original lane
  lanelet::ConstLanelet closest_lanelet{};
  auto clock{rclcpp::Clock{RCL_ROS_TIME}};
  if (!lanelet::utils::query::getClosestLanelet(current_lanes, current_pose, &closest_lanelet)) {
    RCLCPP_ERROR_THROTTLE(
      getLogger(), clock, 1000,
      "Failed to find closest lane! Lane change aborting function is not working!");
    return false;
  }

  // check if lane change path is still safe
  const bool is_path_safe = std::invoke([this, &route_handler, &dynamic_objects, &current_lanes,
                                         &current_pose, &current_twist, &common_parameters]() {
    constexpr double check_distance = 100.0;
    // get lanes used for detection
    const auto & path = status_.lane_change_path;
    const double check_distance_with_path =
      check_distance + path.preparation_length + path.lane_change_length;
    const auto check_lanes = route_handler->getCheckTargetLanesFromPath(
      path.path, status_.lane_change_lanes, check_distance_with_path);

    std::unordered_map<std::string, CollisionCheckDebug> debug_data;

    return lane_change_utils::isLaneChangePathSafe(
      path.path, current_lanes, check_lanes, dynamic_objects, current_pose, current_twist,
      common_parameters, *parameters_, debug_data, false, status_.lane_change_path.acceleration);
  });

  // abort only if velocity is low or vehicle pose is close enough
  if (!is_path_safe) {
    // check vehicle velocity thresh
    const bool is_velocity_low =
      util::l2Norm(current_twist.linear) < parameters_->abort_lane_change_velocity_thresh;
    const bool is_within_original_lane =
      lane_change_utils::isEgoWithinOriginalLane(current_lanes, current_pose, common_parameters);
    if (is_velocity_low && is_within_original_lane) {
      return true;
    }

    const bool is_distance_small =
      lane_change_utils::isEgoDistanceNearToCenterline(closest_lanelet, current_pose, *parameters_);

    // check angle thresh from original lane
    const bool is_angle_diff_small = lane_change_utils::isEgoHeadingAngleLessThanThreshold(
      closest_lanelet, current_pose, *parameters_);

    if (is_distance_small && is_angle_diff_small) {
      return true;
    }
    auto clock{rclcpp::Clock{RCL_ROS_TIME}};
    RCLCPP_WARN_STREAM_THROTTLE(
      getLogger(), clock, 1000,
      "DANGER!!! Path is not safe anymore, but it is too late to abort! Please be cautious");
  }

  return false;
}

bool ExternalRequestLaneChangeModule::hasFinishedLaneChange() const
{
  const auto & current_pose = getEgoPose();
  const auto arclength_current =
    lanelet::utils::getArcCoordinates(status_.lane_change_lanes, current_pose);
  const double travel_distance = arclength_current.length - status_.start_distance;
  const double finish_distance = status_.lane_change_path.preparation_length +
                                 status_.lane_change_path.lane_change_length +
                                 parameters_->lane_change_finish_judge_buffer;
  return travel_distance > finish_distance;
}

void ExternalRequestLaneChangeModule::setObjectDebugVisualization() const
{
  using marker_utils::lane_change_markers::showAllValidLaneChangePath;
  using marker_utils::lane_change_markers::showLerpedPose;
  using marker_utils::lane_change_markers::showObjectInfo;
  using marker_utils::lane_change_markers::showPolygon;
  using marker_utils::lane_change_markers::showPolygonPose;

  debug_marker_.markers.clear();
  const auto add = [this](const MarkerArray & added) {
    tier4_autoware_utils::appendMarkerArray(added, &debug_marker_);
  };

  add(showObjectInfo(object_debug_, "object_debug_info"));
  add(showLerpedPose(object_debug_, "lerp_pose_before_true"));
  add(showPolygonPose(object_debug_, "expected_pose"));
  add(showPolygon(object_debug_, "lerped_polygon"));
  add(showAllValidLaneChangePath(debug_valid_path_, "lane_change_valid_paths"));
}

std::shared_ptr<LaneChangeDebugMsgArray> ExternalRequestLaneChangeModule::get_debug_msg_array()
  const
{
  LaneChangeDebugMsgArray debug_msg_array;
  debug_msg_array.lane_change_info.reserve(object_debug_.size());
  for (const auto & [uuid, debug_data] : object_debug_) {
    LaneChangeDebugMsg debug_msg;
    debug_msg.object_id = uuid;
    debug_msg.allow_lane_change = debug_data.allow_lane_change;
    debug_msg.is_front = debug_data.is_front;
    debug_msg.relative_distance = debug_data.relative_to_ego;
    debug_msg.failed_reason = debug_data.failed_reason;
    debug_msg.velocity = util::l2Norm(debug_data.object_twist.linear);
    debug_msg_array.lane_change_info.push_back(debug_msg);
  }
  lane_change_debug_msg_array_ = debug_msg_array;

  lane_change_debug_msg_array_.header.stamp = clock_->now();
  return std::make_shared<LaneChangeDebugMsgArray>(lane_change_debug_msg_array_);
}

Pose ExternalRequestLaneChangeModule::getEgoPose() const { return planner_data_->self_pose->pose; }
Twist ExternalRequestLaneChangeModule::getEgoTwist() const
{
  return planner_data_->self_odometry->twist.twist;
}
std_msgs::msg::Header ExternalRequestLaneChangeModule::getRouteHeader() const
{
  return planner_data_->route_handler->getRouteHeader();
}
void ExternalRequestLaneChangeModule::generateExtendedDrivableArea(PathWithLaneId & path)
{
  const auto & common_parameters = planner_data_->parameters;
  lanelet::ConstLanelets lanes;
  lanes.reserve(status_.current_lanes.size() + status_.lane_change_lanes.size());
  lanes.insert(lanes.end(), status_.current_lanes.begin(), status_.current_lanes.end());
  lanes.insert(lanes.end(), status_.lane_change_lanes.begin(), status_.lane_change_lanes.end());

  const double & resolution = common_parameters.drivable_area_resolution;
  path.drivable_area = util::generateDrivableArea(
    path, lanes, resolution, common_parameters.vehicle_length, planner_data_);
}

void ExternalRequestLaneChangeModule::updateOutputTurnSignal(BehaviorModuleOutput & output)
{
  const auto turn_signal_info = util::getPathTurnSignal(
    status_.current_lanes, status_.lane_change_path.shifted_path,
    status_.lane_change_path.shift_point, getEgoPose(), getEgoTwist().linear.x,
    planner_data_->parameters);
  output.turn_signal_info.turn_signal.command = turn_signal_info.first.command;

  output.turn_signal_info.turn_signal.command = turn_signal_info.first.command;
  output.turn_signal_info.signal_distance = turn_signal_info.second;
}

void ExternalRequestLaneChangeModule::resetParameters()
{
  clearWaitingApproval();
  removeRTCStatus();
  resetPathCandidate();
  object_debug_.clear();
  debug_marker_.markers.clear();
}

ExternalRequestLaneChangeRightModule::ExternalRequestLaneChangeRightModule(
  const std::string & name, rclcpp::Node & node, std::shared_ptr<LaneChangeParameters> parameters)
: ExternalRequestLaneChangeModule{name, node, parameters, Direction::RIGHT}
{
}

ExternalRequestLaneChangeLeftModule::ExternalRequestLaneChangeLeftModule(
  const std::string & name, rclcpp::Node & node, std::shared_ptr<LaneChangeParameters> parameters)
: ExternalRequestLaneChangeModule{name, node, parameters, Direction::LEFT}
{
}

}  // namespace behavior_path_planner
