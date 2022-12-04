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
ExternalRequestLaneChangeModule::ExternalRequestLaneChangeModule(
  const std::string & name, rclcpp::Node & node, std::shared_ptr<LaneChangeParameters> parameters)
: SceneModuleInterface{name, node},
  parameters_{std::move(parameters)},
  rtc_interface_left_(&node, "ext_request_lane_change_left"),
  rtc_interface_right_(&node, "ext_request_lane_change_right"),
  uuid_left_{generateUUID()},
  uuid_right_{generateUUID()}
{
  steering_factor_interface_ptr_ =
    std::make_unique<SteeringFactorInterface>(&node, "ext_request_lane_change");
}

BehaviorModuleOutput ExternalRequestLaneChangeModule::run()
{
  RCLCPP_DEBUG(getLogger(), "Was waiting approval, and now approved. Do plan().");
  current_state_ = BT::NodeStatus::RUNNING;
  auto output = plan();

  if (!isProper()) {
    current_state_ = BT::NodeStatus::SUCCESS;  // for breaking loop
    return output;
  }

  if (isActivatedLeft()) {
    updateSteeringFactorPtr(output, status_left_);
  } else if (isActivatedRight()) {
    updateSteeringFactorPtr(output, status_right_);
  }

  return output;
}

void ExternalRequestLaneChangeModule::onEntry()
{
  RCLCPP_DEBUG(getLogger(), "LANE_CHANGE onEntry");
  current_state_ = BT::NodeStatus::SUCCESS;
  updateLaneChangeStatusLeft();
  updateLaneChangeStatusRight();
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

  const auto lane_change_lanes_left =
    getLeftLaneChangeLanes(current_lanes, lane_change_lane_length_);
  LaneChangePath selected_path_left;
  const auto [found_valid_path_left, found_safe_path_left] =
    getSafePath(lane_change_lanes_left, check_distance_, selected_path_left);

  const auto lane_change_lanes_right =
    getRightLaneChangeLanes(current_lanes, lane_change_lane_length_);
  LaneChangePath selected_path_right;
  const auto [found_valid_path_right, found_safe_path_right] =
    getSafePath(lane_change_lanes_right, check_distance_, selected_path_right);

  return found_valid_path_left || found_valid_path_right;
}

bool ExternalRequestLaneChangeModule::isExecutionReady() const
{
  if (current_state_ == BT::NodeStatus::RUNNING) {
    return true;
  }

  const auto current_lanes = util::getCurrentLanes(planner_data_);
  const auto lane_change_lanes_left =
    getLeftLaneChangeLanes(current_lanes, lane_change_lane_length_);
  LaneChangePath selected_path_left;
  const auto [found_valid_path_left, found_safe_path_left] =
    getSafePath(lane_change_lanes_left, check_distance_, selected_path_left);

  const auto lane_change_lanes_right =
    getRightLaneChangeLanes(current_lanes, lane_change_lane_length_);
  LaneChangePath selected_path_right;
  const auto [found_valid_path_right, found_safe_path_right] =
    getSafePath(lane_change_lanes_right, check_distance_, selected_path_right);

  const bool is_safe_left = found_valid_path_left && found_safe_path_left;
  const bool is_safe_right = found_valid_path_right && found_safe_path_right;

  return is_safe_left || is_safe_right;
}

BT::NodeStatus ExternalRequestLaneChangeModule::updateState()
{
  RCLCPP_DEBUG(getLogger(), "LANE_CHANGE updateState");
  if (!isProper()) {
    current_state_ = BT::NodeStatus::SUCCESS;
    return current_state_;
  }

  if (isActivatedLeft()) {
    is_activated_ = isActivatedLeft();
    current_state_ = getState(status_left_);
    return current_state_;
  } else if (isActivatedRight()) {
    is_activated_ = isActivatedRight();
    current_state_ = getState(status_right_);
    return current_state_;
  }

  current_state_ = BT::NodeStatus::SUCCESS;
  return current_state_;
}

BT::NodeStatus ExternalRequestLaneChangeModule::getState(const LaneChangeStatus & status) const
{
  if (isAbortConditionSatisfied(status)) {
    if (isNearEndOfLane(status) && isCurrentSpeedLow()) {
      return BT::NodeStatus::RUNNING;
    }
    // cancel lane change path
    return BT::NodeStatus::FAILURE;
  }

  if (hasFinishedLaneChange(status)) {
    return BT::NodeStatus::SUCCESS;
  }
  return BT::NodeStatus::RUNNING;
}

BehaviorModuleOutput ExternalRequestLaneChangeModule::plan()
{
  if (isActivatedLeft()) {
    is_activated_ = isActivatedLeft();
    return getOutput(status_left_);
  } else if (isActivatedRight()) {
    is_activated_ = isActivatedRight();
    return getOutput(status_right_);
  }

  return BehaviorModuleOutput{};
}

BehaviorModuleOutput ExternalRequestLaneChangeModule::getOutput(const LaneChangeStatus & status)
{
  constexpr double resample_interval{1.0};
  PathWithLaneId path =
    util::resamplePathWithSpline(status.lane_change_path.path, resample_interval);

  if (!isValidPath(path, status)) {
    is_proper_ = false;
    return BehaviorModuleOutput{};
  }

  is_proper_ = true;

  generateExtendedDrivableArea(path, status);

  if (isAbortConditionSatisfied(status)) {
    if (isNearEndOfLane(status) && isCurrentSpeedLow()) {
      const auto stop_point = util::insertStopPoint(0.1, &path);
    }
  }

  BehaviorModuleOutput output;
  output.path = std::make_shared<PathWithLaneId>(path);
  updateOutputTurnSignal(output, status);
  return output;
}

CandidateOutput ExternalRequestLaneChangeModule::planCandidate() const
{
  CandidateOutput output;
  const LaneChangePath selected_path = selectLaneChangePath();

  if (selected_path.path.points.empty()) {
    return output;
  }

  const auto & start_idx = selected_path.shift_line.start_idx;
  const auto & end_idx = selected_path.shift_line.end_idx;

  output.path_candidate = selected_path.path;
  output.lateral_shift = selected_path.shifted_path.shift_length.at(end_idx) -
                         selected_path.shifted_path.shift_length.at(start_idx);
  output.start_distance_to_path_change = motion_utils::calcSignedArcLength(
    selected_path.path.points, getEgoPose().position, selected_path.shift_line.start.position);
  output.finish_distance_to_path_change = motion_utils::calcSignedArcLength(
    selected_path.path.points, getEgoPose().position, selected_path.shift_line.end.position);

  updateSteeringFactorPtr(output, selected_path);
  return output;
}

BehaviorModuleOutput ExternalRequestLaneChangeModule::planWaitingApproval()
{
  BehaviorModuleOutput out;
  out.path = std::make_shared<PathWithLaneId>(getReferencePath());
  const auto candidate = planCandidate();
  out.path_candidate = std::make_shared<PathWithLaneId>(candidate.path_candidate);
  updateRTCStatus(candidate);
  waitApproval();
  return out;
}

LaneChangePath ExternalRequestLaneChangeModule::selectLaneChangePath() const
{
  // Get left lane change path
  const auto current_lanes = util::getCurrentLanes(planner_data_);
  const auto lane_change_lanes_left =
    getLeftLaneChangeLanes(current_lanes, lane_change_lane_length_);
  LaneChangePath selected_path_left;
  [[maybe_unused]] const auto [found_valid_path_left, found_safe_path_left] =
    getSafePath(lane_change_lanes_left, check_distance_, selected_path_left);
  selected_path_left.path.header = planner_data_->route_handler->getRouteHeader();

  // Get right lane change path
  const auto lane_change_lanes_right =
    getRightLaneChangeLanes(current_lanes, lane_change_lane_length_);
  LaneChangePath selected_path_right;
  [[maybe_unused]] const auto [found_valid_path_right, found_safe_path_right] =
    getSafePath(lane_change_lanes_right, check_distance_, selected_path_right);
  selected_path_right.path.header = planner_data_->route_handler->getRouteHeader();

  LaneChangePath selected_path;
  if (selected_path_left.path.points.empty() && selected_path_right.path.points.empty()) {
    return selected_path;
  }

  if (selected_path_left.path.points.empty()) {
    return selected_path_right;
  } else if (selected_path_right.path.points.empty()) {
    return selected_path_left;
  } else {
    // TODO(watanabe): Multiple candidate paths
    return selected_path_right;
  }
}

void ExternalRequestLaneChangeModule::updateLaneChangeStatusLeft()
{
  status_left_.current_lanes = util::getCurrentLanes(planner_data_);
  status_left_.lane_change_lanes =
    getLeftLaneChangeLanes(status_left_.current_lanes, lane_change_lane_length_);

  // Find lane change path
  LaneChangePath selected_path;
  [[maybe_unused]] const auto [found_valid_path, found_safe_path] =
    getSafePath(status_left_.lane_change_lanes, check_distance_, selected_path);

  // Update status
  status_left_.lane_change_path = selected_path;
  status_left_.lane_follow_lane_ids = util::getIds(status_left_.current_lanes);
  status_left_.lane_change_lane_ids = util::getIds(status_left_.lane_change_lanes);

  const auto arclength_start =
    lanelet::utils::getArcCoordinates(status_left_.lane_change_lanes, getEgoPose());
  status_left_.start_distance = arclength_start.length;
  status_left_.lane_change_path.path.header = getRouteHeader();
}

void ExternalRequestLaneChangeModule::updateLaneChangeStatusRight()
{
  status_right_.current_lanes = util::getCurrentLanes(planner_data_);
  status_right_.lane_change_lanes =
    getRightLaneChangeLanes(status_right_.current_lanes, lane_change_lane_length_);

  // Find lane change path
  LaneChangePath selected_path;
  [[maybe_unused]] const auto [found_valid_path, found_safe_path] =
    getSafePath(status_right_.lane_change_lanes, check_distance_, selected_path);

  // Update status
  status_right_.lane_change_path = selected_path;
  status_right_.lane_follow_lane_ids = util::getIds(status_right_.current_lanes);
  status_right_.lane_change_lane_ids = util::getIds(status_right_.lane_change_lanes);

  const auto arclength_start =
    lanelet::utils::getArcCoordinates(status_right_.lane_change_lanes, getEgoPose());
  status_right_.start_distance = arclength_start.length;
  status_right_.lane_change_path.path.header = getRouteHeader();
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
    *route_handler, reference_path, current_lanes, common_parameters, num_lane_change,
    optional_lengths);
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

  const auto drivable_lanes = util::generateDrivableLanes(current_lanes);
  const auto expanded_lanes = util::expandLanelets(
    drivable_lanes, parameters_->drivable_area_left_bound_offset,
    parameters_->drivable_area_right_bound_offset);

  reference_path.drivable_area = util::generateDrivableArea(
    reference_path, expanded_lanes, common_parameters.drivable_area_resolution,
    common_parameters.vehicle_length, planner_data_);

  return reference_path;
}

lanelet::ConstLanelets ExternalRequestLaneChangeModule::getLeftLaneChangeLanes(
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
  if (route_handler->getLeftLaneChangeTargetExceptPreferredLane(
        current_check_lanes, &lane_change_lane)) {
    lane_change_lanes = route_handler->getLaneletSequence(
      lane_change_lane, current_pose, lane_change_lane_length, lane_change_lane_length);
  } else {
    lane_change_lanes.clear();
  }

  return lane_change_lanes;
}

lanelet::ConstLanelets ExternalRequestLaneChangeModule::getRightLaneChangeLanes(
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
  if (route_handler->getRightLaneChangeTargetExceptPreferredLane(
        current_check_lanes, &lane_change_lane)) {
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
      route_handler->getGoalPose(), common_parameters.minimum_lane_change_length);

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

bool ExternalRequestLaneChangeModule::isProper() const { return is_proper_; }

bool ExternalRequestLaneChangeModule::isValidPath(
  const PathWithLaneId & path, const LaneChangeStatus & status) const
{
  const auto & route_handler = planner_data_->route_handler;

  // check lane departure
  const auto drivable_lanes = lane_change_utils::generateDrivableLanes(
    *route_handler, util::extendLanes(route_handler, status.current_lanes),
    util::extendLanes(route_handler, status.lane_change_lanes));
  const auto expanded_lanes = util::expandLanelets(
    drivable_lanes, parameters_->drivable_area_left_bound_offset,
    parameters_->drivable_area_right_bound_offset);
  const auto lanelets = util::transformToLanelets(expanded_lanes);

  // check path points are in any lanelets
  for (const auto & point : path.points) {
    bool is_in_lanelet = false;
    for (const auto & lanelet : lanelets) {
      if (lanelet::utils::isInLanelet(point.point.pose, lanelet)) {
        is_in_lanelet = true;
        break;
      }
    }
    if (!is_in_lanelet) {
      RCLCPP_WARN_STREAM_THROTTLE(getLogger(), *clock_, 1000, "path is out of lanes");
      return false;
    }
  }

  // check relative angle
  if (!util::checkPathRelativeAngle(path, M_PI)) {
    RCLCPP_WARN_STREAM_THROTTLE(getLogger(), *clock_, 1000, "path relative angle is invalid");
    return false;
  }

  return true;
}

bool ExternalRequestLaneChangeModule::isNearEndOfLane(const LaneChangeStatus & status) const
{
  const auto & current_pose = getEgoPose();
  const auto minimum_lane_change_length = planner_data_->parameters.minimum_lane_change_length;
  const auto end_of_lane_buffer = planner_data_->parameters.backward_length_buffer_for_end_of_lane;
  const double threshold = end_of_lane_buffer + minimum_lane_change_length;

  return std::max(0.0, util::getDistanceToEndOfLane(current_pose, status.current_lanes)) <
         threshold;
}

bool ExternalRequestLaneChangeModule::isCurrentSpeedLow() const
{
  constexpr double threshold_ms = 10.0 * 1000 / 3600;
  return util::l2Norm(getEgoTwist().linear) < threshold_ms;
}

bool ExternalRequestLaneChangeModule::isAbortConditionSatisfied(
  const LaneChangeStatus & status) const
{
  const auto & route_handler = planner_data_->route_handler;
  const auto current_pose = getEgoPose();
  const auto current_twist = getEgoTwist();
  const auto & dynamic_objects = planner_data_->dynamic_object;
  const auto & common_parameters = planner_data_->parameters;

  const auto & current_lanes = status.current_lanes;

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
  const bool is_path_safe =
    std::invoke([this, &status, &route_handler, &dynamic_objects, &current_lanes, &current_pose,
                 &current_twist, &common_parameters]() {
      constexpr double check_distance = 100.0;
      // get lanes used for detection
      const auto & path = status.lane_change_path;
      const double check_distance_with_path =
        check_distance + path.preparation_length + path.lane_change_length;
      const auto check_lanes = route_handler->getCheckTargetLanesFromPath(
        path.path, status.lane_change_lanes, check_distance_with_path);

      std::unordered_map<std::string, CollisionCheckDebug> debug_data;

      const size_t current_seg_idx = motion_utils::findFirstNearestSegmentIndexWithSoftConstraints(
        path.path.points, current_pose, common_parameters.ego_nearest_dist_threshold,
        common_parameters.ego_nearest_yaw_threshold);
      return lane_change_utils::isLaneChangePathSafe(
        path.path, current_lanes, check_lanes, dynamic_objects, current_pose, current_seg_idx,
        current_twist, common_parameters, *parameters_,
        common_parameters.expected_front_deceleration_for_abort,
        common_parameters.expected_rear_deceleration_for_abort, debug_data, false,
        status.lane_change_path.acceleration);
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

bool ExternalRequestLaneChangeModule::hasFinishedLaneChange(const LaneChangeStatus & status) const
{
  const auto & current_pose = getEgoPose();
  const auto arclength_current =
    lanelet::utils::getArcCoordinates(status.lane_change_lanes, current_pose);
  const double travel_distance = arclength_current.length - status.start_distance;
  const double finish_distance = status.lane_change_path.preparation_length +
                                 status.lane_change_path.lane_change_length +
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

void ExternalRequestLaneChangeModule::updateSteeringFactorPtr(
  const BehaviorModuleOutput & output, const LaneChangeStatus & status)
{
  const auto turn_signal_info = output.turn_signal_info;
  const auto current_pose = getEgoPose();
  const double start_distance = motion_utils::calcSignedArcLength(
    output.path->points, current_pose.position, status.lane_change_path.shift_line.start.position);
  const double finish_distance = motion_utils::calcSignedArcLength(
    output.path->points, current_pose.position, status.lane_change_path.shift_line.end.position);

  const uint16_t steering_factor_direction =
    std::invoke([this, &start_distance, &finish_distance, &turn_signal_info]() {
      if (turn_signal_info.turn_signal.command == TurnIndicatorsCommand::ENABLE_LEFT) {
        waitApprovalLeft(start_distance, finish_distance);
        return SteeringFactor::LEFT;
      }
      if (turn_signal_info.turn_signal.command == TurnIndicatorsCommand::ENABLE_RIGHT) {
        waitApprovalRight(start_distance, finish_distance);
        return SteeringFactor::RIGHT;
      }
      return SteeringFactor::UNKNOWN;
    });

  // TODO(tkhmy) add handle status TRYING
  steering_factor_interface_ptr_->updateSteeringFactor(
    {status.lane_change_path.shift_line.start, status.lane_change_path.shift_line.end},
    {start_distance, finish_distance}, SteeringFactor::LANE_CHANGE, steering_factor_direction,
    SteeringFactor::TURNING, "");
}

void ExternalRequestLaneChangeModule::updateSteeringFactorPtr(
  const CandidateOutput & output, const LaneChangePath & selected_path) const
{
  const uint16_t steering_factor_direction = std::invoke([&output]() {
    if (output.lateral_shift > 0.0) {
      return SteeringFactor::LEFT;
    }
    return SteeringFactor::RIGHT;
  });

  steering_factor_interface_ptr_->updateSteeringFactor(
    {selected_path.shift_line.start, selected_path.shift_line.end},
    {output.start_distance_to_path_change, output.finish_distance_to_path_change},
    SteeringFactor::LANE_CHANGE, steering_factor_direction, SteeringFactor::APPROACHING, "");
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
void ExternalRequestLaneChangeModule::generateExtendedDrivableArea(
  PathWithLaneId & path, const LaneChangeStatus & status)
{
  const auto & common_parameters = planner_data_->parameters;
  const auto & route_handler = planner_data_->route_handler;
  const auto drivable_lanes = lane_change_utils::generateDrivableLanes(
    *route_handler, status.current_lanes, status.lane_change_lanes);
  const auto expanded_lanes = util::expandLanelets(
    drivable_lanes, parameters_->drivable_area_left_bound_offset,
    parameters_->drivable_area_right_bound_offset);

  const double & resolution = common_parameters.drivable_area_resolution;
  path.drivable_area = util::generateDrivableArea(
    path, expanded_lanes, resolution, common_parameters.vehicle_length, planner_data_);
}

void ExternalRequestLaneChangeModule::updateOutputTurnSignal(
  BehaviorModuleOutput & output, const LaneChangeStatus & status)
{
  const auto turn_signal_info = util::getPathTurnSignal(
    status.current_lanes, status.lane_change_path.shifted_path, status.lane_change_path.shift_line,
    getEgoPose(), getEgoTwist().linear.x, planner_data_->parameters);
  output.turn_signal_info.turn_signal.command = turn_signal_info.first.command;

  lane_change_utils::get_turn_signal_info(status.lane_change_path, &output.turn_signal_info);
}

void ExternalRequestLaneChangeModule::resetParameters()
{
  clearWaitingApproval();
  removeRTCStatus();
  steering_factor_interface_ptr_->clearSteeringFactors();
  object_debug_.clear();
  debug_marker_.markers.clear();
}

void ExternalRequestLaneChangeModule::acceptVisitor(
  const std::shared_ptr<SceneModuleVisitor> & visitor) const
{
  if (visitor) {
    visitor->visitExternalRequestLaneChangeModule(this);
  }
}

void SceneModuleVisitor::visitExternalRequestLaneChangeModule(
  const ExternalRequestLaneChangeModule * module) const
{
  ext_request_lane_change_visitor_ = module->get_debug_msg_array();
}
}  // namespace behavior_path_planner
