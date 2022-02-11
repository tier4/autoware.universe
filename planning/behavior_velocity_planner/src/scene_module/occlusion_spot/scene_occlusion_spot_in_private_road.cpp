// Copyright 2020 Tier IV, Inc. All rights reserved.
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

#include <lanelet2_extension/utility/utilities.hpp>
#include <scene_module/occlusion_spot/occlusion_spot_utils.hpp>
#include <scene_module/occlusion_spot/risk_predictive_braking.hpp>
#include <scene_module/occlusion_spot/scene_occlusion_spot_in_private_road.hpp>
#include <utilization/util.hpp>

#include <lanelet2_core/primitives/BasicRegulatoryElements.h>

#include <memory>
#include <vector>

namespace behavior_velocity_planner
{
using occlusion_spot_utils::ROAD_TYPE::PRIVATE;
namespace bg = boost::geometry;
namespace lg = lanelet::geometry;
namespace utils = occlusion_spot_utils;

OcclusionSpotInPrivateModule::OcclusionSpotInPrivateModule(
  const int64_t module_id, [[maybe_unused]] std::shared_ptr<const PlannerData> planner_data,
  const PlannerParam & planner_param, const rclcpp::Logger logger,
  const rclcpp::Clock::SharedPtr clock,
  const rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr publisher)
: SceneModuleInterface(module_id, logger, clock), publisher_(publisher)
{
  param_ = planner_param;
}

bool OcclusionSpotInPrivateModule::modifyPathVelocity(
  autoware_auto_planning_msgs::msg::PathWithLaneId * path,
  [[maybe_unused]] tier4_planning_msgs::msg::StopReason * stop_reason)
{
  debug_data_ = DebugData();
  if (path->points.size() < 2) {
    return true;
  }
  param_.vehicle_info.baselink_to_front = planner_data_->vehicle_info_.max_longitudinal_offset_m;
  param_.vehicle_info.vehicle_width = planner_data_->vehicle_info_.vehicle_width_m;

  const geometry_msgs::msg::Pose ego_pose = planner_data_->current_pose.pose;
  const double ego_velocity = planner_data_->current_velocity->twist.linear.x;
  const auto & lanelet_map_ptr = planner_data_->lanelet_map;
  const auto & traffic_rules_ptr = planner_data_->traffic_rules;
  const auto & occ_grid_ptr = planner_data_->occupancy_grid;
  if (!lanelet_map_ptr || !traffic_rules_ptr || !occ_grid_ptr) {
    return true;
  }

  const double max_range = 50.0;
  PathWithLaneId clipped_path;
  utils::clipPathByLength(*path, clipped_path, max_range);
  PathWithLaneId interp_path;
  utils::splineInterpolate(clipped_path, 1.0, &interp_path, logger_);
  debug_data_.interp_path = interp_path;
  int closest_idx = -1;
  if (!planning_utils::calcClosestIndex<PathWithLaneId>(
        interp_path, ego_pose, closest_idx, param_.dist_thr, param_.angle_thr)) {
    return true;
  }
  // return if ego is final point of interpolated path
  if (closest_idx == static_cast<int>(interp_path.points.size()) - 1) return true;
  DetectionAreaIdx focus_area =
    extractTargetRoadArcLength(lanelet_map_ptr, max_range, *path, PRIVATE);
  if (!focus_area) return true;
  nav_msgs::msg::OccupancyGrid occ_grid = *occ_grid_ptr;
  grid_map::GridMap grid_map;
  grid_utils::denoiseOccupancyGridCV(occ_grid, grid_map, param_.grid);
  if (param_.show_debug_grid) {
    publisher_->publish(occ_grid);
  }
  double offset_from_start_to_ego = utils::offsetFromStartToEgo(interp_path, ego_pose, closest_idx);
  RCLCPP_DEBUG_STREAM_THROTTLE(logger_, *clock_, 3000, "closest_idx : " << closest_idx);
  RCLCPP_DEBUG_STREAM_THROTTLE(
    logger_, *clock_, 3000, "offset_from_start_to_ego : " << offset_from_start_to_ego);
  std::vector<utils::PossibleCollisionInfo> possible_collisions;
  // Note: Don't consider offset from path start to ego here
  utils::generateSidewalkPossibleCollisions(
    possible_collisions, grid_map, interp_path, offset_from_start_to_ego, param_,
    debug_data_.sidewalks);
  utils::filterCollisionByRoadType(possible_collisions, focus_area);
  RCLCPP_DEBUG_STREAM_THROTTLE(
    logger_, *clock_, 3000, "num possible collision:" << possible_collisions.size());
  utils::calcSlowDownPointsForPossibleCollision(0, interp_path, 0.0, possible_collisions);
  // Note: Consider offset from path start to ego here
  utils::handleCollisionOffset(possible_collisions, offset_from_start_to_ego, 0.0);
  // apply safe velocity using ebs and pbs deceleration
  applySafeVelocityConsideringPossibleCollison(
    path, possible_collisions, ego_velocity, param_.private_road, param_);
  debug_data_.z = path->points.front().point.pose.position.z;
  debug_data_.possible_collisions = possible_collisions;
  debug_data_.path_raw = *path;
  return true;
}

}  // namespace behavior_velocity_planner
