// Copyright 2023 TIER IV, Inc. All rights reserved.
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

#include "scene_out_of_lane.hpp"

#include "calculate_slowdown_points.hpp"
#include "debug.hpp"
#include "decisions.hpp"
#include "filter_predicted_objects.hpp"
#include "footprint.hpp"
#include "lanelets_selection.hpp"
#include "overlapping_range.hpp"
#include "types.hpp"

#include <behavior_velocity_planner_common/utilization/debug.hpp>
#include <behavior_velocity_planner_common/utilization/util.hpp>
#include <lanelet2_extension/utility/query.hpp>
#include <lanelet2_extension/utility/utilities.hpp>
#include <tier4_autoware_utils/system/stop_watch.hpp>

#include <lanelet2_core/geometry/LaneletMap.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace behavior_velocity_planner::out_of_lane
{

using visualization_msgs::msg::Marker;
using visualization_msgs::msg::MarkerArray;

OutOfLaneModule::OutOfLaneModule(
  const int64_t module_id, PlannerParam planner_param, const rclcpp::Logger & logger,
  const rclcpp::Clock::SharedPtr clock)
: SceneModuleInterface(module_id, logger, clock), params_(std::move(planner_param))
{
  velocity_factor_.init(VelocityFactor::UNKNOWN);
}

inline double point_to_line_distance(const Point2d & p, const Point2d & p1, const Point2d & p2)
{
  const Point2d p2_vec = {p2.x() - p1.x(), p2.y() - p1.y()};
  const Point2d p_vec = {p.x() - p1.x(), p.y() - p1.y()};

  const auto cross = p2_vec.x() * p_vec.y() - p2_vec.y() * p_vec.x();
  const auto dist_sign = cross < 0.0 ? -1.0 : 1.0;

  const auto c1 = boost::geometry::dot_product(p_vec, p2_vec);
  const auto c2 = boost::geometry::dot_product(p2_vec, p2_vec);
  const auto projection = p1 + (p2_vec * c1 / c2);
  const auto projection_point = Point2d{projection.x(), projection.y()};
  return boost::geometry::distance(p, projection_point) * dist_sign;
}

std::vector<PathPointWithLaneId> reuse_previous_path_points(
  const PathWithLaneId & path, const std::vector<PathPointWithLaneId> & prev_path_points,
  const geometry_msgs::msg::Point & ego_point, const PlannerParam & params)
{
  LineString2d path_ls;
  for (const auto & p : path.points)
    path_ls.emplace_back(p.point.pose.position.x, p.point.pose.position.y);
  /*TODO(Maxime): param */
  constexpr auto max_deviation = 0.5;
  constexpr auto resample_interval = 0.5;
  const auto max_arc_length = std::max(params.slow_dist_threshold, params.stop_dist_threshold);
  std::vector<PathPointWithLaneId> cropped_points;
  const auto ego_is_behind =
    prev_path_points.size() > 1 &&
    motion_utils::calcLongitudinalOffsetToSegment(prev_path_points, 0, ego_point) < 0.0;
  const auto ego_is_far = !prev_path_points.empty() && tier4_autoware_utils::calcDistance2d(
                                                         ego_point, prev_path_points.front()) < 0.0;
  if (!ego_is_behind && !ego_is_far && prev_path_points.size() > 1) {
    const auto first_idx =
      motion_utils::findNearestSegmentIndex(prev_path_points, path.points.front().point.pose);
    const auto deviation =
      motion_utils::calcLateralOffset(prev_path_points, path.points.front().point.pose.position);
    if (first_idx && deviation < max_deviation) {
      for (auto idx = *first_idx; idx < prev_path_points.size(); ++idx) {
        double lateral_offset = std::numeric_limits<double>::max();
        for (auto segment_idx = 0LU; segment_idx + 1 < path_ls.size(); ++segment_idx) {
          const auto distance = point_to_line_distance(
            Point2d(
              prev_path_points[idx].point.pose.position.x,
              prev_path_points[idx].point.pose.position.y),
            path_ls[segment_idx], path_ls[segment_idx + 1]);
          lateral_offset = std::min(distance, lateral_offset);
        }
        if (lateral_offset > max_deviation) break;
        cropped_points.push_back(prev_path_points[idx]);
      }
    }
  }
  if (cropped_points.empty()) {
    const auto resampled_path_points =
      motion_utils::resamplePath(path, resample_interval, true, true, false).points;
    const auto cropped_path =
      max_arc_length <= 0.0
        ? resampled_path_points
        : motion_utils::cropForwardPoints(
            resampled_path_points, resampled_path_points.front().point.pose.position, 0,
            max_arc_length);
    for (const auto & p : cropped_path) cropped_points.push_back(p);
  } else {
    const auto initial_arc_length = motion_utils::calcArcLength(cropped_points);
    const auto max_path_arc_length = motion_utils::calcArcLength(path.points);
    const auto first_arc_length = motion_utils::calcSignedArcLength(
      path.points, path.points.front().point.pose.position,
      cropped_points.back().point.pose.position);
    for (auto arc_length = first_arc_length + resample_interval;
         (max_arc_length <= 0.0 ||
          initial_arc_length + (arc_length - first_arc_length) <= max_arc_length) &&
         arc_length <= max_path_arc_length;
         arc_length += resample_interval) {
      cropped_points.push_back(cropped_points.back());
      cropped_points.back().point.pose =
        motion_utils::calcInterpolatedPose(path.points, arc_length);
    }
  }
  return motion_utils::removeOverlapPoints(cropped_points);
}

bool OutOfLaneModule::modifyPathVelocity(PathWithLaneId * path, StopReason * stop_reason)
{
  debug_data_.reset_data();
  *stop_reason = planning_utils::initializeStopReason(StopReason::OUT_OF_LANE);
  if (!path || path->points.size() < 2) return true;
  tier4_autoware_utils::StopWatch<std::chrono::microseconds> stopwatch;
  stopwatch.tic();
  EgoData ego_data;
  ego_data.pose = planner_data_->current_odometry->pose;
  ego_data.path.points =
    reuse_previous_path_points(*path, prev_path_points_, ego_data.pose.position, params_);
  prev_path_points_ = ego_data.path.points;

  ego_data.first_path_idx =
    motion_utils::findNearestSegmentIndex(ego_data.path.points, ego_data.pose.position);
  std::printf("1st path index = %lu\n", ego_data.first_path_idx);
  ego_data.velocity = planner_data_->current_velocity->twist.linear.x;
  ego_data.max_decel = -planner_data_->max_stop_acceleration_threshold;
  stopwatch.tic("calculate_path_footprints");
  ego_data.current_footprint = calculate_current_ego_footprint(ego_data, params_, true);
  debug_data_.current_footprint = ego_data.current_footprint;
  const auto path_footprints = calculate_path_footprints(ego_data, params_);
  const auto calculate_path_footprints_us = stopwatch.toc("calculate_path_footprints");
  // Calculate lanelets to ignore and consider
  const auto path_lanelets = planning_utils::getLaneletsOnPath(
    ego_data.path, planner_data_->route_handler_->getLaneletMapPtr(),
    planner_data_->current_odometry->pose);
  const auto ignored_lanelets =
    calculate_ignored_lanelets(ego_data, path_lanelets, *planner_data_->route_handler_, params_);
  const auto other_lanelets = calculate_other_lanelets(
    ego_data, path_lanelets, ignored_lanelets, *planner_data_->route_handler_, params_);

  debug_data_.footprints = path_footprints;
  debug_data_.path_lanelets = path_lanelets;
  debug_data_.ignored_lanelets = ignored_lanelets;
  debug_data_.other_lanelets = other_lanelets;
  debug_data_.path = ego_data.path;
  debug_data_.first_path_idx = ego_data.first_path_idx;

  if (params_.skip_if_already_overlapping) {
    const auto overlapped_lanelet_it =
      std::find_if(other_lanelets.begin(), other_lanelets.end(), [&](const auto & ll) {
        return boost::geometry::intersects(
          ll.polygon2d().basicPolygon(), ego_data.current_footprint);
      });
    if (overlapped_lanelet_it != other_lanelets.end()) {
      debug_data_.current_overlapped_lanelets.push_back(*overlapped_lanelet_it);
      RCLCPP_DEBUG(logger_, "Ego is already overlapping a lane, skipping the module ()\n");
      return true;
    }
  }
  // Calculate overlapping ranges
  stopwatch.tic("calculate_overlapping_ranges");
  auto ranges =  // these are already sorted by increasing arc length along the path
    calculate_overlapping_ranges(path_footprints, path_lanelets, other_lanelets, params_);
  const auto calculate_overlapping_ranges_us = stopwatch.toc("calculate_overlapping_ranges");
  // Calculate stop and slowdown points
  stopwatch.tic("calculate_decisions");
  DecisionInputs inputs;
  inputs.ego_data = ego_data;
  inputs.objects = filter_predicted_objects(*planner_data_->predicted_objects, ego_data, params_);
  inputs.route_handler = planner_data_->route_handler_;
  inputs.lanelets = other_lanelets;
  calculate_decisions(ranges, inputs, params_, logger_);
  const auto calculate_decisions_us = stopwatch.toc("calculate_decisions");
  stopwatch.tic("calc_slowdown_points");
  if (  // reset the previous inserted point if the timer expired
    prev_inserted_point_ &&
    (clock_->now() - prev_inserted_point_time_).seconds() > params_.min_decision_duration)
    prev_inserted_point_.reset();
  auto point_to_insert = calculate_slowdown_point(ego_data, ranges, prev_inserted_point_, params_);
  const auto calc_slowdown_points_us = stopwatch.toc("calc_slowdown_points");
  stopwatch.tic("insert_slowdown_points");
  debug_data_.slowdowns.clear();
  if (  // reset the timer if there is no previous inserted point or if we avoid the same lane
    point_to_insert &&
    (!prev_inserted_point_ || prev_inserted_point_->slowdown.lane_to_avoid.id() ==
                                point_to_insert->slowdown.lane_to_avoid.id()))
    prev_inserted_point_time_ = clock_->now();
  // reuse previous stop point if there is no new one or if its velocity is not higher than the new
  // one and its arc length is lower
  const auto should_use_prev_inserted_point = [&]() {
    if (
      point_to_insert && prev_inserted_point_ &&
      prev_inserted_point_->slowdown.velocity <= point_to_insert->slowdown.velocity) {
      const auto arc_length = motion_utils::calcSignedArcLength(
        path->points, 0LU, point_to_insert->point.point.pose.position);
      const auto prev_arc_length = motion_utils::calcSignedArcLength(
        path->points, 0LU, prev_inserted_point_->point.point.pose.position);
      return prev_arc_length < arc_length;
    }
    return !point_to_insert && prev_inserted_point_;
  }();
  if (should_use_prev_inserted_point) {
    // if the path changed the prev point is no longer on the path so we project it
    const auto insert_arc_length = motion_utils::calcSignedArcLength(
      path->points, 0LU, prev_inserted_point_->point.point.pose.position);
    prev_inserted_point_->point.point.pose =
      motion_utils::calcInterpolatedPose(path->points, insert_arc_length);
    // update the target path idx
    prev_inserted_point_->slowdown.target_path_idx = motion_utils::findNearestSegmentIndex(
      ego_data.path.points, prev_inserted_point_->point.point.pose.position);
    point_to_insert = prev_inserted_point_;
  }
  if (point_to_insert) {
    prev_inserted_point_ = point_to_insert;
    RCLCPP_INFO(logger_, "Avoiding lane %lu", point_to_insert->slowdown.lane_to_avoid.id());
    debug_data_.slowdowns = {*point_to_insert};
    auto path_idx = motion_utils::findNearestSegmentIndex(
                      path->points, point_to_insert->point.point.pose.position) +
                    1;
    planning_utils::insertVelocity(
      *path, point_to_insert->point, point_to_insert->slowdown.velocity, path_idx);
    if (point_to_insert->slowdown.velocity == 0.0) {
      tier4_planning_msgs::msg::StopFactor stop_factor;
      stop_factor.stop_pose = point_to_insert->point.point.pose;
      stop_factor.dist_to_stop_pose = motion_utils::calcSignedArcLength(
        path->points, ego_data.pose.position, point_to_insert->point.point.pose.position);
      planning_utils::appendStopReason(stop_factor, stop_reason);
    }
    velocity_factor_.set(
      path->points, planner_data_->current_odometry->pose, point_to_insert->point.point.pose,
      VelocityFactor::UNKNOWN);
  }
  const auto insert_slowdown_points_us = stopwatch.toc("insert_slowdown_points");
  debug_data_.ranges = ranges;

  const auto total_time_us = stopwatch.toc();
  RCLCPP_DEBUG(
    logger_,
    "Total time = %2.2fus\n"
    "\tcalculate_path_footprints = %2.0fus\n"
    "\tcalculate_overlapping_ranges = %2.0fus\n"
    "\tcalculate_decisions = %2.0fus\n"
    "\tcalc_slowdown_points = %2.0fus\n"
    "\tinsert_slowdown_points = %2.0fus\n",
    total_time_us, calculate_path_footprints_us, calculate_overlapping_ranges_us,
    calculate_decisions_us, calc_slowdown_points_us, insert_slowdown_points_us);
  return true;
}

MarkerArray OutOfLaneModule::createDebugMarkerArray()
{
  constexpr auto z = 0.0;
  MarkerArray debug_marker_array;

  debug::add_footprint_markers(
    debug_marker_array, debug_data_.footprints, z, debug_data_.prev_footprints);
  debug::add_current_overlap_marker(
    debug_marker_array, debug_data_.current_footprint, debug_data_.current_overlapped_lanelets, z,
    debug_data_.prev_current_overlapped_lanelets);
  debug::add_lanelet_markers(
    debug_marker_array, debug_data_.path_lanelets, "path_lanelets",
    tier4_autoware_utils::createMarkerColor(0.1, 0.1, 1.0, 0.5), debug_data_.prev_path_lanelets);
  debug::add_lanelet_markers(
    debug_marker_array, debug_data_.ignored_lanelets, "ignored_lanelets",
    tier4_autoware_utils::createMarkerColor(0.7, 0.7, 0.2, 0.5), debug_data_.prev_ignored_lanelets);
  debug::add_lanelet_markers(
    debug_marker_array, debug_data_.other_lanelets, "other_lanelets",
    tier4_autoware_utils::createMarkerColor(0.4, 0.4, 0.7, 0.5), debug_data_.prev_other_lanelets);
  debug::add_range_markers(
    debug_marker_array, debug_data_.ranges, debug_data_.path, debug_data_.first_path_idx, z,
    debug_data_.prev_ranges);
  return debug_marker_array;
}

motion_utils::VirtualWalls OutOfLaneModule::createVirtualWalls()
{
  motion_utils::VirtualWalls virtual_walls;
  motion_utils::VirtualWall wall;
  wall.text = "out_of_lane";
  wall.longitudinal_offset = params_.front_offset;
  for (const auto & slowdown : debug_data_.slowdowns) {
    wall.style = slowdown.slowdown.velocity == 0.0 ? motion_utils::VirtualWallType::stop
                                                   : motion_utils::VirtualWallType::slowdown;
    wall.pose = slowdown.point.point.pose;
    virtual_walls.push_back(wall);
  }
  return virtual_walls;
}

}  // namespace behavior_velocity_planner::out_of_lane
