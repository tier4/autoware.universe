// Copyright 2021 Tier IV, Inc.
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

#ifndef BEHAVIOR_PATH_PLANNER__UTILS__AVOIDANCE__UTIL_HPP_
#define BEHAVIOR_PATH_PLANNER__UTILS__AVOIDANCE__UTIL_HPP_

#include "behavior_path_planner/data_manager.hpp"
#include "behavior_path_planner/utils/avoidance/avoidance_module_data.hpp"

#include <memory>
#include <string>
#include <vector>

namespace behavior_path_planner
{
using behavior_path_planner::PlannerData;

struct PolygonPoint
{
  geometry_msgs::msg::Point point;
  size_t bound_seg_idx;
  double lon_dist_to_segment;
  double lat_dist_to_bound;

  bool is_after(const PolygonPoint & other_point) const
  {
    if (bound_seg_idx == other_point.bound_seg_idx) {
      return other_point.lon_dist_to_segment < lon_dist_to_segment;
    }
    return other_point.bound_seg_idx < bound_seg_idx;
  }

  bool is_outside_bounds(const bool is_on_right) const
  {
    if (is_on_right) {
      return lat_dist_to_bound < 0.0;
    }
    return 0.0 < lat_dist_to_bound;
  };
};

bool isOnRight(const ObjectData & obj);

bool isTargetObjectType(
  const PredictedObject & object, const std::shared_ptr<AvoidanceParameters> & parameters);

double calcShiftLength(
  const bool & is_object_on_right, const double & overhang_dist, const double & avoid_margin);

bool isSameDirectionShift(const bool & is_object_on_right, const double & shift_length);

size_t findPathIndexFromArclength(
  const std::vector<double> & path_arclength_arr, const double target_arc);

ShiftedPath toShiftedPath(const PathWithLaneId & path);

ShiftLineArray toShiftLineArray(const AvoidLineArray & avoid_points);

std::vector<size_t> concatParentIds(
  const std::vector<size_t> & ids1, const std::vector<size_t> & ids2);

double lerpShiftLengthOnArc(double arc, const AvoidLine & al);

void fillLongitudinalAndLengthByClosestEnvelopeFootprint(
  const PathWithLaneId & path, const Point & ego_pos, ObjectData & obj);

double calcEnvelopeOverhangDistance(
  const ObjectData & object_data, const Pose & base_pose, Point & overhang_pose);

void setEndData(
  AvoidLine & al, const double length, const geometry_msgs::msg::Pose & end, const size_t end_idx,
  const double end_dist);

void setStartData(
  AvoidLine & al, const double start_shift_length, const geometry_msgs::msg::Pose & start,
  const size_t start_idx, const double start_dist);

Polygon2d createEnvelopePolygon(
  const ObjectData & object_data, const Pose & closest_pose, const double envelope_buffer);

std::vector<tier4_autoware_utils::Polygon2d> generateObstaclePolygonsForDrivableArea(
  const ObjectDataArray & objects, const std::shared_ptr<AvoidanceParameters> & parameters,
  const double vehicle_width);

double getLongitudinalVelocity(const Pose & p_ref, const Pose & p_target, const double v);

bool isCentroidWithinLanelets(
  const PredictedObject & object, const lanelet::ConstLanelets & target_lanelets);

lanelet::ConstLanelets getTargetLanelets(
  const std::shared_ptr<const PlannerData> & planner_data, lanelet::ConstLanelets & route_lanelets,
  const double left_offset, const double right_offset);

void insertDecelPoint(
  const Point & p_src, const double offset, const double velocity, PathWithLaneId & path,
  boost::optional<Pose> & p_out);

void fillObjectEnvelopePolygon(
  ObjectData & object_data, const ObjectDataArray & registered_objects, const Pose & closest_pose,
  const std::shared_ptr<AvoidanceParameters> & parameters);

void fillObjectMovingTime(
  ObjectData & object_data, ObjectDataArray & stopped_objects,
  const std::shared_ptr<AvoidanceParameters> & parameters);

void updateRegisteredObject(
  ObjectDataArray & registered_objects, const ObjectDataArray & now_objects,
  const std::shared_ptr<AvoidanceParameters> & parameters);

void compensateDetectionLost(
  const ObjectDataArray & registered_objects, ObjectDataArray & now_objects,
  ObjectDataArray & other_objects);

void filterTargetObjects(
  ObjectDataArray & objects, AvoidancePlanningData & data, DebugData & debug,
  const std::shared_ptr<const PlannerData> & planner_data,
  const std::shared_ptr<AvoidanceParameters> & parameters);
}  // namespace behavior_path_planner

#endif  // BEHAVIOR_PATH_PLANNER__UTILS__AVOIDANCE__UTIL_HPP_
