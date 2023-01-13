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

#ifndef BEHAVIOR_PATH_PLANNER__UTIL__DRIVABLE_AREA_EXPANSION__PARAMETERS_HPP_
#define BEHAVIOR_PATH_PLANNER__UTIL__DRIVABLE_AREA_EXPANSION__PARAMETERS_HPP_

#include "behavior_path_planner/util/drivable_area_expansion/types.hpp"

#include <rclcpp/node.hpp>
#include <vehicle_info_util/vehicle_info_util.hpp>

#include <string>
#include <vector>

namespace drivable_area_expansion
{

struct DrivableAreaExpansionParameters
{
  static constexpr auto ENABLED_PARAM = "dynamic_expansion.enabled";
  static constexpr auto MAX_EXP_DIST_PARAM = "dynamic_expansion.max_distance";
  static constexpr auto EGO_EXTRA_OFFSET_FRONT =
    "dynamic_expansion.ego.extra_footprint_offset.front";
  static constexpr auto EGO_EXTRA_OFFSET_REAR = "dynamic_expansion.ego.extra_footprint_offset.rear";
  static constexpr auto EGO_EXTRA_OFFSET_LEFT = "dynamic_expansion.ego.extra_footprint_offset.left";
  static constexpr auto EGO_EXTRA_OFFSET_RIGHT =
    "dynamic_expansion.ego.extra_footprint_offset.right";
  static constexpr auto EGO_USE_CIRCLE_FOOTPRINT = "dynamic_expansion.ego.use_circle_footprint";
  static constexpr auto EGO_COMPENSATE = "dynamic_expansion.ego.compensate";
  static constexpr auto EGO_EXTRA_COMPENSATE_SHIFT = "dynamic_expansion.ego.extra_compensate_shift";
  static constexpr auto DYN_OBJECTS_EXTRA_OFFSET_FRONT =
    "dynamic_expansion.dynamic_objects.extra_footprint_offset.front";
  static constexpr auto DYN_OBJECTS_EXTRA_OFFSET_REAR =
    "dynamic_expansion.dynamic_objects.extra_footprint_offset.rear";
  static constexpr auto DYN_OBJECTS_EXTRA_OFFSET_LEFT =
    "dynamic_expansion.dynamic_objects.extra_footprint_offset.left";
  static constexpr auto DYN_OBJECTS_EXTRA_OFFSET_RIGHT =
    "dynamic_expansion.dynamic_objects.extra_footprint_offset.right";
  static constexpr auto AVOID_DYN_OBJECTS_PARAM = "dynamic_expansion.dynamic_objects.avoid";
  static constexpr auto AVOID_LINESTRING_TYPES_PARAM = "dynamic_expansion.avoid_linestring_types";
  static constexpr auto AVOID_LINESTRING_DIST_PARAM = "dynamic_expansion.avoid_linestring_distance";

  bool enabled = false;
  double max_expansion_distance{};
  std::vector<std::string> avoid_linestring_types{};
  bool avoid_dynamic_objects{};
  double avoid_linestring_dist{};
  double ego_left_offset;
  double ego_right_offset;
  double ego_rear_offset;
  double ego_front_offset;
  double ego_extra_left_offset;
  double ego_extra_right_offset;
  double ego_extra_rear_offset;
  double ego_extra_front_offset;
  bool ego_use_circle_footprint;
  bool ego_compensate;
  double ego_extra_compensate_shift;
  double dynamic_objects_extra_left_offset;
  double dynamic_objects_extra_right_offset;
  double dynamic_objects_extra_rear_offset;
  double dynamic_objects_extra_front_offset;

  DrivableAreaExpansionParameters() = default;
  explicit DrivableAreaExpansionParameters(rclcpp::Node & node) { init(node); }

  void init(rclcpp::Node & node)
  {
    enabled = node.declare_parameter<bool>(ENABLED_PARAM);
    max_expansion_distance = node.declare_parameter<double>(MAX_EXP_DIST_PARAM);
    ego_extra_front_offset = node.declare_parameter<double>(EGO_EXTRA_OFFSET_FRONT);
    ego_extra_rear_offset = node.declare_parameter<double>(EGO_EXTRA_OFFSET_REAR);
    ego_extra_left_offset = node.declare_parameter<double>(EGO_EXTRA_OFFSET_LEFT);
    ego_extra_right_offset = node.declare_parameter<double>(EGO_EXTRA_OFFSET_RIGHT);
    ego_use_circle_footprint = node.declare_parameter<bool>(EGO_USE_CIRCLE_FOOTPRINT);
    ego_compensate = node.declare_parameter<bool>(EGO_COMPENSATE);
    ego_extra_compensate_shift = node.declare_parameter<double>(EGO_EXTRA_COMPENSATE_SHIFT);
    dynamic_objects_extra_front_offset =
      node.declare_parameter<double>(DYN_OBJECTS_EXTRA_OFFSET_FRONT);
    dynamic_objects_extra_rear_offset =
      node.declare_parameter<double>(DYN_OBJECTS_EXTRA_OFFSET_REAR);
    dynamic_objects_extra_left_offset =
      node.declare_parameter<double>(DYN_OBJECTS_EXTRA_OFFSET_LEFT);
    dynamic_objects_extra_right_offset =
      node.declare_parameter<double>(DYN_OBJECTS_EXTRA_OFFSET_RIGHT);
    avoid_linestring_types =
      node.declare_parameter<std::vector<std::string>>(AVOID_LINESTRING_TYPES_PARAM);
    avoid_dynamic_objects = node.declare_parameter<bool>(AVOID_DYN_OBJECTS_PARAM);
    avoid_linestring_dist = node.declare_parameter<double>(AVOID_LINESTRING_DIST_PARAM);

    const auto vehicle_info = vehicle_info_util::VehicleInfoUtil(node).getVehicleInfo();
    ego_left_offset = vehicle_info.vehicle_width_m / 2.0;
    ego_right_offset = -vehicle_info.vehicle_width_m / 2.0;
    ego_rear_offset = -vehicle_info.rear_overhang_m;
    ego_front_offset = vehicle_info.wheel_base_m + vehicle_info.front_overhang_m;
  }
};

}  // namespace drivable_area_expansion
#endif  // BEHAVIOR_PATH_PLANNER__UTIL__DRIVABLE_AREA_EXPANSION__PARAMETERS_HPP_
