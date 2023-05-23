// Copyright 2020 Tier IV, Inc.
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

#ifndef SCENE_WALKWAY_HPP_
#define SCENE_WALKWAY_HPP_

#include "scene_crosswalk.hpp"
#include "util.hpp"

#include <behavior_velocity_planner_common/scene_module_interface.hpp>
#include <lanelet2_extension/utility/query.hpp>
#include <rclcpp/rclcpp.hpp>

#include <autoware_auto_perception_msgs/msg/predicted_objects.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <lanelet2_core/LaneletMap.h>
#include <lanelet2_routing/RoutingGraph.h>
#include <lanelet2_routing/RoutingGraphContainer.h>

#include <memory>
#include <utility>
#include <vector>

namespace behavior_velocity_planner
{

class WalkwayModule : public SceneModuleInterface
{
public:
  struct PlannerParam
  {
    double stop_line_distance;
    double stop_duration_sec;
  };
  WalkwayModule(
    const int64_t module_id, lanelet::ConstLanelet walkway, const PlannerParam & planner_param,
    const rclcpp::Logger & logger, const rclcpp::Clock::SharedPtr clock);

  bool modifyPathVelocity(PathWithLaneId * path, StopReason * stop_reason) override;

  visualization_msgs::msg::MarkerArray createDebugMarkerArray() override;
  visualization_msgs::msg::MarkerArray createVirtualWallMarkerArray() override;

private:
  int64_t module_id_;

  [[nodiscard]] boost::optional<std::pair<double, geometry_msgs::msg::Point>> getStopLine(
    const PathWithLaneId & ego_path, bool & exist_stopline_in_map) const;

  enum class State { APPROACH, STOP, SURPASSED };

  lanelet::ConstLanelet walkway_;

  std::vector<geometry_msgs::msg::Point> path_intersects_;

  // State machine
  State state_;

  // Parameter
  PlannerParam planner_param_;

  // Debug
  DebugData debug_data_;

  std::shared_ptr<motion_utils::VirtualWallMarkerCreator> virtual_wall_marker_creator_walkway_ =
    std::make_shared<motion_utils::VirtualWallMarkerCreator>();
};
}  // namespace behavior_velocity_planner

#endif  // SCENE_WALKWAY_HPP_
