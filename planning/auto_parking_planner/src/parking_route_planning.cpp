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

namespace auto_parking_planner
{

PlanningResult AutoParkingPlanner::planParkingRoute() const
{
  const auto current_pose = getEgoVehiclePose();
  const auto feasible_goal_poses =
    askFeasibleGoalIndex(current_pose.pose, feasible_parking_goal_poses_);

  if (feasible_goal_poses.empty()) {
    const std::string message = "feasible goal pose not found";
    RCLCPP_WARN_STREAM(get_logger(), message);
    return PlanningResult{true, ParkingMissionPlan::Request::CIRCULAR, HADMapRoute(), message};
  }

  route_handler::RouteHandler route_handler(
    parking_map_info_->lanelet_map_ptr, parking_map_info_->traffic_rules_ptr,
    parking_map_info_->routing_graph_ptr);

  lanelet::Lanelet current_lanelet;
  lanelet::utils::query::getClosestLanelet(
    parking_map_info_->road_llts, current_pose.pose, &current_lanelet);

  const auto single_llt_path = lanelet::ConstLanelets{current_lanelet};
  route_handler.setRouteLanelets(single_llt_path);

  HADMapRoute next_route;
  next_route.header.stamp = this->now();
  next_route.header.frame_id = map_frame_;
  next_route.segments = route_handler.createMapSegments(single_llt_path);
  next_route.start_pose = current_pose.pose;
  next_route.goal_pose = feasible_goal_poses.front();  // assuming that poses are sorted
  const auto next_phase = ParkingMissionPlan::Request::END;

  return PlanningResult{true, next_phase, next_route, ""};
}

}  // namespace auto_parking_planner
