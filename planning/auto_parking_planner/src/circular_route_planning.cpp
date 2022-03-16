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
#include "route_handler/route_handler.hpp"

namespace auto_parking_planner
{

PlanningResult AutoParkingPlanner::planCircularRoute() const
{
  const auto current_pose = getEgoVehiclePose();
  if (previous_mode_ == autoware_parking_srvs::srv::ParkingMissionPlan::Request::PARKING) {
    circular_plan_cache_.path_seq.clear();
  }

  if (circular_plan_cache_.path_seq.empty()) {
    planAndCacheCirclingPathSeq(current_pose);
  }
  const auto circular_path = circular_plan_cache_.path_seq.front();
  circular_plan_cache_.path_seq.pop_front();
  circular_plan_cache_.current_path = circular_path;

  route_handler::RouteHandler route_handler(
    parking_map_info_.lanelet_map_ptr, parking_map_info_.traffic_rules_ptr,
    parking_map_info_.routing_graph_ptr);

  // time stamp will be set outsode of this method as this class is not a "rclcpp::Node"
  HADMapRoute next_route;
  next_route.header.frame_id = map_frame_;
  next_route.segments = route_handler.createMapSegments(circular_path);
  next_route.start_pose = current_pose.pose;
  next_route.goal_pose = computeLaneletCenterPose(circular_path.back());
  const auto next_phase = ParkingMissionPlan::Request::PREPARKING;

  return PlanningResult{next_phase, next_route};
}

}  // namespace auto_parking_planner
