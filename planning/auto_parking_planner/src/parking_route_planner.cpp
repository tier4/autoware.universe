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

PlanningResult AutoParkingPlanner::planParkingRoute()
{
  auto freespace_plan_req = std::make_shared<autoware_parking_srvs::srv::FreespacePlan::Request>();

  const auto current_pose = getEgoVehiclePose();
  for (const auto goal_pose : feasible_parking_goal_poses_) {
    PoseStamped start_pose_stamped;
    PoseStamped goal_pose_stampled;

    start_pose.header.frame_id = map_frame_;
    start_pose.pose = start_pose_filtered.at(i);
    goal_pose.header.frame_id = map_frame_;
    goal_pose.pose = goal_pose_filtered.at(i);

    freespace_plan_req->start_poses.push_back(start_pose);
    freespace_plan_req->goal_poses.push_back(goal_pose);
  }
}

}  // namespace auto_parking_planner
