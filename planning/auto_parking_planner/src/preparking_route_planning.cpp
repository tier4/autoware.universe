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

#include <tier4_autoware_utils/tier4_autoware_utils.hpp>

#include <chrono>
#include <iterator>
#include <limits>

namespace auto_parking_planner
{

void getCloseGoalsAndCorrenspondingStarts(
  const Pose & current_pose, const ParkingMapInfo & parking_map_info, double threshold,
  std::vector<Pose> & goal_poses_out, std::vector<Pose> & start_poses_out)
{
  RCLCPP_INFO_STREAM(rclcpp::get_logger("ishida"), "<<debugging .... >>");
  for (const auto & pose : parking_map_info.parking_poses) {
    const auto dist = tier4_autoware_utils::calcDistance2d(pose, current_pose);
    RCLCPP_INFO_STREAM(rclcpp::get_logger("ishida"), "dist: " << dist);
    if (dist < threshold) {
      goal_poses_out.push_back(pose);
    }
  }

  for (const auto & goal_pose : goal_poses_out) {
    lanelet::ConstLanelet llt_closest;
    lanelet::utils::query::getClosestLanelet(parking_map_info.road_llts, goal_pose, &llt_closest);

    const auto centerline = llt_closest.centerline();
    double min_dist_from_goal = std::numeric_limits<double>::infinity();
    Pose best_start_pose;
    for (size_t i = 0; i < centerline.size() - 1; ++i) {
      const Eigen::Vector3d p0 = centerline[i];
      const Eigen::Vector3d g(goal_pose.position.x, goal_pose.position.y, goal_pose.position.z);

      const double dist_from_goal = (g - p0).norm();
      if (dist_from_goal < min_dist_from_goal) {
        // set pos
        best_start_pose.position.x = p0.x();
        best_start_pose.position.y = p0.y();
        best_start_pose.position.z = p0.z();

        // set orientation
        const Eigen::Vector3d p1 = centerline[i + 1];
        const double yaw = std::atan2(p1.y() - p0.y(), p1.x() - p0.x());
        tf2::convert(
          tier4_autoware_utils::createQuaternionFromRPY(0, 0, yaw), best_start_pose.orientation);
      }
    }
    start_poses_out.push_back(best_start_pose);
  }
}

PlanningResult AutoParkingPlanner::planPreparkingRoute() const
{
  const bool connected = freespaceplane_client_->wait_for_service(std::chrono::seconds(10));

  if (!connected) {
    const std::string message = "failed because unable to connect freespace planning server";
    RCLCPP_WARN_STREAM(get_logger(), message);
    return PlanningResult{false, ParkingMissionPlan::Request::END, HADMapRoute(), message};
  }

  while (true) {
    rclcpp::sleep_for(std::chrono::milliseconds(100));

    const auto current_pose = getEgoVehiclePose();

    std::vector<Pose> goal_pose_filtered;
    std::vector<Pose> start_pose_filtered;
    getCloseGoalsAndCorrenspondingStarts(
      current_pose.pose, *parking_map_info_, config_.euclid_threashold_length, goal_pose_filtered,
      start_pose_filtered);

    if (goal_pose_filtered.empty()) {
      RCLCPP_INFO_STREAM(get_logger(), "could not find parking space around here...");
      continue;
    }

    auto freespace_plan_req =
      std::make_shared<autoware_parking_srvs::srv::FreespacePlan::Request>();

    for (size_t i = 0; i < start_pose_filtered.size(); ++i) {
      PoseStamped start_pose;
      PoseStamped goal_pose;

      start_pose.header.frame_id = map_frame_;
      start_pose.pose = start_pose_filtered.at(i);
      goal_pose.header.frame_id = map_frame_;
      goal_pose.pose = goal_pose_filtered.at(i);

      freespace_plan_req->start_poses.push_back(start_pose);
      freespace_plan_req->goal_poses.push_back(goal_pose);
    }

    const auto f = freespaceplane_client_->async_send_request(freespace_plan_req);
    if (std::future_status::ready != f.wait_for(std::chrono::seconds(10))) {
      RCLCPP_WARN_STREAM(get_logger(), "took to long time to obtain freespace planning result..");
      continue;
    }

    const auto & result = f.get();
    RCLCPP_INFO_STREAM(get_logger(), "Obtained fresult from the freespace planning server.");

    bool at_least_one_success = false;
    std::vector<Pose> feasible_goal_poses;
    for (size_t idx = 0; idx < result->successes.size(); idx++) {
      if (result->successes[idx]) {
        at_least_one_success = true;
        feasible_goal_poses.push_back(goal_pose_filtered.at(idx));
      }
    }
    RCLCPP_WARN_STREAM(get_logger(), "ISHIDA: found feasible goals" << feasible_goal_poses.size());
    // assuming goal poses are sorted...
    if (at_least_one_success) {
      route_handler::RouteHandler route_handler(
        parking_map_info_->lanelet_map_ptr, parking_map_info_->traffic_rules_ptr,
        parking_map_info_->routing_graph_ptr);
      lanelet::ConstLanelets preparking_path;

      route_handler.planPathLaneletsBetweenCheckpoints(
        current_pose.pose, start_pose_filtered.front(), &preparking_path);
      route_handler.setRouteLanelets(preparking_path);

      HADMapRoute next_route;
      next_route.header.stamp = this->now();
      next_route.header.frame_id = map_frame_;
      next_route.segments = route_handler.createMapSegments(preparking_path);
      next_route.start_pose = current_pose.pose;
      next_route.goal_pose = start_pose_filtered.front();
      const auto next_phase = ParkingMissionPlan::Request::PARKING;
    }
  }
}

}  // namespace auto_parking_planner
