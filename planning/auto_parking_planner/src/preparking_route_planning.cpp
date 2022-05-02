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

#include <interpolation/spline_interpolation_points_2d.hpp>
#include <tier4_autoware_utils/tier4_autoware_utils.hpp>

#include "autoware_auto_planning_msgs/msg/detail/trajectory__struct.hpp"

#include <chrono>
#include <functional>
#include <iterator>
#include <limits>

namespace auto_parking_planner
{

Pose createTrajectoryBasedInterpolator(
  const Pose & current_pose, const Trajectory & traj, double ahead_dist)
{
  const auto & points = traj.points;
  std::vector<TrajectoryPoint> points_extended;
  for (const auto & p : points) {
    points_extended.push_back(p);
  }

  SplineInterpolationPoints2d itp;
  itp.calcSplineCoefficients(points_extended);
  const auto idx = tier4_autoware_utils::findNearestIndex(points_extended, current_pose.position);

  const auto pt = itp.getSplineInterpolatedPoint(idx, ahead_dist);
  const auto yaw = itp.getSplineInterpolatedYaw(idx, ahead_dist);
  auto pose = Pose();
  pose.position.x = pt.x;
  pose.position.y = pt.y;
  pose.position.z = 0.0;
  pose.orientation = tier4_autoware_utils::createQuaternionFromYaw(yaw);
  return pose;
}

std::vector<Pose> getCandidateGoalPoses(
  const Pose & start_pose, const ParkingMapInfo & parking_map_info, double threshold)
{
  std::vector<Pose> goal_poses;
  for (const auto & pose : parking_map_info.parking_poses) {
    const auto dist = tier4_autoware_utils::calcDistance2d(pose, start_pose);
    if (dist < threshold) {
      goal_poses.push_back(pose);
    }
  }
  return goal_poses;
}

double compute_lookehead_length(
  const autoware_auto_vehicle_msgs::msg::VelocityReport & velocity,
  const AutoParkingConfig & config)
{
  constexpr double timeout = 1.0;
  const double length_cand = velocity.longitudinal_velocity * timeout + config.lookahead_length_min;
  const double length = std::min(length_cand, config.lookahead_length_max);
  return length;
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

    if (this->previousRouteFinished()) {
      const std::string message = "Reached the terminal of a circular trajectory";
      RCLCPP_INFO_STREAM(get_logger(), message);
      return PlanningResult{true, ParkingMissionPlan::Request::CIRCULAR, HADMapRoute(), message};
    }

    const auto current_pose = getEgoVehiclePose();
    if (!sub_msgs_.velocity_ptr_ || !sub_msgs_.traj_ptr_ || sub_msgs_.traj_ptr_->points.empty()) {
      continue;
    }
    const auto lookahead_length = compute_lookehead_length(*sub_msgs_.velocity_ptr_, config_);
    RCLCPP_INFO_STREAM(get_logger(), "lookahead_length: " << lookahead_length << " [m]");
    const auto lookahead_pose =
      createTrajectoryBasedInterpolator(current_pose.pose, *sub_msgs_.traj_ptr_, lookahead_length);

    std::vector<Pose> goal_pose_filtered;
    const auto cand_goal_poses =
      getCandidateGoalPoses(lookahead_pose, *parking_map_info_, config_.euclid_threashold_length);
    if (cand_goal_poses.empty()) {
      RCLCPP_INFO_STREAM(get_logger(), "could not find parking space around here...");
      continue;
    }

    const auto feasible_goal_poses = askFeasibleGoalIndex(lookahead_pose, cand_goal_poses);
    if (feasible_goal_poses.empty()) {
      continue;
    }
    RCLCPP_WARN_STREAM(get_logger(), "found " << feasible_goal_poses.size() << " feasible goals");

    // cache
    feasible_parking_goal_poses_.clear();
    for (const auto & feasible_goal_pose : feasible_goal_poses) {
      feasible_parking_goal_poses_.push_back(feasible_goal_pose);
    }

    route_handler::RouteHandler route_handler(
      parking_map_info_->lanelet_map_ptr, parking_map_info_->traffic_rules_ptr,
      parking_map_info_->routing_graph_ptr);
    lanelet::ConstLanelets preparking_path;

    route_handler.planPathLaneletsBetweenCheckpoints(
      current_pose.pose, lookahead_pose, &preparking_path);
    route_handler.setRouteLanelets(preparking_path);

    HADMapRoute next_route;
    next_route.header.stamp = this->now();
    next_route.header.frame_id = map_frame_;
    next_route.segments = route_handler.createMapSegments(preparking_path);
    next_route.start_pose = current_pose.pose;
    next_route.goal_pose = lookahead_pose;
    const auto next_phase = ParkingMissionPlan::Request::PARKING;
    return PlanningResult{true, next_phase, next_route, ""};
  }
}

}  // namespace auto_parking_planner
