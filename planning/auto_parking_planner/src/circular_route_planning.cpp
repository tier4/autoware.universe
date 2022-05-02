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
#include "lanelet2_core/Forward.h"
#include "lanelet2_core/primitives/Lanelet.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace auto_parking_planner
{

bool is_straight(const lanelet::ConstLanelet & llt)
{
  // straight lanelet has only 4 vertexes
  const lanelet::CompoundPolygon3d llt_poly = llt.polygon3d();
  const auto lb = llt.leftBound2d();
  const auto rb = llt.rightBound2d();
  // TODO (HiroIshida) checking just enter and exit surface can't detect non-straight
  // lanelet if lanelet is winding
  const Eigen::Vector2d entrance_surface = lb.front().basicPoint2d() - rb.front().basicPoint2d();
  const Eigen::Vector2d exit_surface = lb.back().basicPoint2d() - rb.back().basicPoint2d();
  const double angle =
    std::acos(entrance_surface.dot(exit_surface) / entrance_surface.norm() / exit_surface.norm());
  return angle < 3.1415926 * 30 / 180.0;
}

bool isGoingToExit(const lanelet::ConstLanelet & llt, const ParkingMapInfo & parking_map_info)
{
  auto llt_here = llt;
  while (true) {
    const auto it = parking_map_info.llt_type_table.find(llt.id());
    const ParkingLaneletType llt_type = it->second;
    if (llt_type == ParkingLaneletType::EXIT) {
      return true;
    }

    const auto following_llts = parking_map_info.routing_graph_ptr->following(llt_here);
    const bool is_deadend = (following_llts.size() == 0);
    const bool is_forked = (following_llts.size() > 1);
    if (is_forked || is_deadend) {
      return false;
    }

    llt_here = following_llts.front();
  }
}

Pose computeLaneletCenterPose(const lanelet::ConstLanelet & lanelet)
{
  const lanelet::ConstLineString3d center_line = lanelet.centerline();
  // const size_t middle_index = center_line.size() / 2;
  const size_t middle_index = center_line.size() - 1;
  const Eigen::Vector3d middle_point = center_line[middle_index];
  const double middle_yaw = std::atan2(
    center_line[middle_index].y() - center_line[middle_index - 1].y(),
    center_line[middle_index].x() - center_line[middle_index - 1].x());

  // Create pose msg
  auto pose = geometry_msgs::msg::Pose();
  pose.position.x = middle_point.x();
  pose.position.y = middle_point.y();
  pose.position.z = middle_point.z();
  tf2::convert(tier4_autoware_utils::createQuaternionFromRPY(0, 0, middle_yaw), pose.orientation);
  return pose;
}

boost::optional<Pose> getPoseInLaneletWithEnoughForwardMargin(
  const lanelet::ConstLanelet & llt, const ParkingMapInfo & parking_map_info, double vehicle_length)
{
  // If straight pose does not exist inside the lanelet, return booost::none
  const auto center_line = llt.centerline2d();

  const auto llt_next = parking_map_info.routing_graph_ptr->following(llt);

  // must start from 1 !
  for (size_t idx = center_line.size() - 2; idx != (size_t)0; --idx) {
    const auto p = center_line[idx].basicPoint2d();
    const auto p_next = center_line[idx + 1].basicPoint2d();
    const auto diff = p_next - p;
    const auto center_ahead = p + diff.normalized() * vehicle_length * 2.0;
    lanelet::BasicPoint2d pt(center_ahead.x(), center_ahead.y());

    if (!boost::geometry::within(pt, llt.polygon2d())) continue;
    const auto llt_nexts = parking_map_info.routing_graph_ptr->following(llt);
    for (const auto & llt_next : llt_nexts) {
      if (!boost::geometry::within(pt, llt_next.polygon2d())) continue;
    }

    const auto yaw = atan2(diff.y(), diff.x());
    Pose pose_center;
    pose_center.orientation = tier4_autoware_utils::createQuaternionFromYaw(yaw);
    pose_center.position.x = p.x();
    pose_center.position.y = p.y();
    pose_center.position.z = 0.0;
    return pose_center;
  }
  return boost::none;
}

std::deque<lanelet::ConstLanelets> computeCircularPathSequenceIfNoLoop(
  const lanelet::ConstLanelet & llt, const ParkingMapInfo & parking_map_info)
{
  lanelet::ConstLanelets llt_seqeunce{llt};
  while (true) {
    const auto && llts_following =
      parking_map_info.routing_graph_ptr->following(llt_seqeunce.back());
    llt_seqeunce.push_back(llts_following.front());
  }

  if (llt_seqeunce.empty()) {
    return std::deque<lanelet::ConstLanelets>{};
  }
  return std::deque<lanelet::ConstLanelets>{llt_seqeunce};
}

std::deque<lanelet::ConstLanelets> computeCircularPathSequenceIfLoop(
  const lanelet::ConstLanelet & llt, const ParkingMapInfo & parking_map_info,
  const AutoParkingConfig & config)
{
  auto reachable_llts =
    parking_map_info.routing_graph_ptr->reachableSet(llt, std::numeric_limits<double>::infinity());

  // initialize table is visited
  std::unordered_map<size_t, bool> table_is_visited;
  for (const auto & llt : reachable_llts) {
    table_is_visited[llt.id()] = false;
  }
  for (const auto & llt : reachable_llts) {
    if (isGoingToExit(llt, parking_map_info)) table_is_visited[llt.id()] = true;
  }

  const auto is_visited_all = [&]() {
    for (const auto & p : table_is_visited) {
      if (!p.second) return false;
    }
    return true;
  };

  lanelet::ConstLanelets circling_path_whole;
  {
    lanelet::ConstLanelet llt_here = llt;
    table_is_visited[llt_here.id()] = true;
    circling_path_whole.push_back(llt_here);
    while (!is_visited_all()) {
      const auto llts_follow = parking_map_info.routing_graph_ptr->following(llt_here);
      if (llts_follow.empty()) throw std::runtime_error("strange");

      boost::optional<lanelet::ConstLanelet> llt_next = boost::none;

      lanelet::ConstLanelets candidate_next_llts;
      for (const auto & llt_follow : parking_map_info.routing_graph_ptr->following(llt_here)) {
        if (!isGoingToExit(llt_follow, parking_map_info)) candidate_next_llts.push_back(llt_follow);
      }
      if (candidate_next_llts.empty()) throw std::runtime_error("strange..");

      if (candidate_next_llts.size() == 1) {
        llt_next = {candidate_next_llts[0]};
      } else {
        const auto & it = std::find_if(
          candidate_next_llts.begin(), candidate_next_llts.end(),
          [&](const auto & llt) { return table_is_visited[llt.id()] == false; });
        llt_next = {*it};
      }

      llt_here = *llt_next;
      circling_path_whole.push_back(llt_here);
      table_is_visited[llt_here.id()] = true;
    }
  }

  std::deque<lanelet::ConstLanelets> circling_path_seq;
  lanelet::ConstLanelets path_partial;
  for (const auto & llt : circling_path_whole) {
    const auto is_next_loopy = [&](const lanelet::ConstLanelet & llt) -> bool {
      const auto llts_follow = parking_map_info.routing_graph_ptr->following(llt);
      for (const auto & llt : llts_follow) {
        const auto it_same = std::find_if(
          path_partial.begin(), path_partial.end(),
          [&llt](const auto & llt_) { return llt.id() == llt_.id(); });
        if (it_same != path_partial.end()) {
          return true;
        }
      }
      return false;
    };

    if (is_next_loopy(llt)) {
      // When next llt is the
      auto path_partial_new = lanelet::ConstLanelets{path_partial.back()};

      while (true) {
        const auto llt_terminal = path_partial.back();
        const auto pose = getPoseInLaneletWithEnoughForwardMargin(
          llt_terminal, parking_map_info, config.vehicle_length);
        if (pose != boost::none) break;
        path_partial.pop_back();
        path_partial_new.push_back(path_partial.back());
      }

      std::reverse(path_partial_new.begin(), path_partial_new.end());
      circling_path_seq.push_back(path_partial);
      path_partial = path_partial_new;
    }
    path_partial.push_back(llt);
  }

  {
    // Add the final partial path
    while (true) {
      // modify the final partial path by pop_back()
      // so that the last llt is ensured to be "straight"
      const auto llt_terminal = path_partial.back();
      const auto pose = getPoseInLaneletWithEnoughForwardMargin(
        llt_terminal, parking_map_info, config.vehicle_length);
      if (pose != boost::none) break;
      path_partial.pop_back();
    }
    circling_path_seq.push_back(path_partial);
  }

  return circling_path_seq;
}

std::deque<lanelet::ConstLanelets> computeCircularPathSequence(
  const ParkingMapInfo & parking_map_info, const lanelet::ConstLanelet & current_lanelet,
  const AutoParkingConfig & config)
{
  lanelet::ConstLanelets entrance_llts;
  lanelet::ConstLanelets exit_llts;

  for (const auto & llt : parking_map_info.road_llts) {
    const auto it = parking_map_info.llt_type_table.find(llt.id());
    const auto llt_type = it->second;

    if (llt_type == ParkingLaneletType::ENTRANCE) {
      entrance_llts.push_back(llt);
    }
    if (llt_type == ParkingLaneletType::EXIT) {
      exit_llts.push_back(llt);
    }
  }

  if (entrance_llts.size() != 1 || exit_llts.size() != 1) {
    throw std::runtime_error("current impl assumes only one entrance and exit");  // TODO
  }

  if (isGoingToExit(current_lanelet, parking_map_info)) {
    return computeCircularPathSequenceIfNoLoop(current_lanelet, parking_map_info);
  }
  return computeCircularPathSequenceIfLoop(current_lanelet, parking_map_info, config);
}

PlanningResult AutoParkingPlanner::planCircularRoute() const
{
  const auto current_pose = getEgoVehiclePose();

  lanelet::Lanelet current_lanelet;
  lanelet::utils::query::getClosestLanelet(
    parking_map_info_->road_llts, current_pose.pose, &current_lanelet);
  if (!containLanelet(parking_map_info_->focus_region, current_lanelet)) {
    const std::string message = "failed because current lanelet is not inside the parking lot";
    RCLCPP_WARN_STREAM(get_logger(), message);
    return PlanningResult{false, ParkingMissionPlan::Request::END, HADMapRoute(), message};
  }

  if (previous_phase_ == autoware_parking_srvs::srv::ParkingMissionPlan::Request::PARKING) {
    circular_plan_cache_.path_seq.clear();
  }

  if (circular_plan_cache_.path_seq.empty()) {
    const auto path_seq = computeCircularPathSequence(*parking_map_info_, current_lanelet, config_);

    if (path_seq.empty()) {
      const std::string message = "No succeeding path exists";
      RCLCPP_INFO_STREAM(get_logger(), message);
      return PlanningResult{true, ParkingMissionPlan::Request::END, HADMapRoute(), message};
    }

    circular_plan_cache_.path_seq =
      computeCircularPathSequence(*parking_map_info_, current_lanelet, config_);
  }
  const auto circular_path = circular_plan_cache_.path_seq.front();
  circular_plan_cache_.path_seq.pop_front();
  circular_plan_cache_.current_path = circular_path;

  route_handler::RouteHandler route_handler(
    parking_map_info_->lanelet_map_ptr, parking_map_info_->traffic_rules_ptr,
    parking_map_info_->routing_graph_ptr);
  route_handler.setRouteLanelets(
    circular_path);  // TODO(HiroIshida) redundant? maybe should modify route_handler

  const auto goal_pose = getPoseInLaneletWithEnoughForwardMargin(
    circular_path.back(), *parking_map_info_, config_.vehicle_length);
  HADMapRoute next_route;
  next_route.header.stamp = this->now();
  next_route.header.frame_id = map_frame_;
  next_route.segments = route_handler.createMapSegments(circular_path);
  next_route.start_pose = current_pose.pose;
  next_route.goal_pose = goal_pose.get();
  const auto next_phase = ParkingMissionPlan::Request::PREPARKING;

  return PlanningResult{true, next_phase, next_route, ""};
}

}  // namespace auto_parking_planner
