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

Pose computeLaneletCenterPose(const lanelet::ConstLanelet & lanelet)
{
  const lanelet::ConstLineString3d center_line = lanelet.centerline();
  const size_t middle_index = center_line.size() / 2;
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

std::deque<lanelet::ConstLanelets> computeCircularPathSequence(
  const ParkingMapInfo & parking_map_info, const PoseStamped & current_pose)
{
  lanelet::ConstLanelets entrance_llts;
  lanelet::ConstLanelets exit_llts;

  for (const auto & llt_pair : parking_map_info.llt_types) {
    const auto llt = parking_map_info.road_llts[llt_pair.first];
    if (llt_pair.second == ParkingLaneletType::ENTRANCE) {
      entrance_llts.push_back(llt);
    }
    if (llt_pair.second == ParkingLaneletType::EXIT) {
      exit_llts.push_back(llt);
    }
  }
  if (entrance_llts.size() != 1 || exit_llts.size() != 1) {
    throw std::runtime_error("current impl assumes only one entrance and exit");  // TODO
  }

  lanelet::Lanelet current_lanelet;
  lanelet::utils::query::getClosestLanelet(
    parking_map_info.road_llts, current_pose.pose, &current_lanelet);
  if (!std::count(
        parking_map_info.road_llts.begin(), parking_map_info.road_llts.end(), current_lanelet)) {
    throw std::runtime_error("current impl assumes car is already inside");  // TODO
  }

  auto reachable_llts = parking_map_info.routing_graph_ptr->reachableSet(
    current_lanelet, std::numeric_limits<double>::infinity());

  const auto is_leading_to_deadend = [&](const lanelet::ConstLanelet & llt) -> bool {
    // TODO: this implementation is inefficient. Reasonably, we should save the intermiediate result
    // in a cache. However, for small size parking lot, without caching is not necessarly.
    auto llt_here = llt;
    while (true) {
      const auto is_deadend =
        std::find_if(exit_llts.begin(), exit_llts.end(), [&llt_here](const auto & llt_) {
          return llt_here.id() == llt_.id();
        }) != exit_llts.end();
      if (is_deadend) return true;

      const auto & following_llts = parking_map_info.routing_graph_ptr->following(llt_here);
      if (following_llts.size() > 1) return false;
      llt_here = following_llts.front();
    }
  };

  // initialize table is visited
  std::unordered_map<size_t, bool> table_is_visited;
  for (const auto & llt : reachable_llts) {
    table_is_visited[llt.id()] = false;
  }
  for (const auto & llt : reachable_llts) {
    if (is_leading_to_deadend(llt)) table_is_visited[llt.id()] = true;
  }

  const auto is_visited_all = [&]() {
    for (const auto & p : table_is_visited) {
      if (!p.second) return false;
    }
    return true;
  };

  lanelet::ConstLanelets circling_path_whole;
  {
    lanelet::ConstLanelet llt_here = current_lanelet;
    table_is_visited[llt_here.id()] = true;
    circling_path_whole.push_back(llt_here);
    while (!is_visited_all()) {
      const auto llts_follow = parking_map_info.routing_graph_ptr->following(llt_here);
      if (llts_follow.empty()) throw std::runtime_error("strange");

      boost::optional<lanelet::ConstLanelet> llt_next = boost::none;

      lanelet::ConstLanelets candidate_next_llts;
      for (const auto & llt_follow : parking_map_info.routing_graph_ptr->following(llt_here)) {
        if (!is_leading_to_deadend(llt_follow)) candidate_next_llts.push_back(llt_follow);
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

  {
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
        auto path_patial_new = lanelet::ConstLanelets{path_partial.back()};
        if (!is_straight(path_partial.back())) {
          // TODO(HiroIshida) Must iterate until finding straight. But I this two curve lanelet
          // don't exist.
          path_partial.pop_back();
          path_patial_new.insert(path_patial_new.begin(), path_partial.back());
        }

        circling_path_seq.push_back(path_partial);
        path_partial = path_patial_new;
      }
      path_partial.push_back(llt);
    }
    circling_path_seq.push_back(path_partial);
    return circling_path_seq;
  }
}

PlanningResult AutoParkingPlanner::planCircularRoute() const
{
  const auto current_pose = getEgoVehiclePose();
  if (previous_mode_ == autoware_parking_srvs::srv::ParkingMissionPlan::Request::PARKING) {
    circular_plan_cache_.path_seq.clear();
  }

  if (circular_plan_cache_.path_seq.empty()) {
    circular_plan_cache_.path_seq = computeCircularPathSequence(*parking_map_info_, current_pose);
  }
  const auto circular_path = circular_plan_cache_.path_seq.front();
  circular_plan_cache_.path_seq.pop_front();
  circular_plan_cache_.current_path = circular_path;

  route_handler::RouteHandler route_handler(
    parking_map_info_->lanelet_map_ptr, parking_map_info_->traffic_rules_ptr,
    parking_map_info_->routing_graph_ptr);

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
