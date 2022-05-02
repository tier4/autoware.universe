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
#include "circular_graph.hpp"
#include "lanelet2_core/Forward.h"
#include "lanelet2_core/primitives/Lanelet.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace auto_parking_planner
{

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

class LaneletCircularGraph : public CircularGraphBase<lanelet::ConstLanelet>
{
public:
  explicit LaneletCircularGraph(const ParkingMapInfo & info, const AutoParkingConfig & config)
  : info_(info), config_(config)
  {
  }

  std::vector<lanelet::ConstLanelet> getFollowings(const lanelet::ConstLanelet & llt) const override
  {
    return info_.routing_graph_ptr->following(llt);
  }

  std::vector<lanelet::ConstLanelet> getReachables(const lanelet::ConstLanelet & llt) const override
  {
    return info_.routing_graph_ptr->reachableSet(llt, std::numeric_limits<double>::infinity());
  }

  bool is_stoppable(const lanelet::ConstLanelet & llt) const override
  {
    const boost::optional<Pose> pose =
      getPoseInLaneletWithEnoughForwardMargin(llt, info_, config_.vehicle_length);
    const bool is_pose_found = (pose != boost::none);
    return is_pose_found;
  }

  size_t getID(const lanelet::ConstLanelet & llt) const override { return llt.id(); }

  size_t getElementNum() const override { return info_.road_llts.size(); }

  ParkingMapInfo info_;
  AutoParkingConfig config_;
};

std::stack<lanelet::ConstLanelets> computeCircularPathSequence(
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

  const auto graph = LaneletCircularGraph(parking_map_info, config);
  const auto circular_path_seq = graph.planCircularPathSequence(current_lanelet);

  std::stack<lanelet::ConstLanelets> circular_path_stack;
  for (auto reverse_it = circular_path_seq.rbegin(); reverse_it != circular_path_seq.rend();
       reverse_it++) {
    circular_path_stack.push(*reverse_it);
  }
  return circular_path_stack;
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
    circular_path_stack_ = std::stack<lanelet::ConstLanelets>();
  }

  if (circular_path_stack_.empty()) {
    const auto circular_path_stack =
      computeCircularPathSequence(*parking_map_info_, current_lanelet, config_);

    if (circular_path_stack.empty()) {
      const std::string message = "No succeeding path exists";
      RCLCPP_INFO_STREAM(get_logger(), message);
      return PlanningResult{true, ParkingMissionPlan::Request::END, HADMapRoute(), message};
    }
    circular_path_stack_ = circular_path_stack;
  }
  const auto circular_path = circular_path_stack_.top();
  circular_path_stack_.pop();

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
