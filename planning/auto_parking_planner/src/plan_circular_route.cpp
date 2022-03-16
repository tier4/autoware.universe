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

std::vector<HADMapSegment> createMapSegments(const lanelet::ConstLanelets & path_lanelets) const
{
  // Fetched from route_handler
  const auto main_path = getMainLanelets(path_lanelets);

  std::vector<HADMapSegment> route_sections;

  if (main_path.empty()) {
    return route_sections;
  }

  for (const auto & main_llt : main_path) {
    HADMapSegment route_section_msg;
    lanelet::ConstLanelets route_section_lanelets = getNeighborsWithinRoute(main_llt);
    route_section_msg.preferred_primitive_id = main_llt.id();
    for (const auto & section_llt : route_section_lanelets) {
      MapPrimitive p;
      p.id = section_llt.id();
      p.primitive_type = "lane";
      route_section_msg.primitives.push_back(p);
    }
    route_sections.push_back(route_section_msg);
  }
  return route_sections;
}

std::string AutoParkingPlanner::planCircularRoute(boost::optional<HADMapRoute> & route_out)
{
  const auto current_pose = getEgoVehiclePose();
  if (previous_mode_ == autoware_parking_srvs::srv::ParkingMissionPlan::Request::PARKING) {
    circling_path_seq_.clear();
  }

  if (circling_path_seq_.empty()) {
    planAndCacheCirclingPathSeq(current_pose);
  }
  const auto circling_path = circling_path_seq_.front();
  circling_path_seq_.pop_front();

  const auto entire_path = circling_path;
  circling_path_ = circling_path;

  // auto current_to_parklot_path = get_connecting_path(current_lanelet, circling_path_.at(0));
  //  auto entire_path = current_to_parklot_path;
  //  entire_path.insert(entire_path.end(), circling_path_.begin()+1, circling_path_.end());

  route_handler::RouteHandler route_handler(lanelet_map_ptr_, routing_graph_ptr_, entire_path);

  // time stamp will be set outsode of this method as this class is not a "rclcpp::Node"
  autoware_planning_msgs::msg::HADMapRoute route_msg;
  route_msg.header.frame_id = map_frame_;
  route_msg.segments = route_handler.createRouteSections(entire_path);
  route_msg.start_pose = current_pose.pose();
  route_msg.goal_pose = computeLaneletCenterPose(entire_path.back());

  route_out = route_msg;
  return ParkingMissionPlan::Request::PREPARKING;
}

std::vector<PoseStamped> AutoParkingPlanner::getCandidateGoalPoses(const PoseStamped & start_pose)
{
  RCLCPP_INFO_STREAM(
    rclcpp::get_logger("ishida_debug->getCandidateGoalPoses"),
    "1: inside rs-radius: " << config_.reedsshepp_radius
                            << ", radius: " << config_.euclid_threashold_length);
  // TODO get parameter from free space planner
  auto rs_space = freespace_planning_algorithms::ReedsSheppStateSpace(config_.reedsshepp_radius);

  const auto evaluate_reedsshepp_dist = [&](const geometry_msgs::msg::PoseStamped & target_pose) {
    using rs = freespace_planning_algorithms::ReedsSheppStateSpace;
    const auto pose2state = [](const geometry_msgs::msg::PoseStamped & pose) {
      return rs::StateXYT{
        pose.pose.position.x, pose.pose.position.y, tf2::getYaw(pose.pose.orientation)};
    };
    const auto s_start = pose2state(start_pose);
    const auto s_goal = pose2state(target_pose);
    return rs_space.distance(s_start, s_goal);
  };

  // caching the sorted result
  std::sort(
    parking_target_poses_.begin(), parking_target_poses_.end(),
    [&evaluate_reedsshepp_dist](const auto & a, const auto & b) {
      return evaluate_reedsshepp_dist(a) < evaluate_reedsshepp_dist(b);
    });

  const auto predicate_isInBall = [&](const PoseStamped & pose) -> bool {
    return autoware_utils::calcDistance2d(pose, start_pose) < config_.euclid_threashold_length &&
           evaluate_reedsshepp_dist(pose) < config_.reedsshepp_threashold_length;
  };

  std::vector<PoseStamped> target_poses_cand;
  std::copy_if(
    parking_target_poses_.begin(), parking_target_poses_.end(),
    std::back_inserter(target_poses_cand), predicate_isInBall);
  RCLCPP_INFO_STREAM(
    rclcpp::get_logger("ishida_debug->getCandidateGoalPoses"),
    "2: num of cand poses " << parking_target_poses_.size());
  return target_poses_cand;
}

void AutoParkingPlanner::planAndCacheCirclingPathSeq(const PoseStamped & current_pose)
{
  lanelet::ConstLanelets entrance_llts, exit_llts;
  for (const auto & llt_pair : llt_pairs_) {
    if (llt_pair.second == ParkingLaneletType::ENTRANCE) {
      entrance_llts.push_back(llt_pair.first);
    }
    if (llt_pair.second == ParkingLaneletType::EXIT) {
      exit_llts.push_back(llt_pair.first);
    }
  }
  if (entrance_llts.size() != 1 || exit_llts.size() != 1) {
    throw std::runtime_error("current impl assumes only one entrance and exit");  // TODO
  }

  lanelet::Lanelet current_lanelet;
  lanelet::utils::query::getClosestLanelet(road_lanelets_, current_pose.pose, &current_lanelet);
  if (!std::count(sub_road_lanelets_.begin(), sub_road_lanelets_.end(), current_lanelet)) {
    throw std::runtime_error("current impl assumes car is already inside");  // TODO
  }

  auto reachable_llts =
    sub_graph_ptr_->reachableSet(current_lanelet, std::numeric_limits<double>::infinity());

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

      const auto & following_llts = sub_graph_ptr_->following(llt_here);
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
      const auto llts_follow = sub_graph_ptr_->following(llt_here);
      if (llts_follow.size() == 0) throw std::runtime_error("strange");

      boost::optional<lanelet::ConstLanelet> llt_next = boost::none;

      lanelet::ConstLanelets candidate_next_llts;
      for (const auto & llt_follow : sub_graph_ptr_->following(llt_here)) {
        if (!is_leading_to_deadend(llt_follow)) candidate_next_llts.push_back(llt_follow);
      }
      if (candidate_next_llts.size() == 0) throw std::runtime_error("strange..");

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
        const auto llts_follow = routing_graph_ptr_->following(llt);
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
    circling_path_seq_ = circling_path_seq;
  }
}

}  // namespace auto_parking_planner
