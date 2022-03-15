// Copyright 2019 Autoware Foundation
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

#ifndef MISSION_PLANNER__LANELET2_IMPL__MISSION_PLANNER_LANELET2_HPP_
#define MISSION_PLANNER__LANELET2_IMPL__MISSION_PLANNER_LANELET2_HPP_

#include <memory>
#include <string>
#include <vector>

// ROS
#include "std_srvs/srv/trigger.hpp"

#include <rclcpp/rclcpp.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

// Autoware
#include "mission_planner/mission_planner_base.hpp"

#include <route_handler/route_handler.hpp>

#include <autoware_auto_mapping_msgs/msg/had_map_bin.hpp>
#include <autoware_auto_planning_msgs/msg/had_map_route.hpp>

// lanelet
#include <lanelet2_core/LaneletMap.h>
#include <lanelet2_routing/RoutingGraph.h>
#include <lanelet2_traffic_rules/TrafficRulesFactory.h>

namespace mission_planner
{
using RouteSections = std::vector<autoware_auto_mapping_msgs::msg::HADMapSegment>;
class MissionPlannerLanelet2 : public MissionPlanner
{
public:
  explicit MissionPlannerLanelet2(const rclcpp::NodeOptions & node_options);

private:
  bool is_graph_ready_;
  lanelet::LaneletMapPtr lanelet_map_ptr_;
  lanelet::routing::RoutingGraphPtr routing_graph_ptr_;
  lanelet::traffic_rules::TrafficRulesPtr traffic_rules_ptr_;
  lanelet::ConstLanelets road_lanelets_;
  lanelet::ConstLanelets shoulder_lanelets_;
  route_handler::RouteHandler route_handler_;

  rclcpp::Subscription<autoware_auto_mapping_msgs::msg::HADMapBin>::SharedPtr map_subscriber_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_autopark_;
  rclcpp::Client<autoware_parking_srvs::srv::ParkingMissionPlan>::SharedPtr
    parking_mission_plan_client_;
  rclcpp::CallbackGroup::SharedPtr inner_callback_group_;
  rclcpp::CallbackGroup::SharedPtr inner_callback_group2_;

  void mapCallback(const autoware_auto_mapping_msgs::msg::HADMapBin::ConstSharedPtr msg);
  bool autoparkCallback(
    const std::shared_ptr<rmw_request_id_t> request_header,
    const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response);
  bool isGoalValid() const;

  // virtual functions
  bool isRoutingGraphReady() const;
  autoware_auto_planning_msgs::msg::HADMapRoute planRoute();
  void visualizeRoute(const autoware_auto_planning_msgs::msg::HADMapRoute & route) const;
};
}  // namespace mission_planner

#endif  // MISSION_PLANNER__LANELET2_IMPL__MISSION_PLANNER_LANELET2_HPP_
