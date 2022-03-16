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

#ifndef AUTO_PARKING_PLANNER_HPP_
#define AUTO_PARKING_PLANNER_HPP_

#include "autoware_parking_srvs/srv/detail/parking_mission_plan__struct.hpp"
#include "autoware_parking_srvs/srv/freespace_plan.hpp"
#include "autoware_parking_srvs/srv/parking_mission_plan.hpp"
#include "rclcpp/rclcpp.hpp"
#include "route_handler/route_handler.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"

#include "autoware_auto_mapping_msgs/msg/had_map_bin.hpp"
#include "autoware_auto_planning_msgs/msg/detail/had_map_route__struct.hpp"
#include "autoware_auto_planning_msgs/msg/had_map_route.hpp"
#include "autoware_auto_planning_msgs/msg/trajectory.hpp"
#include "autoware_auto_system_msgs/msg/autoware_state.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"

#include <boost/optional.hpp>

#include <deque>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace auto_parking_planner
{

using autoware_auto_mapping_msgs::msg::HADMapBin;
using autoware_auto_planning_msgs::msg::HADMapRoute;
using autoware_auto_planning_msgs::msg::Trajectory;
using autoware_parking_srvs::srv::ParkingMissionPlan;

using geometry_msgs::msg::Pose;
using geometry_msgs::msg::PoseStamped;
using geometry_msgs::msg::TwistStamped;

struct AutoParkingConfig
{
  double lookahead_length;
  double reedsshepp_threashold_length;
  double euclid_threashold_length;
  double reedsshepp_radius;
  double freespace_plan_timeout;
};

struct SubscribedMessages
{
  autoware_auto_mapping_msgs::msg::HADMapBin::ConstSharedPtr map_ptr;
  autoware_auto_system_msgs::msg::AutowareState::ConstSharedPtr state_ptr;
  geometry_msgs::msg::TwistStamped::ConstSharedPtr twist_ptr_;
  autoware_auto_planning_msgs::msg::Trajectory::ConstSharedPtr traj_ptr_;
};

enum class ParkingLaneletType : int { ENTRANCE, EXIT, NORMAL };

struct ParkingMapInfo
{
  lanelet::ConstPolygon3d focus_region;
  lanelet::LaneletMapPtr lanelet_map_ptr;
  lanelet::traffic_rules::TrafficRulesPtr traffic_rules_ptr;
  lanelet::routing::RoutingGraphPtr routing_graph_ptr;
  lanelet::ConstLanelets road_llts;
  std::vector<Pose> parking_poses;
  std::map<size_t, ParkingLaneletType> llt_types;
};

class AutoParkingPlanner : public rclcpp::Node
{
public:
  explicit AutoParkingPlanner(const rclcpp::NodeOptions & node_options);

  // Node stuff
  rclcpp::CallbackGroup::SharedPtr cb_group_;
  rclcpp::CallbackGroup::SharedPtr cb_group_nested_;

  rclcpp::Subscription<autoware_auto_system_msgs::msg::AutowareState>::SharedPtr state_subscriber_;
  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr twist_subscriber_;
  rclcpp::Subscription<autoware_auto_mapping_msgs::msg::HADMapBin>::SharedPtr map_subscriber_;
  rclcpp::Subscription<autoware_auto_planning_msgs::msg::Trajectory>::SharedPtr traj_subscriber_;
  rclcpp::Service<autoware_parking_srvs::srv::ParkingMissionPlan>::SharedPtr srv_parking_mission_;
  rclcpp::Client<autoware_parking_srvs::srv::FreespacePlan>::SharedPtr freespaceplane_client_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  AutoParkingConfig config_;
  SubscribedMessages sub_msgs_;
  ParkingMapInfo parking_map_info_;

  std::string base_link_frame_;
  std::string map_frame_;

  mutable CircularPlanCache circular_plan_cache_;
  boost::optional<std::string> previous_mode_;

  void mapCallback(const autoware_auto_mapping_msgs::msg::HADMapBin::ConstSharedPtr msg);
  void stateCallback(const autoware_auto_system_msgs::msg::AutowareState::ConstSharedPtr msg);
  void twistCallback(const geometry_msgs::msg::TwistStamped::ConstSharedPtr msg);
  void trajCallback(const autoware_auto_planning_msgs::msg::Trajectory::ConstSharedPtr msg);

  bool transformPose(
    const PoseStamped & input_pose, PoseStamped * output_pose,
    const std::string target_frame) const;
  PoseStamped getEgoVehiclePose() const;

  bool parkingMissionPlanCallback(
    const std::shared_ptr<rmw_request_id_t> request_header,
    const std::shared_ptr<autoware_parking_srvs::srv::ParkingMissionPlan::Request> request,
    std::shared_ptr<autoware_parking_srvs::srv::ParkingMissionPlan::Response> response);

  // Related to prepare metdod
  void prepare();
  PlanningResult planCircularRoute() const;  // except circular_plan_cache_
};

/*
bool is_straight(const lanelet::ConstLanelet & llt);

Pose computeLaneletCenterPose(const lanelet::ConstLanelet & lanelet);
boost::optional<Pose> interpolateTrajectory(
  const std::vector<double> & base_x, const std::vector<double> & base_y, const double s_query);

struct AutoParkConfig
{
  double lookahead_length;
  double reedsshepp_threashold_length;
  double euclid_threashold_length;
  double reedsshepp_radius;
};

class AutoParkingPlanner : public rclcpp::Node
{
  enum class ParkingLaneletType : int { ENTRANCE, EXIT, NORMAL };  // add crossload
  using LLTPair = std::pair<lanelet::ConstLanelet, ParkingLaneletType>;

public:
  explicit AutoParkingPlanner(const rclcpp::NodeOptions & node_options);

private:
  // server client
  rclcpp::Service<autoware_parking_srvs::srv::ParkingMissionPlan>::SharedPtr srv_parking_mission_;
  rclcpp::Client<autoware_parking_srvs::srv::FreespacePlan>::SharedPtr freespaceplane_client_;

  // publisher Subscriber
  rclcpp::Publisher<HADMapRoute>::SharedPtr route_publisher_;
  rclcpp::Publisher<PoseStamped>::SharedPtr debug_goal_pose_publisher_;
  rclcpp::Publisher<PoseArray>::SharedPtr debug_target_poses_publisher_;
  rclcpp::Publisher<PoseArray>::SharedPtr debug_target_poses_publisher2_;
  rclcpp::Publisher<Trajectory>::SharedPtr debug_trajectory_publisher_;
  rclcpp::Subscription<autoware_auto_system_msgs::msg::AutowareState>::SharedPtr state_subscriber_;
  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr twist_subscriber_;
  rclcpp::Subscription<autoware_auto_mapping_msgs::msg::HADMapBin>::SharedPtr map_subscriber_;
  rclcpp::Subscription<autoware_auto_planning_msgs::msg::Trajectory>::SharedPtr traj_subscriber_;

  // callback groups
  rclcpp::CallbackGroup::SharedPtr cb_group_;
  rclcpp::CallbackGroup::SharedPtr cb_group_nested_;

  // tf things
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  AutoParkConfig config_;

  // variable sets in callbacks
  lanelet::ConstLanelets road_lanelets_;
  lanelet::LaneletMapPtr lanelet_map_ptr_;
  lanelet::routing::RoutingGraphPtr routing_graph_ptr_;
  lanelet::traffic_rules::TrafficRulesPtr traffic_rules_ptr_;
  autoware_auto_system_msgs::msg::AutowareState::ConstSharedPtr state_ptr_;
  geometry_msgs::msg::TwistStamped::ConstSharedPtr current_twist_;
  Trajectory::ConstSharedPtr trajectory_ptr_;

  // info inside parking lot
  lanelet::LaneletMapConstPtr parking_map_ptr_;
  lanelet::routing::RoutingGraphConstPtr sub_graph_ptr_;
  lanelet::ConstLanelets sub_road_lanelets_;

  std::string base_link_frame_;
  std::string map_frame_;
  double freespace_plan_timeout_;

  boost::optional<std::string> previous_mode_;

  // computation result cache
  boost::optional<HADMapRoute> previous_route_;
  std::vector<PoseStamped> parking_target_poses_;
  std::vector<AutoParkingPlanner::LLTPair> llt_pairs_;
  // TODO(HiroIshida) use previous route !!!!!!!!!!!!!

  std::deque<lanelet::ConstLanelets> circling_path_seq_;
  lanelet::ConstLanelets circling_path_;  // cached by planCircularRoute
  std::vector<PoseStamped> feasible_goal_poses_;
  // PoseStamped::ConstSharedPtr goal_pose_;
  PoseStamped::ConstSharedPtr start_pose_;

  void mapCallback(const autoware_auto_mapping_msgs::msg::HADMapBin::ConstSharedPtr msg);
  void stateCallback(const autoware_auto_system_msgs::msg::AutowareState::ConstSharedPtr msg)
  {
    this->state_ptr_ = msg;
  }
  void twistCallback(const geometry_msgs::msg::TwistStamped::ConstSharedPtr msg)
  {
    this->current_twist_ = msg;
  }
  void trajCallback(const autoware_auto_planning_msgs::msg::Trajectory::ConstSharedPtr msg);

  PoseStamped getEgoVehiclePose();

  bool transformPose(
    const PoseStamped & input_pose, PoseStamped * output_pose, const std::string target_frame);

  bool parkingMissionPlanCallback(
    const std::shared_ptr<rmw_request_id_t> request_header,
    const std::shared_ptr<autoware_parking_srvs::srv::ParkingMissionPlan::Request> request,
    std::shared_ptr<autoware_parking_srvs::srv::ParkingMissionPlan::Response> response);

  // cores
  bool isPreviousRouteFinished();
  void waitForPreviousRouteFinished();
  std::string planPreparkingRoute(boost::optional<HADMapRoute> & route_out);
  std::string planParkingRoute(boost::optional<HADMapRoute> & route_out);
  std::string planCircularRoute(boost::optional<HADMapRoute> & route_out);
  bool reset();
  std::vector<PoseStamped> findParkingTargetPoses();
  void extractSubGraphInfo(const lanelet::ConstPolygon3d & parking_lot);
  void filterLaneletsInsideParkingLot(const lanelet::ConstPolygon3d & parking_lot);
  lanelet::ConstLanelets get_connecting_path(
    const lanelet::ConstLanelet & llt_start, const lanelet::ConstLanelet & llt_end);
  void planAndCacheCirclingPathSeq(const PoseStamped & current_pose);
  HADMapRoute computeCircularRoute(const PoseStamped & current_pose);

  HADMapRoute createPreparkingRouteMsg(
    const PoseStamped & vehicle_pose, const PoseStamped & goal_pose);
  HADMapRoute createFinalRouteMsg(const PoseStamped & vehicle_pose, const PoseStamped & goal_pose);

  // PoseStamped get_nextlane_center(const PoseStamped& vehicle_pose);
  std::function<boost::optional<Pose>(double)> createInterpolatorFromCurrentPose();
  boost::optional<PoseStamped> getAheadPlanningStartPose(double ahead_dist);
  std::vector<PoseStamped> getCandidateGoalPoses(const PoseStamped & start_pose);
};
*/

}  // namespace auto_parking_planner

#endif  // AUTO_PARKING_PLANNER_HPP_
