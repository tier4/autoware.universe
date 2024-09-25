// Copyright 2024 TIER IV, Inc.
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

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <autoware/behavior_path_goal_planner_module/manager.hpp>
#include <autoware/behavior_path_goal_planner_module/pull_over_planner/shift_pull_over.hpp>
#include <autoware/behavior_path_planner/behavior_path_planner_node.hpp>
#include <autoware/behavior_path_planner_common/data_manager.hpp>
#include <autoware/behavior_path_planner_common/utils/path_utils.hpp>
#include <autoware/route_handler/route_handler.hpp>
#include <autoware_lanelet2_extension/io/autoware_osm_parser.hpp>
#include <autoware_lanelet2_extension/projection/mgrs_projector.hpp>
#include <autoware_lanelet2_extension/utility/message_conversion.hpp>

#include <autoware_map_msgs/msg/lanelet_map_bin.hpp>
#include <autoware_planning_msgs/msg/lanelet_primitive.hpp>
#include <autoware_planning_msgs/msg/lanelet_route.hpp>
#include <autoware_planning_msgs/msg/lanelet_segment.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <tier4_planning_msgs/msg/path_with_lane_id.hpp>

#include <lanelet2_io/Io.h>
#include <matplotlibcpp17/pyplot.h>
#include <yaml-cpp/yaml.h>

#include <chrono>
#include <iostream>

using namespace std::chrono_literals;  // NOLINT

namespace
{
geometry_msgs::msg::Pose parse_pose(const YAML::Node & node)
{
  geometry_msgs::msg::Pose pose;
  pose.position.x = node["position"]["x"].as<double>();
  pose.position.y = node["position"]["y"].as<double>();
  pose.position.z = node["position"]["z"].as<double>();
  pose.orientation.x = node["orientation"]["x"].as<double>();
  pose.orientation.y = node["orientation"]["y"].as<double>();
  pose.orientation.z = node["orientation"]["z"].as<double>();
  pose.orientation.w = node["orientation"]["w"].as<double>();
  return pose;
}

autoware_planning_msgs::msg::LaneletPrimitive parse_lanelet_primitive(const YAML::Node & node)
{
  autoware_planning_msgs::msg::LaneletPrimitive primitive;
  primitive.id = node["id"].as<int64_t>();
  primitive.primitive_type = node["primitive_type"].as<std::string>();

  return primitive;
}

std::vector<autoware_planning_msgs::msg::LaneletPrimitive> parse_lanelet_primitives(
  const YAML::Node & node)
{
  std::vector<autoware_planning_msgs::msg::LaneletPrimitive> primitives;
  primitives.reserve(node.size());
  std::transform(node.begin(), node.end(), std::back_inserter(primitives), [&](const auto & p) {
    return parse_lanelet_primitive(p);
  });

  return primitives;
}

std::vector<autoware_planning_msgs::msg::LaneletSegment> parse_segments(const YAML::Node & node)
{
  std::vector<autoware_planning_msgs::msg::LaneletSegment> segments;
  std::transform(node.begin(), node.end(), std::back_inserter(segments), [&](const auto & input) {
    autoware_planning_msgs::msg::LaneletSegment segment;
    segment.preferred_primitive = parse_lanelet_primitive(input["preferred_primitive"]);
    segment.primitives = parse_lanelet_primitives(input["primitives"]);
    return segment;
  });

  return segments;
}
}  // namespace

void loadFromYAML(
  std::shared_ptr<autoware::behavior_path_planner::PlannerData> planner_data,
  const std::string & planner_data_yaml_path)
{
  const auto planner_data_yaml = YAML::LoadFile(planner_data_yaml_path);

  const auto route_node = planner_data_yaml["route"];
  autoware_planning_msgs::msg::LaneletRoute route_msg;
  {
    route_msg.start_pose = ::parse_pose(route_node["start_pose"]);
    route_msg.goal_pose = ::parse_pose(route_node["goal_pose"]);
    route_msg.segments = ::parse_segments(route_node["segments"]);
    route_msg.allow_modification = route_node["allow_modification"].as<bool>();
  }
  planner_data->route_handler->setRoute(route_msg);

  const auto self_odometry_node = planner_data_yaml["self_odometry"];
  auto odom_msg = std::make_shared<nav_msgs::msg::Odometry>();
  {
    odom_msg->pose.pose = ::parse_pose(self_odometry_node["pose"]["pose"]);
    // TODO(soblin): covariance, twist, accel, etc.
  }
  planner_data->self_odometry = odom_msg;
}

void plot_path_with_lane_id(
  matplotlibcpp17::axes::Axes & axes, const tier4_planning_msgs::msg::PathWithLaneId path)
{
  std::vector<double> xs, ys;
  for (const auto & point : path.points) {
    xs.push_back(point.point.pose.position.x);
    ys.push_back(point.point.pose.position.y);
  }
  axes.plot(Args(xs, ys), Kwargs("color"_a = "red", "linewidth"_a = 1.0));
}

void plot_lanelet(
  matplotlibcpp17::axes::Axes & axes, lanelet::ConstLanelet lanelet,
  const std::string & color = "blue", const double linewidth = 0.5)
{
  const auto lefts = lanelet.leftBound();
  const auto rights = lanelet.rightBound();
  std::vector<double> xs_left, ys_left;
  for (const auto & point : lefts) {
    xs_left.push_back(point.x());
    ys_left.push_back(point.y());
  }

  std::vector<double> xs_right, ys_right;
  for (const auto & point : rights) {
    xs_right.push_back(point.x());
    ys_right.push_back(point.y());
  }

  std::vector<double> xs_center, ys_center;
  for (const auto & point : lanelet.centerline()) {
    xs_center.push_back(point.x());
    ys_center.push_back(point.y());
  }

  axes.plot(Args(xs_left, ys_left), Kwargs("color"_a = color, "linewidth"_a = linewidth));
  axes.plot(Args(xs_right, ys_right), Kwargs("color"_a = color, "linewidth"_a = linewidth));
  axes.plot(
    Args(xs_center, ys_center),
    Kwargs("color"_a = "black", "linewidth"_a = linewidth, "linestyle"_a = "dashed"));
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  const auto road_shoulder_test_map_path =
    ament_index_cpp::get_package_share_directory("autoware_behavior_path_planner_common") +
    "/test_map/road_shoulder/lanelet2_map.osm";

  lanelet::ErrorMessages errors{};
  lanelet::projection::MGRSProjector projector{};
  const lanelet::LaneletMapPtr lanelet_map_ptr =
    lanelet::load(road_shoulder_test_map_path, projector, &errors);
  if (!errors.empty()) {
    for (const auto & error : errors) {
      std::cout << error << std::endl;
    }
    return 1;
  }

  autoware_map_msgs::msg::LaneletMapBin map_bin;
  lanelet::utils::conversion::toBinMsg(
    lanelet_map_ptr, &map_bin);  // TODO(soblin): pass lanelet_map_ptr to RouteHandler

  auto node_options = rclcpp::NodeOptions{};
  node_options.parameter_overrides(
    std::vector<rclcpp::Parameter>{{"launch_modules", std::vector<std::string>{}}});
  node_options.arguments(std::vector<std::string>{
    "--ros-args", "--params-file",
    ament_index_cpp::get_package_share_directory("autoware_behavior_path_planner") +
      "/config/behavior_path_planner.param.yaml",
    "--params-file",
    ament_index_cpp::get_package_share_directory("autoware_behavior_path_planner") +
      "/config/drivable_area_expansion.param.yaml",
    "--params-file",
    ament_index_cpp::get_package_share_directory("autoware_behavior_path_planner") +
      "/config/scene_module_manager.param.yaml",
    "--params-file",
    ament_index_cpp::get_package_share_directory("autoware_test_utils") +
      "/config/test_common.param.yaml",
    "--params-file",
    ament_index_cpp::get_package_share_directory("autoware_test_utils") +
      "/config/test_nearest_search.param.yaml",
    "--params-file",
    ament_index_cpp::get_package_share_directory("autoware_test_utils") +
      "/config/test_vehicle_info.param.yaml",
    "--params-file",
    ament_index_cpp::get_package_share_directory("autoware_behavior_path_goal_planner_module") +
      "/config/goal_planner.param.yaml"});
  auto node = rclcpp::Node::make_shared("plot_map", node_options);

  // init PlannerData from saved file
  auto planner_data = std::make_shared<autoware::behavior_path_planner::PlannerData>();
  planner_data->init_parameters(*node);
  planner_data->route_handler->setMap(map_bin);
  loadFromYAML(
    planner_data, ament_index_cpp::get_package_share_directory("autoware_behavior_path_planner") +
                    "/config/planner_data.yaml");

  // get current_route lanelet
  lanelet::ConstLanelet current_route_lanelet;
  planner_data->route_handler->getClosestLaneletWithinRoute(
    planner_data->self_odometry->pose.pose, &current_route_lanelet);
  std::cout << "current_route_lanelet is " << current_route_lanelet.id() << std::endl;

  // generate reference path
  const auto reference_path =
    autoware::behavior_path_planner::utils::getReferencePath(current_route_lanelet, planner_data);

  // generate vehicle_info
  const auto vehicle_info = autoware::vehicle_info_utils::VehicleInfoUtils(*node).getVehicleInfo();

  autoware::lane_departure_checker::LaneDepartureChecker lane_departure_checker{};
  lane_departure_checker.setVehicleInfo(vehicle_info);
  autoware::lane_departure_checker::Param lane_departure_checker_params;
  auto goal_planner_parameter =
    autoware::behavior_path_planner::GoalPlannerModuleManager::initGoalPlannerParameters(
      node.get(), "goal_planner.");
  lane_departure_checker_params.footprint_extra_margin =
    goal_planner_parameter.lane_departure_check_expansion_margin;
  lane_departure_checker.setParam(lane_departure_checker_params);

  // geneate ShiftPullOverPlanner and pull_over_path
  auto shift_pull_over_planner = autoware::behavior_path_planner::ShiftPullOver(
    *node, goal_planner_parameter, lane_departure_checker);
  const auto pull_over_path_opt = shift_pull_over_planner.plan(
    planner_data, reference_path, planner_data->route_handler->getGoalPose());

  pybind11::scoped_interpreter guard{};
  auto plt = matplotlibcpp17::pyplot::import();
  auto [fig, ax] = plt.subplots();

  const std::vector<lanelet::Id> ids{17756, 17757, 18493, 18494, 18496, 18497,
                                     18492, 18495, 18498, 18499, 18500};
  for (const auto & id : ids) {
    const auto lanelet = lanelet_map_ptr->laneletLayer.get(id);
    plot_lanelet(ax, lanelet);
  }

  plot_path_with_lane_id(ax, reference_path.path);
  std::cout << pull_over_path_opt.has_value() << std::endl;
  if (pull_over_path_opt) {
    const auto & pull_over_path = pull_over_path_opt.value();
    const auto full_path = pull_over_path.getFullPath();
    plot_path_with_lane_id(ax, full_path);
  }
  ax.set_aspect(Args("equal"));
  plt.show();

  rclcpp::shutdown();
  return 0;
}
