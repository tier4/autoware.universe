// Copyright 2020 Tier IV, Inc.
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

#ifndef DUMMY_PERCEPTION_PUBLISHER__RANDOM_OBJECTS_PUBLISHER_HPP_
#define DUMMY_PERCEPTION_PUBLISHER__RANDOM_OBJECTS_PUBLISHER_HPP_

#include <lanelet2_extension/utility/message_conversion.hpp>
#include <lanelet2_extension/utility/query.hpp>
#include <rclcpp/rclcpp.hpp>

#include <autoware_auto_mapping_msgs/msg/had_map_bin.hpp>
#include <autoware_auto_perception_msgs/msg/detected_objects.hpp>
#include <autoware_auto_planning_msgs/msg/path.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tier4_perception_msgs/msg/detected_objects_with_feature.hpp>

#include <lanelet2_core/LaneletMap.h>
#include <lanelet2_core/geometry/BoundingBox.h>
#include <lanelet2_core/geometry/Lanelet.h>
#include <lanelet2_core/geometry/Point.h>
#include <lanelet2_routing/RoutingGraph.h>
#include <lanelet2_traffic_rules/TrafficRulesFactory.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <memory>
#include <random>
#include <vector>

class VehicleObject
{
public:
  VehicleObject(
    const lanelet::ConstLanelet & start_lanelet, const geometry_msgs::msg::Pose & initial_pose,
    const double initial_vel)
  : nearest_lanelet(start_lanelet), pose(initial_pose), velocity(initial_vel)
  {
  }

  lanelet::ConstLanelet nearest_lanelet;
  geometry_msgs::msg::Pose pose;
  double velocity;
};

class RandomObjectsPublisher : public rclcpp::Node
{
public:
  RandomObjectsPublisher();

private:
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  std::vector<VehicleObject> vehicle_objects_;
  double update_hz_;

  rclcpp::Subscription<autoware_auto_planning_msgs::msg::Path>::SharedPtr path_sub_;
  rclcpp::Subscription<autoware_auto_mapping_msgs::msg::HADMapBin>::SharedPtr map_sub_;

  autoware_auto_planning_msgs::msg::Path::ConstSharedPtr path_ptr_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_pub_;
  rclcpp::Publisher<tier4_perception_msgs::msg::DetectedObjectsWithFeature>::SharedPtr
    detected_object_with_feature_pub_;

  // Lanelet Map Pointers
  std::shared_ptr<lanelet::LaneletMap> lanelet_map_ptr_;
  std::shared_ptr<lanelet::routing::RoutingGraph> routing_graph_ptr_;
  std::shared_ptr<lanelet::traffic_rules::TrafficRules> traffic_rules_ptr_;
  lanelet::ConstLanelets lanelets_;

  rclcpp::TimerBase::SharedPtr timer_;

  void onMap(const autoware_auto_mapping_msgs::msg::HADMapBin::ConstSharedPtr msg);
  void onTimer();
  void spawnVehicleObject();
  void step();
  void validateObject();
};

#endif  // DUMMY_PERCEPTION_PUBLISHER__RANDOM_OBJECTS_PUBLISHER_HPP_
