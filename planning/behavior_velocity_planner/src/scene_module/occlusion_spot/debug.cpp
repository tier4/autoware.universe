// Copyright 2021 Tier IV, Inc.
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

#include <scene_module/occlusion_spot/occlusion_spot_utils.hpp>
#include <scene_module/occlusion_spot/scene_occlusion_spot_in_private_road.hpp>
#include <scene_module/occlusion_spot/scene_occlusion_spot_in_public_road.hpp>
#include <tier4_autoware_utils/ros/marker_helper.hpp>
#include <utilization/marker_helper.hpp>
#include <utilization/util.hpp>

#include <string>
#include <vector>

namespace behavior_velocity_planner
{
namespace
{
visualization_msgs::msg::Marker makeArrowMarker(
  const occlusion_spot_utils::PossibleCollisionInfo & possible_collision, int id)
{
  visualization_msgs::msg::Marker debug_marker;
  debug_marker.header.frame_id = "map";
  debug_marker.ns = "occlusion spot arrow";
  debug_marker.id = id;
  debug_marker.type = visualization_msgs::msg::Marker::ARROW;
  debug_marker.pose.orientation = tier4_autoware_utils::createMarkerOrientation(0, 0, 0, 1.0);
  debug_marker.scale = tier4_autoware_utils::createMarkerScale(0.05, 0.2, 0.5);
  debug_marker.color = tier4_autoware_utils::createMarkerColor(0.1, 0.1, 0.1, 0.5);
  debug_marker.lifetime = rclcpp::Duration::from_seconds(0.5);
  geometry_msgs::msg::Point obs_point, intersection_point{};
  obs_point = possible_collision.obstacle_info.position;
  obs_point.z += 1;
  intersection_point = possible_collision.intersection_pose.position, intersection_point.z += 1;
  debug_marker.points = {obs_point, intersection_point};
  return debug_marker;
}

std::vector<visualization_msgs::msg::Marker> makeSlowDownMarkers(
  const occlusion_spot_utils::PossibleCollisionInfo & possible_collision,
  const std::string road_type, int id)
{
  // virtual wall
  std::vector<visualization_msgs::msg::Marker> debug_markers;
  visualization_msgs::msg::Marker wall_marker;
  wall_marker.header.frame_id = "map";
  wall_marker.ns = "occlusion spot slow down";
  wall_marker.lifetime = rclcpp::Duration::from_seconds(0.5);
  wall_marker.type = visualization_msgs::msg::Marker::CUBE;
  wall_marker.action = visualization_msgs::msg::Marker::ADD;
  wall_marker.id = id;
  // cylinder at collision point
  wall_marker.pose = possible_collision.intersection_pose;
  wall_marker.pose.position.z += 1.0;
  wall_marker.scale = tier4_autoware_utils::createMarkerScale(0.1, 5.0, 2.0);
  wall_marker.color = tier4_autoware_utils::createMarkerColor(1.0, 1.0, 0.0, 0.5);

  wall_marker.lifetime = rclcpp::Duration::from_seconds(0.5);
  debug_markers.emplace_back(wall_marker);

  // slow down reason marker
  visualization_msgs::msg::Marker slowdown_reason_marker;
  slowdown_reason_marker.header.frame_id = "map";
  slowdown_reason_marker.ns = "slow factor_text";
  slowdown_reason_marker.id = id;
  slowdown_reason_marker.lifetime = rclcpp::Duration::from_seconds(0.5);
  slowdown_reason_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
  slowdown_reason_marker.action = visualization_msgs::msg::Marker::ADD;
  slowdown_reason_marker.pose = possible_collision.intersection_pose;
  slowdown_reason_marker.scale = tier4_autoware_utils::createMarkerScale(0.0, 0.0, 1.0);
  slowdown_reason_marker.color = tier4_autoware_utils::createMarkerColor(1.0, 1.0, 1.0, 0.999);
  slowdown_reason_marker.text = "occlusion spot";
  debug_markers.emplace_back(slowdown_reason_marker);
  slowdown_reason_marker.scale = tier4_autoware_utils::createMarkerScale(0.0, 0.0, 0.5);
  slowdown_reason_marker.id = id + 100;
  slowdown_reason_marker.text = "\n \n" + road_type;
  debug_markers.emplace_back(slowdown_reason_marker);
  debug_markers.push_back(makeArrowMarker(possible_collision, id));
  return debug_markers;
}

std::vector<visualization_msgs::msg::Marker> makeCollisionMarkers(
  const occlusion_spot_utils::PossibleCollisionInfo & possible_collision, int id, bool show_text)
{
  std::vector<visualization_msgs::msg::Marker> debug_markers;
  visualization_msgs::msg::Marker debug_marker;
  debug_marker.header.frame_id = "map";
  debug_marker.ns = "collision_point";
  debug_marker.id = id;
  // cylinder at collision_point point
  debug_marker.type = visualization_msgs::msg::Marker::CYLINDER;
  debug_marker.pose = possible_collision.collision_path_point.pose;
  debug_marker.scale = tier4_autoware_utils::createMarkerScale(1.0, 1.0, 0.5);
  debug_marker.color = tier4_autoware_utils::createMarkerColor(1.0, 0.0, 0.0, 0.5);

  debug_marker.lifetime = rclcpp::Duration::from_seconds(0.5);
  debug_markers.push_back(debug_marker);
  // cylinder at obstacle point
  debug_marker.ns = "obstacle";
  debug_marker.type = visualization_msgs::msg::Marker::CYLINDER;
  debug_marker.pose.position = possible_collision.obstacle_info.position;
  debug_marker.color = tier4_autoware_utils::createMarkerColor(0.5, 0.5, 0.5, 0.5);
  debug_marker.scale = tier4_autoware_utils::createMarkerScale(1.0, 1.0, 1.0);
  debug_markers.push_back(debug_marker);
  if (show_text) {
    // info text at obstacle point
    debug_marker.ns = "info_obstacle";
    debug_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    debug_marker.pose = possible_collision.collision_path_point.pose;
    debug_marker.scale.z = 1.0;
    debug_marker.color = tier4_autoware_utils::createMarkerColor(1.0, 1.0, 0.0, 1.0);
    std::ostringstream string_stream;
    string_stream << "(s,d,v)=(" << possible_collision.arc_lane_dist_at_collision.length << " , "
                  << possible_collision.arc_lane_dist_at_collision.distance << " , "
                  << possible_collision.collision_path_point.longitudinal_velocity_mps << ")";
    debug_marker.text = string_stream.str();
    debug_markers.push_back(debug_marker);
  }
  return debug_markers;
}

visualization_msgs::msg::MarkerArray makePolygonMarker(
  const std::vector<lanelet::BasicPolygon2d> & polygons, double z)
{
  visualization_msgs::msg::MarkerArray debug_markers;
  visualization_msgs::msg::Marker debug_marker;
  debug_marker.header.frame_id = "map";
  debug_marker.id = 0;
  debug_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
  debug_marker.action = visualization_msgs::msg::Marker::ADD;
  debug_marker.pose.position = tier4_autoware_utils::createMarkerPosition(0.0, 0.0, z);
  debug_marker.pose.orientation = tier4_autoware_utils::createMarkerOrientation(0, 0, 0, 1.0);
  debug_marker.scale = tier4_autoware_utils::createMarkerScale(0.1, 0.1, 0.1);
  debug_marker.color = tier4_autoware_utils::createMarkerColor(1.0, 0.0, 1.0, 0.3);
  debug_marker.lifetime = rclcpp::Duration::from_seconds(0.1);
  debug_marker.ns = "sidewalk";
  for (const auto & poly : polygons) {
    for (const auto & p : poly) {
      geometry_msgs::msg::Point point =
        tier4_autoware_utils::createMarkerPosition(p.x(), p.y(), 0.0);
      debug_marker.points.push_back(point);
    }
    debug_markers.markers.push_back(debug_marker);
    debug_marker.id++;
    debug_marker.points.clear();
  }
  return debug_markers;
}

visualization_msgs::msg::MarkerArray createPathMarkerArray(
  const PathWithLaneId & path, const std::string & ns, const int64_t lane_id, const double r,
  const double g, const double b)
{
  visualization_msgs::msg::MarkerArray msg;
  int32_t uid = planning_utils::bitShift(lane_id);
  int32_t i = 0;
  for (const auto & p : path.points) {
    visualization_msgs::msg::Marker marker{};
    marker.header.frame_id = "map";
    marker.ns = ns;
    marker.id = uid + i++;
    marker.lifetime = rclcpp::Duration::from_seconds(0.3);
    marker.type = visualization_msgs::msg::Marker::ARROW;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose = p.point.pose;
    marker.scale = createMarkerScale(0.6, 0.3, 0.3);
    if (std::find(p.lane_ids.begin(), p.lane_ids.end(), lane_id) != p.lane_ids.end()) {
      // if p.lane_ids has lane_id
      marker.color = createMarkerColor(r, g, b, 0.999);
    } else {
      marker.color = createMarkerColor(0.5, 0.5, 0.5, 0.999);
    }
    msg.markers.push_back(marker);
  }

  return msg;
}

template <class T>
visualization_msgs::msg::MarkerArray createMarkers(
  T & debug_data, [[maybe_unused]] const int64_t module_id_)
{
  // add slow down markers for occlusion spot
  visualization_msgs::msg::MarkerArray occlusion_spot_slowdown_markers;
  auto & possible_collisions = debug_data.possible_collisions;
  // sort collision
  std::sort(
    possible_collisions.begin(), possible_collisions.end(),
    [](
      occlusion_spot_utils::PossibleCollisionInfo pc1,
      occlusion_spot_utils::PossibleCollisionInfo pc2) {
      return pc1.arc_lane_dist_at_collision.length < pc2.arc_lane_dist_at_collision.length;
    });

  // draw virtual wall markers
  int id = 0;
  for (const auto & possible_collision : possible_collisions) {
    std::vector<visualization_msgs::msg::Marker> collision_markers =
      makeSlowDownMarkers(possible_collision, debug_data.road_type, id++);
    occlusion_spot_slowdown_markers.markers.insert(
      occlusion_spot_slowdown_markers.markers.end(), collision_markers.begin(),
      collision_markers.end());
  }

  // draw obstacle collision
  id = 0;
  for (const auto & possible_collision : possible_collisions) {
    // debug marker
    std::vector<visualization_msgs::msg::Marker> collision_markers =
      makeCollisionMarkers(possible_collision, id++, true);
    occlusion_spot_slowdown_markers.markers.insert(
      occlusion_spot_slowdown_markers.markers.end(), collision_markers.begin(),
      collision_markers.end());
    id++;
  }
  return occlusion_spot_slowdown_markers;
}

}  // namespace

visualization_msgs::msg::MarkerArray OcclusionSpotInPublicModule::createDebugMarkerArray()
{
  const auto current_time = this->clock_->now();

  visualization_msgs::msg::MarkerArray debug_marker_array;
  appendMarkerArray(createMarkers(debug_data_, module_id_), current_time, &debug_marker_array);
  return debug_marker_array;
}
visualization_msgs::msg::MarkerArray OcclusionSpotInPrivateModule::createDebugMarkerArray()
{
  const auto current_time = this->clock_->now();

  visualization_msgs::msg::MarkerArray debug_marker_array;
  appendMarkerArray(createMarkers(debug_data_, module_id_), current_time, &debug_marker_array);
  appendMarkerArray(
    makePolygonMarker(debug_data_.sidewalks, debug_data_.z), current_time, &debug_marker_array);
  appendMarkerArray(
    createPathMarkerArray(debug_data_.path_raw, "path_raw", 0, 0.0, 1.0, 1.0), current_time,
    &debug_marker_array);
  appendMarkerArray(
    createPathMarkerArray(debug_data_.interp_path, "path_interp", 0, 0.0, 1.0, 1.0), current_time,
    &debug_marker_array);
  return debug_marker_array;
}
}  // namespace behavior_velocity_planner
