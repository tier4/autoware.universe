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
#include "lanelet2_extension/utility/message_conversion.hpp"
#include "lanelet2_extension/utility/utilities.hpp"
#include "tier4_autoware_utils/tier4_autoware_utils.hpp"

#include <stdexcept>

namespace auto_parking_planner
{

bool containLanelet(const lanelet::ConstPolygon3d & polygon, const lanelet::ConstLanelet & llt)
{
  // if one of the vertexes of the lanlet is contained by the polygon
  // this function returns true
  const lanelet::CompoundPolygon3d llt_poly = llt.polygon3d();
  for (const auto & pt : llt_poly) {
    if (lanelet::geometry::within(pt, polygon.basicPolygon())) {
      return true;
    }
  }
  return false;
}

std::map<size_t, ParkingLaneletType> build_llt_type_table(
  lanelet::routing::RoutingGraphPtr routing_graph_ptr, const lanelet::ConstLanelets & road_llts)
{
  auto table = std::map<size_t, ParkingLaneletType>();

  for (const auto & llt : road_llts) {
    if (routing_graph_ptr->following(llt).empty()) {
      table[llt.id()] = ParkingLaneletType::EXIT;
    } else if (routing_graph_ptr->previous(llt).empty()) {
      table[llt.id()] = ParkingLaneletType::ENTRANCE;
    } else {
      table[llt.id()] = ParkingLaneletType::NORMAL;
    }
    throw std::logic_error("logically strange");
  }
  return table;
}

std::vector<Pose> get_possible_parking_poses(lanelet::LaneletMapPtr lanelet_map_ptr)
{
  std::vector<Pose> poses;
  const lanelet::ConstLineStrings3d parking_spaces =
    lanelet::utils::query::getAllParkingSpaces(lanelet_map_ptr);
  for (const auto & parking_space : parking_spaces) {
    lanelet::ConstPolygon3d polygon;
    lanelet::utils::lineStringWithWidthToPolygon(parking_space, &polygon);

    Eigen::Vector3d p0 = polygon[0];
    Eigen::Vector3d p1 = polygon[1];
    Eigen::Vector3d p2 = polygon[2];
    Eigen::Vector3d p3 = polygon[3];

    const auto center = (p0 + p1 + p2 + p3) * 0.25;
    const auto parking_space_d0 = (p1 - p0).norm();
    const auto parking_space_d1 = (p2 - p1).norm();

    const double yaw =
      (parking_space_d1 > parking_space_d0
         ? std::atan2(p1.y() - p0.y(), p1.x() - p0.x()) + M_PI * 0.5
         : std::atan2(p2.y() - p1.y(), p2.x() - p1.x()) + M_PI * 0.5);

    Pose pose;
    pose.position.x = center.x();
    pose.position.y = center.y();
    tf2::convert(tier4_autoware_utils::createQuaternionFromRPY(0, 0, yaw), pose.orientation);

    Pose pose_back = pose;
    tf2::convert(tier4_autoware_utils::createQuaternionFromRPY(0, 0, -yaw), pose_back.orientation);
    poses.push_back(pose);
  }
  return poses;
}

void build_parking_map_info(
  lanelet::LaneletMapPtr lanelet_map_ptr, const lanelet::ConstPolygon3d & focus_region,
  ParkingMapInfo & parking_map_info)
{
  lanelet::traffic_rules::TrafficRulesPtr traffic_rules_ptr;
  traffic_rules_ptr = lanelet::traffic_rules::TrafficRulesFactory::create(
    lanelet::Locations::Germany, lanelet::Participants::Vehicle);

  lanelet::Lanelets llts_inside;
  lanelet::ConstLanelets all_lanelets = lanelet::utils::query::laneletLayer(lanelet_map_ptr);
  for (const lanelet::ConstLanelet & llt : all_lanelets) {
    if (containLanelet(focus_region, llt)) {
      llts_inside.push_back(lanelet_map_ptr->laneletLayer.get(llt.id()));
    }
  }

  const lanelet::LaneletMapPtr sub_lanelet_map_ptr =
    lanelet::utils::createSubmap(llts_inside)->laneletMap();
  const lanelet::routing::RoutingGraphPtr sub_routing_graph_ptr =
    lanelet::routing::RoutingGraph::build(*sub_lanelet_map_ptr, *traffic_rules_ptr);
  const auto road_llts = lanelet::utils::query::roadLanelets(all_lanelets);
  const auto parking_poses = get_possible_parking_poses(sub_lanelet_map_ptr);
  const auto llt_types = build_llt_type_table(sub_routing_graph_ptr, road_llts);

  parking_map_info.lanelet_map_ptr = sub_lanelet_map_ptr;
  parking_map_info.routing_graph_ptr = sub_routing_graph_ptr;
  parking_map_info.traffic_rules_ptr = traffic_rules_ptr;
  parking_map_info.focus_region = focus_region;
  parking_map_info.road_llts = road_llts;
  parking_map_info.parking_poses = parking_poses;
  parking_map_info.llt_types = llt_types;
}

void AutoParkingPlanner::prepare()
{
  lanelet::LaneletMapPtr lanelet_map_ptr;
  lanelet::utils::conversion::fromBinMsg(*sub_msgs_.map_ptr, lanelet_map_ptr);

  const auto all_parking_lots = lanelet::utils::query::getAllParkingLots(lanelet_map_ptr);
  const auto nearest_parking_lot = all_parking_lots[0];  // TODO(HiroIshida): temp

  build_parking_map_info(lanelet_map_ptr, nearest_parking_lot, parking_map_info_);
}

}  // namespace auto_parking_planner
