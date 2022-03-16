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

void build_partial_map_info(
  lanelet::LaneletMapPtr lanelet_map_ptr, const lanelet::ConstPolygon3d & focus_region,
  PartialMapInfo & partial_map_info)
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
  const auto llt_types = build_llt_type_table(sub_routing_graph_ptr, road_llts);

  partial_map_info.lanelet_map_ptr = sub_lanelet_map_ptr;
  partial_map_info.routing_graph_ptr = sub_routing_graph_ptr;
  partial_map_info.traffic_rules_ptr = traffic_rules_ptr;
  partial_map_info.focus_region = focus_region;
  partial_map_info.road_llts = road_llts;
  partial_map_info.llt_types = llt_types;
}

void AutoParkingPlanner::prepare()
{
  lanelet::LaneletMapPtr lanelet_map_ptr;
  lanelet::utils::conversion::fromBinMsg(*sub_msgs_.map_ptr, lanelet_map_ptr);

  const auto all_parking_lots = lanelet::utils::query::getAllParkingLots(lanelet_map_ptr);
  const auto nearest_parking_lot = all_parking_lots[0];  // TODO(HiroIshida): temp

  build_partial_map_info(lanelet_map_ptr, nearest_parking_lot, partial_map_info_);
}

}  // namespace auto_parking_planner
