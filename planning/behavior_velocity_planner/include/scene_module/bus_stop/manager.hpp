// Copyright 2023 TIER IV, Inc.
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

#ifndef SCENE_MODULE__BUS_STOP__MANAGER_HPP_
#define SCENE_MODULE__BUS_STOP__MANAGER_HPP_

#include <rclcpp/rclcpp.hpp>
#include <scene_module/bus_stop/scene.hpp>
#include <scene_module/bus_stop/state_machine.hpp>
#include <scene_module/bus_stop/turn_indicator.hpp>
#include <scene_module/scene_module_interface.hpp>

#include <autoware_auto_planning_msgs/msg/path_with_lane_id.hpp>

#include <functional>
#include <memory>

namespace behavior_velocity_planner
{
namespace bus_stop
{
class BusStopModuleManager : public SceneModuleManagerInterfaceWithRTC
{
public:
  explicit BusStopModuleManager(rclcpp::Node & node);
  const char * getModuleName() override { return "bus_stop"; }

private:
  BusStopModule::PlannerParam planner_param_;

  void launchNewModules(const autoware_auto_planning_msgs::msg::PathWithLaneId & path) override;

  std::function<bool(const std::shared_ptr<SceneModuleInterface> &)> getModuleExpiredFunction(
    const autoware_auto_planning_msgs::msg::PathWithLaneId & path) override;

  std::shared_ptr<TurnIndicator> turn_indicator_;
  std::shared_ptr<StateMachine> state_machine_;
  rclcpp::Node & node_;
};
}  // namespace bus_stop
}  // namespace behavior_velocity_planner

#endif  // SCENE_MODULE__BUS_STOP__MANAGER_HPP_
