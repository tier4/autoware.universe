// Copyright 2023 Tier IV, Inc.
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

#include "feature_manager_panel.hpp"

#include <rclcpp/rclcpp.hpp>

#include <ctime>
#include <filesystem>
#include <iostream>

using std::placeholders::_1;
using std::placeholders::_2;

AutowareFeatureManagerPanel::AutowareFeatureManagerPanel(QWidget * parent)
: rviz_common::Panel(parent)
{
  auto * layout = new QVBoxLayout;

  const auto addCheckbox = [&](auto & checkbox, const auto & title, const auto callback) {
    checkbox = new QCheckBox(title);
    layout->addWidget(checkbox);
    connect(checkbox, &QCheckBox::toggled, this, callback);
  };

  addCheckbox(
    checkbox_stop_line_, "Stop Line (behavior_velocity_planner)",
    &AutowareFeatureManagerPanel::onCheckStopLineFeature);
  addCheckbox(
    checkbox_crosswalk_, "Crosswalk (behavior_velocity_planner)",
    &AutowareFeatureManagerPanel::onCheckCrosswalkFeature);

  setLayout(layout);
}

void AutowareFeatureManagerPanel::onCheckStopLineFeature(bool checked)
{
  std::cerr << "\n\n stop line is now " << checked << "!!\n\n" << std::endl;

  loadOrUnloadBehaviorVelocityModule(
    checked, "behavior_velocity_planner::StopLineModulePlugin", "stop_line");
}

void AutowareFeatureManagerPanel::loadOrUnloadBehaviorVelocityModule(
  const bool load, const std::string & load_name, const std::string & unload_name)
{
  if (load) {
    const auto callback =
      [&](rclcpp::Client<behavior_velocity_planner::srv::LoadPlugin>::SharedFuture future) {
        if (!future.get()->success) {
          std::cerr << load_name << ": load failed" << std::endl;
        }
      };
    const auto req = std::make_shared<behavior_velocity_planner::srv::LoadPlugin::Request>();
    req->plugin_name = load_name;
    behavior_velocity_load_client_->async_send_request(req, callback);
  } else {
    const auto callback =
      [&](rclcpp::Client<behavior_velocity_planner::srv::UnloadPlugin>::SharedFuture future) {
        if (!future.get()->success) {
          std::cerr << unload_name << ": unload failed\n\n" << std::endl;
        }
      };
    const auto req = std::make_shared<behavior_velocity_planner::srv::UnloadPlugin::Request>();
    req->plugin_name = unload_name;
    behavior_velocity_unload_client_->async_send_request(req, callback);
  }
}

void AutowareFeatureManagerPanel::onCheckCrosswalkFeature(bool checked)
{
  std::cerr << "\n\n stop line is now " << checked << "!!\n\n" << std::endl;
  loadOrUnloadBehaviorVelocityModule(
    checked, "behavior_velocity_planner::CrosswalkModulePlugin", "crosswalk");
}

void AutowareFeatureManagerPanel::onInitialize()
{
  raw_node_ = this->getDisplayContext()->getRosNodeAbstraction().lock()->get_raw_node();

  const std::string behavior_velocity_ns =
    "/planning/scenario_planning/lane_driving/behavior_planning/behavior_velocity_planner/";
  behavior_velocity_load_client_ =
    raw_node_->create_client<behavior_velocity_planner::srv::LoadPlugin>(
      behavior_velocity_ns + "service/load_plugin");
  behavior_velocity_unload_client_ =
    raw_node_->create_client<behavior_velocity_planner::srv::UnloadPlugin>(
      behavior_velocity_ns + "service/unload_plugin");
}

void AutowareFeatureManagerPanel::save(rviz_common::Config config) const
{
  Panel::save(config);
}

void AutowareFeatureManagerPanel::load(const rviz_common::Config & config)
{
  Panel::load(config);
}

AutowareFeatureManagerPanel::~AutowareFeatureManagerPanel() = default;

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(AutowareFeatureManagerPanel, rviz_common::Panel)
