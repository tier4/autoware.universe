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

  checkbox_stop_line_ = new QCheckBox("Stop Line (behavior_velocity_planner)");
  layout->addWidget(checkbox_stop_line_);

  checkbox_crosswalk_ = new QCheckBox("Crosswalk (behavior_velocity_planner)");
  layout->addWidget(checkbox_crosswalk_);

  setLayout(layout);

  connect(
    checkbox_stop_line_, &QCheckBox::toggled, this,
    &AutowareFeatureManagerPanel::onCheckStopLineFeature);
  connect(
    checkbox_crosswalk_, &QCheckBox::toggled, this,
    &AutowareFeatureManagerPanel::onCheckCrosswalkFeature);
}

void AutowareFeatureManagerPanel::onCheckStopLineFeature(bool checked)
{
  std::cerr << "\n\n stop line is now " << checked << "!!\n\n" << std::endl;

  behavior_velocity_planner::CrosswalkModulePlugin

    crosswalk

      behavior_velocity_planner::StopLineModulePlugin

        stop_line
}

void AutowareFeatureManagerPanel::onCheckCrosswalkFeature(bool checked)
{
  if (checked) {
    behavior_velocity_load_client_->async_send_request();
  }
  std::cerr << "\n\n stop line is now " << checked << "!!\n\n" << std::endl;
}

void AutowareFeatureManagerPanel::onInitialize()
{
  raw_node_ = this->getDisplayContext()->getRosNodeAbstraction().lock()->get_raw_node();

  behavior_velocity_ns =
    "/planning/scenario_planning/lane_driving/behavior_planning/behavior_velocity_planner/";
  behavior_velocity_load_client_ =
    raw_node_->create_client<behavior_velocity_planner::srv::LoadPlugin>(
      behavior_velocity_ns + "service/load_plugin");
  behavior_velocity_unload_client_ =
    raw_node_->create_client<behavior_velocity_planner::srv::UnloadPlugin>(
      behavior_velocity_ns + "service/unload_plugin");

  // capture_service_ = raw_node_->create_service<std_srvs::srv::Trigger>(
  //   "/debug/service/capture_screen",
  //   std::bind(&AutowareFeatureManagerPanel::onCaptureTrigger, this, _1, _2));
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
