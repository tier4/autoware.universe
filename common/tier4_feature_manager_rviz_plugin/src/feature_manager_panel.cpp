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
    checkbox->setChecked(true);  // initialize
    connect(checkbox, &QCheckBox::toggled, this, callback);
  };

  const auto addCheckbox_UnSupported = [&](
                                         auto & checkbox, const auto & title, const auto callback) {
    checkbox = new QCheckBox(title);
    layout->addWidget(checkbox);
    checkbox->setChecked(true);
    checkbox->setEnabled(false);  // gray out (disabled)
    connect(checkbox, &QCheckBox::toggled, this, callback);
  };

  // behavior path planner
  addCheckbox_UnSupported(
    checkbox_start_, "[WIP] behavior_path_planner: Start",
    &AutowareFeatureManagerPanel::onCheckStartFeature);
  addCheckbox_UnSupported(
    checkbox_goal_, "[WIP] behavior_path_planner: Goal",
    &AutowareFeatureManagerPanel::onCheckGoalFeature);
  addCheckbox_UnSupported(
    checkbox_lane_change_, "[WIP] behavior_path_planner: Lane Change",
    &AutowareFeatureManagerPanel::onCheckLaneChangeFeature);
  addCheckbox_UnSupported(
    checkbox_avoidance_, "[WIP] behavior_path_planner: Avoidance",
    &AutowareFeatureManagerPanel::onCheckAvoidanceFeature);
  addCheckbox_UnSupported(
    checkbox_side_shift_, "[WIP] behavior_path_planner: Side Shift",
    &AutowareFeatureManagerPanel::onCheckSideShiftFeature);

  // behavior velocity planner
  addCheckbox(
    checkbox_crosswalk_, "behavior_velocity_planner: Crosswalk",
    &AutowareFeatureManagerPanel::onCheckCrosswalkFeature);
  addCheckbox(
    checkbox_walkway_, "behavior_velocity_planner: Walkway",
    &AutowareFeatureManagerPanel::onCheckWalkwayFeature);
  addCheckbox(
    checkbox_traffic_light_, "behavior_velocity_planner: Traffic Light",
    &AutowareFeatureManagerPanel::onCheckTrafficLightFeature);
  addCheckbox(
    checkbox_intersection_, "behavior_velocity_planner: Intersection",
    &AutowareFeatureManagerPanel::onCheckIntersectionFeature);
  addCheckbox(
    checkbox_merge_from_private_, "behavior_velocity_planner: Merge From Private",
    &AutowareFeatureManagerPanel::onCheckMergeFromPrivateFeature);
  addCheckbox(
    checkbox_blind_spot_, "behavior_velocity_planner: Blind Spot",
    &AutowareFeatureManagerPanel::onCheckBlindSpotFeature);
  addCheckbox(
    checkbox_detection_area_, "behavior_velocity_planner: Detection Area",
    &AutowareFeatureManagerPanel::onCheckDetectionAreaFeature);
  addCheckbox(
    checkbox_virtual_traffic_light_, "behavior_velocity_planner: Virtual Traffic Light",
    &AutowareFeatureManagerPanel::onCheckVirtualTrafficLightFeature);
  addCheckbox(
    checkbox_no_stopping_area_, "behavior_velocity_planner: No Stopping Area",
    &AutowareFeatureManagerPanel::onCheckNoStoppingAreaFeature);
  addCheckbox(
    checkbox_stop_line_, "behavior_velocity_planner: Stop Line",
    &AutowareFeatureManagerPanel::onCheckStopLineFeature);
  addCheckbox(
    checkbox_occlusion_spot_, "behavior_velocity_planner: Occlusion Spot",
    &AutowareFeatureManagerPanel::onCheckOcclusionSpotFeature);
  addCheckbox(
    checkbox_run_out_, "behavior_velocity_planner: Run Out",
    &AutowareFeatureManagerPanel::onCheckRunOutFeature);
  addCheckbox(
    checkbox_speed_bump_, "behavior_velocity_planner: Speed Bump",
    &AutowareFeatureManagerPanel::onCheckSpeedBumpFeature);
  addCheckbox(
    checkbox_out_of_lane_, "behavior_velocity_planner: Out of Lane",
    &AutowareFeatureManagerPanel::onCheckOutOfLaneFeature);
  addCheckbox(
    checkbox_no_drivable_lane_, "behavior_velocity_planner: No Drivable Lane",
    &AutowareFeatureManagerPanel::onCheckNoDrivableLaneFeature);

  // motion
  addCheckbox_UnSupported(
    checkbox_path_smoothing_, "[WIP] Path Smoothing",
    &AutowareFeatureManagerPanel::onCheckPathSmoothingFeature);
  addCheckbox_UnSupported(
    checkbox_obstacle_cruise_, "[WIP] Obstacle Cruise",
    &AutowareFeatureManagerPanel::onCheckObstacleCruiseFeature);
  addCheckbox_UnSupported(
    checkbox_obstacle_stop_, "[WIP] Obstacle Stop",
    &AutowareFeatureManagerPanel::onCheckObstacleStopFeature);
  addCheckbox_UnSupported(
    checkbox_obstacle_decel_, "[WIP] Obstacle Decel",
    &AutowareFeatureManagerPanel::onCheckObstacleDecelFeature);
  addCheckbox_UnSupported(
    checkbox_decel_on_curve_, "[WIP] Decel on Curve",
    &AutowareFeatureManagerPanel::onCheckDecelOnCurveFeature);
  addCheckbox_UnSupported(
    checkbox_decel_on_curve_for_obstacle_, "[WIP] Decel on Curve for Obstacles",
    &AutowareFeatureManagerPanel::onCheckDecelOnCurveForObstaclesFeature);
  addCheckbox_UnSupported(
    checkbox_surround_check_, "[WIP] Surround Check",
    &AutowareFeatureManagerPanel::onCheckSurroundCheckFeature);

  setLayout(layout);
}

// ***************** behavior path planner ************************
void AutowareFeatureManagerPanel::onCheckStartFeature(bool checked)
{
  (void)checked;
  std::cerr << __func__ << ": NOT SUPPORTED YET" << std::endl;
}
void AutowareFeatureManagerPanel::onCheckGoalFeature(bool checked)
{
  (void)checked;
  std::cerr << __func__ << ": NOT SUPPORTED YET" << std::endl;
}
void AutowareFeatureManagerPanel::onCheckLaneChangeFeature(bool checked)
{
  (void)checked;
  std::cerr << __func__ << ": NOT SUPPORTED YET" << std::endl;
}
void AutowareFeatureManagerPanel::onCheckAvoidanceFeature(bool checked)
{
  (void)checked;
  std::cerr << __func__ << ": NOT SUPPORTED YET" << std::endl;
}
void AutowareFeatureManagerPanel::onCheckSideShiftFeature(bool checked)
{
  (void)checked;
  std::cerr << __func__ << ": NOT SUPPORTED YET" << std::endl;
}

// ***************** behavior velocity planner ************************
void AutowareFeatureManagerPanel::onCheckCrosswalkFeature(bool checked)
{
  loadOrUnloadBehaviorVelocityModule(
    checked, "behavior_velocity_planner::CrosswalkModulePlugin", "crosswalk");
}
void AutowareFeatureManagerPanel::onCheckWalkwayFeature(bool checked)
{
  loadOrUnloadBehaviorVelocityModule(
    checked, "behavior_velocity_planner::WalkwayModulePlugin", "walkway");
}
void AutowareFeatureManagerPanel::onCheckTrafficLightFeature(bool checked)
{
  loadOrUnloadBehaviorVelocityModule(
    checked, "behavior_velocity_planner::TrafficLightModulePlugin", "traffic_light");
}
void AutowareFeatureManagerPanel::onCheckIntersectionFeature(bool checked)
{
  loadOrUnloadBehaviorVelocityModule(
    checked, "behavior_velocity_planner::IntersectionModulePlugin", "intersection");
}
void AutowareFeatureManagerPanel::onCheckMergeFromPrivateFeature(bool checked)
{
  loadOrUnloadBehaviorVelocityModule(
    checked, "behavior_velocity_planner::MergeFromPrivateModulePlugin", "merge_from_private");
}
void AutowareFeatureManagerPanel::onCheckBlindSpotFeature(bool checked)
{
  loadOrUnloadBehaviorVelocityModule(
    checked, "behavior_velocity_planner::BlindSpotModulePlugin", "blind_spot");
}
void AutowareFeatureManagerPanel::onCheckDetectionAreaFeature(bool checked)
{
  loadOrUnloadBehaviorVelocityModule(
    checked, "behavior_velocity_planner::DetectionAreaModulePlugin", "detection_area");
}
void AutowareFeatureManagerPanel::onCheckVirtualTrafficLightFeature(bool checked)
{
  loadOrUnloadBehaviorVelocityModule(
    checked, "behavior_velocity_planner::VirtualTrafficLightModulePlugin", "virtual_traffic_light");
}
void AutowareFeatureManagerPanel::onCheckNoStoppingAreaFeature(bool checked)
{
  loadOrUnloadBehaviorVelocityModule(
    checked, "behavior_velocity_planner::NoStoppingAreaModulePlugin", "no_stopping_area");
}
void AutowareFeatureManagerPanel::onCheckStopLineFeature(bool checked)
{
  loadOrUnloadBehaviorVelocityModule(
    checked, "behavior_velocity_planner::StopLineModulePlugin", "stop_line");
}
void AutowareFeatureManagerPanel::onCheckOcclusionSpotFeature(bool checked)
{
  loadOrUnloadBehaviorVelocityModule(
    checked, "behavior_velocity_planner::OcclusionSpotModulePlugin", "occlusion_spot");
}
void AutowareFeatureManagerPanel::onCheckRunOutFeature(bool checked)
{
  loadOrUnloadBehaviorVelocityModule(
    checked, "behavior_velocity_planner::RunOutModulePlugin", "run_out");
}
void AutowareFeatureManagerPanel::onCheckSpeedBumpFeature(bool checked)
{
  loadOrUnloadBehaviorVelocityModule(
    checked, "behavior_velocity_planner::SpeedBumpModulePlugin", "speed_bump");
}
void AutowareFeatureManagerPanel::onCheckOutOfLaneFeature(bool checked)
{
  loadOrUnloadBehaviorVelocityModule(
    checked, "behavior_velocity_planner::OutOfLaneModulePlugin", "out_of_lane");
}
void AutowareFeatureManagerPanel::onCheckNoDrivableLaneFeature(bool checked)
{
  loadOrUnloadBehaviorVelocityModule(
    checked, "behavior_velocity_planner::NoDrivableLaneModulePlugin", "no_drivable_lane");
}

// ***************** motion planner ************************

void AutowareFeatureManagerPanel::onCheckPathSmoothingFeature(bool checked)
{
  (void)checked;
  std::cerr << __func__ << ": NOT SUPPORTED YET" << std::endl;
}
void AutowareFeatureManagerPanel::onCheckObstacleCruiseFeature(bool checked)
{
  (void)checked;
  std::cerr << __func__ << ": NOT SUPPORTED YET" << std::endl;
}
void AutowareFeatureManagerPanel::onCheckObstacleStopFeature(bool checked)
{
  (void)checked;
  std::cerr << __func__ << ": NOT SUPPORTED YET" << std::endl;
}
void AutowareFeatureManagerPanel::onCheckObstacleDecelFeature(bool checked)
{
  (void)checked;
  std::cerr << __func__ << ": NOT SUPPORTED YET" << std::endl;
}
void AutowareFeatureManagerPanel::onCheckDecelOnCurveFeature(bool checked)
{
  (void)checked;
  std::cerr << __func__ << ": NOT SUPPORTED YET" << std::endl;
}
void AutowareFeatureManagerPanel::onCheckDecelOnCurveForObstaclesFeature(bool checked)
{
  (void)checked;
  std::cerr << __func__ << ": NOT SUPPORTED YET" << std::endl;
}
void AutowareFeatureManagerPanel::onCheckSurroundCheckFeature(bool checked)
{
  (void)checked;
  std::cerr << __func__ << ": NOT SUPPORTED YET" << std::endl;
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

AutowareFeatureManagerPanel::~AutowareFeatureManagerPanel() = default;

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(AutowareFeatureManagerPanel, rviz_common::Panel)
