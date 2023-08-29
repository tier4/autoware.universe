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

#include <iostream>

using std::placeholders::_1;
using std::placeholders::_2;

FeatureManager::FeatureManager(QWidget * parent) : rviz_common::Panel(parent)
{
  auto * layout = new QVBoxLayout;

  // Your widget for the QScrollArea
  QWidget * contentWidget = new QWidget;
  contentWidget->setLayout(layout);

  const auto addCheckbox = [&](auto & name, const auto & title, const auto callback) {
    auto * checkbox = new QCheckBox(title);
    layout->addWidget(checkbox);
    checkbox->setChecked(true);  // initialize
    connect(checkbox, &QCheckBox::toggled, this, callback);
    checkbox_storage_.emplace(name, checkbox);
  };

  const auto addCheckbox_UnSupported = [&](auto & name, const auto & title, const auto callback) {
    auto * checkbox = new QCheckBox(title);
    layout->addWidget(checkbox);
    checkbox->setChecked(true);
    checkbox->setEnabled(false);  // gray out (disabled)
    connect(checkbox, &QCheckBox::toggled, this, callback);
    checkbox_storage_.emplace(name, checkbox);
  };

  const auto addLabel = [&](const QString & label_str) {
    auto * label = new QLabel(label_str);
    layout->addWidget(label);
  };

  addLabel("===== Planning =====");

  // behavior path planner
  addLabel("----- behavior path planner -----");
  addCheckbox_UnSupported("start_planner", "[WIP] Start", &FeatureManager::onCheckStart);
  addCheckbox_UnSupported("goal_planner", "[WIP] Goal", &FeatureManager::onCheckGoal);
  addCheckbox_UnSupported("lane_change", "[WIP] Lane Change", &FeatureManager::onCheckLaneChange);
  addCheckbox_UnSupported("avoidance", "[WIP] Avoidance", &FeatureManager::onCheckAvoidance);
  addCheckbox_UnSupported("side_shift", "[WIP] Side Shift", &FeatureManager::onCheckSideShift);

  // behavior velocity planner
  addLabel("----- behavior velocity planner -----");
  addCheckbox("crosswalk", "Crosswalk", &FeatureManager::onCheckCrosswalk);
  addCheckbox("walkway", "Walkway", &FeatureManager::onCheckWalkway);
  addCheckbox("traffic_light", "Traffic Light", &FeatureManager::onCheckTrafficLight);
  addCheckbox("intersection", "Intersection", &FeatureManager::onCheckIntersection);
  addCheckbox("merge_from_private", "Merge From Private", &FeatureManager::onCheckMergeFromPrivate);
  addCheckbox("blind_spot", "Blind Spot", &FeatureManager::onCheckBlindSpot);
  addCheckbox("detection_area", "Detection Area", &FeatureManager::onCheckDetectionArea);
  addCheckbox(
    "virtual_traffic_light", "Virtual Traffic Light", &FeatureManager::onCheckVirtualTrafficLight);
  addCheckbox("no_stopping_area", "No Stopping Area", &FeatureManager::onCheckNoStoppingArea);
  addCheckbox("stop_line", "Stop Line", &FeatureManager::onCheckStopLine);
  addCheckbox("occlusion_spot", "Occlusion Spot", &FeatureManager::onCheckOcclusionSpot);
  addCheckbox("run_out", "Run Out", &FeatureManager::onCheckRunOut);
  addCheckbox("speed_bump", "Speed Bump", &FeatureManager::onCheckSpeedBump);
  addCheckbox("out_of_lane", "Out of Lane", &FeatureManager::onCheckOutOfLane);
  addCheckbox("no_drivable_lane", "No Drivable Lane", &FeatureManager::onCheckNoDrivableLane);

  // motion
  addLabel("----- motion planner -----");

  addCheckbox_UnSupported(
    "path_smoothing", "[WIP] Path Smoothing", &FeatureManager::onCheckPathSmoothing);
  addCheckbox_UnSupported(
    "obstacle_cruise", "[WIP] Obstacle Cruise", &FeatureManager::onCheckObstacleCruise);
  addCheckbox_UnSupported(
    "obstacle_stop", "[WIP] Obstacle Stop", &FeatureManager::onCheckObstacleStop);
  addCheckbox_UnSupported(
    "obstacle_decel", "[WIP] Obstacle Decel", &FeatureManager::onCheckObstacleDecel);
  addCheckbox_UnSupported(
    "decel_on_curve", "[WIP] Decel on Curve", &FeatureManager::onCheckDecelOnCurve);
  addCheckbox_UnSupported(
    "decel_on_curve_for_obstacle", "[WIP] Decel on Curve for Obstacles",
    &FeatureManager::onCheckDecelOnCurveForObstacles);
  addCheckbox_UnSupported(
    "surround_check", "[WIP] Surround Check", &FeatureManager::onCheckSurroundCheck);

  addCheckbox_UnSupported(
    "planing_validator", "[WIP] Trajectory Validation",
    &FeatureManager::onCheckTrajectoryValidation);

  addLabel("===== Control =====");

  addCheckbox_UnSupported(
    "slope_compensation", "[WIP] Slope Compensation", &FeatureManager::onCheckSlopeCompensation);
  addCheckbox_UnSupported(
    "steer_offset", "[WIP] Steering Offset Remove", &FeatureManager::onCheckSteerOffsetRemover);

  // Add a stretch at the end to push everything to the top
  layout->addStretch(1);

  // Create the scroll area and set contentWidget as its child
  QScrollArea * scrollArea = new QScrollArea(this);
  scrollArea->setWidget(contentWidget);
  scrollArea->setWidgetResizable(true);

  // New main layout for the parent widget
  QVBoxLayout * mainLayout = new QVBoxLayout;
  mainLayout->addWidget(scrollArea);

  setLayout(mainLayout);
}

// ***************** behavior path planner ************************
void FeatureManager::onCheckStart(bool checked)
{
  (void)checked;
  std::cerr << __func__ << ": NOT SUPPORTED YET" << std::endl;
}
void FeatureManager::onCheckGoal(bool checked)
{
  (void)checked;
  std::cerr << __func__ << ": NOT SUPPORTED YET" << std::endl;
}
void FeatureManager::onCheckLaneChange(bool checked)
{
  (void)checked;
  std::cerr << __func__ << ": NOT SUPPORTED YET" << std::endl;
}
void FeatureManager::onCheckAvoidance(bool checked)
{
  (void)checked;
  std::cerr << __func__ << ": NOT SUPPORTED YET" << std::endl;
}
void FeatureManager::onCheckSideShift(bool checked)
{
  (void)checked;
  std::cerr << __func__ << ": NOT SUPPORTED YET" << std::endl;
}

// ***************** behavior velocity planner ************************
void FeatureManager::onCheckCrosswalk(bool checked)
{
  loadOrUnloadBehaviorVelocityModule(
    checked, "behavior_velocity_planner::CrosswalkModulePlugin", "crosswalk");
}
void FeatureManager::onCheckWalkway(bool checked)
{
  loadOrUnloadBehaviorVelocityModule(
    checked, "behavior_velocity_planner::WalkwayModulePlugin", "walkway");
}
void FeatureManager::onCheckTrafficLight(bool checked)
{
  loadOrUnloadBehaviorVelocityModule(
    checked, "behavior_velocity_planner::TrafficLightModulePlugin", "traffic_light");
}
void FeatureManager::onCheckIntersection(bool checked)
{
  loadOrUnloadBehaviorVelocityModule(
    checked, "behavior_velocity_planner::IntersectionModulePlugin", "intersection");
}
void FeatureManager::onCheckMergeFromPrivate(bool checked)
{
  loadOrUnloadBehaviorVelocityModule(
    checked, "behavior_velocity_planner::MergeFromPrivateModulePlugin", "merge_from_private");
}
void FeatureManager::onCheckBlindSpot(bool checked)
{
  loadOrUnloadBehaviorVelocityModule(
    checked, "behavior_velocity_planner::BlindSpotModulePlugin", "blind_spot");
}
void FeatureManager::onCheckDetectionArea(bool checked)
{
  loadOrUnloadBehaviorVelocityModule(
    checked, "behavior_velocity_planner::DetectionAreaModulePlugin", "detection_area");
}
void FeatureManager::onCheckVirtualTrafficLight(bool checked)
{
  loadOrUnloadBehaviorVelocityModule(
    checked, "behavior_velocity_planner::VirtualTrafficLightModulePlugin", "virtual_traffic_light");
}
void FeatureManager::onCheckNoStoppingArea(bool checked)
{
  loadOrUnloadBehaviorVelocityModule(
    checked, "behavior_velocity_planner::NoStoppingAreaModulePlugin", "no_stopping_area");
}
void FeatureManager::onCheckStopLine(bool checked)
{
  loadOrUnloadBehaviorVelocityModule(
    checked, "behavior_velocity_planner::StopLineModulePlugin", "stop_line");
}
void FeatureManager::onCheckOcclusionSpot(bool checked)
{
  loadOrUnloadBehaviorVelocityModule(
    checked, "behavior_velocity_planner::OcclusionSpotModulePlugin", "occlusion_spot");
}
void FeatureManager::onCheckRunOut(bool checked)
{
  loadOrUnloadBehaviorVelocityModule(
    checked, "behavior_velocity_planner::RunOutModulePlugin", "run_out");
}
void FeatureManager::onCheckSpeedBump(bool checked)
{
  loadOrUnloadBehaviorVelocityModule(
    checked, "behavior_velocity_planner::SpeedBumpModulePlugin", "speed_bump");
}
void FeatureManager::onCheckOutOfLane(bool checked)
{
  loadOrUnloadBehaviorVelocityModule(
    checked, "behavior_velocity_planner::OutOfLaneModulePlugin", "out_of_lane");
}
void FeatureManager::onCheckNoDrivableLane(bool checked)
{
  loadOrUnloadBehaviorVelocityModule(
    checked, "behavior_velocity_planner::NoDrivableLaneModulePlugin", "no_drivable_lane");
}

// ***************** motion planner ************************

void FeatureManager::onCheckPathSmoothing(bool checked)
{
  (void)checked;
  std::cerr << __func__ << ": NOT SUPPORTED YET" << std::endl;
}
void FeatureManager::onCheckObstacleCruise(bool checked)
{
  (void)checked;
  std::cerr << __func__ << ": NOT SUPPORTED YET" << std::endl;
}
void FeatureManager::onCheckObstacleStop(bool checked)
{
  (void)checked;
  std::cerr << __func__ << ": NOT SUPPORTED YET" << std::endl;
}
void FeatureManager::onCheckObstacleDecel(bool checked)
{
  (void)checked;
  std::cerr << __func__ << ": NOT SUPPORTED YET" << std::endl;
}
void FeatureManager::onCheckDecelOnCurve(bool checked)
{
  (void)checked;
  std::cerr << __func__ << ": NOT SUPPORTED YET" << std::endl;
}
void FeatureManager::onCheckDecelOnCurveForObstacles(bool checked)
{
  (void)checked;
  std::cerr << __func__ << ": NOT SUPPORTED YET" << std::endl;
}
void FeatureManager::onCheckSurroundCheck(bool checked)
{
  (void)checked;
  std::cerr << __func__ << ": NOT SUPPORTED YET" << std::endl;
}
void FeatureManager::onCheckTrajectoryValidation(bool checked)
{
  (void)checked;
  std::cerr << __func__ << ": NOT SUPPORTED YET" << std::endl;
}

// ***************** control ************************
void FeatureManager::onCheckSlopeCompensation(bool checked)
{
  (void)checked;
  std::cerr << __func__ << ": NOT SUPPORTED YET" << std::endl;
}
void FeatureManager::onCheckSteerOffsetRemover(bool checked)
{
  (void)checked;
  std::cerr << __func__ << ": NOT SUPPORTED YET" << std::endl;
}

void FeatureManager::onInitialize()
{
  raw_node_ = this->getDisplayContext()->getRosNodeAbstraction().lock()->get_raw_node();

  const std::string behavior_velocity_ns =
    "/planning/scenario_planning/lane_driving/behavior_planning/behavior_velocity_planner/";
  behavior_velocity_load_client_ =
    raw_node_->create_client<LoadPlugin>(behavior_velocity_ns + "service/load_plugin");
  behavior_velocity_unload_client_ =
    raw_node_->create_client<UnloadPlugin>(behavior_velocity_ns + "service/unload_plugin");
}

void FeatureManager::save(rviz_common::Config config) const
{
  Panel::save(config);
}

void FeatureManager::load(const rviz_common::Config & config)
{
  Panel::load(config);
}

void FeatureManager::loadOrUnloadBehaviorVelocityModule(
  const bool load, const std::string & load_name, const std::string & unload_name)
{
  if (load) {
    const auto callback = [&](rclcpp::Client<LoadPlugin>::SharedFuture future) {
      if (!future.get()->success) {
        std::cerr << load_name << ": load failed" << std::endl;
      }
    };
    const auto req = std::make_shared<LoadPlugin::Request>();
    req->plugin_name = load_name;
    behavior_velocity_load_client_->async_send_request(req, callback);
  } else {
    const auto callback = [&](rclcpp::Client<UnloadPlugin>::SharedFuture future) {
      if (!future.get()->success) {
        std::cerr << unload_name << ": unload failed\n\n" << std::endl;
      }
    };
    const auto req = std::make_shared<UnloadPlugin::Request>();
    req->plugin_name = unload_name;
    behavior_velocity_unload_client_->async_send_request(req, callback);
  }
}

FeatureManager::~FeatureManager() = default;

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(FeatureManager, rviz_common::Panel)
