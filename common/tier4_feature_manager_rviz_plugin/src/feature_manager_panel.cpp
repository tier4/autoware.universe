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
  // Create main layout for the scroll area
  QVBoxLayout * mainLayout = new QVBoxLayout;

  // Create a content widget and layout for the items
  QWidget * contentWidget = new QWidget;
  QVBoxLayout * contentLayout = new QVBoxLayout;

  const auto addCheckbox_impl = [&](
                                  QLayout * layout, auto & name, const auto & displayed_title,
                                  const auto & tooltip, const auto callback, const bool enable) {
    auto * checkbox = new QCheckBox(displayed_title);
    checkbox->setToolTip(tooltip);  // Set the tooltip
    layout->addWidget(checkbox);
    checkbox->setChecked(true);    // initialize
    checkbox->setEnabled(enable);  // gray out (disabled)
    connect(checkbox, &QCheckBox::toggled, this, callback);
    checkbox_storage_.emplace(name, checkbox);
  };

  const auto addCheckbox_UnSupported = [&](
                                         QLayout * layout, auto & name,
                                         const auto & displayed_title, const auto & tooltip,
                                         const auto callback) {
    addCheckbox_impl(layout, name, displayed_title, tooltip, callback, false);
  };

  const auto addCheckbox = [&](
                             QLayout * layout, auto & name, const auto & displayed_title,
                             const auto & tooltip, const auto callback) {
    addCheckbox_impl(layout, name, displayed_title, tooltip, callback, true);
  };

  const auto addLabel = [&](QLayout * layout, const QString & label_str) {
    auto * label = new QLabel(label_str);
    layout->addWidget(label);
  };

  // ========== Localization Group ==========
  QGroupBox * LocalizationGroup = new QGroupBox("Localization");
  QVBoxLayout * LocalizationLayout = new QVBoxLayout;

  addCheckbox_UnSupported(
    LocalizationLayout, "localization_error_monitor", "[WIP] Localization Error Monitor",
    "Diagnose the status of your localization module and report the results.",
    &FeatureManager::onCheckLocalizationErrorMonitor);

  LocalizationGroup->setLayout(LocalizationLayout);
  contentLayout->addWidget(LocalizationGroup);

  // ========== Perception Group ==========
  QGroupBox * PerceptionGroup = new QGroupBox("Perception");
  QVBoxLayout * PerceptionLayout = new QVBoxLayout;

  addCheckbox_UnSupported(
    PerceptionLayout, "map_based_prediction", "[WIP] Map Based Prediction",
    "calculates the predicted path for all detected objects based on the load connection "
    "information.",
    &FeatureManager::onCheckMapBasedPrediction);

  PerceptionGroup->setLayout(PerceptionLayout);
  contentLayout->addWidget(PerceptionGroup);

  // ========== Control Group ==========
  QGroupBox * planningGroup = new QGroupBox("Planning");
  QVBoxLayout * planningLayout = new QVBoxLayout;

  // mission planner
  addLabel(planningLayout, "---------- mission planner ----------");
  addCheckbox_UnSupported(
    planningLayout, "goal_validation", "[WIP] Goal Validation",
    "rejects the invalid goal, e.g. the foot print is not in the lane.",
    &FeatureManager::onCheckGoalValidation);

  // behavior path planner
  addLabel(planningLayout, "---------- behavior path planner ----------");
  addCheckbox_UnSupported(
    planningLayout, "start_planner", "[WIP] Start",
    "generate efficient and smooth starting path (pull out, freespace, etc) with safety check.",
    &FeatureManager::onCheckStart);
  addCheckbox_UnSupported(
    planningLayout, "goal_planner", "[WIP] Goal",
    "generate efficient and smooth goal path (pull over, freespace, etc) with safety check.",
    &FeatureManager::onCheckGoal);
  addCheckbox_UnSupported(
    planningLayout, "lane_change", "[WIP] Lane Change",
    "automatically generate lane change path with safety check when it is needed.",
    &FeatureManager::onCheckLaneChange);
  addCheckbox_UnSupported(
    planningLayout, "avoidance", "[WIP] Avoidance",
    "automatically generate avoidance path to avoid the collision with detected objects.",
    &FeatureManager::onCheckAvoidance);
  addCheckbox_UnSupported(
    planningLayout, "side_shift", "[WIP] Side Shift",
    "generate shifted path on the lateral direction with the instructed direction and distance.",
    &FeatureManager::onCheckSideShift);

  // behavior velocity planner
  addLabel(planningLayout, "---------- behavior velocity planner ----------");
  addCheckbox(
    planningLayout, "crosswalk", "Crosswalk",
    "stop or decelerate around a crosswalk considering the situation.",
    &FeatureManager::onCheckCrosswalk);
  addCheckbox(
    planningLayout, "walkway", "Walkway", "stop once and go if the walkway is free.",
    &FeatureManager::onCheckWalkway);
  addCheckbox(
    planningLayout, "traffic_light", "Traffic Light", "stop when the traffic light indicates stop.",
    &FeatureManager::onCheckTrafficLight);
  addCheckbox(
    planningLayout, "intersection", "Intersection",
    "stop and decelerate around intersection considering the oncoming vehicle.",
    &FeatureManager::onCheckIntersection);
  addCheckbox(
    planningLayout, "intersection_stuck", "Intersection Stuck Vehicle",
    "stop in front of the intersection if another vehicle is stopped in the intersection.",
    &FeatureManager::onCheckIntersection);
  addCheckbox(
    planningLayout, "intersection_occlusion", "Intersection Occlusion",
    "Go slowly to see the occluded area in the intersection. Go into the intersection after the "
    "safety is confirmed.",
    &FeatureManager::onCheckIntersection);
  addCheckbox(
    planningLayout, "merge_from_private", "Merge From Private",
    "stop once and go after the safety is confirmed on the merging.",
    &FeatureManager::onCheckMergeFromPrivate);
  addCheckbox(
    planningLayout, "blind_spot", "Blind Spot",
    "check the blind spot when the vehicle entries into the intersection. Stop if it detects a "
    "danger situation.",
    &FeatureManager::onCheckBlindSpot);
  addCheckbox(
    planningLayout, "detection_area", "Detection Area",
    "Stop at a pre-defined stop line when any obstacle in on the pre-defined detection area.",
    &FeatureManager::onCheckDetectionArea);
  addCheckbox(
    planningLayout, "virtual_traffic_light", "Virtual Traffic Light", "write me...",
    &FeatureManager::onCheckVirtualTrafficLight);
  addCheckbox(
    planningLayout, "no_stopping_area", "No Stopping Area", "write me...",
    &FeatureManager::onCheckNoStoppingArea);
  addCheckbox(
    planningLayout, "stop_line", "Stop Line", "write me...", &FeatureManager::onCheckStopLine);
  addCheckbox(
    planningLayout, "occlusion_spot", "Occlusion Spot", "write me...",
    &FeatureManager::onCheckOcclusionSpot);
  addCheckbox(planningLayout, "run_out", "Run Out", "write me...", &FeatureManager::onCheckRunOut);
  addCheckbox(
    planningLayout, "speed_bump", "Speed Bump", "write me...", &FeatureManager::onCheckSpeedBump);
  addCheckbox(
    planningLayout, "out_of_lane", "Out of Lane", "write me...", &FeatureManager::onCheckOutOfLane);
  addCheckbox(
    planningLayout, "no_drivable_lane", "No Drivable Lane", "write me...",
    &FeatureManager::onCheckNoDrivableLane);

  // motion
  addLabel(planningLayout, "---------- motion planner ----------");

  addCheckbox_UnSupported(
    planningLayout, "path_smoothing", "[WIP] Path Smoothing", "write me...",
    &FeatureManager::onCheckPathSmoothing);
  addCheckbox_UnSupported(
    planningLayout, "obstacle_cruise", "[WIP] Obstacle Cruise", "write me...",
    &FeatureManager::onCheckObstacleCruise);
  addCheckbox_UnSupported(
    planningLayout, "obstacle_stop", "[WIP] Obstacle Stop", "write me...",
    &FeatureManager::onCheckObstacleStop);
  addCheckbox_UnSupported(
    planningLayout, "obstacle_decel", "[WIP] Obstacle Decel", "write me...",
    &FeatureManager::onCheckObstacleDecel);
  addCheckbox_UnSupported(
    planningLayout, "decel_on_curve", "[WIP] Decel on Curve", "write me...",
    &FeatureManager::onCheckDecelOnCurve);
  addCheckbox_UnSupported(
    planningLayout, "decel_on_curve_for_obstacle", "[WIP] Decel on Curve for Obstacles",
    "write me...", &FeatureManager::onCheckDecelOnCurveForObstacles);
  addCheckbox_UnSupported(
    planningLayout, "surround_check", "[WIP] Surround Check", "write me...",
    &FeatureManager::onCheckSurroundCheck);

  addCheckbox_UnSupported(
    planningLayout, "planing_validator", "[WIP] Trajectory Validation", "write me...",
    &FeatureManager::onCheckTrajectoryValidation);

  planningGroup->setLayout(planningLayout);
  contentLayout->addWidget(planningGroup);

  // ========== Control Group ==========
  QGroupBox * controlGroup = new QGroupBox("Control");
  QVBoxLayout * controlLayout = new QVBoxLayout;

  addCheckbox_UnSupported(
    controlLayout, "slope_compensation", "[WIP] Slope Compensation",
    "Modify the target acceleration considering the slope angle.",
    &FeatureManager::onCheckSlopeCompensation);
  addCheckbox_UnSupported(
    controlLayout, "steer_offset", "[WIP] Steering Offset Removal",
    "Compensate the steering offset.", &FeatureManager::onCheckSteerOffsetRemover);

  controlGroup->setLayout(controlLayout);
  contentLayout->addWidget(controlGroup);

  // After adding all the groups to contentLayout, add a stretch at the end.
  // This will make the Layout "top-stuffed".
  contentLayout->addStretch(1);

  // Assign content layout to content widget
  contentWidget->setLayout(contentLayout);

  // Create the scroll area and set contentWidget as its child
  QScrollArea * scrollArea = new QScrollArea;
  scrollArea->setWidget(contentWidget);
  scrollArea->setWidgetResizable(true);
  mainLayout->addWidget(scrollArea);

  setLayout(mainLayout);
}
// ***************** localization ************************
void FeatureManager::onCheckLocalizationErrorMonitor(bool checked)
{
  (void)checked;
  std::cerr << __func__ << ": NOT SUPPORTED YET" << std::endl;
}

// ***************** perception ************************
void FeatureManager::onCheckMapBasedPrediction(bool checked)
{
  (void)checked;
  std::cerr << __func__ << ": NOT SUPPORTED YET" << std::endl;
}

// ***************** mission planner ************************
void FeatureManager::onCheckGoalValidation(bool checked)
{
  (void)checked;
  std::cerr << __func__ << ": NOT SUPPORTED YET" << std::endl;
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
