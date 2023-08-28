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

#ifndef FEATURE_MANAGER_PANEL_HPP_
#define FEATURE_MANAGER_PANEL_HPP_

// Qt
#include <QApplication>
#include <QDesktopWidget>
#include <QDir>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMainWindow>
#include <QPushButton>
#include <QScreen>
#include <QSpinBox>
#include <QTimer>

// horibe
#include <QCheckBox>
#include <behavior_velocity_planner/srv/load_plugin.hpp>
#include <behavior_velocity_planner/srv/unload_plugin.hpp>

// rviz
#include <rviz_common/display_context.hpp>
#include <rviz_common/panel.hpp>
#include <rviz_common/render_panel.hpp>
#include <rviz_common/ros_integration/ros_node_abstraction_iface.hpp>
#include <rviz_common/view_manager.hpp>
#include <rviz_rendering/render_window.hpp>

#include <memory>
#include <string>
#include <vector>

class QLineEdit;

class AutowareFeatureManagerPanel : public rviz_common::Panel
{
  Q_OBJECT

public:
  explicit AutowareFeatureManagerPanel(QWidget * parent = nullptr);
  ~AutowareFeatureManagerPanel() override;
  void onInitialize() override;
  void save(rviz_common::Config config) const override;
  void load(const rviz_common::Config & config) override;

public Q_SLOTS:
  // behavior path planner
  void onCheckStartFeature(bool checked);
  void onCheckGoalFeature(bool checked);
  void onCheckLaneChangeFeature(bool checked);
  void onCheckAvoidanceFeature(bool checked);
  void onCheckSideShiftFeature(bool checked);

  // behavior velocity planner
  void onCheckCrosswalkFeature(bool checked);
  void onCheckWalkwayFeature(bool checked);
  void onCheckTrafficLightFeature(bool checked);
  void onCheckIntersectionFeature(bool checked);
  void onCheckMergeFromPrivateFeature(bool checked);
  void onCheckBlindSpotFeature(bool checked);
  void onCheckDetectionAreaFeature(bool checked);
  void onCheckVirtualTrafficLightFeature(bool checked);
  void onCheckNoStoppingAreaFeature(bool checked);
  void onCheckStopLineFeature(bool checked);
  void onCheckOcclusionSpotFeature(bool checked);
  void onCheckRunOutFeature(bool checked);
  void onCheckSpeedBumpFeature(bool checked);
  void onCheckOutOfLaneFeature(bool checked);
  void onCheckNoDrivableLaneFeature(bool checked);

  // motion
  void onCheckPathSmoothingFeature(bool checked);
  void onCheckObstacleCruiseFeature(bool checked);
  void onCheckObstacleStopFeature(bool checked);
  void onCheckObstacleDecelFeature(bool checked);
  void onCheckDecelOnCurveFeature(bool checked);
  void onCheckDecelOnCurveForObstaclesFeature(bool checked);
  void onCheckSurroundCheckFeature(bool checked);

private:
  // behavior path planner
  QCheckBox * checkbox_start_;
  QCheckBox * checkbox_goal_;
  QCheckBox * checkbox_lane_change_;
  QCheckBox * checkbox_avoidance_;
  QCheckBox * checkbox_side_shift_;

  // behavior velocity planner
  QCheckBox * checkbox_crosswalk_;
  QCheckBox * checkbox_walkway_;
  QCheckBox * checkbox_traffic_light_;
  QCheckBox * checkbox_intersection_;
  QCheckBox * checkbox_merge_from_private_;
  QCheckBox * checkbox_blind_spot_;
  QCheckBox * checkbox_detection_area_;
  QCheckBox * checkbox_virtual_traffic_light_;
  QCheckBox * checkbox_no_stopping_area_;
  QCheckBox * checkbox_stop_line_;
  QCheckBox * checkbox_occlusion_spot_;
  QCheckBox * checkbox_run_out_;
  QCheckBox * checkbox_speed_bump_;
  QCheckBox * checkbox_out_of_lane_;
  QCheckBox * checkbox_no_drivable_lane_;

  // motion
  QCheckBox * checkbox_path_smoothing_;
  QCheckBox * checkbox_obstacle_cruise_;
  QCheckBox * checkbox_obstacle_stop_;
  QCheckBox * checkbox_obstacle_decel_;
  QCheckBox * checkbox_decel_on_curve_;
  QCheckBox * checkbox_decel_on_curve_for_obstacle_;
  QCheckBox * checkbox_surround_check_;

  rclcpp::Client<behavior_velocity_planner::srv::LoadPlugin>::SharedPtr
    behavior_velocity_load_client_;
  rclcpp::Client<behavior_velocity_planner::srv::UnloadPlugin>::SharedPtr
    behavior_velocity_unload_client_;

  void loadOrUnloadBehaviorVelocityModule(
    const bool load, const std::string & load_name, const std::string & unload_name);

protected:
  rclcpp::Node::SharedPtr raw_node_;
};

#endif  // FEATURE_MANAGER_PANEL_HPP_
