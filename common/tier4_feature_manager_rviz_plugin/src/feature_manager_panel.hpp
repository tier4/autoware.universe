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
#include <QCheckBox>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QMainWindow>
#include <QPushButton>
#include <QScreen>
#include <QScrollArea>
#include <behavior_velocity_planner/srv/load_plugin.hpp>
#include <behavior_velocity_planner/srv/unload_plugin.hpp>
#include <rviz_common/display_context.hpp>
#include <rviz_common/panel.hpp>

#include <memory>
#include <string>
#include <vector>

using behavior_velocity_planner::srv::LoadPlugin;
using behavior_velocity_planner::srv::UnloadPlugin;

class FeatureManager : public rviz_common::Panel
{
  Q_OBJECT

public:
  explicit FeatureManager(QWidget * parent = nullptr);
  ~FeatureManager() override;
  void onInitialize() override;
  void save(rviz_common::Config config) const override;
  void load(const rviz_common::Config & config) override;

public Q_SLOTS:
  // planning mission planner
  void onCheckGoalValidation(bool checked);

  // planning behavior path planner
  void onCheckStart(bool checked);
  void onCheckGoal(bool checked);
  void onCheckLaneChange(bool checked);
  void onCheckAvoidance(bool checked);
  void onCheckSideShift(bool checked);

  // planning behavior velocity planner
  void onCheckCrosswalk(bool checked);
  void onCheckWalkway(bool checked);
  void onCheckTrafficLight(bool checked);
  void onCheckIntersection(bool checked);
  void onCheckMergeFromPrivate(bool checked);
  void onCheckBlindSpot(bool checked);
  void onCheckDetectionArea(bool checked);
  void onCheckVirtualTrafficLight(bool checked);
  void onCheckNoStoppingArea(bool checked);
  void onCheckStopLine(bool checked);
  void onCheckOcclusionSpot(bool checked);
  void onCheckRunOut(bool checked);
  void onCheckSpeedBump(bool checked);
  void onCheckOutOfLane(bool checked);
  void onCheckNoDrivableLane(bool checked);

  // planning motion
  void onCheckPathSmoothing(bool checked);
  void onCheckObstacleCruise(bool checked);
  void onCheckObstacleStop(bool checked);
  void onCheckObstacleDecel(bool checked);
  void onCheckDecelOnCurve(bool checked);
  void onCheckDecelOnCurveForObstacles(bool checked);
  void onCheckSurroundCheck(bool checked);

  // planning others
  void onCheckTrajectoryValidation(bool checked);

  // control
  void onCheckSlopeCompensation(bool checked);
  void onCheckSteerOffsetRemover(bool checked);

private:
  std::unordered_map<std::string, QCheckBox *> checkbox_storage_;

  rclcpp::Client<LoadPlugin>::SharedPtr behavior_velocity_load_client_;
  rclcpp::Client<UnloadPlugin>::SharedPtr behavior_velocity_unload_client_;

  void loadOrUnloadBehaviorVelocityModule(
    const bool load, const std::string & load_name, const std::string & unload_name);

protected:
  rclcpp::Node::SharedPtr raw_node_;
};

#endif  // FEATURE_MANAGER_PANEL_HPP_
