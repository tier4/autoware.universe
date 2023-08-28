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
#include <opencv2/opencv.hpp>
#include <rviz_common/display_context.hpp>
#include <rviz_common/panel.hpp>
#include <rviz_common/render_panel.hpp>
#include <rviz_common/ros_integration/ros_node_abstraction_iface.hpp>
#include <rviz_common/view_manager.hpp>
#include <rviz_rendering/render_window.hpp>

// ros
#include <std_srvs/srv/trigger.hpp>

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
  void onCheckStopLineFeature(bool checked);
  void onCheckCrosswalkFeature(bool checked);

private:
  QCheckBox * checkbox_stop_line_;
  QCheckBox * checkbox_crosswalk_;

  rclcpp::Client<behavior_velocity_planner::srv::LoadPlugin>::SharedPtr
    behavior_velocity_load_client_;
  rclcpp::Client<behavior_velocity_planner::srv::UnloadPlugin>::SharedPtr
    behavior_velocity_unload_client_;

protected:
  rclcpp::Node::SharedPtr raw_node_;
};

#endif  // FEATURE_MANAGER_PANEL_HPP_
