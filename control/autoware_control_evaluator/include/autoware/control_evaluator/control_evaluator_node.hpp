// Copyright 2024 Tier IV, Inc.
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

#ifndef AUTOWARE__CONTROL_EVALUATOR__CONTROL_EVALUATOR_NODE_HPP_
#define AUTOWARE__CONTROL_EVALUATOR__CONTROL_EVALUATOR_NODE_HPP_

#include "autoware/control_evaluator/metrics/deviation_metrics.hpp"

#include <rclcpp/rclcpp.hpp>
#include <route_handler/route_handler.hpp>
#include <tier4_autoware_utils/tier4_autoware_utils.hpp>

#include "geometry_msgs/msg/accel_with_covariance_stamped.hpp"
#include <autoware_auto_planning_msgs/msg/route.hpp>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include <deque>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace control_diagnostics
{

using autoware_auto_planning_msgs::msg::Trajectory;
using diagnostic_msgs::msg::DiagnosticArray;
using diagnostic_msgs::msg::DiagnosticStatus;
using geometry_msgs::msg::Point;
using geometry_msgs::msg::Pose;
using nav_msgs::msg::Odometry;
using LaneletMapBin = autoware_auto_mapping_msgs::msg::HADMapBin;
using LaneletRoute = autoware_auto_planning_msgs::msg::HADMapRoute;
using geometry_msgs::msg::AccelWithCovarianceStamped;
using sensor_msgs::msg::Imu;


/**
 * @brief Node for control evaluation
 */
class ControlEvaluatorNode : public rclcpp::Node
{
public:
  explicit ControlEvaluatorNode(const rclcpp::NodeOptions & node_options);
  DiagnosticStatus generateLateralDeviationDiagnosticStatus(
    const Trajectory & traj, const Point & ego_point);
  DiagnosticStatus generateYawDeviationDiagnosticStatus(
    const Trajectory & traj, const Pose & ego_pose);
  std::optional<DiagnosticStatus> generateStopDiagnosticStatus(
    const DiagnosticArray & diag, const std::string & function_name);

  DiagnosticStatus generateAEBDiagnosticStatus(const DiagnosticStatus & diag);
  DiagnosticStatus generateLaneletDiagnosticStatus(const Pose & ego_pose) const;
  DiagnosticStatus generateKinematicStateDiagnosticStatus(
    const Odometry & odom, const Imu & imu);

  void onDiagnostics(const DiagnosticArray::ConstSharedPtr diag_msg);
  void onTimer();

private:
  // The diagnostics cycle is faster than timer, and each node publishes diagnostic separately.
  // takeData() in onTimer() with a polling subscriber will miss a topic, so save all topics with
  // onDiagnostics().
  rclcpp::Subscription<DiagnosticArray>::SharedPtr control_diag_sub_;

  autoware::universe_utils::InterProcessPollingSubscriber<Odometry> odometry_sub_{
    this, "~/input/odometry"};
  // autoware::universe_utils::InterProcessPollingSubscriber<AccelWithCovarianceStamped> accel_sub_{
  //   this, "~/input/acceleration"};
  autoware::universe_utils::InterProcessPollingSubscriber<Imu> imu_sub_{
    this, "~/input/imu"};
  autoware::universe_utils::InterProcessPollingSubscriber<Trajectory> traj_sub_{
    this, "~/input/trajectory"};
  autoware::universe_utils::InterProcessPollingSubscriber<
    LaneletRoute, autoware::universe_utils::polling_policy::Newest>
    route_subscriber_{this, "~/input/route", rclcpp::QoS{1}.transient_local()};
  autoware::universe_utils::InterProcessPollingSubscriber<
    LaneletMapBin, autoware::universe_utils::polling_policy::Newest>
    vector_map_subscriber_{this, "~/input/vector_map", rclcpp::QoS{1}.transient_local()};

  rclcpp::Publisher<DiagnosticArray>::SharedPtr metrics_pub_;

  // update Route Handler
  void getRouteData();

  // Calculator
  // Metrics
  std::deque<rclcpp::Time> stamps_;

  // queue for diagnostics and time stamp
  std::deque<std::pair<DiagnosticStatus, rclcpp::Time>> diag_queue_;
  const std::vector<std::string> target_functions_ = {"autonomous_emergency_braking"};

  route_handler::RouteHandler route_handler_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::optional<Imu> prev_imu_{std::nullopt};
};
}  // namespace control_diagnostics

#endif  // AUTOWARE__CONTROL_EVALUATOR__CONTROL_EVALUATOR_NODE_HPP_
