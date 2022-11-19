// Copyright 2022 The Autoware Foundation.
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

#ifndef SYSTEM_IDENTIFICATION_INCLUDE_SYSID_NODE_HPP_
#define SYSTEM_IDENTIFICATION_INCLUDE_SYSID_NODE_HPP_

// ROS headers
#include "rcl_interfaces/msg/set_parameters_result.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"

// Autoware Headers
#include "common/types.hpp"
#include "system_identification/msg/sys_id_steering_vars.hpp"

#include <vehicle_info_util/vehicle_info_util.hpp>

#include "autoware_auto_control_msgs/msg/ackermann_control_command.hpp"
#include "autoware_auto_vehicle_msgs/msg/steering_report.hpp"
#include "autoware_auto_vehicle_msgs/msg/vehicle_odometry.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "utils_act/act_utils.hpp"

namespace sysid
{

struct sParameters
{
  double sysid_dt{0.01};
};

// Using declerations
using ControlCommand = autoware_auto_control_msgs::msg::AckermannControlCommand;
using VelocityMsg = nav_msgs::msg::Odometry;
using autoware_auto_vehicle_msgs::msg::SteeringReport;
using system_identification::msg::SysIDSteeringVars;
using vehicle_info_util::VehicleInfoUtil;

class SystemIdentificationNode : public rclcpp::Node
{
 public:
  /**
   * @brief constructor
   */
  explicit SystemIdentificationNode(const rclcpp::NodeOptions &node_options);

  /**
   * @brief destructor
   */
  ~SystemIdentificationNode() override = default;

 private:
  sParameters params_node_{};

  //!< @brief timer to update after a given interval
  rclcpp::TimerBase::SharedPtr timer_;

  // Subscribers
  rclcpp::Subscription<ControlCommand>::SharedPtr sub_control_cmds_;

  //!< @brief subscription for current velocity
  rclcpp::Subscription<VelocityMsg>::SharedPtr sub_current_velocity_ptr_;

  //!< @brief subscription for current velocity
  rclcpp::Subscription<SteeringReport>::SharedPtr sub_current_steering_ptr_;

  // Publishers
  rclcpp::Publisher<SysIDSteeringVars>::SharedPtr pub_sysid_input_;

  /**
   * Node storage
   * */
  // Input messages
  std::shared_ptr<SysIDSteeringVars> current_sysid_input_cmd_{nullptr};

  // Node Methods
  //!< initialize timer to work in real, simulation, and replay
  void initTimer(double period_s);

  /**
   * @brief compute and publish the sysid input signals with a constant control period
   */
  void onTimer();

  /**
   * @brief Publish message.
   * */

  void publishSysIDCommand();
};

}  // namespace sysid

#endif  // SYSTEM_IDENTIFICATION_INCLUDE_SYSID_NODE_HPP_
