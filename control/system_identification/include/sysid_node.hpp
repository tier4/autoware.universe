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
#include "tf2/utils.h"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "rclcpp_components/register_node_macro.hpp"

#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "tf2_msgs/msg/tf_message.hpp"

// Autoware Headers
#include "common/types.hpp"
#include "system_identification/msg/sys_id_steering_vars.hpp"

#include <vehicle_info_util/vehicle_info_util.hpp>

#include "autoware_auto_control_msgs/msg/ackermann_control_command.hpp"
#include "autoware_auto_vehicle_msgs/msg/steering_report.hpp"
#include "autoware_auto_vehicle_msgs/msg/vehicle_odometry.hpp"
#include "autoware_auto_planning_msgs/msg/trajectory.hpp"

#include "utils_act/act_utils.hpp"
#include "utils_act/act_utils_eigen.hpp"
#include "signal_processing/lowpass_filter.hpp"
#include "input_library/input_lib.hpp"

namespace sysid
{

struct sCommonParametersInputLib
{
  double sysid_dt{0.01};
  double maximum_amplitude{0.1};
  double minimum_speed{1.};  // signal activated after this speed
  double maximum_speed{3.};  // signal zeroed after this speed
  double tstart{0.5};  // signal started to generate after this amount of time.
};

enum class InputType : int
{
  IDENTITY = 0,
  STEP = 1,
  PRBS = 2,
  FWNOISE = 3,
  SUMSINs = 4,
};

// Using declarations
using ControlCommand = autoware_auto_control_msgs::msg::AckermannControlCommand;
using VelocityMsg = nav_msgs::msg::Odometry;
using autoware_auto_vehicle_msgs::msg::SteeringReport;
using autoware_auto_planning_msgs::msg::Trajectory;
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
  //!< @brief input library common parameters.
  sCommonParametersInputLib common_input_lib_params_{};

  // DATA MEMBERS
  //!< @brief measured pose
  geometry_msgs::msg::PoseStamped::SharedPtr current_pose_ptr_;

  //!< @brief measured velocity
  nav_msgs::msg::Odometry::SharedPtr current_velocity_ptr_;
  double current_vx_{};

  //!< @brief measured steering
  autoware_auto_vehicle_msgs::msg::SteeringReport::SharedPtr current_steering_ptr_;

  //!< @brief reference trajectory
  autoware_auto_planning_msgs::msg::Trajectory::SharedPtr current_trajectory_ptr_;

  //!< @brief buffer for transforms
  tf2::BufferCore m_tf_buffer_{tf2::BUFFER_CORE_DEFAULT_CACHE_TIME};
  tf2_ros::TransformListener m_tf_listener_{m_tf_buffer_};

  //!< @brief timer to update after a given interval
  rclcpp::TimerBase::SharedPtr timer_;

  //!< @brief input class wrapper with a default identity input 0.
  sysid::InputWrapper input_wrapper_{sysid::InputIdentity{}};

  // Subscribers
  // rclcpp::Subscription<ControlCommand>::SharedPtr sub_control_cmds_;

  //!< @brief subscription for current velocity
  rclcpp::Subscription<VelocityMsg>::SharedPtr sub_velocity_;

  //!< @brief subscription for current steering
  rclcpp::Subscription<SteeringReport>::SharedPtr sub_vehicle_steering_;

  //!< @brief subscription for current trajectory
  rclcpp::Subscription<Trajectory>::SharedPtr sub_trajectory_;

  // Publishers
  rclcpp::Publisher<ControlCommand>::SharedPtr pub_control_cmd_;
  rclcpp::Publisher<SysIDSteeringVars>::SharedPtr pub_sysid_debug_vars_;

  /**
   * Node storage
   * */
  // Input messages
  std::shared_ptr<SysIDSteeringVars> current_sysid_vars_{nullptr};
  std::shared_ptr<ControlCommand> current_sysid_cmd_{nullptr};

  // Node Methods
  //!< initialize timer to work in real, simulation, and replay
  void initTimer(double period_s);

  bool isDataReady() const;
  bool updateCurrentPose();

  /**
   * @brief compute and publish the sysid input signals with a constant control period
   */
  void onTimer();

  InputType getInputType(int const &input_id);
  void loadParams(InputType const &input_type);

  void onTrajectory(const autoware_auto_planning_msgs::msg::Trajectory::SharedPtr msg);
  void onVelocity(const nav_msgs::msg::Odometry::SharedPtr msg);
  void onSteering(const autoware_auto_vehicle_msgs::msg::SteeringReport::SharedPtr msg);

  /**
   * @brief Publish message.
   * */
  void publishSysIDCommand();
};

}  // namespace sysid

#endif  // SYSTEM_IDENTIFICATION_INCLUDE_SYSID_NODE_HPP_
