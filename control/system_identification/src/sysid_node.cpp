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

#include "sysid_node.hpp"

namespace sysid
{

sysid::SystemIdentificationNode::SystemIdentificationNode(const rclcpp::NodeOptions &node_options)
  : Node("system_dentification_node", node_options)
{
  using std::placeholders::_1;

  initTimer(common_input_lib_params_.sysid_dt);

  // Publishers
  // Initialize the publishers.
  pub_control_cmd_ = create_publisher<ControlCommand>("~/output/control_cmd", 1);

  pub_sysid_debug_vars_ =
    create_publisher<SysIDSteeringVars>("~/output/system_identification/lateral_cmd", 1);

  // Subscribers
  sub_trajectory_ = create_subscription<Trajectory>("~/input/reference_trajectory", rclcpp::QoS{1},
                                                    std::bind(&SystemIdentificationNode::onTrajectory, this, _1));

  sub_velocity_ = create_subscription<VelocityMsg>("~/input/current_velocity", rclcpp::QoS{1},
                                                   std::bind(&SystemIdentificationNode::onVelocity, this, _1));

  sub_vehicle_steering_ = create_subscription<SteeringReport>("~/input/current_steering", rclcpp::QoS{1},
                                                              std::bind(&SystemIdentificationNode::onSteering,
                                                                        this, _1));
}

void SystemIdentificationNode::initTimer(double period_s)
{
  const auto period_ns =
    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(period_s));

  timer_ = rclcpp::create_timer(
    this, get_clock(), period_ns, std::bind(&SystemIdentificationNode::onTimer, this));
}

/***
 * @brief timer callback.
 */
void SystemIdentificationNode::onTimer()
{
  RCLCPP_INFO(this->get_logger(), "%s", "in onTimer ....");
  ns_utils::print("In on Timer ....");

  /** Publish input message*/
  SysIDSteeringVars msg;
  msg.sysid_steering_input = 1.;

  current_sysid_input_cmd_ = std::make_shared<SysIDSteeringVars>(msg);
  publishSysIDCommand();
}
void SystemIdentificationNode::publishSysIDCommand()
{
  current_sysid_input_cmd_->stamp = this->now();
  pub_sysid_debug_vars_->publish(*current_sysid_input_cmd_);
}
bool SystemIdentificationNode::isDataReady() const
{
  if (!current_velocity_ptr_)
  {
    RCLCPP_DEBUG(get_logger(), "Waiting for the  current_velocity = %d",
                 current_velocity_ptr_ != nullptr);
    return false;
  }

  if (!current_steering_ptr_)
  {
    RCLCPP_DEBUG(get_logger(), "Waiting for the current_steering = %d", current_steering_ptr_ != nullptr);
    return false;
  }

  if (!current_trajectory_ptr_)
  {
    RCLCPP_DEBUG(get_logger(), " Waiting for the current trajectory = %d", current_trajectory_ptr_ != nullptr);
    return false;
  }

  return true;
}
bool SystemIdentificationNode::updateCurrentPose()
{
  geometry_msgs::msg::TransformStamped transform;
  try
  {
    transform =
      m_tf_buffer_.lookupTransform(current_trajectory_ptr_->header.frame_id, "base_link", tf2::TimePointZero);
  } catch (tf2::TransformException &ex)
  {
    RCLCPP_WARN_SKIPFIRST_THROTTLE(get_logger(), *this->get_clock(), 5000 /*ms*/, "%s", ex.what());
    RCLCPP_WARN_SKIPFIRST_THROTTLE(get_logger(), *this->get_clock(), 5000 /*ms*/,
                                   "%s", m_tf_buffer_.allFramesAsString().c_str());
    return false;
  }
  geometry_msgs::msg::PoseStamped ps;
  ps.header = transform.header;
  ps.pose.position.x = transform.transform.translation.x;
  ps.pose.position.y = transform.transform.translation.y;
  ps.pose.position.z = transform.transform.translation.z;
  ps.pose.orientation = transform.transform.rotation;
  current_pose_ptr_ = std::make_shared<geometry_msgs::msg::PoseStamped>(ps);

  return true;

}
void SystemIdentificationNode::onTrajectory(autoware_auto_planning_msgs::msg::Trajectory::SharedPtr const msg)
{

  ns_utils::print("In on Trajectory ....");
  current_trajectory_ptr_ = msg;
}
void SystemIdentificationNode::onVelocity(nav_msgs::msg::Odometry::SharedPtr const msg)
{
  ns_utils::print("In on Velocity ....");
  current_velocity_ptr_ = msg;
}
void SystemIdentificationNode::onSteering(autoware_auto_vehicle_msgs::msg::SteeringReport::SharedPtr const msg)
{
  ns_utils::print("In on Steering ....");
  current_steering_ptr_ = msg;
}

}  // namespace sysid

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(sysid::SystemIdentificationNode)
