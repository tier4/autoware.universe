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

  // Subscribers

  // Publishers
  pub_sysid_debug_vars_ =
    create_publisher<SysIDSteeringVars>("~/output/system_identification/lateral_cmd", 1);
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
  return false;
}
bool SystemIdentificationNode::updateCurrentPose()
{
  return false;
}

}  // namespace sysid

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(sysid::SystemIdentificationNode)
