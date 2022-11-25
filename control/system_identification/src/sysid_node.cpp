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

  auto const &sys_id_frequency_hz = declare_parameter<double>("identification_frequency", 100.);
  common_input_lib_params_.sysid_dt = 1. / sys_id_frequency_hz;
  initTimer(common_input_lib_params_.sysid_dt);

  // Define input type
  int input_type_id = static_cast<int>(declare_parameter<long>("default_input_class", 0));
  auto input_type = getInputType(input_type_id);
  loadParams(input_type);

  // Publishers
  // Initialize the publishers.
  pub_control_cmd_ = create_publisher<ControlCommand>("~/output/control_cmd", 1);

  pub_sysid_debug_vars_ =
    create_publisher<SysIDSteeringVars>("~/output/system_identification_vars", 1);

  pub_predicted_traj_ = create_publisher<Trajectory>("~/output/predicted_trajectory", 1);

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

  RCLCPP_WARN_SKIPFIRST_THROTTLE(get_logger(), *this->get_clock(), 1000 /*ms*/, "In SYSID onTimer ....");

  if (!isDataReady() || !updateCurrentPose())
  {
    // ns_utils::print("Data is not ready ");
    return;
  }

  /* Publish input messages - for sysid analysis */
  SysIDSteeringVars sysid_vars_msg;
  sysid_vars_msg.sysid_steering_input = 1.;
  current_sysid_vars_ = std::make_shared<SysIDSteeringVars>(sysid_vars_msg);

  // Compute the control signals.
  ControlCommand sysid_cmd_msg;

  auto const &sysid_steering_input = input_wrapper_.generateInput(current_vx_);
  auto const &sysid_acc_input = getLongitudinalControl();

  sysid_cmd_msg.lateral.steering_tire_angle = static_cast<float >(sysid_steering_input);
  sysid_cmd_msg.longitudinal.acceleration = static_cast<float>(sysid_acc_input);
  sysid_cmd_msg.longitudinal.speed = static_cast<float>(sysid_acc_input);

  current_sysid_cmd_ = std::make_shared<ControlCommand>(sysid_cmd_msg);

  // ns_utils::print("Current sysid input ... ", sysid_steering_input);
  publishSysIDCommand();
}
void SystemIdentificationNode::publishSysIDCommand()
{
  current_sysid_vars_->stamp = this->now();
  current_sysid_cmd_->stamp = this->now();

  // publish messages.
  pub_control_cmd_->publish(*current_sysid_cmd_);
  pub_sysid_debug_vars_->publish(*current_sysid_vars_);
  pub_predicted_traj_->publish(*current_trajectory_ptr_);
}

bool SystemIdentificationNode::isDataReady()
{
  if (!current_velocity_ptr_)
  {
    RCLCPP_WARN_SKIPFIRST_THROTTLE(get_logger(),
                                   *this->get_clock(),
                                   1000 /*ms*/,
                                   "Waiting for the  current_velocity = %d",
                                   current_velocity_ptr_ != nullptr);

    return false;
  }

  if (!current_steering_ptr_)
  {
    RCLCPP_WARN_SKIPFIRST_THROTTLE(get_logger(),
                                   *this->get_clock(),
                                   1000 /*ms*/,
                                   "Waiting for the current_steering = %d",
                                   current_steering_ptr_ != nullptr);
    return false;
  }

  if (!current_trajectory_ptr_)
  {
    RCLCPP_WARN_SKIPFIRST_THROTTLE(get_logger(),
                                   *this->get_clock(),
                                   1000 /*ms*/,
                                   " Waiting for the current trajectory = %d",
                                   current_trajectory_ptr_ != nullptr);
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
    RCLCPP_WARN_SKIPFIRST_THROTTLE(get_logger(), *this->get_clock(), 1000 /*ms*/, "%s", ex.what());
    RCLCPP_WARN_SKIPFIRST_THROTTLE(get_logger(), *this->get_clock(), 1000 /*ms*/,
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
  RCLCPP_WARN_SKIPFIRST_THROTTLE(get_logger(), *this->get_clock(), 1000 /*ms*/, "In SYSID onTrajectory ....");
  current_trajectory_ptr_ = msg;
}
void SystemIdentificationNode::onVelocity(nav_msgs::msg::Odometry::SharedPtr const msg)
{
  RCLCPP_WARN_SKIPFIRST_THROTTLE(get_logger(), *this->get_clock(), 1000 /*ms*/, "In SYSID onVelocity ....");
  // ns_utils::print("In on Velocity ....");
  current_velocity_ptr_ = msg;
  current_vx_ = static_cast<double>(current_velocity_ptr_->twist.twist.linear.x);
}

void SystemIdentificationNode::onSteering(autoware_auto_vehicle_msgs::msg::SteeringReport::SharedPtr const msg)
{
  // ns_utils::print("In on Steering ....");
  RCLCPP_WARN_SKIPFIRST_THROTTLE(get_logger(), *this->get_clock(), 1000 /*ms*/, "In SYSID onSteering ....");
  current_steering_ptr_ = msg;
}
void SystemIdentificationNode::loadParams(InputType const &input_type)
{

  // Load common parameters.
  common_input_lib_params_.maximum_amplitude = declare_parameter<double>("common_variables.signal_magnitude", 0.);

  auto const &vstart = declare_parameter<double>("common_variables.min_speed", 0.);
  auto const &vmax = declare_parameter<double>("common_variables.max_speed", 0.);

  common_input_lib_params_.minimum_speed = sysid::kmh2ms(vstart);
  common_input_lib_params_.maximum_speed = sysid::kmh2ms(vmax);

  common_input_lib_params_.tstart = declare_parameter<double>("common_variables.time_start_after", 0.);
  auto const &ctrl_period = common_input_lib_params_.sysid_dt;

  // Longitudinal control parameters.
  common_input_lib_params_.target_speed = declare_parameter<double>("target_speed", 1.) / 3.6;  // [m/s]
  common_input_lib_params_.p_of_pid = declare_parameter<double>("pcontrol_coeff", 0.1); // [km/h]

  if (input_type == InputType::STEP)
  {
    sysid::sStepParameters step_params;
    step_params.start_time = common_input_lib_params_.tstart;
    step_params.max_amplitude = common_input_lib_params_.maximum_amplitude;

    step_params.step_period = declare_parameter<double>("step_input_params.step_period", 1.);
    step_params.step_direction_flag = declare_parameter<int8_t>("step_input_params.step_direction", 0);

    sysid::InpStepUpDown step_up_down_input_type(common_input_lib_params_.minimum_speed,
                                                 common_input_lib_params_.maximum_speed,
                                                 step_params);
    // Change the input wrapper class.
    input_wrapper_ = sysid::InputWrapper{step_up_down_input_type};
    return;
  }

  if (input_type == InputType::PRBS)
  {
    // const size_t prbs_n = static_cast<size_t>( node_->declare_parameter<int>("prbs_input_params.prbs_type_n", 8));
    const double estimated_rise_time = declare_parameter<double>("prbs_input_params.estimated_rise_time", 0.5);

    sysid::sPRBSparams prbs_params(common_input_lib_params_.tstart,
                                   estimated_rise_time,
                                   ctrl_period,
                                   PRBS_N);

    prbs_params.max_amplitude = common_input_lib_params_.maximum_amplitude;

    sysid::InpPRBS<PRBS_N> prbs_input_type(common_input_lib_params_.minimum_speed,
                                           common_input_lib_params_.maximum_speed,
                                           prbs_params);

    // Change the input wrapper class.
    input_wrapper_ = sysid::InputWrapper{prbs_input_type};
    return;
  }
  if (input_type == InputType::FWNOISE)
  {

    auto const &lowpass_filter_cutoff_hz = declare_parameter<double>("fwnoise_input_params.noise_cutoff_frq_hz", 5.);

    auto const &lowpass_filter_order = declare_parameter<int>("fwnoise_input_params.lowpass_filter_order", 3);
    auto const &noise_mean = declare_parameter<double>("fwnoise_input_params.noise_mean", 0.);
    auto const &noise_std = declare_parameter<double>("fwnoise_input_params.noise_std", 0.1);

    sysid::sFilteredWhiteNoiseParameters params{};
    params.start_time = common_input_lib_params_.tstart;
    params.cutoff_frequency_hz = lowpass_filter_cutoff_hz;
    params.sampling_frequency_hz = 1. / ctrl_period;
    params.max_amplitude = common_input_lib_params_.maximum_amplitude;
    params.filter_order = static_cast<int>(lowpass_filter_order);
    params.noise_mean = noise_mean;
    params.noise_stddev = noise_std;

    sysid::InpFilteredWhiteNoise filtered_white_noise_input_type(common_input_lib_params_.minimum_speed,
                                                                 common_input_lib_params_.maximum_speed,
                                                                 params);

    // Change the input wrapper class.
    input_wrapper_ = sysid::InputWrapper{filtered_white_noise_input_type};
    return;
  }

  if (input_type == InputType::SUMSINs)
  {
    sysid::sSumOfSinParameters params;
    params.start_time = common_input_lib_params_.tstart;
    auto const &temp_frequency_band = declare_parameter<std::vector<double>>("sumof_sinusoids_params"
                                                                             ".frequency_band_hz",
                                                                             std::vector<double>{1., 10.});

    params.frequency_band = std::array<double, 2>{temp_frequency_band[0], temp_frequency_band[1]};
    params.num_of_sins = declare_parameter<int>("sumof_sinusoids_params.num_of_sinuses", 5);


    // noise definitions
    params.add_noise = declare_parameter<bool>("sumof_sinusoids_params.add_noise", false);
    params.noise_mean = declare_parameter<double>("sumof_sinusoids_params.noise_mean", 0.);
    params.noise_stddev = declare_parameter<double>("sumof_sinusoids_params.noise_std", 0.01);

    params.max_amplitude = common_input_lib_params_.maximum_amplitude;

    sysid::InpSumOfSinusoids inp_sum_of_sinusoids_type(common_input_lib_params_.minimum_speed,
                                                       common_input_lib_params_.maximum_speed,
                                                       params);

    // Change the input wrapper class.
    input_wrapper_ = sysid::InputWrapper{inp_sum_of_sinusoids_type};
  }

}
InputType SystemIdentificationNode::getInputType(int const &input_id)
{
  if (input_id == 0)
  { return InputType::IDENTITY; }

  if (input_id == 1)
  { return InputType::STEP; }

  if (input_id == 2)
  { return InputType::PRBS; }

  if (input_id == 3)
  { return InputType::FWNOISE; }

  if (input_id == 4)
  { return InputType::SUMSINs; }

  return InputType::IDENTITY;
}

/**
 * @brief Longitudinal control signal.
 * */
double SystemIdentificationNode::getLongitudinalControl() const
{
  auto const &target_vx_ms = common_input_lib_params_.target_speed;
  auto const &error_vx = current_vx_ - target_vx_ms;
  auto const &K = common_input_lib_params_.p_of_pid;
  return -K * error_vx;
}

}  // namespace sysid

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(sysid::SystemIdentificationNode)
