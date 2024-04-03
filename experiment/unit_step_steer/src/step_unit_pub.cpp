#include "rclcpp/rclcpp.hpp"
#include "autoware_auto_control_msgs/msg/ackermann_control_command.hpp"
#include "autoware_adapi_v1_msgs/msg/operation_mode_state.hpp" // src/autoware/autoware_adapi_msgs/autoware_adapi_v1_msgs/operation_mode/msg/OperationModeState.msg
#include <limits>
#include <chrono>
#include "rclcpp/qos.hpp"


class ControlCommandPublisher : public rclcpp::Node
{
public:
    ControlCommandPublisher() 
    : Node("control_command_publisher"), start_time_(this->now())
    {
        auto qos = rclcpp::QoS(rclcpp::KeepLast(10));
        qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
        qos.durability(RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL);
        publisher_ = this->create_publisher<autoware_auto_control_msgs::msg::AckermannControlCommand>("/control/trajectory_follower/control_cmd", qos);
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&ControlCommandPublisher::publish_control_command, this));
        // Set up the subscriber for the operation mode state
        operation_mode_subscriber_ = this->create_subscription<autoware_adapi_v1_msgs::msg::OperationModeState>(
            "/api/operation_mode/state",
            10,
            std::bind(&ControlCommandPublisher::operation_mode_callback, this, std::placeholders::_1));
    }

private:
    void publish_control_command()
    {
        auto control_command = autoware_auto_control_msgs::msg::AckermannControlCommand();

        if (operation_mode_ != 2) {
            control_command.lateral.steering_tire_angle = 0.0;
            control_command.lateral.steering_tire_rotation_rate = 0.0;
            control_command.longitudinal.acceleration = 0.0;
            control_command.longitudinal.jerk = 0.0;
            control_command.longitudinal.speed = 0.001;
            start_time_ = this->get_clock()->now();
        } else {
            if ((this->get_clock()->now() - start_time_).seconds() < 10.0) {
                // For the first 10 seconds, publish zeros
                control_command.lateral.steering_tire_angle = 0.0;
                control_command.lateral.steering_tire_rotation_rate = 0.0;
                control_command.longitudinal.acceleration = 0.0;
                control_command.longitudinal.jerk = 0.0;
                control_command.longitudinal.speed = 0.001;
            } else {
                // After 10 seconds, publish the actual values
                control_command.lateral.steering_tire_angle = 0.5;
                control_command.lateral.steering_tire_rotation_rate = std::numeric_limits<double>::infinity();
                control_command.longitudinal.acceleration = 0.0; // Set to your desired value
                control_command.longitudinal.jerk = 0.0; // Set to your desired value
                control_command.longitudinal.speed = 0.001; // Set to your desired value
            }
        }

        // Set the time stamp
        control_command.stamp = this->get_clock()->now();
        control_command.lateral.stamp = this->get_clock()->now();
        control_command.longitudinal.stamp = this->get_clock()->now();

        // RCLCPP_INFO(this->get_logger(), "Publishing control command at time '%u.%09u'", control_command.stamp.sec, control_command.stamp.nanosec);
        publisher_->publish(control_command);
    }
    
    void operation_mode_callback(const autoware_adapi_v1_msgs::msg::OperationModeState::SharedPtr msg)
    {
        operation_mode_ = msg->mode;
    }

    rclcpp::Publisher<autoware_auto_control_msgs::msg::AckermannControlCommand>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Time start_time_;
    rclcpp::Subscription<autoware_adapi_v1_msgs::msg::OperationModeState>::SharedPtr operation_mode_subscriber_;
    int operation_mode_{0};
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ControlCommandPublisher>());
    rclcpp::shutdown();
    return 0;
}
