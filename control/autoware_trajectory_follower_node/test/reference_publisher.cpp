#include <chrono>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <autoware_planning_msgs/msg/trajectory.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/accel.hpp>

using namespace std::chrono_literals;

class TrajectoryPublisher : public rclcpp::Node
{
public:
  TrajectoryPublisher()
  : Node("trajectory_publisher")
  {
    publisher_ = this->create_publisher<autoware_planning_msgs::msg::Trajectory>("controller/input/reference_trajectory", 10);
    timer_ = this->create_wall_timer(500ms, std::bind(&TrajectoryPublisher::publish_trajectory, this));
  }

private:
  void publish_trajectory()
  {
    auto message = autoware_planning_msgs::msg::Trajectory();
    message.header.stamp = this->now();
    message.header.frame_id = "map";

    // Create and add trajectory points
    message.points.push_back(create_trajectory_point(-1.0, -1.0, 1.0f));
    message.points.push_back(create_trajectory_point(0.0, 0.0, 1.0f));
    message.points.push_back(create_trajectory_point(1.0, -1.0, 1.0f));
    message.points.push_back(create_trajectory_point(2.0, -2.0, 1.0f));

    RCLCPP_INFO(this->get_logger(), "Publishing trajectory with %zu points", message.points.size());
    publisher_->publish(message);
  }

  autoware_planning_msgs::msg::TrajectoryPoint create_trajectory_point(double x, double y, float velocity)
  {
    autoware_planning_msgs::msg::TrajectoryPoint point;
    point.pose.position.x = x;
    point.pose.position.y = y;
    point.pose.orientation = geometry_msgs::msg::Quaternion();
    point.twist.linear.x = velocity;
    point.accel.linear.x = 0.0;
    return point;
  }

  rclcpp::Publisher<autoware_planning_msgs::msg::Trajectory>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TrajectoryPublisher>());
  rclcpp::shutdown();
  return 0;
}
